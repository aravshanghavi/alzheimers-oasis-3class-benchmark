"""Shared conventions for the exploratory experiments added on 29 September 2026.

WHY THIS MODULE EXISTS
----------------------
a25 through a38 were written in one night against a deadline. Each of them needs
the same four things: an AUC that handles ties correctly, metadata models whose
preprocessing does not see the held-out fold, participant pooling that is checked
rather than assumed, and a bootstrap that pairs every model on the same draw. If
each script carried its own copy, the copies would drift, and the paper would
print two numbers for the same quantity. That already happened once in this
project, with the loss-share ratio, and it cost a day of reconciliation.

So the machinery lives here, once, and every new script imports it.

WHAT IS REUSED RATHER THAN REBUILT
----------------------------------
a23_protocol_audit.py already carries Cohort, MetaFitter, the null collector and
the small frame helpers, and its MetaFitter is already fold honest. Those are
imported here rather than copied, so a fix in a23 reaches every script. a23
guards its main() behind __name__, so importing it runs nothing.

WHAT IS NEW HERE
----------------
midrank_auc        a17's AUC gives tied scores distinct ranks, which makes a
                   constant predictor score 0.4958 instead of 0.5000. That is
                   defect D3. This version uses midranks and is asserted against
                   the constant case in __main__.
stratified_participant_bootstrap
                   paired AUC differences need resampling WITHIN class strata,
                   because an unstratified draw can empty the 23-participant
                   CDR >= 1 class and make the AUC undefined.
cnn_features       the two log-probability-ratio features used when a network
                   score is stacked into a metadata model. This is a stacked
                   CNN score, never the network itself, and the label matters.
changelog_entry    every one of these experiments is exploratory and declared
                   after the primary results were known. Each script prints the
                   exact line the first author must paste into
                   REGISTRY_CHANGELOG.md, so the declaration cannot be
                   paraphrased differently in five places.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence as _ABCSequence
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analysis._determinism import seeded_rng                                    # noqa: E402
from analysis.a23_protocol_audit import (Cohort, MetaFitter, confusion,         # noqa: E402
                                         constant_frame, find_full_tree,
                                         null_replicates, participant_frame,
                                         per_class_f1, prob_frame,
                                         FULL_EXPERIMENTS,
                                         EXTENDED_RESTRICTED_EXPERIMENTS,
                                         ENV_INCLUDE_NEW_ARMS,
                                         include_new_arms_requested,
                                         resolve_restricted_experiments,
                                         PERMNULL_TREE, ARM_ORDER)
from analysis.a23_protocol_audit import \
    RESTRICTED_EXPERIMENTS as DEFAULT_RESTRICTED_EXPERIMENTS                    # noqa: E402

EPS = 1e-12
DECLARED_ON = "added 29 Sep 2026 after the primary results were known, exploratory"


# ---------------------------------------------------------------------------
# Which restricted experiments the importing scripts see
# ---------------------------------------------------------------------------
#
# Fifteen scripts do `Cohort("restricted", RESTRICTED_EXPERIMENTS, ...)`. Giving
# each of them its own copy of the resolution logic is how the two spellings of
# the loss-share ratio happened, so the decision is made once, here.
#
# The name below is resolved WHEN IT IS READ rather than when this module is
# imported. That is what lets a script's own --include-new-arms flag take effect:
# the flag is parsed after the import has already happened, and rebinding a name
# a script imported by value would not reach the call site. Reading it late also
# keeps the warning about a missing experiment next to the run that triggers it
# instead of at import time.
#
# The default is DEFAULT_RESTRICTED_EXPERIMENTS, exactly the four experiments
# a23 has always named, in that order.

_INCLUDE_NEW_ARMS: Optional[bool] = None
_RESOLVED: Dict[bool, List[str]] = {}


def set_include_new_arms(flag: Optional[bool] = True) -> None:
    """Ask for the three 29 September arms for the rest of this process.

    A false or missing flag is NOT a refusal. It leaves the decision to
    SREP_INCLUDE_NEW_ARMS, so a script may call
    set_include_new_arms(args.include_new_arms) unconditionally without
    silently overriding the variable its read-out was launched with.
    """
    global _INCLUDE_NEW_ARMS
    if flag:
        _INCLUDE_NEW_ARMS = True


def restricted_experiments(include_new_arms: Optional[bool] = None) -> List[str]:
    """The restricted-cohort experiment list, default unless extension is asked.

    Resolved once per process for each answer, so the skip warnings print once
    rather than on every read of RESTRICTED_EXPERIMENTS.
    """
    flag = bool(_INCLUDE_NEW_ARMS if include_new_arms is None else include_new_arms)
    if flag not in _RESOLVED:
        _RESOLVED[flag] = resolve_restricted_experiments(flag)
    return list(_RESOLVED[flag])


class _RestrictedExperiments(_ABCSequence):
    """A list of experiment names that reads its contents at the last moment.

    Behaves as a plain sequence of strings: iterate it, index it, len it, or
    compare it with a list. It is not mutable, because the only thing allowed to
    change what it holds is set_include_new_arms or the environment variable.
    """

    def __getitem__(self, i):
        return restricted_experiments()[i]

    def __len__(self) -> int:
        return len(restricted_experiments())

    def __iter__(self):
        return iter(restricted_experiments())

    def __contains__(self, item) -> bool:
        return item in restricted_experiments()

    def __eq__(self, other) -> bool:
        try:
            return restricted_experiments() == list(other)
        except TypeError:
            return NotImplemented

    def __ne__(self, other) -> bool:
        eq = self.__eq__(other)
        return eq if eq is NotImplemented else not eq

    def __hash__(self):
        return hash(tuple(restricted_experiments()))

    def __add__(self, other):
        return restricted_experiments() + list(other)

    def __repr__(self) -> str:
        return repr(restricted_experiments())


RESTRICTED_EXPERIMENTS = _RestrictedExperiments()


# ---------------------------------------------------------------------------
# AUC with midranks
# ---------------------------------------------------------------------------

def midranks(x: np.ndarray) -> np.ndarray:
    """Ranks from 1, with tied values sharing the average of their positions."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    r = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def midrank_auc(scores: np.ndarray, positive: np.ndarray) -> float:
    """AUC of `scores` for separating `positive` from the rest, ties handled.

    Returns NaN when either class is empty, which happens in an unstratified
    bootstrap draw on the 23-participant CDR >= 1 class. Callers that cannot
    tolerate NaN should use stratified_participant_bootstrap.
    """
    s = np.asarray(scores, dtype=float)
    pos = np.asarray(positive, dtype=bool)
    n1 = int(pos.sum())
    n0 = int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = midranks(s)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def pairwise_auc(scores: np.ndarray, labels: np.ndarray,
                 positive_class: int, negative_class: int) -> float:
    """AUC restricted to two label classes, everything else dropped."""
    labels = np.asarray(labels)
    m = (labels == positive_class) | (labels == negative_class)
    if not m.any():
        return float("nan")
    return midrank_auc(np.asarray(scores, dtype=float)[m], labels[m] == positive_class)


# ---------------------------------------------------------------------------
# Stacked network score
# ---------------------------------------------------------------------------

def cnn_features(probs: np.ndarray) -> np.ndarray:
    """log(p1/p0) and log(p2/p0), the two free coordinates of a 3-class simplex.

    Clipped at EPS on both sides before the ratio. A participant whose mean
    probability for the reference class underflows would otherwise contribute an
    infinity that silently dominates the standardisation.
    """
    p = np.clip(np.asarray(probs, dtype=float), EPS, 1.0)
    return np.column_stack([np.log(p[:, 1] / p[:, 0]), np.log(p[:, 2] / p[:, 0])])


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def stratified_participant_bootstrap(labels: np.ndarray, n_boot: int, tag: str,
                                     base_seed: int = 12345,
                                     **parts) -> np.ndarray:
    """Index matrix of shape (n_boot, n) resampling participants WITHIN class.

    Every model is scored on the same row, so differences stay paired. Stratifying
    keeps each class's count fixed at its observed value, which is what makes an
    AUC difference well defined on a cohort holding 23 CDR >= 1 participants.

    The seed comes from _determinism.seeded_rng, so the stream depends only on the
    tag and its parts, never on how many draws another call site has taken.
    """
    labels = np.asarray(labels)
    n = len(labels)
    rng = seeded_rng(tag, base_seed, **parts)
    strata = [np.flatnonzero(labels == k) for k in np.unique(labels)]
    out = np.empty((n_boot, n), dtype=np.int64)
    for b in range(n_boot):
        pos = 0
        for idx in strata:
            m = len(idx)
            out[b, pos:pos + m] = rng.choice(idx, size=m, replace=True)
            pos += m
    return out


def unstratified_participant_bootstrap(n: int, n_boot: int, tag: str,
                                       base_seed: int = 12345, **parts) -> np.ndarray:
    """Plain participant-clustered resample, for metrics that tolerate it."""
    rng = seeded_rng(tag, base_seed, **parts)
    return rng.integers(0, n, size=(n_boot, n))


def percentile_interval(draws: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    lo, hi = np.nanpercentile(draws, [100 * alpha / 2.0, 100 * (1 - alpha / 2.0)])
    return float(lo), float(hi)


def bootstrap_p(draws: np.ndarray) -> Dict[str, object]:
    """Two-sided bootstrap p, floored at 1/B, with the floor flagged.

    A bootstrap over B resamples cannot resolve a probability below 1/B, so a
    literal 0.0000 in a table is a resolution artefact rather than a finding.
    This mirrors _common.paired_difference so the two never disagree.
    """
    d = np.asarray(draws, dtype=float)
    d = d[~np.isnan(d)]
    b = len(d)
    if b == 0:
        return {"p_bootstrap": float("nan"), "p_at_resolution_floor": False, "n_bootstrap": 0}
    raw = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
    return {"p_bootstrap": float(min(max(raw, 1.0 / b), 1.0)),
            "p_at_resolution_floor": bool(raw < 1.0 / b),
            "n_bootstrap": int(b)}


# ---------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------

def changelog_entry(experiment_id: str, comparison: str, metric: str,
                    unit: str, when: str = "2026-09-29") -> str:
    """The exact REGISTRY_CHANGELOG.md line the author must paste.

    Printed rather than written. Nothing here should edit the registry on its
    own: a declaration the author did not make is not a declaration.
    """
    return ("\n".join([
        f"## {when} {experiment_id}",
        f"- comparison: {comparison}",
        f"- metric: {metric}",
        f"- unit: {unit}",
        f"- status: {DECLARED_ON}",
    ]))


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    const = np.ones(50)
    pos = np.zeros(50, dtype=bool)
    pos[:17] = True
    got = midrank_auc(const, pos)
    assert got == 0.5, f"constant predictor must score exactly 0.5, got {got!r}"

    perfect = np.where(pos, 1.0, 0.0)
    assert midrank_auc(perfect, pos) == 1.0
    assert midrank_auc(-perfect, pos) == 0.0

    s = rng.normal(size=200)
    p = rng.random(200) < 0.3
    naive_n1 = int(p.sum()); naive_n0 = 200 - naive_n1
    wins = sum(1.0 if a > b else 0.5 if a == b else 0.0
               for a in s[p] for b in s[~p]) / (naive_n1 * naive_n0)
    assert abs(midrank_auc(s, p) - wins) < 1e-12, "midrank AUC must equal the win fraction"

    tied = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
    assert np.allclose(midranks(tied), [1.5, 1.5, 3.5, 3.5, 5.0])

    lab = np.array([0] * 85 + [1] * 58 + [2] * 23)
    B = stratified_participant_bootstrap(lab, 7, "selftest")
    assert B.shape == (7, 166)
    for row in B:
        c = np.bincount(lab[row], minlength=3)
        assert tuple(c) == (85, 58, 23), f"strata not preserved: {c}"
    again = stratified_participant_bootstrap(lab, 7, "selftest")
    assert np.array_equal(B, again), "the same tag must give the same draws"
    other = stratified_participant_bootstrap(lab, 7, "selftest", cohort="full")
    assert not np.array_equal(B, other), "different parts must give different draws"

    f = cnn_features(np.array([[0.5, 0.3, 0.2], [1.0, 0.0, 0.0]]))
    assert np.isfinite(f).all(), "cnn_features must never emit inf"

    assert list(RESTRICTED_EXPERIMENTS) == list(DEFAULT_RESTRICTED_EXPERIMENTS), \
        "the default restricted list must be a23's four experiments, in order"
    assert RESTRICTED_EXPERIMENTS == list(DEFAULT_RESTRICTED_EXPERIMENTS)
    assert len(RESTRICTED_EXPERIMENTS) == 4 and "exp10_age60_main" in RESTRICTED_EXPERIMENTS
    assert RESTRICTED_EXPERIMENTS[0] == "exp10_age60_main"

    print("_protocol_lib self test: all assertions passed")
