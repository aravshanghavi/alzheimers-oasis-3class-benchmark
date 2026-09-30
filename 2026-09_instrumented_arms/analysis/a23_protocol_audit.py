#!/usr/bin/env python3
"""a23: the paper's proposed evaluation protocol, run as one audit.

    python analysis/a23_protocol_audit.py
    python analysis/a23_protocol_audit.py --quick        # 200 resamples, smoke test
    python analysis/a23_protocol_audit.py --boot 5000

WHY THIS SCRIPT EXISTS
----------------------
The manuscript proposes five requirements that any imaging-only dementia
classifier should have to clear before its numbers are believed. Those five
requirements are currently spread over a17, a18, a20, a21 and a22, each with
its own cohort, its own unit of analysis and its own output file. A reviewer
who wants to check that the paper applies its own protocol has to reconcile
five scripts by hand, and so does the next person who adds an arm.

So the protocol is implemented once, here, and every arm is put through all
five requirements on both cohorts with one verdict per cell.

    R1  Does the arm beat a constant majority-class predictor?
    R2  Does it beat the MEASURED permuted-label null?
    R3  Do the participant unit and the image unit agree?
    R4  Does it beat cross-validated metadata regressions on the same folds?
    R5  Does its advantage survive restricting the cohort to who a clinician
        would actually have assessed?

WHAT THIS SCRIPT IS NOT
-----------------------
It is not a replacement for a20, which owns the registered objective and
gamma-control comparisons, or for a21, which owns the age decomposition, or
for a22, which owns the permutation null. Where one of those already computes
a number, this script reads or reproduces it rather than inventing a second
version. Nothing here writes to any file those scripts own, and nothing here
calls REGISTRY.write_audit(): re-registering comparisons another script already
registered would inflate the declared family counts and turn a clean audit into
a protocol violation.

THE TWO CODE FINGERPRINTS
-------------------------
The full-cohort arms were produced by one tree (8378f120) and the restricted
arms by another (cb7ee123). _common.runs_frame refuses to mix fingerprints,
which is the right default for a results table and the wrong behaviour for an
audit whose whole point is to put the two cohorts side by side. So this script
does not call runs_frame. It loads each cohort separately, records the
fingerprint of every arm in the output, and prints a warning naming which arms
came from which tree. A cross-fingerprint comparison is labelled as one in the
table rather than being silently made or silently refused.

MULTIPLICITY
------------
A comparison that belongs to a declared registry family gets that family's
corrected interval through paired_difference(family=...). A comparison that
does not belong to one gets a nominal 95 percent interval and the string
"undeclared" in the family column, so that no reader can mistake it for a
pre-registered test. Several requirements here are computed at BOTH units of
analysis; only the unit the declared family was actually realised at keeps the
family, because counting a second unit into the same declared size would break
the declaration the registry exists to hold.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import analysis.discover as dsc                                              # noqa: E402
from analysis._common import (N_CLASSES, SHORT, SuffStats, clustered_bootstrap,  # noqa: E402
                              paired_difference, pooled_predictions, save_table)
from analysis._determinism import seed_for                                    # noqa: E402
from analysis._paths import EXPERIMENTS_DIR, OUT, metadata_xlsx               # noqa: E402
from analysis._registry import REGISTRY                                       # noqa: E402

SOURCE = "analysis/a23_protocol_audit.py"
N_BOOT = 2000
QUICK_BOOT = 200
BASE_SEED = 12345

CONSTANT = "Constant majority"
FA = "FA-FL"

# The restricted cohort is exp10-exp13 in THIS tree. The full cohort lives in
# the 2026-08 tree; --full-from overrides the search.
RESTRICTED_EXPERIMENTS = ["exp10_age60_main", "exp11_age60_gamma137",
                          "exp12_age60_gamma200", "exp13_age60_gamma269"]
FULL_EXPERIMENTS = ["exp01_main_sweep", "exp02_class_balanced",
                    "exp07_focal_gamma137", "exp08_focal_gamma200"]
FULL_TREE_CANDIDATES = ["2026-08_srep_revision_participant_level",
                        "2026-09_tier1_verified_stack"]
PERMNULL_TREE = "2026-09_permutation_null"

# The three arms added on 29 September 2026: exp15 the unweighted reference,
# exp16 the initialisation-variance replicate set, exp17 the multi-slice arm.
# They are OPT IN and the default list above is never touched, because every
# table from t20 to t42 was computed from those four experiments and has to stay
# reproducible byte for byte. Ask for these with --include-new-arms, or by
# exporting SREP_INCLUDE_NEW_ARMS=1, which is the route that works through
# run_musts.py and run_shoulds.py: those two forward only --quick to the scripts
# they launch, so a command-line flag cannot reach a23 through them.
EXTENDED_RESTRICTED_EXPERIMENTS = ["exp15_age60_unweighted",
                                   "exp16_age60_init_variance",
                                   "exp17_age60_multislice"]
ENV_INCLUDE_NEW_ARMS = "SREP_INCLUDE_NEW_ARMS"

# arm_label adds a qualifier only when a run departs from the configuration the
# published tables were built from. Every COMPLETE run of exp10 to exp13 and of
# exp01, exp02, exp07 and exp08 has init_seed 42 and one context slice and sits
# in LABEL_PLAIN_EXPERIMENTS, so all three qualifiers stay empty there and every
# published arm label is unchanged.
DEFAULT_INIT_SEED = 42
DEFAULT_CONTEXT_SLICES = 1
LABEL_PLAIN_EXPERIMENTS = frozenset(RESTRICTED_EXPERIMENTS) | frozenset(FULL_EXPERIMENTS)

ARM_ORDER = ["Weighted CE", "LDAM", "Class-Balanced", "Focal g=1.37",
             "Focal g=2.00", "Focal g=2.69", "Focal g=3.00", FA]

# Requirement 4. Every model is fitted on the SAME grouped folds as the CNNs,
# with a17's softmax regression, so that the contrast is a contrast of
# information and not of estimator or of split.
META_MODELS: List[Tuple[str, List[str]]] = [
    ("META age only", ["Age"]),
    ("META sex only", ["M/F"]),
    ("META age+sex", ["Age", "M/F"]),
    ("META eTIV only", ["eTIV"]),
    ("META nWBV only", ["nWBV"]),
    ("META MMSE only", ["MMSE"]),
    ("META age+sex+eTIV+nWBV", ["Age", "M/F", "eTIV", "nWBV"]),
    ("META age+sex+eTIV+nWBV+MMSE", ["Age", "M/F", "eTIV", "nWBV", "MMSE"]),
]
META_REFERENCE = "META age+sex+eTIV+nWBV"

# A metadata reference is NOT INTERPRETABLE when more than this fraction of any one
# of its predictors had to be imputed. The rule was already in this file, inside the
# console print; it is named here so that R4 and the printed legend both use the one
# definition instead of the verdict quietly following a different rule from the text.
NOT_INTERPRETABLE_IMPUTED_FRACTION = 0.2

METRICS = ["accuracy", "macro_f1", "ece", "brier"]
UNITS = ["participant", "image"]

UNDECLARED = "undeclared"

# Requirement 5. The declared family was realised at the image level on accuracy, and
# only for the arms a21 has a row for. Every R5 row says which of the three cases it is
# in rather than letting a fallback look like a corrected interval.
R5_FAMILY = "age60_train_matched_vs_evaluated"
R5_BASIS_DECLARED = f"declared family {R5_FAMILY}, corrected level"
R5_BASIS_FALLBACK = "FALLBACK nominal 95 percent, no a21 row for this arm"
R5_BASIS_OFF_CELL = ("nominal 95 percent, family realised only at unit image, "
                     "metric accuracy")
R5_BASIS_NA = "not applicable, no full-cohort counterpart"
R5_BASIS_CONTEXT = "not applicable, context row, never differenced"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def warn(msg: str) -> None:
    print(f"  !! {msg}")


def include_new_arms_requested(flag: Optional[bool] = None) -> bool:
    """Whether the caller asked for the three 29 September arms.

    A true flag wins. A false or missing flag falls through to
    SREP_INCLUDE_NEW_ARMS, so a script can pass its own flag unconditionally
    without overriding the variable the read-out was launched with.
    """
    if flag:
        return True
    raw = os.environ.get(ENV_INCLUDE_NEW_ARMS, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def has_complete_runs(experiments_dir: Path, experiment: str) -> bool:
    """Whether one experiment folder holds at least one COMPLETE run.

    Read straight from the manifests rather than through discover.load_runs,
    which raises on duplicate specs. A check meant to keep the flag safe to pass
    while the GPU is still writing cannot itself be a way to crash.
    """
    root = Path(experiments_dir) / experiment / "outputs"
    if not root.is_dir():
        return False
    for mf in sorted(root.glob("*/manifest.json")):
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:                                   # noqa: BLE001
            continue
        if m.get("status") == "COMPLETE":
            return True
    return False


def resolve_restricted_experiments(include_new_arms: Optional[bool] = None,
                                   experiments_dir: Optional[Path] = None,
                                   verbose: bool = True) -> List[str]:
    """RESTRICTED_EXPERIMENTS unchanged by default, plus the new arms when asked.

    Returning the default list untouched is the whole contract of this function.
    An experiment named in EXTENDED_RESTRICTED_EXPERIMENTS with no COMPLETE run
    is skipped with a warning, never raised on, so --include-new-arms can be
    passed before the queue has finished and still produce the ordinary tables.
    """
    if not include_new_arms_requested(include_new_arms):
        return list(RESTRICTED_EXPERIMENTS)
    root = Path(experiments_dir) if experiments_dir is not None else EXPERIMENTS_DIR
    out = list(RESTRICTED_EXPERIMENTS)
    for exp in EXTENDED_RESTRICTED_EXPERIMENTS:
        if has_complete_runs(root, exp):
            out.append(exp)
        elif verbose:
            warn(f"include-new-arms: {exp} has no COMPLETE run under {root}; skipping it")
    if verbose and out == list(RESTRICTED_EXPERIMENTS):
        warn("include-new-arms was requested but none of the new experiments has a "
             "COMPLETE run yet, so every table below is the four-experiment table")
    return out


def imputed_fraction(miss: Dict[str, int], n: int) -> float:
    """The worst single predictor's imputed share, which is what the rule reads."""
    return (max(miss.values()) / n) if (miss and n) else 0.0


def not_interpretable(miss: Dict[str, int], n: int) -> bool:
    """The file's own rule, stated once instead of twice.

    More than NOT_INTERPRETABLE_IMPUTED_FRACTION of any one predictor imputed makes
    the model mostly a constant in that predictor, so its score measures the
    imputation and not the predictor. A model the script refuses to interpret is not
    allowed to decide an R4 verdict.
    """
    return imputed_fraction(miss, n) > NOT_INTERPRETABLE_IMPUTED_FRACTION


def experiment_tag(experiment: str) -> str:
    """exp16_age60_init_variance -> exp16. Short enough to read in a table cell."""
    return str(experiment).split("_", 1)[0]


def arm_label(manifest: Dict) -> str:
    """One name per objective, identical across both cohorts.

    a20 and a22 use these strings. Inventing a second spelling here would make
    the two tables impossible to join, which is the failure mode this audit is
    supposed to remove rather than add to.

    The objective alone stopped being a unique name the moment a second
    experiment trained the same objective. exp16 trains WCE at seeds 88, 101,
    256 and 2024, and exp15 trains ClassBalanced at beta 0.999, so keying only
    off the loss type would pool them into exp10's `Weighted CE` and
    `Class-Balanced` and double count those folds without saying so. A run
    therefore carries a bracketed qualifier naming whatever departs from the
    configuration the published tables were built from:

        the experiment, when it is not in LABEL_PLAIN_EXPERIMENTS
        s<seed>,       when init_seed is not DEFAULT_INIT_SEED
        <n>sl,         when context_slices is not DEFAULT_CONTEXT_SLICES

    so exp16's WCE at seed 88 reads `Weighted CE [exp16 s88]`, exp15 reads
    `Class-Balanced [exp15]`, and exp17's multi-slice FA-FL reads
    `FA-FL [exp17 3sl]`. Every COMPLETE run of exp10 to exp13 and of exp01,
    exp02, exp07 and exp08 is seed 42, one slice and in
    LABEL_PLAIN_EXPERIMENTS, so all three qualifiers are empty there and every
    label already in t20 through t42 comes out exactly as before.
    """
    t = manifest["loss"]["type"]
    if t == "FocalLoss":
        base = f"Focal g={float(manifest['loss']['params']['gamma']):.2f}"
    else:
        base = {"WCE": "Weighted CE", "LDAM": "LDAM",
                "ClassBalanced": "Class-Balanced", "FA_FL": FA}.get(t, t)

    quals: List[str] = []
    exp = manifest.get("experiment")
    if exp and exp not in LABEL_PLAIN_EXPERIMENTS:
        quals.append(experiment_tag(exp))
    seed = manifest.get("init_seed")
    if seed is not None and int(seed) != DEFAULT_INIT_SEED:
        quals.append(f"s{int(seed)}")
    slices = manifest.get("context_slices")
    if slices is not None and int(slices) != DEFAULT_CONTEXT_SLICES:
        quals.append(f"{int(slices)}sl")
    return base if not quals else f"{base} [{' '.join(quals)}]"


def participant_frame(pooled: pd.DataFrame) -> pd.DataFrame:
    """One row per participant: mean predicted distribution, then argmax.

    This is the manuscript's Table 4 rule and the unit the paper declares as
    primary. It is NOT the same as pooling images, because participants carry
    between 122 and 488 images each, so image pooling silently weights people
    by how many times they were scanned.
    """
    cols = [f"p{c}" for c in range(N_CLASSES)]
    g = pooled.groupby("participant_id", sort=True)
    out = g[cols].mean()
    out["label"] = g["label"].first()
    out = out.reset_index()
    out["pred"] = out[cols].to_numpy().argmax(axis=1)
    return out[["participant_id", "label", "pred"] + cols]


def constant_frame(template: pd.DataFrame) -> Tuple[pd.DataFrame, int, np.ndarray]:
    """A predictor that always names the majority class of the rows it is scored on.

    Handed the majority class and the class prior for free, from the very labels
    it is scored against. Deliberately generous: an arm that cannot beat this
    has not earned a claim.
    """
    counts = np.array([(template["label"].to_numpy() == c).sum() for c in range(N_CLASSES)],
                      dtype=float)
    prior = counts / counts.sum()
    maj = int(counts.argmax())
    out = template[["participant_id", "label"]].copy()
    out["pred"] = maj
    for c in range(N_CLASSES):
        out[f"p{c}"] = prior[c]
    return out, maj, prior


def prob_frame(probs: np.ndarray, pids: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"participant_id": pids, "label": labels.astype(int)})
    out["pred"] = probs.argmax(axis=1)
    for c in range(N_CLASSES):
        out[f"p{c}"] = probs[:, c]
    return out


def per_class_f1(cm: np.ndarray) -> np.ndarray:
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    return np.divide(2 * tp, denom, out=np.zeros(N_CLASSES), where=denom > 0)


def confusion(pred: np.ndarray, lab: np.ndarray) -> np.ndarray:
    cm = np.zeros((N_CLASSES, N_CLASSES))
    np.add.at(cm, (lab, pred), 1.0)
    return cm


# ---------------------------------------------------------------------------
# Loading a cohort
# ---------------------------------------------------------------------------

class Cohort:
    """Every arm of one cohort, pooled over its five grouped test folds."""

    def __init__(self, name: str, experiments: Sequence[str], experiments_dir: Path):
        self.name = name
        self.experiments_dir = Path(experiments_dir)
        self.arms: Dict[str, pd.DataFrame] = {}          # image level
        self.part: Dict[str, pd.DataFrame] = {}          # participant level
        self.scaled: Dict[str, pd.DataFrame] = {}        # image level, temperature scaled
        self.fingerprint: Dict[str, str] = {}
        self.run_ids: List[str] = []
        self.missing_scaled: List[str] = []

        prev = dsc.EXPERIMENTS
        try:
            dsc.EXPERIMENTS = self.experiments_dir
            frames = []
            for exp in experiments:
                try:
                    f = dsc.load_runs(exp)
                except Exception as exc:                 # noqa: BLE001
                    warn(f"{name}: could not load {exp} ({type(exc).__name__}: {exc})")
                    continue
                if not f.empty:
                    f = f.copy()
                    f["_experiment_dir"] = exp
                    frames.append(f)
            if not frames:
                raise FileNotFoundError(
                    f"no COMPLETE runs for {list(experiments)} under {self.experiments_dir}")
            runs = pd.concat(frames, ignore_index=True)
            labels = []
            for _, r in runs.iterrows():
                m = json.loads((Path(r["run_dir"]) / "manifest.json").read_text(encoding="utf-8"))
                labels.append(arm_label(m))
            runs["arm"] = labels
            self.runs = runs
            self.run_ids = list(runs["run_id"])

            for arm in sorted(set(labels)):
                sub = runs[runs["arm"] == arm]
                loss = sub["loss"].iloc[0]
                if sub["loss"].nunique() != 1:
                    warn(f"{name}/{arm}: {sub['loss'].nunique()} loss types under one arm label; "
                         f"skipping this arm")
                    continue
                if len(sub) != 5:
                    warn(f"{name}/{arm}: {len(sub)} folds, expected 5; skipping this arm")
                    continue
                self.arms[arm] = pooled_predictions(sub, loss)
                self.part[arm] = participant_frame(self.arms[arm])
                fps = sorted(sub["code_fingerprint"].dropna().unique())
                self.fingerprint[arm] = fps[0][:8] if len(fps) == 1 else "MIXED"
                if all((Path(d) / "predictions_test_temperature_scaled.npz").exists()
                       for d in sub["run_dir"]):
                    self.scaled[arm] = pooled_predictions(sub, loss, temperature_scaled=True)
                else:
                    self.missing_scaled.append(arm)
        finally:
            dsc.EXPERIMENTS = prev

        if not self.arms:
            raise FileNotFoundError(f"{name}: no usable arms under {self.experiments_dir}")

        ref = set(next(iter(self.arms.values()))["participant_id"])
        for a, f in self.arms.items():
            if set(f["participant_id"]) != ref:
                raise SystemExit(f"{name}/{a} covers a different participant set; cannot pair")
        self.participants = np.array(sorted(ref))
        self.n_participants = len(self.participants)
        self.n_images = len(next(iter(self.arms.values())))

        first = next(iter(self.arms.values()))
        g = first.groupby("participant_id", sort=True)
        self.labels = g["label"].first().to_numpy().astype(int)
        self.folds = g["fold"].first().to_numpy().astype(int)
        self.image_counts = g.size().to_numpy().astype(int)

    def unit(self, u: str) -> Dict[str, pd.DataFrame]:
        return self.part if u == "participant" else self.arms

    def order(self) -> List[str]:
        known = [a for a in ARM_ORDER if a in self.arms]
        return known + [a for a in sorted(self.arms) if a not in known]


def find_full_tree(explicit: Optional[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.is_dir() else None
    for cand in FULL_TREE_CANDIDATES:
        p = REPO_ROOT.parent / cand / "experiments"
        if p.is_dir() and any(p.glob("exp01_main_sweep/outputs/*/manifest.json")):
            return p
    local = EXPERIMENTS_DIR
    if any(local.glob("exp01_main_sweep/outputs/*/manifest.json")):
        return local
    return None


# ---------------------------------------------------------------------------
# Metadata reference models (R4)
# ---------------------------------------------------------------------------

def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_softmax(X: np.ndarray, y: np.ndarray, it: int = 1200, lr: float = 0.5,
                l2: float = 1e-3) -> np.ndarray:
    """Full-batch gradient descent multinomial logistic regression.

    Copied from a17 rather than imported from scikit-learn, which is not
    installed and is not worth adding for eight small regressions. Same
    iterations, learning rate and ridge term as a17, so a17's age-only and
    nWBV-only numbers reproduce here.
    """
    W = np.zeros((X.shape[1], N_CLASSES))
    onehot = np.eye(N_CLASSES)[y]
    for _ in range(it):
        W -= lr * (X.T @ (softmax(X @ W) - onehot) / len(y) + l2 * W)
    return W


class MetaFitter:
    """Cross-validated metadata regressions on one cohort's grouped folds.

    Imputation is declared, not silent. a17 median-imputes and says nothing
    about how many values that touched; MMSE is absent for 147 of the 347
    full-cohort participants, which is 42 percent, so a median-imputed MMSE
    model on that cohort is mostly a constant and must not be read as a measure
    of what MMSE knows. The counts travel with every row.

    The preprocessing is FOLD HONEST. The median used to impute and the mean and
    standard deviation used to standardise are computed on the TRAINING folds of
    each split and then applied to the held-out fold. Doing it once over the whole
    cohort, as a17 does, lets the held-out participants shape the scaling and the
    fill value of the very rows the model is then scored on. That is transductive,
    and it tilts the metadata baselines upward, which is exactly the wrong direction
    for a protocol whose point is that the CNN has to beat them honestly.

    NOTE: a17_participant_baseline.py still does this the transductive way, whole
    cohort at once. a17 is not changed here. So a23's metadata numbers will differ
    slightly from a17's for the same model on the same folds, and the difference is
    the leak, not a bug in either file.
    """

    def __init__(self, pids: np.ndarray, y: np.ndarray, folds: np.ndarray, meta: pd.DataFrame):
        self.pids, self.y, self.folds = pids, y, folds
        self.meta = meta.reindex(pids)
        self.imputed: Dict[str, int] = {}
        self.empty_train_column: Dict[str, List[int]] = {}
        self.n = len(pids)

    def column(self, name: str) -> Tuple[np.ndarray, int]:
        """The raw column. Missing values stay NaN; the fold loop fills them."""
        if name == "M/F":
            raw = self.meta["M/F"]
            v = pd.Series(np.where(raw.astype(str).str.upper().str.startswith("M"), 1.0, 0.0),
                          index=raw.index)
            v[raw.isna()] = np.nan
        else:
            v = pd.to_numeric(self.meta[name], errors="coerce")
        n_missing = int(v.isna().sum())
        self.imputed[name] = n_missing
        return v.to_numpy(dtype=float), n_missing

    def fit(self, cols: Sequence[str]) -> Tuple[np.ndarray, Dict[str, int]]:
        mat, miss = [], {}
        for c in cols:
            v, n_missing = self.column(c)
            mat.append(v)
            miss[c] = n_missing
        V_raw = np.column_stack(mat)
        Q = np.zeros((len(V_raw), N_CLASSES))
        for f in np.unique(self.folds):
            tr, te = self.folds != f, self.folds == f
            V = V_raw.copy()
            for j, c in enumerate(cols):
                col = V[:, j]
                seen = col[tr][~np.isnan(col[tr])]
                if seen.size == 0:
                    # Not one training participant has this value, so there is nothing
                    # to impute from. Hold the column at zero for this fold and record
                    # it. Borrowing the held-out fold's median here would be the leak
                    # this method exists to remove.
                    fill = 0.0
                    self.empty_train_column.setdefault(c, []).append(int(f))
                else:
                    fill = float(np.median(seen))
                col[np.isnan(col)] = fill
                mu = float(col[tr].mean())
                sd = float(col[tr].std())
                if sd == 0.0:
                    sd = 1.0          # constant on the training folds, so it carries nothing
                V[:, j] = (col - mu) / sd
            X = np.column_stack([np.ones(len(V)), V])
            Q[te] = softmax(X[te] @ fit_softmax(X[tr], self.y[tr]))
        return Q, miss


# ---------------------------------------------------------------------------
# The permuted-label null (R2)
# ---------------------------------------------------------------------------

def load_runfilter(tree: Path):
    """Import the permutation tree's own production filter by file path.

    Both trees have a package called `src`, and this process has already
    imported this tree's. Adding the other tree to sys.path would shadow it and
    change what discover.py sees, so the module is loaded by location under a
    private name instead.
    """
    f = tree / "src" / "runfilter.py"
    if not f.is_file():
        return None
    spec = importlib.util.spec_from_file_location("_a23_permnull_runfilter", f)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def collect_null(tree: Path) -> Tuple[Dict[int, List[Path]], List[Tuple[str, str]]]:
    """Production permuted runs, keyed by permutation seed. a22's rule, reused.

    run_permnull.py --smoke writes a real COMPLETE run with a real seed and fold
    into the same directory. Counting one would push a replicate to six folds
    and drop it from the null without saying so.
    """
    rf = load_runfilter(tree)
    base = tree / "experiments" / "exp14_permnull_wce" / "outputs"
    if rf is None or not base.is_dir():
        return {}, []
    by_spec, excluded = {}, []
    for d in sorted(base.iterdir()):
        if not d.is_dir() or not (d / "_COMPLETE").exists():
            continue
        mf = d / "manifest.json"
        if not mf.is_file():
            continue
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:                                   # noqa: BLE001
            excluded.append((d.name, "unreadable manifest"))
            continue
        if not (m.get("permutation") or {}).get("enabled"):
            continue
        why = rf.production_reason(m)
        if why is not None:
            excluded.append((d.name, why))
            continue
        spec = rf.permuted_spec(m)
        if spec is None:
            excluded.append((d.name, "seed or fold missing"))
            continue
        started = m.get("started_utc", "")
        prev = by_spec.get(spec)
        if prev is None or started > prev[0]:
            by_spec[spec] = (started, d)
    out: Dict[int, List[Path]] = {}
    for (seed, _fold), (_s, d) in sorted(by_spec.items()):
        out.setdefault(seed, []).append(d)
    return out, excluded


def null_replicates(tree: Path) -> Tuple[pd.DataFrame, List[Tuple[str, str]], int]:
    """One row per (replicate, unit) with accuracy, macro-F1 and per-class F1."""
    by_seed, excluded = collect_null(tree)
    rows, n_runs = [], 0
    for seed, dirs in sorted(by_seed.items()):
        if len(dirs) != 5:
            excluded.append((f"seed {seed}", f"{len(dirs)} folds, need 5"))
            continue
        n_runs += len(dirs)
        parts = []
        for d in dirs:
            z = np.load(d / "predictions_test.npz", allow_pickle=False)
            parts.append(prob_frame(z["probabilities"].astype(np.float64),
                                    z["participant_id"].astype(str), z["labels"]))
        img = pd.concat(parts, ignore_index=True)
        for unit, frame in (("participant", participant_frame(img)), ("image", img)):
            cm = confusion(frame["pred"].to_numpy(), frame["label"].to_numpy())
            f1 = per_class_f1(cm)
            rows.append({"perm_seed": seed, "unit": unit,
                         "accuracy": 100.0 * np.trace(cm) / cm.sum(),
                         "macro_f1": float(f1.mean()),
                         **{f"f1_{SHORT[c]}": float(f1[c]) for c in range(N_CLASSES)}})
    return pd.DataFrame(rows), excluded, n_runs


# ---------------------------------------------------------------------------
# Bootstrap bookkeeping
# ---------------------------------------------------------------------------

def boot_block(stats: Dict[str, SuffStats], _metric: str, _n_boot: int, _tag: str,
               **parts) -> Dict[str, Dict]:
    """One paired bootstrap. The seed names its call site, per _determinism.py.

    Every entry in `stats` is resampled on the SAME participant draw, so any
    difference taken from this block is paired. Entries may mix units: a
    participant-level frame and an image-level frame over the same people share
    a participant index, and resampling that index resamples both correctly.
    """
    return clustered_bootstrap(stats, _metric, n_boot=_n_boot,
                               seed=seed_for(BASE_SEED, _tag, **parts))


def diff_row(boot: Dict[str, Dict], a: str, b: str, family: Optional[str]) -> Dict:
    fam = family if family else None
    d = paired_difference(boot, a, b, family=fam)
    d["family"] = family or UNDECLARED
    d["alpha_used"] = (REGISTRY.alpha_corrected(family) if family else 0.05)
    return d


def favours(d: Dict, higher_is_better: bool) -> bool:
    """PASS means the interval excludes zero in the arm's favour, not merely that
    it excludes zero. An arm significantly WORSE than the baseline is a FAIL."""
    if not d["excludes_zero"]:
        return False
    return d["ci_low"] > 0 if higher_is_better else d["ci_high"] < 0


HIGHER_BETTER = {"accuracy": True, "macro_f1": True, "ece": False, "brier": False}


# ---------------------------------------------------------------------------
# Requirement 1
# ---------------------------------------------------------------------------

# The declared arm-vs-constant families were realised by a20 at the IMAGE level
# (t20_pooled_metrics.csv reports the constant predictor at 51.8349 percent,
# which is the image-level majority share of the restricted cohort, not the
# participant-level 51.2048). Applying the same family to a second unit would
# double its realised size against a declaration that fixes it, so the
# participant-level half of R1 is reported as undeclared.
CONSTANT_FAMILY = {
    ("full", "accuracy"): "arm_vs_constant_accuracy",
    ("full", "macro_f1"): "arm_vs_constant_macro_f1",
    ("restricted", "accuracy"): "age60_arm_vs_constant_accuracy",
    ("restricted", "macro_f1"): "age60_arm_vs_constant_macro_f1",
}


def requirement_1(coh: Cohort, boots: Dict, long: List[Dict]) -> Dict[Tuple[str, str], bool]:
    verdict: Dict[Tuple[str, str], bool] = {}
    for unit in UNITS:
        for metric in ("accuracy", "macro_f1"):
            b = boots[(unit, metric)]
            fam = CONSTANT_FAMILY.get((coh.name, metric)) if unit == "image" else None
            for arm in coh.order():
                d = diff_row(b, arm, CONSTANT, fam)
                ok = favours(d, HIGHER_BETTER[metric])
                verdict[(arm, f"{unit}:{metric}")] = ok
                long.append({"requirement": "R1", "cohort": coh.name, "unit": unit,
                             "metric": metric, "arm": arm, "reference": CONSTANT,
                             "arm_value": b[arm]["point"], "reference_value": b[CONSTANT]["point"],
                             "difference": d["diff"], "ci_low": d["ci_low"],
                             "ci_high": d["ci_high"], "ci_level": d["ci_level"],
                             "p_bootstrap": d["p_bootstrap"],
                             "p_at_resolution_floor": d["p_at_resolution_floor"],
                             "family": d["family"], "alpha_used": d["alpha_used"],
                             "code_fingerprint": coh.fingerprint.get(arm, ""),
                             "verdict": "PASS" if ok else "FAIL",
                             "note": ("family realised at the image level by a20; the "
                                      "participant-level half is undeclared"
                                      if unit == "participant" else "")})
    return verdict


# ---------------------------------------------------------------------------
# Requirement 2
# ---------------------------------------------------------------------------

def requirement_2(coh: Cohort, null: pd.DataFrame, n_runs: int,
                  long: List[Dict]) -> Dict[str, Dict]:
    """Every arm against the MEASURED null, on macro-F1 and on each class's F1.

    The null is five replicates. That is the number the smallest reportable
    p-value comes from, and it is the number every statement below has to be
    read against: 1/(5+1) = 0.1667 is the floor, so "outside all five
    replicates" is the strongest available sentence and "p < 0.05" is not
    available at any effect size. The standard deviation of five numbers is
    itself uncertain by roughly SD/sqrt(2(n-1)) = SD/2.83, so every z below
    carries that uncertainty and no z is reported without it.
    """
    out: Dict[str, Dict] = {}
    if null.empty:
        return out
    n_rep = int(null["perm_seed"].nunique())
    p_floor = 1.0 / (n_rep + 1)
    quantities = ["macro_f1"] + [f"f1_{s}" for s in SHORT]

    for unit in UNITS:
        sub = null[null["unit"] == unit]
        if sub.empty:
            continue
        for arm in coh.order():
            frame = coh.unit(unit)[arm]
            cm = confusion(frame["pred"].to_numpy(), frame["label"].to_numpy())
            f1 = per_class_f1(cm)
            values = {"macro_f1": float(f1.mean()),
                      **{f"f1_{SHORT[c]}": float(f1[c]) for c in range(N_CLASSES)}}
            for q in quantities:
                nv = sub[q].to_numpy(dtype=float)
                mean, sd = float(nv.mean()), float(nv.std(ddof=1))
                sd_se = sd / np.sqrt(2.0 * (n_rep - 1)) if n_rep > 1 else float("nan")
                v = values[q]
                ge = int((nv >= v).sum())
                outside = bool(v > nv.max())
                z = (v - mean) / sd if sd > 0 else float("nan")
                ok = outside
                if unit == "participant" and q == "macro_f1":
                    out[arm] = {"pass": ok, "n_rep": n_rep}
                long.append({"requirement": "R2", "cohort": coh.name, "unit": unit,
                             "metric": q, "arm": arm, "reference": "permuted-label null",
                             "arm_value": v, "reference_value": mean,
                             "difference": v - mean,
                             "null_sd": sd, "null_sd_standard_error": sd_se,
                             "null_min": float(nv.min()), "null_max": float(nv.max()),
                             "z_against_null_sd": z,
                             "outside_observed_null_range": outside,
                             "null_ge_arm": ge,
                             "p_permutation": (1 + ge) / (1 + n_rep),
                             "p_permutation_floor": p_floor,
                             "n_null_replicates": n_rep, "n_null_runs": n_runs,
                             "min_detectable_effect": float(nv.max() - mean),
                             "family": UNDECLARED, "alpha_used": np.nan,
                             "code_fingerprint": coh.fingerprint.get(arm, ""),
                             "verdict": "PASS" if ok else "FAIL",
                             "note": (f"z carries the uncertainty of an SD estimated from "
                                      f"{n_rep} replicates (SE {sd_se:.4f}). With {n_rep} "
                                      f"replicates the smallest reportable p is {p_floor:.4f}. "
                                      f"The minimum detectable effect, meaning the smallest "
                                      f"value that clears every replicate, is "
                                      f"{float(nv.max() - mean):+.4f} above the null mean.")})
    return out


# ---------------------------------------------------------------------------
# Requirement 3
# ---------------------------------------------------------------------------

def requirement_3(coh: Cohort, boots: Dict, long: List[Dict]) -> Tuple[Dict[str, bool], List[str]]:
    """Participant unit against image unit, same arms, same cohort, four metrics."""
    agree: Dict[str, bool] = {a: True for a in coh.order()}
    flags: List[str] = []
    for metric in METRICS:
        bp, bi = boots[("participant", metric)], boots[("image", metric)]
        arms = coh.order()
        rank_p = pd.Series({a: bp[a]["point"] for a in arms}).rank(
            ascending=not HIGHER_BETTER[metric])
        rank_i = pd.Series({a: bi[a]["point"] for a in arms}).rank(
            ascending=not HIGHER_BETTER[metric])
        reordered = bool((rank_p != rank_i).any())
        if reordered:
            flags.append(f"{coh.name}/{metric}: the two units rank the arms differently")
        for arm in arms:
            dp = paired_difference(bp, arm, CONSTANT)
            di = paired_difference(bi, arm, CONSTANT)
            sign_p = int(np.sign(dp["diff"]))
            sign_i = int(np.sign(di["diff"]))
            sign_disagree = bool(sign_p != sign_i)
            if sign_disagree:
                agree[arm] = False
                flags.append(f"{coh.name}/{metric}/{arm}: the two units disagree on the SIGN "
                             f"of the difference against the constant predictor")
            long.append({"requirement": "R3", "cohort": coh.name, "unit": "participant_vs_image",
                         "metric": metric, "arm": arm, "reference": "unit of analysis",
                         "arm_value": bp[arm]["point"], "reference_value": bi[arm]["point"],
                         "difference": bp[arm]["point"] - bi[arm]["point"],
                         "ci_low": np.nan, "ci_high": np.nan,
                         "rank_participant": float(rank_p[arm]),
                         "rank_image": float(rank_i[arm]),
                         "units_reorder_arms": reordered,
                         "units_disagree_on_sign_vs_constant": sign_disagree,
                         "family": UNDECLARED, "alpha_used": np.nan,
                         "code_fingerprint": coh.fingerprint.get(arm, ""),
                         "verdict": "FAIL" if sign_disagree else "PASS",
                         "note": ("participant value minus image value. The two are not "
                                  "paired estimates of one quantity, so no interval is "
                                  "given on their difference; they are two different "
                                  "estimands and the question is whether they agree.")})
    return agree, flags


# ---------------------------------------------------------------------------
# Requirement 4
# ---------------------------------------------------------------------------

def requirement_4(coh: Cohort, meta_probs: Dict[str, np.ndarray], miss: Dict[str, Dict[str, int]],
                  boots: Dict, long: List[Dict]) -> Tuple[Dict[str, bool], List[str], List[str]]:
    """Every arm against every metadata reference, on the same folds and people.

    A reference the script itself prints as NOT INTERPRETABLE does not get a vote.
    The full-cohort MMSE models are the case that matters: MMSE is missing for 147 of
    the 347 full-cohort participants, so those models are mostly their own imputed
    median, and letting one of them veto an arm would be a verdict decided by a number
    the script has already said cannot be read. Excluded references stay in the long
    table with their flag, and every summary row names which references were counted.
    """
    verdict: Dict[str, bool] = {}
    counted = [n for n in meta_probs if not not_interpretable(miss[n], coh.n_participants)]
    dropped = [n for n in meta_probs if n not in counted]
    b_f1 = boots[("participant", "macro_f1")]
    for arm in coh.order():
        beats_all = True
        for name in meta_probs:
            in_verdict = name in counted
            fam = ("arm_vs_metadata"
                   if (coh.name == "full" and name == META_REFERENCE) else None)
            for metric in ("accuracy", "macro_f1"):
                b = boots[("participant", metric)]
                d = diff_row(b, arm, name, fam)
                ok = favours(d, HIGHER_BETTER[metric])
                if metric == "macro_f1" and in_verdict and not ok:
                    beats_all = False
                m = miss[name]
                long.append({"requirement": "R4", "cohort": coh.name, "unit": "participant",
                             "metric": metric, "arm": arm, "reference": name,
                             "arm_value": b[arm]["point"], "reference_value": b[name]["point"],
                             "difference": d["diff"], "ci_low": d["ci_low"],
                             "ci_high": d["ci_high"], "ci_level": d["ci_level"],
                             "p_bootstrap": d["p_bootstrap"],
                             "p_at_resolution_floor": d["p_at_resolution_floor"],
                             "family": d["family"], "alpha_used": d["alpha_used"],
                             "reference_median_imputed": json.dumps(m),
                             "reference_max_imputed_fraction":
                                 imputed_fraction(m, coh.n_participants),
                             "reference_interpretable": in_verdict,
                             "counted_in_R4_verdict": in_verdict,
                             "fold_honest_preprocessing": True,
                             "code_fingerprint": coh.fingerprint.get(arm, ""),
                             "verdict": "PASS" if ok else "FAIL",
                             "note": ("reference fitted by cross-validated softmax regression "
                                      "on the same grouped folds as the CNNs; the imputation "
                                      "median and the standardising mean and SD come from the "
                                      "TRAINING folds only, counts in reference_median_imputed"
                                      + ("" if in_verdict else
                                         "; NOT INTERPRETABLE by the >"
                                         f"{NOT_INTERPRETABLE_IMPUTED_FRACTION:.0%} imputed rule, "
                                         "so this row is reported but does not enter the R4 "
                                         "verdict"))})
        verdict[arm] = beats_all
        long.append({"requirement": "R4", "cohort": coh.name, "unit": "participant",
                     "metric": "macro_f1", "arm": arm,
                     "reference": "ALL INTERPRETABLE metadata references",
                     "arm_value": b_f1[arm]["point"], "reference_value": np.nan,
                     "family": UNDECLARED, "alpha_used": np.nan,
                     "fold_honest_preprocessing": True,
                     "r4_references_counted": "; ".join(counted),
                     "r4_references_excluded_not_interpretable": "; ".join(dropped),
                     "r4_n_references_counted": len(counted),
                     "r4_n_references_excluded": len(dropped),
                     "code_fingerprint": coh.fingerprint.get(arm, ""),
                     "verdict": "PASS" if beats_all else "FAIL",
                     "note": ("PASS only if the arm beats every INTERPRETABLE metadata "
                              "reference on macro-F1. A reference with more than "
                              f"{NOT_INTERPRETABLE_IMPUTED_FRACTION:.0%} of any one predictor "
                              "imputed is excluded from this verdict and named in "
                              "r4_references_excluded_not_interpretable")})
    return verdict, counted, dropped


# ---------------------------------------------------------------------------
# Requirement 5
# ---------------------------------------------------------------------------

def requirement_5(full: Optional[Cohort], restricted: Cohort, r1: Dict[str, Dict],
                  n_boot: int, long: List[Dict]) -> Dict[str, Optional[bool]]:
    """Full against restricted, per arm, per metric.

    The two cohorts are different people, so the honest paired contrast is the
    one a21 defines: the full-cohort model scored on the restricted cohort (b)
    against the restricted-cohort model scored on the same people (c). Those
    two share a participant set and are resampled together. The unpaired
    full-on-full value (a) is reported beside them as context and is never
    differenced in a test.

    a21 already decomposes (a) minus (b) into the test-composition effect and
    (b) minus (c) into the training effect, for accuracy. Those columns are
    read from t21_age_confound.csv rather than recomputed, because two scripts
    computing the same decomposition is how the two eventually disagree.
    """
    verdict: Dict[str, Optional[bool]] = {}
    keep = set(restricted.participants)

    decomp = {}
    t21 = OUT / "t21_age_confound.csv"
    if t21.is_file():
        try:
            d21 = pd.read_csv(t21)
            for _, r in d21.iterrows():
                decomp[str(r["arm"])] = {
                    "a21_test_composition_effect_a_minus_b":
                        float(r["test_composition_effect_a_minus_b"]),
                    "a21_training_effect_b_minus_c": float(r["training_effect_b_minus_c"]),
                    "a21_registered": bool(r.get("registered", False))}
        except Exception as exc:                                   # noqa: BLE001
            warn(f"could not read {t21.name} ({exc}); reporting raw differences only")

    shared = [a for a in restricted.order() if full is not None and a in full.arms]
    for arm in restricted.order():
        pass_r = r1["restricted"].get((arm, "participant:macro_f1"), False) and \
                 r1["restricted"].get((arm, "image:macro_f1"), False)
        if full is None or arm not in full.arms:
            verdict[arm] = None
            long.append({"requirement": "R5", "cohort": "full_vs_restricted", "unit": "image",
                         "metric": "macro_f1", "arm": arm, "reference": "full cohort",
                         "family": UNDECLARED, "alpha_used": np.nan, "verdict": "NA",
                         "r5_interval_basis": R5_BASIS_NA,
                         "note": "no full-cohort counterpart for this arm"})
            continue
        pass_f = r1["full"].get((arm, "participant:macro_f1"), False) and \
                 r1["full"].get((arm, "image:macro_f1"), False)
        verdict[arm] = bool(pass_f and pass_r)

    if not shared:
        return verdict

    fallback = [a for a in shared if a not in decomp]
    if fallback:
        warn(f"R5 interval basis: {len(fallback)} of {len(shared)} shared arms have no row in "
             f"t21_age_confound.csv, so their image-level accuracy interval falls back to a "
             f"nominal 95 percent instead of the declared {R5_FAMILY} correction: "
             f"{', '.join(fallback)}. Column r5_interval_basis names the basis on every R5 row.")

    # (b) and (c) on the same people, paired.
    for unit in UNITS:
        stats: Dict[str, SuffStats] = {}
        for arm in shared:
            b_frame = full.unit(unit)[arm]
            b_frame = b_frame[b_frame["participant_id"].isin(keep)].reset_index(drop=True)
            stats[f"{arm} | trained full"] = SuffStats(b_frame)
            stats[f"{arm} | trained restricted"] = SuffStats(restricted.unit(unit)[arm])
        for metric in METRICS:
            b = boot_block(stats, metric, n_boot, "a23:R5_paired", unit=unit, metric=metric)
            for arm in shared:
                kb, kc = f"{arm} | trained full", f"{arm} | trained restricted"
                if unit == "image" and metric == "accuracy":
                    fam = R5_FAMILY if arm in decomp else None
                    basis = R5_BASIS_DECLARED if arm in decomp else R5_BASIS_FALLBACK
                else:
                    fam, basis = None, R5_BASIS_OFF_CELL
                d = diff_row(b, kc, kb, fam)
                row = {"requirement": "R5", "cohort": "full_vs_restricted", "unit": unit,
                       "metric": metric, "arm": arm,
                       "reference": "trained full, scored on the restricted cohort",
                       "arm_value": b[kc]["point"], "reference_value": b[kb]["point"],
                       "difference": d["diff"], "ci_low": d["ci_low"], "ci_high": d["ci_high"],
                       "ci_level": d["ci_level"], "p_bootstrap": d["p_bootstrap"],
                       "p_at_resolution_floor": d["p_at_resolution_floor"],
                       "family": d["family"], "alpha_used": d["alpha_used"],
                       "code_fingerprint": f"{restricted.fingerprint.get(arm,'')} vs "
                                           f"{full.fingerprint.get(arm,'')}",
                       "cross_fingerprint": (restricted.fingerprint.get(arm)
                                             != full.fingerprint.get(arm)),
                       "r5_interval_basis": basis,
                       "verdict": "", "note": ""}
                row.update(decomp.get(arm, {}))
                row["note"] = ("decomposition columns read from a21 (t21_age_confound.csv), "
                               "accuracy only, not recomputed here"
                               if arm in decomp else
                               "a21 owns the composition/training decomposition; it is not "
                               "recomputed here and no a21 row exists for this arm")
                long.append(row)

    # (a) for context, on its own participant set, never differenced.
    for unit in UNITS:
        stats_a = {arm: SuffStats(full.unit(unit)[arm]) for arm in shared}
        for metric in METRICS:
            b = boot_block(stats_a, metric, n_boot, "a23:R5_context", unit=unit, metric=metric)
            for arm in shared:
                long.append({"requirement": "R5", "cohort": "full", "unit": unit,
                             "metric": metric, "arm": arm,
                             "reference": "context, trained full and scored full",
                             "arm_value": b[arm]["point"], "reference_value": np.nan,
                             "ci_low": b[arm]["ci_low"], "ci_high": b[arm]["ci_high"],
                             "family": UNDECLARED, "alpha_used": np.nan,
                             "code_fingerprint": full.fingerprint.get(arm, ""),
                             "verdict": "", "r5_interval_basis": R5_BASIS_CONTEXT,
                             "note": "computed on 347 participants; never differenced against "
                                     "the restricted-cohort rows"})
    return verdict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_boots(coh: Cohort, meta_probs: Dict[str, np.ndarray], n_boot: int) -> Dict:
    boots: Dict = {}
    for unit in UNITS:
        stats: Dict[str, SuffStats] = {a: SuffStats(f) for a, f in coh.unit(unit).items()}
        template = coh.unit(unit)[coh.order()[0]]
        const, maj, prior = constant_frame(template)
        stats[CONSTANT] = SuffStats(const)
        if unit == "participant":
            for name, Q in meta_probs.items():
                stats[name] = SuffStats(prob_frame(Q, coh.participants, coh.labels))
        for metric in METRICS:
            boots[(unit, metric)] = boot_block(stats, metric, n_boot, "a23:main",
                                               cohort=coh.name, unit=unit, metric=metric)
        boots[(unit, "_majority")] = (maj, prior)
    return boots


def main() -> int:
    ap = argparse.ArgumentParser(description="The five-requirement protocol audit.")
    ap.add_argument("--boot", type=int, default=N_BOOT,
                    help=f"bootstrap resamples (default {N_BOOT}, matching a20)")
    ap.add_argument("--quick", action="store_true",
                    help=f"smoke test with {QUICK_BOOT} resamples; intervals are NOT reportable")
    ap.add_argument("--full-from", default=None, metavar="EXPERIMENTS_DIR",
                    help="experiments folder holding the full-cohort runs")
    ap.add_argument("--permnull-from", default=None, metavar="TREE",
                    help=f"root of the {PERMNULL_TREE} tree")
    ap.add_argument("--include-new-arms", action="store_true",
                    help=f"also read {', '.join(EXTENDED_RESTRICTED_EXPERIMENTS)} when they "
                         f"have COMPLETE runs. OFF by default, which reproduces t23 exactly. "
                         f"Also settable with {ENV_INCLUDE_NEW_ARMS}=1")
    args = ap.parse_args()

    n_boot = QUICK_BOOT if args.quick else args.boot
    if args.quick:
        print("QUICK MODE. %d resamples. Intervals are a smoke test, not results." % n_boot)

    print(f"a23  protocol audit, {n_boot:,} resamples, base seed {BASE_SEED}")
    print(f"     restricted arms from {EXPERIMENTS_DIR}")

    restricted_experiments = resolve_restricted_experiments(args.include_new_arms)
    if restricted_experiments != list(RESTRICTED_EXPERIMENTS):
        print("     INCLUDING NEW ARMS: " + ", ".join(restricted_experiments))
    restricted = Cohort("restricted", restricted_experiments, EXPERIMENTS_DIR)
    print(f"     restricted: {len(restricted.arms)} arms, "
          f"{restricted.n_participants} participants, {restricted.n_images:,} images")

    full_dir = find_full_tree(args.full_from)
    full: Optional[Cohort] = None
    if full_dir is None:
        warn("no full-cohort experiments folder found. R1, R3 and R4 run on the restricted "
             "cohort only and R5 reports NA. Pass --full-from to point at one.")
    else:
        try:
            full = Cohort("full", FULL_EXPERIMENTS, full_dir)
            print(f"     full:       {len(full.arms)} arms, {full.n_participants} participants, "
                  f"{full.n_images:,} images, from {full_dir}")
        except Exception as exc:                                    # noqa: BLE001
            warn(f"full cohort unusable ({type(exc).__name__}: {exc}). Continuing without it.")
            full = None

    cohorts = [c for c in (full, restricted) if c is not None]

    # -- fingerprints, named rather than refused --------------------------------
    by_fp: Dict[str, List[str]] = {}
    for c in cohorts:
        for arm, fp in c.fingerprint.items():
            by_fp.setdefault(fp, []).append(f"{c.name}/{arm}")
    if len(by_fp) > 1:
        print()
        warn(f"the arms span {len(by_fp)} code fingerprints. _common.runs_frame would refuse "
             f"this; an audit that puts two cohorts side by side cannot, so every row carries "
             f"its arm's fingerprint and the cross-fingerprint rows are flagged.")
        for fp, arms in sorted(by_fp.items()):
            print(f"       {fp}  {len(arms):>2} arm(s): {', '.join(sorted(arms))}")

    for c in cohorts:
        if c.missing_scaled:
            warn(f"{c.name}: no temperature-scaled predictions for "
                 f"{', '.join(c.missing_scaled)} (a23 does not use them; a24 does)")

    # -- metadata ---------------------------------------------------------------
    meta_raw = pd.read_excel(metadata_xlsx())
    meta_raw["pid"] = meta_raw["ID"].astype(str).str.replace(r"_MR\d+$", "", regex=True)
    meta_raw = meta_raw.drop_duplicates("pid").set_index("pid")

    meta_probs: Dict[str, Dict[str, np.ndarray]] = {}
    meta_miss: Dict[str, Dict[str, Dict[str, int]]] = {}
    print("\n     metadata reference models, cross-validated on the SAME grouped folds.")
    print("     Imputation median and standardising mean and SD are computed on the TRAINING")
    print("     folds of each split only, so no held-out value reaches the preprocessing.")
    print(f"     NOT INTERPRETABLE means more than {NOT_INTERPRETABLE_IMPUTED_FRACTION:.0%} of "
          f"one predictor was imputed.")
    for c in cohorts:
        fitter = MetaFitter(c.participants, c.labels, c.folds, meta_raw)
        meta_probs[c.name], meta_miss[c.name] = {}, {}
        for name, cols in META_MODELS:
            Q, miss = fitter.fit(cols)
            meta_probs[c.name][name] = Q
            meta_miss[c.name][name] = miss
            bad = {k: v for k, v in miss.items() if v}
            if bad:
                frac = imputed_fraction(miss, c.n_participants)
                print(f"       {c.name:<11} {name:<30} imputed {bad} "
                      f"of {c.n_participants} ({frac:.0%}), from the training folds only"
                      + ("   NOT INTERPRETABLE" if not_interpretable(miss, c.n_participants)
                         else ""))
        if not any(any(m.values()) for m in meta_miss[c.name].values()):
            print(f"       {c.name:<11} no missing values in any predictor; nothing imputed")
        for col, folds_hit in sorted(fitter.empty_train_column.items()):
            warn(f"{c.name}: predictor {col} had no non-missing TRAINING value in fold(s) "
                 f"{sorted(set(folds_hit))}; it was held at zero there and contributes nothing "
                 f"but the intercept for those folds")

    # -- bootstraps -------------------------------------------------------------
    long: List[Dict] = []
    boots: Dict[str, Dict] = {}
    for c in cohorts:
        print(f"\n     {c.name}: participant-clustered bootstrap over {len(c.arms)} arms "
              f"plus baselines ...")
        boots[c.name] = build_boots(c, meta_probs[c.name], n_boot)

    # -- R1 ---------------------------------------------------------------------
    r1 = {c.name: requirement_1(c, boots[c.name], long) for c in cohorts}

    # -- R2 ---------------------------------------------------------------------
    pn_tree = (Path(args.permnull_from).expanduser() if args.permnull_from
               else REPO_ROOT.parent / PERMNULL_TREE)
    null, excluded, n_null_runs = (pd.DataFrame(), [], 0)
    if pn_tree.is_dir():
        null, excluded, n_null_runs = null_replicates(pn_tree)
    r2: Dict[str, Dict[str, Dict]] = {c.name: {} for c in cohorts}
    if null.empty:
        warn(f"no usable permuted-label replicates under {pn_tree}. R2 is NA for every arm.")
    else:
        n_rep = int(null["perm_seed"].nunique())
        print(f"\n     permuted-label null: {n_rep} replicates from {n_null_runs} production "
              f"runs, p floor 1/({n_rep}+1) = {1.0/(n_rep+1):.4f}")
        for name, why in excluded:
            print(f"       excluded {name}: {why}")
        r2["restricted"] = requirement_2(restricted, null, n_null_runs, long)
        if full is not None:
            warn("the permuted-label runs were trained on the RESTRICTED cohort, so there is "
                 "no measured null for the full cohort. R2 is NA for every full-cohort arm.")

    # -- R3 ---------------------------------------------------------------------
    r3, unit_flags = {}, []
    for c in cohorts:
        agree, flags = requirement_3(c, boots[c.name], long)
        r3[c.name] = agree
        unit_flags += flags

    # -- R4 ---------------------------------------------------------------------
    r4: Dict[str, Dict[str, bool]] = {}
    r4_counted: Dict[str, List[str]] = {}
    r4_dropped: Dict[str, List[str]] = {}
    for c in cohorts:
        v4, counted4, dropped4 = requirement_4(c, meta_probs[c.name], meta_miss[c.name],
                                               boots[c.name], long)
        r4[c.name], r4_counted[c.name], r4_dropped[c.name] = v4, counted4, dropped4

    # -- R5 ---------------------------------------------------------------------
    r5 = requirement_5(full, restricted, r1, n_boot, long)

    # -- image-count distribution ----------------------------------------------
    for c in cohorts:
        q = np.percentile(c.image_counts, [0, 25, 50, 75, 100])
        long.append({"requirement": "R3", "cohort": c.name, "unit": "cohort",
                     "metric": "images_per_participant", "arm": "", "reference": "",
                     "arm_value": float(c.image_counts.mean()),
                     "images_min": int(q[0]), "images_p25": float(q[1]),
                     "images_median": float(q[2]), "images_p75": float(q[3]),
                     "images_max": int(q[4]), "family": UNDECLARED, "verdict": "",
                     "note": "image pooling weights each participant by this count; "
                             "participant pooling does not"})

    # -- tables -----------------------------------------------------------------
    long_df = pd.DataFrame(long)
    lead = ["requirement", "cohort", "unit", "metric", "arm", "reference", "verdict",
            "arm_value", "reference_value", "difference", "ci_low", "ci_high", "ci_level",
            "p_bootstrap", "family", "alpha_used", "code_fingerprint"]
    long_df = long_df[[c for c in lead if c in long_df.columns]
                      + [c for c in long_df.columns if c not in lead]]
    long_df.to_csv(OUT / "t23_protocol_audit_long.csv", index=False, float_format="%.6f")
    print(f"\n    -> t23_protocol_audit_long.csv  ({len(long_df):,} comparisons)")

    rows = []
    for c in cohorts:
        for arm in c.order():
            r1_pass = all(r1[c.name].get((arm, f"{u}:{m}"), False)
                          for u in UNITS for m in ("accuracy", "macro_f1"))
            r2_cell = r2[c.name].get(arm)
            r5_cell = r5.get(arm)
            n_out = int(((long_df["requirement"] == "R2") & (long_df["cohort"] == c.name) &
                         (long_df["arm"] == arm) & (long_df["unit"] == "participant") &
                         (long_df.get("outside_observed_null_range", False) == True)).sum()) \
                if "outside_observed_null_range" in long_df.columns else 0
            rows.append({
                "arm": arm, "cohort": c.name,
                "code_fingerprint": c.fingerprint.get(arm, ""),
                "R1_beats_constant": "PASS" if r1_pass else "FAIL",
                "R2_beats_measured_null": ("NA" if r2_cell is None
                                           else "PASS" if r2_cell["pass"] else "FAIL"),
                "R3_units_agree": "PASS" if r3[c.name].get(arm, False) else "FAIL",
                "R4_beats_metadata": "PASS" if r4[c.name].get(arm, False) else "FAIL",
                "R5_survives_cohort": ("NA" if r5_cell is None
                                       else "PASS" if r5_cell else "FAIL"),
                "R1_accuracy_participant":
                    "PASS" if r1[c.name].get((arm, "participant:accuracy")) else "FAIL",
                "R1_accuracy_image":
                    "PASS" if r1[c.name].get((arm, "image:accuracy")) else "FAIL",
                "R1_macro_f1_participant":
                    "PASS" if r1[c.name].get((arm, "participant:macro_f1")) else "FAIL",
                "R1_macro_f1_image":
                    "PASS" if r1[c.name].get((arm, "image:macro_f1")) else "FAIL",
                "R2_quantities_outside_null": n_out,
                "n_participants": c.n_participants, "n_images": c.n_images,
                "accuracy_participant": boots[c.name][("participant", "accuracy")][arm]["point"],
                "macro_f1_participant": boots[c.name][("participant", "macro_f1")][arm]["point"],
                "accuracy_image": boots[c.name][("image", "accuracy")][arm]["point"],
                "macro_f1_image": boots[c.name][("image", "macro_f1")][arm]["point"],
                "ece_participant": boots[c.name][("participant", "ece")][arm]["point"],
                "brier_participant": boots[c.name][("participant", "brier")][arm]["point"],
            })
    summary = pd.DataFrame(rows)
    save_table(summary, "t23_protocol_audit", float_fmt="%.6f")

    # -- one screen -------------------------------------------------------------
    print("\n" + "=" * 96)
    print("PROTOCOL AUDIT, one row per arm. PASS means the evidence clears the requirement.")
    print("=" * 96)
    # R1 is printed as four sub-verdicts rather than as one AND over four tests.
    # The split IS the result on this data: macro-F1 clears the constant predictor
    # at both units for every arm and accuracy clears it at neither, so collapsing
    # the four into one cell prints a uniform FAIL column and hides the only
    # pattern in the grid. The combined cell stays in the CSV.
    cols = [("R1_accuracy_participant", "R1acc/P"), ("R1_accuracy_image", "R1acc/I"),
            ("R1_macro_f1_participant", "R1f1/P"), ("R1_macro_f1_image", "R1f1/I"),
            ("R2_beats_measured_null", "R2"), ("R3_units_agree", "R3"),
            ("R4_beats_metadata", "R4"), ("R5_survives_cohort", "R5")]
    widths = [9, 9, 9, 9, 6, 6, 6, 6]
    for c in cohorts:
        s = summary[summary["cohort"] == c.name]
        print(f"\n  cohort {c.name}   n={c.n_participants} participants, {c.n_images:,} images, "
              f"images per participant {c.image_counts.min()} to {c.image_counts.max()}")
        print(f"  {'arm':<16}{'fp':>10}"
              + "".join(f"{h:>{w}}" for (_, h), w in zip(cols, widths)))
        for _, r in s.iterrows():
            print(f"  {r['arm']:<16}{r['code_fingerprint']:>10}"
                  + "".join(f"{r[k]:>{w}}" for (k, _), w in zip(cols, widths)))

    print("\n  R1 beats a constant majority-class predictor, interval excluding zero in")
    print("     the arm's favour. Four separate cells: accuracy (acc) and macro-F1 (f1),")
    print("     each at the participant unit (P) and the image unit (I). The AND of all")
    print("     four is column R1_beats_constant in t23_protocol_audit.csv; it is not")
    print("     printed here because one collapsed cell cannot show which half failed.")
    print("  R2 macro-F1 above every one of the measured permuted-label replicates.")
    print("  R3 the participant unit and the image unit agree on the sign of every")
    print("     difference against the constant predictor.")
    print("  R4 beats EVERY INTERPRETABLE metadata reference model on macro-F1, same folds,")
    print("     same people, with the metadata preprocessing fitted on the training folds only.")
    print(f"     NOT INTERPRETABLE is the rule already used above: more than "
          f"{NOT_INTERPRETABLE_IMPUTED_FRACTION:.0%} of any one")
    print("     predictor imputed. Such a reference stays in t23_protocol_audit_long.csv with")
    print("     its flag and is left out of the R4 verdict rather than silently dropped.")
    for c in cohorts:
        if r4_dropped[c.name]:
            print(f"     cohort {c.name}: EXCLUDED from the R4 verdict as not interpretable "
                  f"({len(r4_dropped[c.name])} of {len(r4_dropped[c.name]) + len(r4_counted[c.name])}): "
                  f"{', '.join(r4_dropped[c.name])}")
        else:
            print(f"     cohort {c.name}: every metadata reference was interpretable, "
                  f"none excluded")
    print("  R5 clears R1 on macro-F1 on BOTH cohorts, so the advantage is not an artefact")
    print("     of scoring participants a clinician would never have assessed.")
    # R1 is one cell over four sub-tests. Saying which one failed is the
    # difference between a verdict a reader can act on and one they cannot.
    for c in cohorts:
        s = summary[summary["cohort"] == c.name]
        fails = {k: int((s[f"R1_{k}"] == "FAIL").sum())
                 for k in ("accuracy_participant", "accuracy_image",
                           "macro_f1_participant", "macro_f1_image")}
        if any(fails.values()):
            print(f"\n  R1 breakdown, cohort {c.name} ({len(s)} arms). Arms failing each "
                  f"sub-test:")
            for k, v in fails.items():
                print(f"    {k:<24} {v} of {len(s)}")

    if unit_flags:
        print(f"\n  UNIT DISAGREEMENTS ({len(unit_flags)}):")
        for f in unit_flags[:20]:
            print(f"    {f}")
        if len(unit_flags) > 20:
            print(f"    ... and {len(unit_flags) - 20} more, in t23_protocol_audit_long.csv")
    if args.quick:
        print("\n  QUICK MODE: %d resamples. Re-run without --quick before quoting anything."
              % n_boot)
    print(f"\n  outputs in {OUT}")
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        raise SystemExit(main())
