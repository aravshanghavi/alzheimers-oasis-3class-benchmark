#!/usr/bin/env python3
"""How much of the full-cohort result was age, decomposed per arm.

    python analysis/a21_age_confound.py
    python analysis/a21_age_confound.py --preliminary   # see below, read the warning

THE THREE NUMBERS
-----------------
For each objective there are three distinct quantities, and the paper has so far
reported only the first while discussing the third.

    (a) trained on the full cohort, tested on the full cohort
        The submitted headline.

    (b) trained on the full cohort, tested on the CDR-assessed age-60+ subset
        The same model, scored only on the participants a clinician would
        actually have assessed. Nothing about training changed.

    (c) trained on the age-60+ cohort, tested on the age-60+ cohort
        The honest comparison, and the only one of the three that has never
        been run before exp10-exp13.

(a) minus (b) is the TEST-SET composition effect: how much of the headline came
from scoring easy participants that a real cohort would not contain. (b) minus
(c) is the TRAINING-SET effect: how much came from a model that had 181 under-60
and 147 CDR-blank participants available to learn an age contrast from.

Reporting only (a) and (c) conflates the two and lets a reader attribute the
whole drop to either one. This script separates them.

WHAT IS REGISTERED AND WHAT IS NOT
----------------------------------
Only (b) versus (c) is registered, as age60_train_matched_vs_evaluated, with
method='none' because the two sides come from different training cohorts and the
family is declared descriptive. (a) is reported for context and carries no test.

THE FINGERPRINT RULE, AND WHAT LIFTS IT
---------------------------------------
(b) and (c) are only comparable when both sides were produced by the same code.
Matching fingerprints prove that. Nothing else does by assertion.

There is one other way to earn it, and it has to be demonstrated rather than
argued: show that this tree reproduces the other tree's run exactly, for the
same spec and seed, under deterministic execution. tools/compare_equivalence.py
does that and writes analysis/outputs/equivalence_attestation.json. This script
reads that file and will register a comparison ONLY for an arm the attestation
lists as IDENTICAL. An arm that was never tested gets no verdict, because the
diagnostics code branches on the criterion type and a demonstration for one loss
says nothing about another.

    --full-from PATH   read the full-cohort arms from another tree. Registered
                       if every arm is attested, refused otherwise.
    --preliminary PATH the same, but forced unregistered: a disclosed look with
                       no verdicts, for deciding whether the runs are worth it.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analysis._common import (N_CLASSES, SuffStats, clustered_bootstrap,          # noqa: E402
                              paired_difference, save_table)
from analysis._figure_md import Finding, Provenance, emit_figure                  # noqa: E402
from analysis._paths import EXPERIMENTS_DIR, OUT                                  # noqa: E402
from analysis._registry import REGISTRY                                           # noqa: E402
from analysis.discover import load_predictions, load_runs                         # noqa: E402

RESTRICTED = ["exp10_age60_main"]
FULL = ["exp01_main_sweep", "exp02_class_balanced"]
FAMILY = "age60_train_matched_vs_evaluated"

N_BOOT = 2000
SEED = 12345
SOURCE = "analysis/a21_age_confound.py"

# The five objectives the family declares. The gamma rungs are not here: exp11,
# exp12 and exp13 have no full-cohort counterpart in this tree.
OBJECTIVES = {"WCE": "Weighted CE", "LDAM": "LDAM", "FocalLoss": "Focal g=3.00",
              "FA_FL": "FA-FL", "ClassBalanced": "Class-Balanced"}
ARM_ORDER = ["Weighted CE", "LDAM", "Class-Balanced", "Focal g=3.00", "FA-FL"]


ATTESTATION = REPO_ROOT / "analysis" / "outputs" / "equivalence_attestation.json"


def load_attestation() -> tuple:
    """Which arms have been DEMONSTRATED to reproduce the other tree exactly.

    Returns (set of attested arm labels, the parsed file or None). Only specs
    whose status is IDENTICAL count. A file that says INCOMPLETE still attests
    the arms it did test, and those may be registered while the others may not.
    """
    if not ATTESTATION.is_file():
        return set(), None
    try:
        att = json.loads(ATTESTATION.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"{ATTESTATION} is unreadable ({exc}). Delete it and re-run "
                         f"tools/compare_equivalence.py.")
    return {r["arm"] for r in att.get("specs", []) if r.get("status") == "IDENTICAL"}, att


def _runs(experiments: List[str], what: str) -> pd.DataFrame:
    frames = [load_runs(e) for e in experiments]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise SystemExit(
            f"no COMPLETE {what} runs ({', '.join(experiments)}) under {EXPERIMENTS_DIR}.\n"
            f"The full-cohort arms have to exist in THIS tree, under this tree's code "
            f"fingerprint, for the comparison to mean anything. Launch them with:\n"
            f"    run_all.bat --experiments exp01_main_sweep exp02_class_balanced "
            f"--supersede-previous\n"
            f"That is 25 runs at roughly 31 minutes each, about 13 GPU-hours.\n"
            f"To take a rough look first using full-cohort runs from another tree, see "
            f"--preliminary in this file's docstring.")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["loss"].isin(OBJECTIVES)].copy()
    df["arm"] = df["loss"].map(OBJECTIVES)
    return df


def pool(runs: pd.DataFrame, arm: str, keep: Optional[set] = None) -> pd.DataFrame:
    sub = runs[runs["arm"] == arm].sort_values("fold")
    if sub.empty:
        raise SystemExit(f"no runs for arm {arm!r}")
    parts = []
    for _, r in sub.iterrows():
        f = load_predictions(r["run_dir"], "test")
        f["fold"] = int(r["fold"])
        parts.append(f[["participant_id", "label", "pred", "fold"]
                       + [f"p{c}" for c in range(N_CLASSES)]])
    pooled = pd.concat(parts, ignore_index=True)
    dup = pooled.groupby("participant_id")["fold"].nunique()
    if (dup > 1).any():
        raise SystemExit(f"arm {arm!r}: {int((dup > 1).sum())} participant(s) in more than one "
                         f"test fold; refusing to pool")
    if keep is not None:
        pooled = pooled[pooled["participant_id"].isin(keep)].reset_index(drop=True)
    return pooled


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full-from", metavar="EXPERIMENTS_DIR", default=None,
                    help="read the full-cohort arms from another tree. Registered only if "
                         "equivalence_attestation.json attests every arm.")
    ap.add_argument("--preliminary", metavar="EXPERIMENTS_DIR", default=None,
                    help="as --full-from, but forced unregistered and stamped PRELIMINARY.")
    args = ap.parse_args()
    other_tree = args.preliminary or args.full_from
    force_preliminary = args.preliminary is not None

    restricted = _runs(RESTRICTED, "age-60")
    fp_r = sorted(restricted["code_fingerprint"].dropna().unique())
    if len(fp_r) > 1:
        raise SystemExit(f"the age-60 runs span {len(fp_r)} fingerprints; fix that first")

    attested, attestation = load_attestation()

    alt = None
    if other_tree:
        import analysis.discover as dsc
        alt = Path(other_tree).expanduser().resolve()
        if not alt.is_dir():
            raise SystemExit(f"that path is not a directory: {alt}")
        dsc.EXPERIMENTS = alt
        if force_preliminary:
            print("=" * 78)
            print("PRELIMINARY MODE, forced. The full-cohort arms are being read from")
            print(f"  {alt}")
            print("No registered comparison is computed and nothing is written to the")
            print("registry. Every number below is a look, not a result.")
            print("=" * 78)

    full = _runs(FULL, "full-cohort")
    fp_f = sorted(full["code_fingerprint"].dropna().unique())
    same_code = (len(fp_f) == 1 and fp_f[0] == fp_r[0])

    # Matching fingerprints, or a demonstration covering every arm. Nothing else.
    missing_attestation = [a for a in ARM_ORDER if a not in attested]
    registered = (not force_preliminary) and (same_code or not missing_attestation)

    if not registered and not force_preliminary:
        lines = [f"the full-cohort runs carry fingerprint(s) {[f[:8] for f in fp_f]} and the "
                 f"age-60 runs carry {fp_r[0][:8]}, so comparing them would attribute a code "
                 f"difference to the cohort."]
        if attestation is None:
            lines.append("No equivalence attestation exists. Run tools\\compare_equivalence.py "
                         "to see which objectives still need a full-cohort run here.")
        else:
            lines.append(f"The attestation ({attestation.get('verdict')}) covers "
                         f"{sorted(attested)} but not {missing_attestation}. Each untested arm "
                         f"needs one full-cohort run in this tree, about 29 minutes each; "
                         f"tools\\compare_equivalence.py prints the exact commands.")
        lines.append("Or use --preliminary for a disclosed, unregistered look.")
        raise SystemExit("\n".join(lines))

    if registered and not same_code:
        print(f"     cross-tree comparison permitted by "
              f"analysis/outputs/equivalence_attestation.json")
        print(f"       verdict {attestation['verdict']}, all {len(ARM_ORDER)} arms reproduce "
              f"the other tree exactly at fold {attestation['fold']}")

    missing = [a for a in ARM_ORDER if a not in set(full["arm"]) or a not in set(restricted["arm"])]
    if missing:
        raise SystemExit("these arms are missing from one side: " + ", ".join(missing))

    # The restricted cohort is DEFINED by whom exp10 actually trained and tested on,
    # rather than by re-deriving the age filter here. One definition, one place.
    keep = set(pool(restricted, ARM_ORDER[0])["participant_id"])
    print(f"a21  restricted cohort: {len(keep)} participants (from the exp10 runs)")

    pooled_c = {a: pool(restricted, a) for a in ARM_ORDER}                  # (c)
    pooled_b = {a: pool(full, a, keep=keep) for a in ARM_ORDER}             # (b)
    pooled_a = {a: pool(full, a) for a in ARM_ORDER}                        # (a)

    for a in ARM_ORDER:
        for name, frame in (("c", pooled_c[a]), ("b", pooled_b[a])):
            if set(frame["participant_id"]) != keep:
                raise SystemExit(f"arm {a!r} side ({name}) does not cover the restricted cohort")
    n_full = len(set(pooled_a[ARM_ORDER[0]]["participant_id"]))
    print(f"     full cohort: {n_full} participants")

    # One bootstrap over the restricted cohort, with both sides of every arm in it,
    # so (b) and (c) are resampled together and their difference stays paired.
    stats_bc = {}
    for a in ARM_ORDER:
        stats_bc[f"{a} | trained full"] = SuffStats(pooled_b[a])
        stats_bc[f"{a} | trained age60"] = SuffStats(pooled_c[a])
    print(f"     participant-clustered bootstrap, {N_BOOT:,} resamples, seed {SEED} ...")
    boot_bc = clustered_bootstrap(stats_bc, "accuracy", n_boot=N_BOOT, seed=SEED)
    # (a) lives on a different participant set, so it gets its own bootstrap and is
    # reported as context rather than differenced against the others.
    boot_a = clustered_bootstrap({a: SuffStats(pooled_a[a]) for a in ARM_ORDER},
                                 "accuracy", n_boot=N_BOOT, seed=SEED)

    rows, findings, comps = [], [], []
    for a in ARM_ORDER:
        kb, kc = f"{a} | trained full", f"{a} | trained age60"
        acc_a = boot_a[a]["point"]
        acc_b = boot_bc[kb]["point"]
        acc_c = boot_bc[kc]["point"]
        d = paired_difference(boot_bc, kc, kb, family=FAMILY if registered else None)
        rows.append({
            "arm": a,
            "a_train_full_test_full": acc_a,
            "b_train_full_test_age60": acc_b,
            "c_train_age60_test_age60": acc_c,
            "test_composition_effect_a_minus_b": acc_a - acc_b,
            "training_effect_b_minus_c": acc_b - acc_c,
            "c_minus_b": d["diff"], "ci_low": d["ci_low"], "ci_high": d["ci_high"],
            "p_bootstrap": d["p_bootstrap"],
            "registered": registered,
        })
        findings.append(Finding(
            label=f"{a}: accuracy, trained age-60 minus trained full (both tested age-60)",
            value=d["diff"], ci=(d["ci_low"], d["ci_high"]), p=d["p_bootstrap"],
            family=FAMILY if registered else None,
            comparison_id=f"{a}_train_matched_vs_evaluated" if registered else None,
            fmt="{:+.3f}"))
        findings.append(Finding(label=f"{a}: test-composition effect (a minus b), accuracy points",
                                value=acc_a - acc_b, fmt="{:+.3f}"))

    stamp = "PRELIMINARY " if not registered else ""
    prov = Provenance(run_ids=list(restricted["run_id"]) + list(full["run_id"]),
                      code_fingerprint=fp_r[0], cohort="age60", unit="participant",
                      n_participants=len(keep), n_bootstrap=N_BOOT, rng_seed=SEED,
                      source_script=SOURCE)

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    xs = np.arange(len(ARM_ORDER)); w = 0.26
    for i, (key, lab, col) in enumerate([
            ("a_train_full_test_full", "(a) train full, test full", "#90a4ae"),
            ("b_train_full_test_age60", "(b) train full, test age-60", "#546e7a"),
            ("c_train_age60_test_age60", "(c) train age-60, test age-60", "#1565c0")]):
        ax.bar(xs + (i - 1) * w, [r[key] for r in rows], width=w, label=lab, color=col)
    ax.axhline(51.83, color="#b71c1c", ls="--", lw=1.2,
               label="majority-class share of the age-60 cohort")
    ax.set_xticks(xs); ax.set_xticklabels(ARM_ORDER, rotation=18, ha="right", fontsize=9)
    ax.set_ylabel("pooled accuracy (%)"); ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8); ax.set_title(f"{stamp}Decomposing the full-cohort headline")
    fig.tight_layout()

    emit_figure("f21_age_confound", fig,
                title=f"{stamp}Decomposing the full-cohort headline",
                shows=("Pooled accuracy for each objective under three conditions: trained and "
                       "tested on the full cohort, trained on the full cohort and tested on the "
                       "CDR-assessed age-60+ subset, and trained and tested on that subset. The "
                       "dashed line is the majority-class share of the restricted cohort."),
                provenance=prov, findings=findings,
                not_shown=("Condition (a) is computed on 347 participants and the other two on "
                           f"{len(keep)}. It is drawn for context and is never differenced "
                           "against them in a test.",
                           "Calibration. This figure is about accuracy, which is where the age "
                           "confound shows up most directly."),
                extra_caveats=(
                    ("Both sides are resampled on the same participant draw, so the (b) versus "
                     "(c) difference is paired."),
                    *(() if registered else (
                        "PRELIMINARY. The full-cohort arms came from a different tree under a "
                        "different code fingerprint, so part of any difference here may be code "
                        "rather than cohort. No comparison on this figure is registered and no "
                        "verdict is attached. Do not quote these numbers.",)),
                    *((f"The two sides come from different trees. That is permitted here "
                       f"because every arm was demonstrated to reproduce the other tree's run "
                       f"exactly, to the last recorded digit, at fold "
                       f"{attestation['fold']} with seed {attestation['init_seed']} under "
                       f"deterministic execution. The per-metric differences are recorded in "
                       f"analysis/outputs/equivalence_attestation.json, generated "
                       f"{attestation['generated_utc']}.",)
                      if (registered and not same_code) else ()),
                    *((f"This family is declared with method='none', so its interval is an "
                       f"ordinary 95% interval and no multiplicity correction is applied. It is "
                       f"descriptive by declaration.",) if registered else ())))
    plt.close(fig)

    save_table(pd.DataFrame(rows), "t21_age_confound" + ("" if registered else "_PRELIMINARY"))
    if registered:
        REGISTRY.write_audit()

    print(f"\n{'arm':<16} {'(a) full':>9} {'(b) full->60':>13} {'(c) 60->60':>11} "
          f"{'a-b':>8} {'b-c':>8}")
    for r in rows:
        print(f"{r['arm']:<16} {r['a_train_full_test_full']:>9.2f} "
              f"{r['b_train_full_test_age60']:>13.2f} {r['c_train_age60_test_age60']:>11.2f} "
              f"{r['test_composition_effect_a_minus_b']:>8.2f} "
              f"{r['training_effect_b_minus_c']:>8.2f}")
    print(f"\n     outputs in {OUT}")
    if not registered:
        print("     PRELIMINARY: nothing above is a result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
