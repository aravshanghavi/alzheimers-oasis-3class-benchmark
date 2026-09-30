#!/usr/bin/env python3
"""The registered comparisons for the train-matched age-60 arms (exp10-exp13).

    python analysis/a20_age60_registered.py

WHY THIS IS A SEPARATE SCRIPT
-----------------------------
a01 through a18 are written against the full-cohort experiment names. a01, a02,
a03 and a06 call runs_frame(), whose default is ("exp01_main_sweep",
"exp02_class_balanced"); the rest name exp03, exp04, exp05, exp07 and exp08
explicitly. None of them can see this cohort, and retrofitting eight scripts to
take an arm set would risk changing results in the tree they were validated on.

WHAT IT COMPUTES
----------------
Pooled participant-level predictions (five grouped test folds concatenated gives
every one of the 166 participants exactly once), participant-clustered bootstrap
with every arm resampled on the SAME draw so differences stay paired, intervals
at each family's multiplicity-corrected level, and verdicts issued by the
registry rather than by this file.

THE FOUR QUESTIONS, AND WHICH FAMILY OWNS EACH
----------------------------------------------
1. Does FA-FL differ from the objectives that are not focal loss?
       age60_objective_ece, age60_objective_macro_f1        (3 each)
2. Does FA-FL differ from focal loss at ANY constant exponent?
       age60_gamma_control_ece, age60_gamma_control_macro_f1 (4 each)
3. Did any arm learn anything a constant predictor does not know?
       age60_arm_vs_constant_accuracy, _macro_f1             (8 each)
4. Where does FA-FL sit on the gamma ladder?
       Descriptive. A ranking, not a test. Five folds cannot power a test of a
       change in ranking, and registering one would let a null be misread as
       evidence that no migration happened.

THE DUAL THRESHOLD ON THE OBJECTIVE FAMILIES
--------------------------------------------
Those two families were corrected on 2026-09-20 from four comparisons to three,
because the original declaration counted focal-3.0 in both the objective family
and the gamma ladder. That relaxes their threshold from 0.0125 to 0.016667. To
remove any benefit the correction could confer, every comparison in them is
reported against both levels and one that clears only 0.016667 is marked NOT
ROBUST here and must not be called significant in the paper.
"""
from __future__ import annotations

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

from analysis._common import (CLASS_NAMES, N_CLASSES, SuffStats, clustered_bootstrap,   # noqa: E402
                              paired_difference, save_table)
from analysis._figure_md import Finding, Provenance, emit_figure                        # noqa: E402
from analysis._paths import OUT                                                         # noqa: E402
from analysis._registry import REGISTRY                                                 # noqa: E402
from analysis.discover import load_predictions, load_runs                               # noqa: E402

EXPERIMENTS = ["exp10_age60_main", "exp11_age60_gamma137",
               "exp12_age60_gamma200", "exp13_age60_gamma269"]

N_BOOT = 2000
SEED = 12345
SOURCE = "analysis/a20_age60_registered.py"

FA = "FA-FL"
BASELINE = "Constant majority"
# The pre-declared strict level for the objective families, before the
# 2026-09-20 correction removed focal-3.0 from them. Reported alongside the
# corrected level so the relaxation cannot quietly manufacture a result.
OBJECTIVE_STRICT_ALPHA = 0.0125

ARM_ORDER = ["Weighted CE", "LDAM", "Class-Balanced",
             "Focal g=1.37", "Focal g=2.00", "Focal g=2.69", "Focal g=3.00", FA]
NON_FOCAL = ["Weighted CE", "LDAM", "Class-Balanced"]
FOCAL_LADDER = [("Focal g=1.37", 1.37), ("Focal g=2.00", 2.00),
                ("Focal g=2.69", 2.69), ("Focal g=3.00", 3.00)]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def arm_label(manifest: Dict) -> str:
    t = manifest["loss"]["type"]
    if t == "FocalLoss":
        return f"Focal g={float(manifest['loss']['params']['gamma']):.2f}"
    return {"WCE": "Weighted CE", "LDAM": "LDAM",
            "ClassBalanced": "Class-Balanced", "FA_FL": FA}.get(t, t)


def load_arms() -> Tuple[pd.DataFrame, Dict]:
    frames = [load_runs(e) for e in EXPERIMENTS]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise SystemExit(
            "no COMPLETE runs for " + ", ".join(EXPERIMENTS) + ".\nRun run_all.bat first.")
    runs = pd.concat(frames, ignore_index=True)

    arms, effg, cohorts = [], {}, set()
    for _, r in runs.iterrows():
        m = json.loads((Path(r["run_dir"]) / "manifest.json").read_text(encoding="utf-8"))
        a = arm_label(m)
        arms.append(a)
        cr = m.get("criterion") or {}
        if cr.get("effective_gamma") is not None:
            effg.setdefault(a, []).append(float(cr["effective_gamma"]))
        c = m.get("cohort", {}) or {}
        cohorts.add((c.get("filter"), c.get("participants_after"), c.get("images_after")))
    runs["arm"] = arms

    fps = sorted(runs["code_fingerprint"].dropna().unique())
    if len(fps) > 1:
        raise SystemExit(f"refusing to compare across {len(fps)} code fingerprints: "
                         f"{[f[:8] for f in fps]}")
    if len(cohorts) > 1:
        raise SystemExit(f"refusing to pool across {len(cohorts)} cohort definitions: {cohorts}")

    missing = [a for a in ARM_ORDER if a not in set(arms)]
    if missing:
        raise SystemExit("these declared arms have no runs: " + ", ".join(missing))

    coh = sorted(cohorts)[0]
    meta = {"fingerprint": fps[0] if fps else "",
            "cohort_filter": coh[0], "n_participants": coh[1], "n_images": coh[2],
            "effective_gamma": {a: float(np.mean(v)) for a, v in effg.items()}}
    return runs, meta


def pool_arm(runs: pd.DataFrame, arm: str) -> pd.DataFrame:
    sub = runs[runs["arm"] == arm].sort_values("fold")
    parts = []
    for _, r in sub.iterrows():
        f = load_predictions(r["run_dir"], "test")
        f["fold"] = int(r["fold"])
        parts.append(f[["participant_id", "label", "pred", "fold"]
                       + [f"p{c}" for c in range(N_CLASSES)]])
    pooled = pd.concat(parts, ignore_index=True)
    dup = pooled.groupby("participant_id")["fold"].nunique()
    if (dup > 1).any():
        raise SystemExit(f"{int((dup > 1).sum())} participant(s) of arm {arm!r} appear in more "
                         f"than one test fold; refusing to pool")
    return pooled


def constant_baseline(template: pd.DataFrame) -> Tuple[pd.DataFrame, int, np.ndarray]:
    """A predictor that always names the cohort's majority class.

    It is handed the majority class and the class prior for free, computed from
    the very labels it is scored against. That is deliberately generous: a CNN
    that cannot beat a baseline given free access to the answer key's marginal
    distribution has not earned a claim.
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


# ---------------------------------------------------------------------------
# Comparison bookkeeping
# ---------------------------------------------------------------------------

class Comparison:
    """One registered comparison: the numbers, the Finding, and the verdict."""

    def __init__(self, family: str, cid: str, label: str, boot, a: str, b: str,
                 fmt: str = "{:+.4f}", strict_alpha: Optional[float] = None):
        d = paired_difference(boot, a, b, family=family)
        self.family, self.cid, self.label = family, cid, label
        self.a, self.b, self.d = a, b, d
        self.finding = Finding(label=label, value=d["diff"],
                               ci=(d["ci_low"], d["ci_high"]), p=d["p_bootstrap"],
                               family=family, comparison_id=cid, fmt=fmt)
        self.verdict = self.finding.resolve()
        self.strict_alpha = strict_alpha
        # Robust only if it would also survive the stricter pre-correction level.
        self.robust = (None if strict_alpha is None
                       else bool(self.verdict.significant and d["p_bootstrap"] < strict_alpha))

    def row(self) -> Dict:
        return {"family": self.family, "comparison": self.cid, "label": self.label,
                "arm_a": self.a, "arm_b": self.b,
                "difference": self.d["diff"],
                "ci_low": self.d["ci_low"], "ci_high": self.d["ci_high"],
                "ci_level": self.d["ci_level"],
                "p_bootstrap": self.d["p_bootstrap"],
                "p_at_resolution_floor": self.d["p_at_resolution_floor"],
                "alpha_corrected": self.verdict.alpha_corrected,
                "significant": self.verdict.significant,
                "p_and_ci_disagree": self.verdict.disagreement,
                "strict_alpha": self.strict_alpha,
                "robust_at_strict_alpha": self.robust}


def forest(comparisons: List[Comparison], title: str, xlabel: str, stem: str,
           shows: str, prov: Provenance, not_shown, extra_caveats=()):
    fig, ax = plt.subplots(figsize=(9.0, 0.62 * len(comparisons) + 2.4))
    ys = np.arange(len(comparisons))[::-1]
    for y, c in zip(ys, comparisons):
        sig = c.verdict.significant
        col = "#1b5e20" if sig else "#616161"
        ax.plot([c.d["ci_low"], c.d["ci_high"]], [y, y], color=col, lw=2.2)
        ax.plot([c.d["diff"]], [y], "o", color=col, ms=7)
    ax.axvline(0.0, color="#b71c1c", ls="--", lw=1.2)
    ax.set_yticks(ys)
    ax.set_yticklabels([c.label for c in comparisons], fontsize=9)
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.25, axis="x")
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    png, md = emit_figure(stem, fig, title=title, shows=shows, provenance=prov,
                          findings=[c.finding for c in comparisons],
                          not_shown=not_shown, extra_caveats=extra_caveats)
    plt.close(fig)
    return png, md


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    runs, meta = load_arms()
    print(f"a20  {len(runs)} runs, {len(ARM_ORDER)} arms, cohort "
          f"{meta['cohort_filter']} n={meta['n_participants']}")

    pooled = {a: pool_arm(runs, a) for a in ARM_ORDER}
    base_frame, maj, prior = constant_baseline(pooled[ARM_ORDER[0]])
    pooled[BASELINE] = base_frame

    # Every arm must cover exactly the same people, or the pairing is a fiction.
    ref = set(pooled[ARM_ORDER[0]]["participant_id"])
    for a, f in pooled.items():
        if set(f["participant_id"]) != ref:
            raise SystemExit(f"arm {a!r} covers a different participant set; cannot pair")
    print(f"     pooled {len(ref)} participants per arm; majority class "
          f"{maj} ({CLASS_NAMES[maj]}), prior {np.round(prior, 4).tolist()}")

    stats = {a: SuffStats(f) for a, f in pooled.items()}
    print(f"     participant-clustered bootstrap, {N_BOOT:,} resamples, seed {SEED} ...")
    boot = {m: clustered_bootstrap(stats, m, n_boot=N_BOOT, seed=SEED)
            for m in ("ece", "macro_f1", "accuracy")}

    prov = Provenance(run_ids=list(runs["run_id"]), code_fingerprint=meta["fingerprint"],
                      cohort="age60" if meta["cohort_filter"] == "cdr_age" else "full",
                      unit="participant", n_participants=meta["n_participants"],
                      n_images=meta["n_images"], n_bootstrap=N_BOOT, rng_seed=SEED,
                      source_script=SOURCE)

    comps: List[Comparison] = []

    # -- 1. FA-FL against the objectives that are not focal loss -------------
    obj_ece = [Comparison("age60_objective_ece", f"FA_FL_vs_{a}", f"ECE: {FA} minus {a}",
                          boot["ece"], FA, a, strict_alpha=OBJECTIVE_STRICT_ALPHA)
               for a in NON_FOCAL]
    obj_f1 = [Comparison("age60_objective_macro_f1", f"FA_FL_vs_{a}",
                         f"macro-F1: {FA} minus {a}", boot["macro_f1"], FA, a,
                         strict_alpha=OBJECTIVE_STRICT_ALPHA)
              for a in NON_FOCAL]

    # -- 2. FA-FL against every constant exponent ----------------------------
    gam_ece = [Comparison("age60_gamma_control_ece", f"FA_FL_vs_{a}", f"ECE: {FA} minus {a}",
                          boot["ece"], FA, a) for a, _ in FOCAL_LADDER]
    gam_f1 = [Comparison("age60_gamma_control_macro_f1", f"FA_FL_vs_{a}",
                         f"macro-F1: {FA} minus {a}", boot["macro_f1"], FA, a)
              for a, _ in FOCAL_LADDER]

    # -- 3. Every arm against the constant predictor -------------------------
    con_acc = [Comparison("age60_arm_vs_constant_accuracy", f"{a}_vs_constant",
                          f"accuracy: {a} minus constant", boot["accuracy"], a, BASELINE,
                          fmt="{:+.3f}") for a in ARM_ORDER]
    con_f1 = [Comparison("age60_arm_vs_constant_macro_f1", f"{a}_vs_constant",
                         f"macro-F1: {a} minus constant", boot["macro_f1"], a, BASELINE)
              for a in ARM_ORDER]
    comps += obj_ece + obj_f1 + gam_ece + gam_f1 + con_acc + con_f1

    # -- figures -------------------------------------------------------------
    DUAL = ("These three comparisons are reported against two thresholds. Their family was "
            f"corrected on 2026-09-20 from four comparisons to three, which relaxed the "
            f"corrected level from {OBJECTIVE_STRICT_ALPHA} to "
            f"{REGISTRY.alpha_corrected('age60_objective_ece'):.6g}. The table "
            f"t20_comparisons.csv carries a robust_at_strict_alpha column, and a comparison "
            f"that clears only the relaxed level is not to be described as significant.")
    PAIRED = ("Every arm is resampled on the SAME participant draw, so the differences are "
              "paired and the interval is on the difference rather than on two independent "
              "estimates.")
    CKPT = ("All metrics are computed at the checkpoint selected on validation accuracy. a19 "
            "shows every arm reaches roughly half its final calibration error around epoch 5 "
            "or 6 and then degrades until selection stops it near epoch 11. The selection rule "
            "is identical across arms, so it does not favour one, but these ECE values describe "
            "the selected checkpoint and not the best the objective can do.")

    forest(obj_ece + obj_f1,
           f"{FA} against the non-focal objectives",
           "difference (negative favours FA-FL on ECE, positive on macro-F1)",
           "f20a_objective_comparisons",
           shows=("Paired differences between FA-FL and each non-focal objective, pooled over "
                  "all five grouped test folds. Bars are bootstrap intervals at the family's "
                  "multiplicity-corrected level, not at 95 percent."),
           prov=prov,
           not_shown=("Comparisons against focal loss. Those belong to the gamma-control "
                      "family and appear in f20b.",
                      "Anything about the full cohort. Every number here is from models "
                      "trained and tested on the CDR-assessed age-60-and-over cohort."),
           extra_caveats=(DUAL, PAIRED, CKPT))

    forest(gam_ece + gam_f1,
           f"{FA} against focal loss at four constant exponents",
           "difference (negative favours FA-FL on ECE, positive on macro-F1)",
           "f20b_gamma_control",
           shows=("Paired differences between FA-FL and constant-gamma focal loss at 1.37, "
                  "2.00, 2.69 and 3.00, pooled over the five grouped test folds. Intervals are "
                  "at this family's corrected level."),
           prov=prov,
           not_shown=("Whether a constant exponent OUTSIDE this ladder would match FA-FL. The "
                      "ladder brackets FA-FL's measured effective exponent of "
                      f"{meta['effective_gamma'].get(FA, float('nan')):.3f} and the 2.69 rung "
                      "sits within 0.04 of it, but it cannot rule out every scalar.",
                      "That matching on a pooled metric means the two objectives behave the "
                      "same way during training. a19 measures that directly and finds they do "
                      "not."),
           extra_caveats=(PAIRED, CKPT))

    forest(con_acc + con_f1,
           "Every arm against a predictor that always names the majority class",
           "difference (positive means the arm beats the constant predictor)",
           "f20c_arm_vs_constant",
           shows=("Paired differences between each arm and a constant majority-class predictor "
                  "on accuracy (percentage points) and macro-F1, pooled over the five grouped "
                  "test folds, at this family's corrected level."),
           prov=prov,
           not_shown=("Clinical usefulness. Beating a constant predictor is the floor, not a "
                      "claim.",
                      "Any comparison between arms. This family asks only whether each arm "
                      "learned something the class prior does not already contain."),
           extra_caveats=(PAIRED,
                          f"The constant predictor is given the majority class and the class "
                          f"prior computed from the same labels it is scored against. That is "
                          f"deliberately generous to the baseline.",
                          "Accuracy and macro-F1 can disagree here by construction: a constant "
                          "predictor scores the majority share on accuracy and about "
                          f"{2 * prior[maj] / (1 + prior[maj]):.3f} divided by three on "
                          "macro-F1. Read both."))

    # -- the gamma ladder, descriptive --------------------------------------
    gam_x = [g for _, g in FOCAL_LADDER]
    gam_y = [boot["ece"][a]["point"] for a, _ in FOCAL_LADDER]
    gam_lo = [boot["ece"][a]["ci_low"] for a, _ in FOCAL_LADDER]
    gam_hi = [boot["ece"][a]["ci_high"] for a, _ in FOCAL_LADDER]
    fa_eff = meta["effective_gamma"].get(FA, float("nan"))
    fa_pt = boot["ece"][FA]["point"]

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.errorbar(gam_x, gam_y, yerr=[np.array(gam_y) - np.array(gam_lo),
                                    np.array(gam_hi) - np.array(gam_y)],
                fmt="o-", color="#37474f", capsize=4, lw=1.6, label="constant-gamma focal")
    ax.axhspan(boot["ece"][FA]["ci_low"], boot["ece"][FA]["ci_high"],
               color="#1565c0", alpha=0.13, lw=0)
    ax.axhline(fa_pt, color="#1565c0", lw=2.0, label=f"{FA} (pooled ECE)")
    ax.axvline(fa_eff, color="#1565c0", ls=":", lw=1.4,
               label=f"{FA} effective exponent {fa_eff:.3f}")
    ax.set_xlabel("constant focusing exponent gamma")
    ax.set_ylabel("pooled participant-level ECE")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax.set_title("Where FA-FL sits on the constant-gamma ladder")
    fig.tight_layout()

    nearest = min(FOCAL_LADDER, key=lambda ag: abs(boot["ece"][ag[0]]["point"] - fa_pt))
    ladder_findings = [Finding(label=f"pooled ECE, {a}", value=boot['ece'][a]['point'],
                               ci=(boot['ece'][a]['ci_low'], boot['ece'][a]['ci_high']),
                               fmt="{:.4f}") for a, _ in FOCAL_LADDER]
    ladder_findings.append(Finding(label=f"pooled ECE, {FA}", value=fa_pt,
                                   ci=(boot['ece'][FA]['ci_low'], boot['ece'][FA]['ci_high']),
                                   fmt="{:.4f}"))
    ladder_findings.append(Finding(label=f"{FA} measured effective exponent",
                                   value=fa_eff, fmt="{:.4f}"))
    ladder_findings.append(Finding(label="nearest constant-gamma arm by pooled ECE",
                                   value=nearest[1], fmt="{:.2f}"))
    emit_figure("f20d_gamma_ladder", fig,
                title="Where FA-FL sits on the constant-gamma ladder",
                shows=("Pooled participant-level ECE for each constant-gamma focal arm against "
                       "its exponent, with FA-FL drawn as a horizontal line and band and its "
                       "measured effective exponent as a vertical line. Intervals are "
                       "participant-clustered bootstrap percentiles."),
                provenance=prov, findings=ladder_findings,
                not_shown=("A test. Which arm is nearest is a ranking read off point estimates, "
                           "and five folds cannot power a test of a change in ranking between "
                           "cohorts. The registered question is whether FA-FL differs from each "
                           "rung, and that is f20b.",),
                extra_caveats=(
                    "No comparison on this figure is registered, so no verdict is attached to "
                    "any of them. Treating the nearest-arm reading as a result would be exactly "
                    "the post-hoc selection the registry exists to prevent.", PAIRED, CKPT))
    plt.close(fig)

    # -- the Demented class, descriptive ------------------------------------
    # NOT pre-registered and NOT tested. It is here because it is the class the
    # paper is about, because a19 predicts exactly this pattern from the
    # training-time loss shares, and because leaving out an inconvenient
    # descriptive result is worse than reporting one without a verdict.
    boot_dem = clustered_bootstrap(stats, "per_class_f1", n_boot=N_BOOT, seed=SEED, per_class=2)
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    xs = np.arange(len(ARM_ORDER))
    pts = [boot_dem[a]["point"] for a in ARM_ORDER]
    lo = [p_ - boot_dem[a]["ci_low"] for p_, a in zip(pts, ARM_ORDER)]
    hi = [boot_dem[a]["ci_high"] - p_ for p_, a in zip(pts, ARM_ORDER)]
    cols = ["#1565c0" if a == FA else "#607d8b" for a in ARM_ORDER]
    ax.bar(xs, pts, color=cols, width=0.62)
    ax.errorbar(xs, pts, yerr=[lo, hi], fmt="none", ecolor="#263238", capsize=4, lw=1.3)
    ax.set_xticks(xs)
    ax.set_xticklabels(ARM_ORDER, rotation=22, ha="right", fontsize=9)
    ax.set_ylabel("pooled F1 on the Demented class")
    ax.grid(alpha=0.25, axis="y")
    ax.set_title("Performance on the rarest class, which is the class the method targets")
    fig.tight_layout()
    dem_findings = [Finding(label=f"Demented-class F1, {a}", value=boot_dem[a]["point"],
                            ci=(boot_dem[a]["ci_low"], boot_dem[a]["ci_high"]), fmt="{:.4f}")
                    for a in ARM_ORDER]
    dem_findings += [Finding(label=f"Demented-class recall, {a}",
                             value=float(stats[a].metrics()["per_class_recall"][2]),
                             fmt="{:.4f}") for a in ARM_ORDER]
    emit_figure("f20e_dementia_class", fig,
                title="Performance on the rarest class",
                shows=("Pooled participant-level F1 on the Demented class for each arm, with "
                       "participant-clustered bootstrap intervals. FA-FL is highlighted. "
                       "Per-class recall for the same arms is listed in the findings."),
                provenance=prov, findings=dem_findings,
                not_shown=("A significance verdict. None of these comparisons is pre-registered, "
                           "so none is tested here. They are reported as description.",
                           "Why the pattern arises. a19 measures the training-time loss share "
                           "per class and is where that question is answered."),
                extra_caveats=(
                    "This figure is descriptive and was produced after the registered "
                    "comparisons. It is included because the Demented class is the clinical "
                    "target of the method, and because omitting it would be selective "
                    "reporting. No claim is made from it without a pre-registered test.",
                    "23 of the 166 participants are Demented, roughly four or five per fold. "
                    "These intervals are wide for that reason and small differences between "
                    "adjacent arms should not be read as real.", PAIRED, CKPT))
    plt.close(fig)

    # -- tables --------------------------------------------------------------
    rows = []
    for a in ARM_ORDER + [BASELINE]:
        m = stats[a].metrics()
        rows.append({"arm": a,
                     "effective_gamma": meta["effective_gamma"].get(a),
                     "ece": boot["ece"][a]["point"],
                     "ece_ci_low": boot["ece"][a]["ci_low"],
                     "ece_ci_high": boot["ece"][a]["ci_high"],
                     "macro_f1": boot["macro_f1"][a]["point"],
                     "macro_f1_ci_low": boot["macro_f1"][a]["ci_low"],
                     "macro_f1_ci_high": boot["macro_f1"][a]["ci_high"],
                     "accuracy": boot["accuracy"][a]["point"],
                     "accuracy_ci_low": boot["accuracy"][a]["ci_low"],
                     "accuracy_ci_high": boot["accuracy"][a]["ci_high"],
                     "brier": m["brier"],
                     # The constant predictor is calibrated by construction on the
                     # full sample: it reports the class prior and scores exactly the
                     # majority share, so its ECE point estimate is ~0 while every
                     # resample moves accuracy off the prior and raises it. The point
                     # estimate therefore falls outside its own interval. That is a
                     # property of the baseline, not an error, and it is inert because
                     # the baseline appears in no ECE family.
                     "notes": ("ECE is degenerate for a prior-reporting constant predictor; "
                               "point estimate lies outside its own bootstrap interval by "
                               "construction. Not used in any ECE comparison."
                               if a == BASELINE else ""),
                     "f1_dementia": float(m["per_class_f1"][2]),
                     "recall_dementia": float(m["per_class_recall"][2]),
                     "n_participants": m["n_participants"], "n_images": m["n_images"]})
    save_table(pd.DataFrame(rows), "t20_pooled_metrics")
    save_table(pd.DataFrame([c.row() for c in comps]), "t20_comparisons", float_fmt="%.6f")

    # -- registry audit ------------------------------------------------------
    audit = REGISTRY.audit()
    bad = {k: v for k, v in audit.items()
           if k.startswith("age60") and v["status"] == "PROTOCOL VIOLATION"}
    print("\n     registry audit (age60 families):")
    for k in sorted(audit):
        if k.startswith("age60"):
            v = audit[k]
            print(f"       {k:<38} declared {v['declared']:>2}  computed {v['seen']:>2}  "
                  f"{v['status']}")
    REGISTRY.write_audit()

    sig = [c for c in comps if c.verdict.significant]
    print(f"\n     {len(comps)} registered comparisons, {len(sig)} distinguishable from zero "
          f"after family correction")
    for c in sig:
        tag = "" if c.robust is not False else "   [NOT ROBUST at the pre-correction level]"
        print(f"       {c.label:<46} {c.d['diff']:+.4f}  p={c.d['p_bootstrap']:.4g}{tag}")
    print(f"\n     tables and figures in {OUT}")
    if bad:
        print("\n     PROTOCOL VIOLATION in: " + ", ".join(sorted(bad)))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
