#!/usr/bin/env python3
"""exp05 vs exp01: what session-level grouping was actually worth.

The manuscript's central methodological claim is that partitioning by scan
session rather than by participant leaks and inflates results. exp05 reruns the
identical pipeline with group_by=session -- the submitted protocol -- so the
difference against exp01 measures that claim instead of asserting it.

One caveat governs how this must be read, and it is stated in the output rather
than buried: session grouping produces 366 groups against 347, so the five folds
contain DIFFERENT people. The comparison is therefore between two pooled
estimates over the same cohort, not a paired fold-by-fold contrast, and the
bootstrap here resamples participants from each pooled set independently.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

from analysis._common import (CLASS_NAMES, N_CLASSES, OUT, SuffStats,  # noqa: E402
                              save_table)
from analysis.discover import load_predictions, load_runs            # noqa: E402
from src.figures import INK_MUTED, SERIES, _style, _save              # noqa: E402

N_BOOT = 2000
LOSS = "FA_FL"


def pool_allowing_repeats(runs: pd.DataFrame, loss: str) -> pd.DataFrame:
    """Pool the five test folds WITHOUT the participant-disjointness check.

    analysis/_common.pooled_predictions() refuses to pool when a participant
    appears in more than one test fold, which is the correct guard for exp01 --
    there it would mean the partitioning is broken. Here it is the point: under
    session-level grouping the two sessions of a repeat participant land in
    different folds by construction, so that guard would reject exp05 outright.

    What must still hold is that every SESSION is tested exactly once, so the
    pooled set covers the cohort once and no image is counted twice. That is
    asserted below instead.
    """
    sub = runs[runs["loss"] == loss].sort_values("fold")
    if sub.empty:
        raise SystemExit(f"no runs for loss {loss!r}")
    parts = []
    for _, r in sub.iterrows():
        frame = load_predictions(r["run_dir"], "test")
        probs = frame[[f"p{c}" for c in range(N_CLASSES)]].to_numpy()
        frame = frame[["participant_id", "label"]].copy()
        frame["pred"] = probs.argmax(axis=1)
        for c in range(N_CLASSES):
            frame[f"p{c}"] = probs[:, c]
        frame["fold"] = int(r["fold"])
        parts.append(frame)
    pooled = pd.concat(parts, ignore_index=True)

    # sessions, not participants, must be disjoint across folds
    sess = pd.concat([pd.read_csv(Path(r["run_dir"]) / "fold_participants.csv")
                      .query("split == 'test'").assign(fold=int(r["fold"]))
                      for _, r in sub.iterrows()], ignore_index=True)
    reused = sess.groupby("session_id")["fold"].nunique()
    if (reused > 1).any():
        raise SystemExit(f"{int((reused > 1).sum())} session(s) tested in more than one fold "
                         f"-- exp05 folds are not session-disjoint, refusing to pool")
    return pooled


def boot_ci(st: SuffStats, metric: str, cls=None, n_boot=N_BOOT, seed=7):
    rng = np.random.default_rng(seed)
    pull = lambda m: (m["per_class_f1"][cls] if cls is not None else m[metric])
    draws = np.array([pull(st.metrics(rng.integers(0, st.n_p, st.n_p))) for _ in range(n_boot)])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return pull(st.metrics()), float(lo), float(hi), draws


def main() -> int:
    print("[a08] leakage ablation -- session-level vs participant-level grouping")
    clean_runs = load_runs("exp01_main_sweep")
    leak_runs = load_runs("exp05_leakage_ablation")
    if leak_runs.empty:
        print("  exp05_leakage_ablation has no completed runs -- skipping")
        return 0
    clean_runs = clean_runs[clean_runs["loss"] == LOSS]
    if clean_runs.empty:
        print(f"  no exp01 runs for {LOSS} to compare against -- skipping")
        return 0

    # how much leakage the flawed protocol actually produced
    leaked = []
    for _, r in leak_runs.iterrows():
        fp = Path(r["run_dir"]) / "fold_participants.csv"
        d = pd.read_csv(fp)
        tr = set(d[d["split"] == "train"]["participant_id"])
        te = set(d[d["split"] == "test"]["participant_id"])
        va = set(d[d["split"] == "val"]["participant_id"])
        leaked.append({"fold": int(r["fold"]),
                       "train_test_overlap": len(tr & te),
                       "train_val_overlap": len(tr & va),
                       "test_participants": len(te)})
    lk = pd.DataFrame(leaked).sort_values("fold")
    save_table(lk, "leakage_counts")

    # cohort structure, measured rather than asserted: how many participants carry
    # more than one session, and which classes they belong to
    roster = pd.concat([pd.read_csv(Path(r["run_dir"]) / "fold_participants.csv")
                        for _, r in leak_runs.iterrows()],
                       ignore_index=True)[["participant_id", "session_id", "class_3"]]
    roster = roster.drop_duplicates()
    n_participants = roster["participant_id"].nunique()
    n_sessions = roster["session_id"].nunique()
    per_p = roster.groupby("participant_id")["session_id"].nunique()
    repeats = per_p[per_p > 1].index
    repeat_classes = (roster[roster["participant_id"].isin(repeats)]
                      .drop_duplicates("participant_id")["class_3"]
                      .value_counts().sort_index())
    repeat_desc = ", ".join(f"{int(v)} {CLASS_NAMES[int(k)]}" for k, v in repeat_classes.items()) \
        if len(repeats) else "none"


    clean = SuffStats(pool_allowing_repeats(clean_runs, LOSS))
    leak = SuffStats(pool_allowing_repeats(leak_runs, LOSS))
    print(f"  participant-level: {clean.n_p} participants, {int(clean.n_img.sum())} images")
    print(f"  session-level    : {leak.n_p} participants, {int(leak.n_img.sum())} images")

    rows = []
    for metric, cls, label in [("accuracy", None, "Accuracy (%)"), ("macro_f1", None, "Macro-F1"),
                               ("per_class_f1", 0, "F1 Non Demented"),
                               ("per_class_f1", 1, "F1 Very Mild"),
                               ("per_class_f1", 2, "F1 Demented"),
                               ("ece", None, "ECE"), ("brier", None, "Brier")]:
        c_pt, c_lo, c_hi, c_d = boot_ci(clean, metric, cls, seed=7)
        l_pt, l_lo, l_hi, l_d = boot_ci(leak, metric, cls, seed=91)
        diff = l_d - c_d              # independent pooled sets, so an unpaired difference
        dlo, dhi = np.percentile(diff, [2.5, 97.5])
        # Floored at 1/len(diff), as in a09 and _common.paired_difference.
        # Note also that this contrast is UNPAIRED across two pooled sets that
        # share essentially the same 347 participants, so its interval is
        # conservative (too wide), not merely "wider". a09 supersedes it.
        p = min(max(2.0 * min((diff <= 0).mean(), (diff >= 0).mean()), 1.0 / len(diff)), 1.0)
        rows.append({"Metric": label,
                     "Participant-level": c_pt, "CI low (participant)": c_lo,
                     "CI high (participant)": c_hi,
                     "Session-level": l_pt, "CI low (session)": l_lo,
                     "CI high (session)": l_hi,
                     "Inflation (session - participant)": l_pt - c_pt,
                     "CI low (diff)": float(dlo), "CI high (diff)": float(dhi),
                     "p (bootstrap)": float(min(p, 1.0)),
                     "CI excludes 0": bool(dlo > 0 or dhi < 0)})
    tab = pd.DataFrame(rows)
    save_table(tab, "leakage_ablation")

    show = tab[tab["Metric"].isin(["Accuracy (%)", "Macro-F1", "F1 Demented"])]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    x = np.arange(len(show))
    ax.bar(x - 0.19, show["Participant-level"], width=0.36, color=SERIES[0],
           label="participant-level grouping (correct)")
    ax.bar(x + 0.19, show["Session-level"], width=0.36, color=SERIES[1],
           label="session-level grouping (submitted protocol)")
    ax.set_xticks(x, list(show["Metric"]))
    ax.set_ylabel("pooled estimate")
    ax.set_title("What the flawed partitioning was worth", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED,
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncols=2)
    _style(ax); _save(fig, OUT / "fig_leakage_ablation.png")
    print("    -> fig_leakage_ablation.png")

    tt = int(lk["train_test_overlap"].sum())
    note = (
        f"Session-level grouping placed {lk['train_test_overlap'].min()}-"
        f"{lk['train_test_overlap'].max()} participants in BOTH the training and test "
        f"partition of each fold ({tt} participant-folds in total), because "
        f"{len(repeats)} of the {n_participants} participants contributed more than one "
        f"imaging session and the grouping key included the session number.\n\n"
        f"IMPORTANT: the two protocols produce different partitions ({n_sessions} groups "
        f"against {n_participants}), so their folds hold different people. This is a "
        "comparison of two pooled estimates over the same cohort, not a paired fold-by-fold "
        "contrast, and the bootstrap resamples each pooled set independently. Intervals are "
        "therefore wider than a paired design would give.\n\n"
        + "\n".join(
            f"{r['Metric']}: participant-level {r['Participant-level']:.4f}, session-level "
            f"{r['Session-level']:.4f}, difference {r['Inflation (session - participant)']:+.4f} "
            f"[{r['CI low (diff)']:.4f}, {r['CI high (diff)']:.4f}], p = {r['p (bootstrap)']:.4f}"
            for _, r in tab.iterrows())
        + f"\n\nInterpret the magnitude honestly. Only {len(repeats)} of {n_participants} "
          f"participants carry a repeat session ({repeat_desc}), so the numerical effect on "
          "this cohort is bounded by construction. The argument for participant-level "
          "partitioning rests on protocol validity, not on the size of the distortion "
          "measured here; a cohort with routine repeat imaging (OASIS-2, ADNI) would be "
          "affected far more.\n")
    (OUT / "leakage_note.txt").write_text(note, encoding="utf-8")
    print("    -> leakage_note.txt")
    print(tab[["Metric", "Participant-level", "Session-level",
               "Inflation (session - participant)", "p (bootstrap)",
               "CI excludes 0"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
