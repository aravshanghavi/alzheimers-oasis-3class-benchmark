#!/usr/bin/env python3
"""Tables 1-4: cohort, overall performance, per-class, and the full aggregate.

SD convention is fixed once here and stated in every caption. The submitted
manuscript reported different SDs in Table 2 and Table 4 for the same runs
because they were computed in two places; that cannot recur.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analysis._common import (CLASS_NAMES, LOSS_LABEL, LOSS_ORDER, N_CLASSES, OUT,  # noqa: E402
                              SHORT, SuffStats, cohort_config_from_runs, fold_metrics,
                              pooled_predictions, runs_frame, save_table)
from src.data import cohort_summary, scan_dataset  # noqa: E402

DDOF = 1          # sample SD. Stated in every caption; used nowhere else.


def table1_cohort(runs: pd.DataFrame) -> pd.DataFrame:
    cfg = cohort_config_from_runs(runs)
    df = scan_dataset(cfg["data"]["root"], cfg["data"]["original_classes"], cfg["data"]["class_map"])
    cs = cohort_summary(df)
    part = df.groupby("participant_id")["class_3"].first()
    sess = df.groupby("session_id")["class_3"].first()
    rows = []
    for c, name in enumerate(CLASS_NAMES):
        rows.append({"Class": name,
                     "Participants": int((part == c).sum()),
                     "Sessions": int((sess == c).sum()),
                     "Images": int((df["class_3"] == c).sum())})
    rows.append({"Class": "Total", "Participants": int(len(part)),
                 "Sessions": int(len(sess)), "Images": int(len(df))})
    out = pd.DataFrame(rows)
    (OUT / "table1_cohort_note.txt").write_text(
        f"Participants are the unit of analysis and of partitioning. The cohort holds "
        f"{cs['participants']} participants across {cs['sessions']} imaging sessions: "
        f"{cs['participants_multi_session']} participants (all Non-Demented, the OASIS-1 "
        f"reliability subset) were scanned twice. The submitted manuscript reported the "
        f"session count ({cs['sessions']}) as a subject count.\n"
        f"Each session contributes {cs['slice_positions']} slice positions "
        f"(index {cs['slice_range'][0]}-{cs['slice_range'][1]}) from "
        f"{min(cs['acquisitions_per_session'])}-{max(cs['acquisitions_per_session'])} "
        f"MPRAGE acquisitions; repeated acquisitions of the same slice are near-duplicates, "
        f"so images are far from independent observations.\n", encoding="utf-8")
    return out


def main() -> int:
    print("[a01] tables")
    runs = runs_frame()
    fm = fold_metrics(runs)

    print("  Table 1: cohort")
    try:
        save_table(table1_cohort(runs), "table1_cohort", float_fmt="%.0f")
    except Exception as exc:                                   # noqa: BLE001
        print(f"    !! could not rebuild Table 1 -- the image folder recorded in the "
              f"manifests is not reachable from here ({exc}). Every other table is "
              f"computed from saved predictions and is unaffected.")

    # ---- Tables 2 and 3: fold-wise mean +/- SD --------------------------------
    print("  Tables 2-3: fold-wise mean +/- SD (ddof=%d)" % DDOF)
    t2, t3 = [], []
    for L in LOSS_ORDER:
        g = fm[fm["loss"] == L]
        if g.empty:
            continue
        t2.append({"Loss": LOSS_LABEL[L],
                   "Accuracy (%)": g["accuracy"].mean(), "Accuracy SD": g["accuracy"].std(ddof=DDOF),
                   "Macro-F1": g["macro_f1"].mean(), "Macro-F1 SD": g["macro_f1"].std(ddof=DDOF),
                   "Folds": len(g)})
        row = {"Loss": LOSS_LABEL[L]}
        for c in range(N_CLASSES):
            row[f"F1 {CLASS_NAMES[c]}"] = g[f"f1_{SHORT[c]}"].mean()
            row[f"SD {SHORT[c]}"] = g[f"f1_{SHORT[c]}"].std(ddof=DDOF)
        t3.append(row)
    save_table(pd.DataFrame(t2), "table2_overall")
    save_table(pd.DataFrame(t3), "table3_perclass")

    # ---- Table 4: pooled estimate over the whole cohort -----------------------
    print("  Table 4: pooled over all participants (each tested exactly once)")
    rows = []
    for L in LOSS_ORDER:
        if fm[fm["loss"] == L].empty:
            continue
        raw = SuffStats(pooled_predictions(runs, L)).metrics()
        try:
            ts = SuffStats(pooled_predictions(runs, L, temperature_scaled=True)).metrics()
            ece_ts, brier_ts = ts["ece"], ts["brier"]
        except SystemExit:
            ece_ts = brier_ts = float("nan")
        g = fm[fm["loss"] == L]
        rows.append({"Loss": LOSS_LABEL[L],
                     "Accuracy (%)": raw["accuracy"], "Macro-F1": raw["macro_f1"],
                     **{f"F1 {SHORT[c]}": raw["per_class_f1"][c] for c in range(N_CLASSES)},
                     "ECE": raw["ece"], "Brier": raw["brier"],
                     "ECE post-T": ece_ts, "Brier post-T": brier_ts,
                     "Mean T": g["temperature"].mean(),
                     "Participants": raw["n_participants"], "Images": raw["n_images"]})
    t4 = pd.DataFrame(rows)
    save_table(t4, "table4_pooled")

    # fold-level detail, so every aggregate above is traceable
    save_table(fm.drop(columns=["run_dir"]), "table_foldwise_detail")

    caption = (
        "Table 2/3: mean and standard deviation across the five participant-level "
        f"cross-validation folds (sample SD, ddof={DDOF}).\n"
        "Table 4: POOLED estimate. Because grouped 5-fold CV tests every participant "
        "exactly once, the union of the five test folds is the entire cohort "
        f"({int(t4['Participants'].iloc[0])} participants, {int(t4['Images'].iloc[0]):,} images). "
        "The pooled figure is the one to lead with: a single fold's minority-class metric "
        "rests on four or five people, whereas the pooled estimate uses all 23 Demented "
        "participants.\n"
        "'post-T' columns apply a single temperature fitted on that fold's VALIDATION split "
        "and applied to its test split. Accuracy is unchanged by construction.\n")
    (OUT / "tables_caption.txt").write_text(caption, encoding="utf-8")
    print("    -> tables_caption.txt")
    print(t4.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
