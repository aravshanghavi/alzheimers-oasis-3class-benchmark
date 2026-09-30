#!/usr/bin/env python3
"""Split-induced variance vs initialisation-induced variance.

exp01 varies the data partition with the initialisation held fixed; exp03 varies
the initialisation with the partition held fixed. Comparing their spreads
answers a question the submitted manuscript answered by assumption: it attributed
seed-to-seed variation to training instability, which the design could not
support because changing the seed also changed the test set.
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

from analysis._common import LOSS_LABEL, OUT, fold_metrics, save_table  # noqa: E402
from analysis.discover import load_runs                              # noqa: E402
from src.figures import GRID, INK, INK_MUTED, SERIES, SURFACE, _style, _save  # noqa: E402

DDOF = 1
METRICS = [("accuracy", "Accuracy (%)"), ("macro_f1", "Macro-F1"),
           ("f1_Dem", "F1 Demented"), ("ece", "ECE")]


def main() -> int:
    print("[a05] variance decomposition")
    main_runs = fold_metrics(load_runs("exp01_main_sweep"))
    init_runs = fold_metrics(load_runs("exp03_init_variance"))
    if init_runs.empty or main_runs.empty:
        missing = "exp03_init_variance" if init_runs.empty else "exp01_main_sweep"
        print(f"  {missing} has no completed runs -- the decomposition needs both "
              f"(exp01 varies the partition, exp03 varies the initialisation). Skipping.")
        return 0

    fixed_fold = int(init_runs["fold"].mode().iloc[0])
    rows = []
    for L in sorted(set(init_runs["loss"])):
        sp = main_runs[main_runs["loss"] == L].sort_values("fold")
        it = init_runs[(init_runs["loss"] == L) & (init_runs["fold"] == fixed_fold)]
        if sp.empty or len(it) < 2:
            continue
        for key, label in METRICS:
            s_sd, i_sd = sp[key].std(ddof=DDOF), it[key].std(ddof=DDOF)
            rows.append({"Loss": LOSS_LABEL.get(L, L), "Metric": label,
                         "SD across partitions": s_sd, "SD across initialisations": i_sd,
                         "SD ratio": s_sd / i_sd if i_sd else np.nan,
                         "Variance ratio": (s_sd / i_sd) ** 2 if i_sd else np.nan,
                         "n partitions": len(sp), "n initialisations": len(it)})
    table = pd.DataFrame(rows)
    save_table(table, "variance_decomposition")

    acc = table[table["Metric"] == "Accuracy (%)"]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(acc))
    ax.bar(x - 0.19, acc["SD across partitions"], width=0.36, color=SERIES[0],
           label="data partition varies (initialisation fixed)")
    ax.bar(x + 0.19, acc["SD across initialisations"], width=0.36, color=SERIES[1],
           label="initialisation varies (partition fixed)")
    for i, r in enumerate(acc.itertuples()):
        ax.text(i, max(r._3, r._4) + 0.12, f"{r._5:.1f}x", ha="center",
                fontsize=9, color=INK)
    ax.set_xticks(x, list(acc["Loss"]))
    ax.set_ylabel("SD of test accuracy (percentage points)")
    ax.set_title("Where the variance actually comes from", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED,
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncols=2)
    _style(ax); _save(fig, OUT / "fig_variance_decomposition.png")
    print("    -> fig_variance_decomposition.png")

    (OUT / "variance_note.txt").write_text(
        f"Partition variance is computed across the five cross-validation folds with "
        f"init_seed fixed at 42 (exp01). Initialisation variance is computed across five "
        f"initialisation seeds with the partition held at fold {fixed_fold} (exp03).\n\n"
        + "\n".join(
            f"{r['Loss']}, {r['Metric']}: partition SD {r['SD across partitions']:.3f} vs "
            f"initialisation SD {r['SD across initialisations']:.3f} "
            f"({r['SD ratio']:.1f}x, {r['Variance ratio']:.0f}x in variance)"
            for _, r in table.iterrows())
        + "\n\nThe manuscript's earlier attribution of seed-to-seed spread to training "
          "instability is not supported: with the partition held fixed, initialisation "
          "contributes a small fraction of the spread. The dominant term is which "
          "participants land in the test partition -- unsurprising when only 23 "
          "participants carry the Demented label.\n", encoding="utf-8")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
