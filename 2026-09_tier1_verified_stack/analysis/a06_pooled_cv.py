#!/usr/bin/env python3
"""Pooled confusion matrix, ROC and PR over the whole cohort.

Grouped 5-fold CV tests every participant exactly once, so the union of the five
test folds is the entire cohort. These replace the submitted manuscript's
Figures 2 and 4, which came from a single favourable run -- the emphasis
Reviewer 3 objected to in comment 6.
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
from sklearn.metrics import (auc, average_precision_score,           # noqa: E402
                             precision_recall_curve, roc_curve)

from analysis._common import (CLASS_NAMES, LOSS_LABEL, LOSS_ORDER, N_CLASSES,  # noqa: E402
                              OUT, SuffStats, fold_metrics, pooled_predictions,
                              runs_frame, save_table)
from src.figures import (INK, INK_MUTED, SERIES, confusion_matrix_figure,  # noqa: E402
                         _style, _save)


def main() -> int:
    print("[a06] pooled cross-validated results")
    runs = runs_frame()
    fm = fold_metrics(runs)
    losses = [L for L in LOSS_ORDER if not fm[fm["loss"] == L].empty]

    rows = []
    for L in losses:
        pooled = pooled_predictions(runs, L)
        st = SuffStats(pooled)
        m = st.metrics()
        probs = pooled[[f"p{c}" for c in range(N_CLASSES)]].to_numpy()
        y = pooled["label"].to_numpy()

        confusion_matrix_figure(
            m["confusion_matrix"], CLASS_NAMES,
            f"Pooled confusion matrix — {LOSS_LABEL[L]}\n"
            f"all {m['n_participants']} participants, {m['n_images']:,} images",
            OUT / f"fig_confusion_pooled_{L}.png")

        aucs, aps = [], []
        for c in range(N_CLASSES):
            yb = (y == c).astype(int)
            fpr, tpr, _ = roc_curve(yb, probs[:, c])
            aucs.append(auc(fpr, tpr))
            aps.append(average_precision_score(yb, probs[:, c]))
        rows.append({"Loss": LOSS_LABEL[L], "Accuracy (%)": m["accuracy"],
                     "Macro-F1": m["macro_f1"],
                     **{f"AUC {CLASS_NAMES[c]}": aucs[c] for c in range(N_CLASSES)},
                     **{f"AP {CLASS_NAMES[c]}": aps[c] for c in range(N_CLASSES)},
                     "Macro AUC": float(np.mean(aucs)), "Macro AP": float(np.mean(aps))})

        if L == "FA_FL":
            for kind in ("roc", "pr"):
                fig, ax = plt.subplots(figsize=(6.0, 5.6))
                for c in range(N_CLASSES):
                    yb = (y == c).astype(int)
                    if kind == "roc":
                        fpr, tpr, _ = roc_curve(yb, probs[:, c])
                        ax.plot(fpr, tpr, color=SERIES[c], lw=2,
                                label=f"{CLASS_NAMES[c]} (AUC {aucs[c]:.3f})")
                    else:
                        pr, rc, _ = precision_recall_curve(yb, probs[:, c])
                        ax.plot(rc, pr, color=SERIES[c], lw=2,
                                label=f"{CLASS_NAMES[c]} (AP {aps[c]:.3f})")
                        ax.axhline(yb.mean(), color=SERIES[c], lw=1, ls=(0, (2, 3)), alpha=0.55)
                if kind == "roc":
                    ax.plot([0, 1], [0, 1], color=INK_MUTED, lw=1, ls=(0, (4, 3)), label="chance")
                    ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
                    ax.set_title(f"ROC, pooled over all participants — {LOSS_LABEL[L]}",
                                 fontsize=10, loc="left")
                else:
                    ax.set_xlabel("recall"); ax.set_ylabel("precision")
                    ax.set_title(f"Precision-recall, pooled — {LOSS_LABEL[L]}\n"
                                 "(dotted = class prevalence, the no-skill floor)",
                                 fontsize=10, loc="left")
                ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
                ax.legend(frameon=False, fontsize=8, labelcolor=INK_MUTED,
                          loc="upper center", bbox_to_anchor=(0.5, -0.14), ncols=2)
                _style(ax); _save(fig, OUT / f"fig_{kind}_pooled.png")
            print("    -> fig_roc_pooled.png / fig_pr_pooled.png")

    save_table(pd.DataFrame(rows), "pooled_discrimination")
    print(f"    -> {len(losses)} pooled confusion matrices")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
