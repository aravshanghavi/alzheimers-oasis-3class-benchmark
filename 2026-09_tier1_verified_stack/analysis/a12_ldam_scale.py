#!/usr/bin/env python3
"""LDAM is evaluated at a different logit scale from the one it was trained at.

LDAMLoss.forward optimises F.cross_entropy(s * output, targets) with s = 30, but
evaluate.predict computes softmax(output) on the UNSCALED logits. The network is
therefore trained so that 30x its logits are well scaled, and then evaluated at
one thirtieth of that. The consequence is severe apparent under-confidence, and
LDAM's raw ECE of 0.2878 is the largest single number in the calibration
comparison.

Because softmax(s*z) = p^s / sum(p^s), the correctly scaled probabilities are
recoverable from the stored ones with no retraining: it is temperature scaling at
T = 1/s.

This script reports LDAM both ways so the manuscript can state which convention
it uses. Dependency light: numpy and pandas only.
"""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd
import sys as _sys, pathlib as _pl  # noqa: E402
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root on path
from analysis._paths import EXPERIMENTS_DIR, OUT as _OUT, CACHE_NPZ, metadata_xlsx  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BINS, S = 15, 30.0


def rescale(p, s):
    """softmax(s*z) from p = softmax(z). Exact, no logits needed."""
    q = np.power(np.clip(p.astype(np.float64), 1e-300, 1.0), s)
    return q / q.sum(axis=1, keepdims=True)


def ece(conf, correct, n_bins=BINS):
    e = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, e[1:-1], right=True), 0, n_bins - 1)
    n = np.bincount(idx, minlength=n_bins).astype(float)
    sc = np.bincount(idx, weights=conf, minlength=n_bins)
    so = np.bincount(idx, weights=correct, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = np.abs(np.where(n > 0, so / n, 0.0) - np.where(n > 0, sc / n, 0.0))
    return float((n / n.sum() * gap).sum())


def brier(p, y):
    oh = np.eye(p.shape[1])[y]
    return float(((p - oh) ** 2).sum(axis=1).mean())


def main() -> int:
    print("[a12] LDAM logit-scale convention")
    parts = []
    for mf in sorted(EXPERIMENTS_DIR.glob("exp01_main_sweep/outputs/*/manifest.json")):
        m = json.loads(mf.read_text())
        if m.get("status") != "COMPLETE" or m["loss"]["type"] != "LDAM":
            continue
        d = np.load(mf.parent / "predictions_test.npz", allow_pickle=False)
        parts.append((d["probabilities"], d["labels"].astype(int)))
    if not parts:
        print("  no completed LDAM runs found")
        return 0
    p = np.concatenate([a for a, _ in parts])
    y = np.concatenate([b for _, b in parts])
    pred = p.argmax(1)
    correct = (pred == y).astype(float)

    rows = []
    for name, probs in [("as evaluated  softmax(z)", p),
                        (f"as trained    softmax({S:.0f}z)", rescale(p, S))]:
        conf = probs.max(1)
        assert (probs.argmax(1) == pred).all(), "rescaling changed argmax"
        rows.append({"Convention": name,
                     "Mean confidence": float(conf.mean()),
                     "Accuracy": float(correct.mean()),
                     "|Offset|": abs(float(correct.mean() - conf.mean())),
                     "ECE": ece(conf, correct),
                     "Brier": brier(probs, y)})
    t = pd.DataFrame(rows)
    out = ROOT / "analysis" / "outputs"
    t.to_csv(out / "ldam_scale.csv", index=False)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n  Accuracy and every F1 are identical under both conventions, because")
    print("  scaling the logits by a positive constant cannot change the argmax.")

    note = (
        "LDAM LOGIT SCALE\n"
        "================\n\n"
        f"LDAM trains on cross_entropy(s*z) with s = {S:.0f}; inference computes softmax(z).\n"
        "The reported probabilities are therefore taken at 1/s of the scale the model was\n"
        "optimised at, which manufactures under-confidence. Since softmax(s*z) = p^s/sum(p^s),\n"
        "both conventions are recoverable from the stored probabilities.\n\n"
        + t.to_string(index=False, float_format=lambda v: f"{v:.4f}") + "\n\n"
        "Accuracy, macro-F1 and every per-class F1 are unchanged, because a positive scaling\n"
        "of the logits cannot change the argmax. Only probability-based metrics move.\n\n"
        "HOW TO REPORT. State the convention explicitly. If the as-evaluated column is kept,\n"
        "say that LDAM's calibration error reflects an implementation convention rather than a\n"
        "property of the objective, and do not lead the calibration comparison with the FA-FL\n"
        "versus LDAM gap. Reporting both columns is the honest option and costs one table row.\n"
    )
    (out / "ldam_scale_note.txt").write_text(note, encoding="utf-8")
    print("    -> ldam_scale.csv / ldam_scale_note.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
