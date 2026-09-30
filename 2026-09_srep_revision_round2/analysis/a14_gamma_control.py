#!/usr/bin/env python3
"""Does FA-FL's calibration come from frequency adaptation, or just from gamma?

StandardFocalLoss and FrequencyAdaptiveFocalLoss in this codebase apply the
IDENTICAL alpha weighting. The only difference is that focal uses one global
gamma while FA-FL uses gamma_t = gamma_base + lambda*alpha_t, giving per-class
effective gammas of about [1.367, 2.809, 5.824].

The majority class holds 78 percent of the images and sets the model's overall
confidence level. FA-FL applies gamma 1.367 to it; the exp01 focal baseline
applies 3.0. Gamma suppresses gradient from confident predictions, so a lower
gamma on the dominant class raises confidence, and confidence is what ECE
measures. A single constant gamma of 1.37 therefore isolates exactly one
variable.

Compares four arms:
    FA-FL                      exp01
    Focal gamma = 3.0          exp01   (the submitted baseline)
    Focal gamma = 1.37         exp07   (mechanism control)
    Focal gamma = 2.0          exp08   (canonical Lin et al. value)

Reports pooled and mean-of-folds ECE, mean confidence, the confidence/accuracy
offset, and a paired participant-clustered bootstrap of FA-FL against the
gamma 1.37 arm. Prints a verdict saying which way to write the manuscript.

numpy and pandas only. No GPU, no retraining.
"""
import json, glob
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NC, BINS, NBOOT = 3, 15, 2000

ARMS = [
    ("FA-FL",            "exp01_main_sweep",     "FA_FL"),
    ("Focal gamma 3.0",  "exp01_main_sweep",     "FocalLoss"),
    ("Focal gamma 1.37", "exp07_focal_gamma137", "FocalLoss"),
    ("Focal gamma 2.0",  "exp08_focal_gamma200", "FocalLoss"),
]


def load(exp, loss):
    """Per-fold arrays for one arm, or None if the experiment has not run."""
    out = []
    for mf in sorted((ROOT / "experiments").glob(f"{exp}/outputs/*/manifest.json")):
        m = json.loads(mf.read_text())
        if m.get("status") != "COMPLETE" or m["loss"]["type"] != loss:
            continue
        d = np.load(mf.parent / "predictions_test.npz", allow_pickle=False)
        out.append({"fold": m["split"]["fold"],
                    "p": d["probabilities"], "y": d["labels"].astype(int),
                    "g": d["participant_id"].astype(str),
                    "gamma": m["loss"]["params"].get("gamma")})
    if not out:
        return None
    seen, uniq = set(), []
    for r in sorted(out, key=lambda r: (r["fold"], r["p"].shape[0])):
        if r["fold"] in seen:
            continue
        seen.add(r["fold"]); uniq.append(r)
    return uniq


def ece_of(p, y):
    conf, ok = p.max(1), (p.argmax(1) == y).astype(float)
    e = np.linspace(0, 1, BINS + 1)
    i = np.clip(np.digitize(conf, e[1:-1], right=True), 0, BINS - 1)
    n = np.bincount(i, minlength=BINS).astype(float)
    sc = np.bincount(i, weights=conf, minlength=BINS)
    so = np.bincount(i, weights=ok, minlength=BINS)
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = np.abs(np.where(n > 0, so / n, 0) - np.where(n > 0, sc / n, 0))
    return float((n / n.sum() * gap).sum())


class Suff:
    """Per-participant additive statistics, so the bootstrap is cheap."""
    def __init__(self, P, Y, G):
        self.u, inv = np.unique(G, return_inverse=True)
        self.n = len(self.u)
        yh = P.argmax(1)
        self.cm = np.zeros((self.n, NC, NC)); np.add.at(self.cm, (inv, Y, yh), 1.)
        conf, ok = P.max(1), (yh == Y).astype(float)
        e = np.linspace(0, 1, BINS + 1)
        b = np.clip(np.digitize(conf, e[1:-1], right=True), 0, BINS - 1)
        self.bn = np.zeros((self.n, BINS)); np.add.at(self.bn, (inv, b), 1.)
        self.bc = np.zeros((self.n, BINS)); np.add.at(self.bc, (inv, b), conf)
        self.bo = np.zeros((self.n, BINS)); np.add.at(self.bo, (inv, b), ok)
        self.cs = np.zeros(self.n); np.add.at(self.cs, inv, conf)
        self.os = np.zeros(self.n); np.add.at(self.os, inv, ok)

    def m(self, idx=None):
        s = slice(None) if idx is None else idx
        cm = self.cm[s].sum(0); tp = np.diag(cm)
        sup, pr, T = cm.sum(1), cm.sum(0), cm.sum()
        p = np.divide(tp, pr, out=np.zeros(NC), where=pr > 0)
        r = np.divide(tp, sup, out=np.zeros(NC), where=sup > 0)
        dn = p + r; f1 = np.divide(2 * p * r, dn, out=np.zeros(NC), where=dn > 0)
        n = self.bn[s].sum(0); c = self.bc[s].sum(0); k = self.bo[s].sum(0); tot = n.sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            gap = np.abs(np.where(n > 0, k / n, 0) - np.where(n > 0, c / n, 0))
        return dict(acc=100 * tp.sum() / T, mf1=float(f1.mean()), f1=f1,
                    ece=float((n / tot * gap).sum()),
                    conf=float(self.cs[s].sum() / tot), accf=float(self.os[s].sum() / tot))


def main() -> int:
    print("[a14] gamma control: is FA-FL's calibration frequency adaptation, or gamma?")
    arms, missing = {}, []
    for name, exp, loss in ARMS:
        d = load(exp, loss)
        if d is None:
            missing.append((name, exp)); continue
        arms[name] = d
        g = {r["gamma"] for r in d}
        print(f"  {name:18s} {len(d)} folds   gamma recorded in manifests: {g}")
    for name, exp in missing:
        print(f"  {name:18s} NOT RUN  ({exp})")
    if "Focal gamma 1.37" not in arms:
        print("\n  The mechanism control has not been run. Launch:")
        print("    python experiments\\exp07_focal_gamma137\\run.py")
        return 0

    rows, stats = [], {}
    for name, folds in arms.items():
        P = np.concatenate([f["p"] for f in folds])
        Y = np.concatenate([f["y"] for f in folds])
        G = np.concatenate([f["g"] for f in folds])
        stats[name] = Suff(P, Y, G)
        m = stats[name].m()
        fold_ece = [ece_of(f["p"], f["y"]) for f in folds]
        rows.append({"Arm": name, "Accuracy": m["acc"], "Macro-F1": m["mf1"],
                     "F1 V.Mild": m["f1"][1], "F1 Dem": m["f1"][2],
                     "ECE pooled": m["ece"],
                     "ECE mean-of-folds": float(np.mean(fold_ece)),
                     "SD folds": float(np.std(fold_ece, ddof=1)),
                     "Mean conf": m["conf"], "|Offset|": abs(m["accf"] - m["conf"])})
    t = pd.DataFrame(rows)
    t.to_csv(ROOT / "analysis" / "outputs" / "gamma_control.csv", index=False)
    print()
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    ref, ctl = "FA-FL", "Focal gamma 1.37"
    if not np.array_equal(stats[ref].u, stats[ctl].u):
        print("\n  [warn] arms cover different participants; paired test skipped")
        return 0
    rng = np.random.default_rng(4242); n = stats[ref].n
    da = np.empty(NBOOT); db = np.empty(NBOOT)
    for i in range(NBOOT):
        idx = rng.integers(0, n, n)
        da[i] = stats[ref].m(idx)["ece"]; db[i] = stats[ctl].m(idx)["ece"]
    d = da - db
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1.0 / NBOOT)
    pt = stats[ref].m()["ece"] - stats[ctl].m()["ece"]
    sig = (lo > 0 or hi < 0)

    print(f"\nPAIRED, participant-clustered, {NBOOT} resamples")
    print(f"  ECE, FA-FL minus Focal(gamma=1.37): {pt:+.4f}  [{lo:+.4f}, {hi:+.4f}]  p = {p:.4f}")

    print("\n" + "=" * 78)
    if sig and pt < 0:
        print("  MECHANISM SUPPORTED.")
        print("  A single constant gamma matched to FA-FL's majority-class value does NOT")
        print("  reproduce its calibration. Frequency adaptation is doing work that one")
        print("  equivalent gamma cannot. You may state the mechanism in the manuscript,")
        print("  citing this control by name.")
    elif sig and pt > 0:
        print("  MECHANISM CONTRADICTED, AND THE CONTROL IS BETTER.")
        print("  Constant gamma 1.37 calibrates BETTER than FA-FL. Report it as a finding:")
        print("  the gain attributed to frequency adaptation is available from a single")
        print("  lower gamma, and more of it. Drop the mechanistic claim entirely.")
    else:
        print("  MECHANISM NOT SUPPORTED.")
        print("  Constant gamma 1.37 matches FA-FL's calibration within noise. The")
        print("  advantage is a property of the majority-class gamma, not of frequency")
        print("  adaptation. Report FA-FL's calibration DESCRIPTIVELY and state that a")
        print("  single equivalent gamma reproduces it. This is still publishable and it")
        print("  is a stronger paper than one making a mechanism claim a reviewer can break.")
    print("=" * 78)
    print("\n  Either way, report Focal at gamma 2.0 as the canonical baseline, so nobody")
    print("  can argue the closest competitor was set to a non-standard value.")
    print("\n    -> analysis/outputs/gamma_control.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
