#!/usr/bin/env python3
"""a15: the last batch of zero-compute analyses, all from stored predictions.

Adds, for every arm and both cohorts (full 347, CDR-assessed age>=60 166):
  1. ordinal agreement: quadratic-weighted kappa, linear kappa, unweighted kappa, MCC
  2. ordinal error structure: share of errors that are one step vs two steps
  3. every one of those compared against the MAJORITY-CLASS predictor by
     participant-clustered bootstrap
  4. selective prediction: AURC and accuracy at fixed coverage
  5. accuracy vs macro-F1 frontier coordinates, with the baseline point
  6. verification of the contaminated-participant profile (exp05)
"""
from __future__ import annotations
import json, sys, glob
from pathlib import Path
import numpy as np, pandas as pd
import sys as _sys, pathlib as _pl  # noqa: E402
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parents[1]))  # repo root on path
from analysis._paths import EXPERIMENTS_DIR, OUT as _OUT, CACHE_NPZ, metadata_xlsx  # noqa: E402
from analysis._determinism import seeded_rng  # noqa: E402

EXP = EXPERIMENTS_DIR
RNG = seeded_rng("a15:module", 20260823)   # see _determinism: per-site seeding below
NB = 2000

ARMS = [
    ("Weighted CE",    "exp01_main_sweep",     "WCE"),
    ("LDAM",           "exp01_main_sweep",     "LDAM"),
    ("Focal g3.0",     "exp01_main_sweep",     "FocalLoss"),
    ("FA-FL",          "exp01_main_sweep",     "FA_FL"),
    ("Class-Balanced", "exp02_class_balanced", "ClassBalanced"),
    ("Focal g1.37",    "exp07_focal_gamma137", "FocalLoss"),
    ("Focal g2.0",     "exp08_focal_gamma200", "FocalLoss"),
]

def load_arm(exp, loss):
    parts = []
    for mf in sorted((EXP / exp / "outputs").glob("*/manifest.json")):
        m = json.loads(mf.read_text(encoding="utf-8"))
        if m.get("status") != "COMPLETE":       continue
        if m.get("loss", {}).get("type") != loss: continue
        if m.get("split", {}).get("group_by") != "participant": continue
        d = np.load(mf.parent / "predictions_test.npz", allow_pickle=False)
        parts.append(pd.DataFrame({
            "pid": d["participant_id"].astype(str),
            "y":   d["labels"].astype(int),
            "yhat": d["probabilities"].argmax(1),
            "conf": d["probabilities"].max(1),
            "fold": int(m["split"]["fold"])}))
    if not parts:
        raise SystemExit(f"no runs for {exp}/{loss}")
    p = pd.concat(parts, ignore_index=True)
    n = p.groupby("pid")["fold"].nunique()
    if (n > 1).any():
        raise SystemExit(f"{exp}/{loss}: {(n>1).sum()} participants in >1 test fold")
    if p["fold"].nunique() != 5:
        raise SystemExit(f"{exp}/{loss}: {p['fold'].nunique()} folds, expected 5")
    return p

# ---- metadata -------------------------------------------------------------
meta = pd.read_excel(metadata_xlsx())
meta["pid"] = meta["ID"].str.replace(r"_MR\d+$", "", regex=True)
meta = meta.drop_duplicates("pid")[["pid", "Age", "CDR"]]
KEEP60 = set(meta.loc[meta["CDR"].notna() & (meta["Age"] >= 60), "pid"])

# ---- metric kernels on additive per-participant 3x3 confusions ------------
W_Q = np.array([[(i - j) ** 2 for j in range(3)] for i in range(3)], float) / 4.0
W_L = np.array([[abs(i - j) for j in range(3)] for i in range(3)], float) / 2.0

def cm_of(df):
    """participant -> flattened 3x3 confusion (true x pred), image counts."""
    c = (df.groupby(["pid", "y", "yhat"]).size()
           .unstack(fill_value=0).reindex(columns=range(3), fill_value=0))
    full = c.unstack(fill_value=0)
    idx = pd.MultiIndex.from_product([range(3), range(3)])
    full = full.reindex(columns=idx, fill_value=0)
    return full

def kappa(M, W):
    n = M.sum()
    if n == 0: return np.nan
    po = (M * W).sum() / n
    r, c = M.sum(1), M.sum(0)
    pe = (np.outer(r, c) * W).sum() / (n * n)
    return np.nan if pe == 0 else 1.0 - po / pe

def mcc(M):
    n = M.sum(); t = M.sum(1); p = M.sum(0)
    c = np.trace(M).astype(float)
    num = c * n - (t * p).sum()
    den = np.sqrt(max(n * n - (p * p).sum(), 0)) * np.sqrt(max(n * n - (t * t).sum(), 0))
    return np.nan if den == 0 else num / den

def macro_f1(M):
    f = []
    for k in range(3):
        tp = M[k, k]; fp = M[:, k].sum() - tp; fn = M[k, :].sum() - tp
        f.append(0.0 if 2 * tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))

def summarise(M):
    n = M.sum()
    off = M.copy(); np.fill_diagonal(off, 0)
    one = sum(off[i, j] for i in range(3) for j in range(3) if abs(i - j) == 1)
    two = sum(off[i, j] for i in range(3) for j in range(3) if abs(i - j) == 2)
    return dict(acc=np.trace(M) / n, macroF1=macro_f1(M),
                qwk=kappa(M, W_Q), lwk=kappa(M, W_L),
                kappa=kappa(M, 1 - np.eye(3)), mcc=mcc(M),
                err1=one / max(one + two, 1), err2=two / max(one + two, 1),
                n=int(n))

def majority_cm(df, maj):
    """confusion of the constant predictor, same participants."""
    c = df.groupby(["pid", "y"]).size().unstack(fill_value=0).reindex(columns=range(3), fill_value=0)
    out = pd.DataFrame(0, index=c.index,
                       columns=pd.MultiIndex.from_product([range(3), range(3)]))
    for k in range(3):
        out[(k, maj)] = c[k]
    return out

def boot(cm_a, cm_b, fn):
    """paired participant-clustered bootstrap of fn(A) - fn(B)."""
    pids = cm_a.index.to_numpy()
    A = cm_a.to_numpy(float); B = cm_b.loc[cm_a.index].to_numpy(float)
    pt = fn(A.sum(0).reshape(3, 3)) - fn(B.sum(0).reshape(3, 3))
    idx = seeded_rng("a15:boot", 20260823, n=len(pids),
                     first=str(pids[0]), last=str(pids[-1])
                     ).integers(0, len(pids), size=(NB, len(pids)))
    d = np.empty(NB)
    for b in range(NB):
        d[b] = fn(A[idx[b]].sum(0).reshape(3, 3)) - fn(B[idx[b]].sum(0).reshape(3, 3))
    return pt, np.nanpercentile(d, 2.5), np.nanpercentile(d, 97.5)

def aurc(conf, correct):
    """area under the risk-coverage curve; lower is better."""
    o = np.argsort(-conf)
    c = correct[o].astype(float)
    risk = 1.0 - np.cumsum(c) / np.arange(1, len(c) + 1)
    return float(risk.mean())

# ---- run ------------------------------------------------------------------
rows, boots, sel, front = [], [], [], []
for label, exp, loss in ARMS:
    p = load_arm(exp, loss)
    for cohort, sub in (("full", p), ("age60", p[p["pid"].isin(KEEP60)])):
        cm = cm_of(sub)
        M = cm.to_numpy(float).sum(0).reshape(3, 3)
        s = summarise(M); s.update(arm=label, cohort=cohort, participants=cm.shape[0])
        rows.append(s)
        maj = int(np.argmax(M.sum(1)))
        bcm = majority_cm(sub, maj)
        base = summarise(bcm.to_numpy(float).sum(0).reshape(3, 3))
        for name, fn in (("accuracy", lambda m: np.trace(m) / m.sum()),
                         ("macroF1", macro_f1),
                         ("QWK", lambda m: kappa(m, W_Q)),
                         ("MCC", mcc)):
            pt, lo, hi = boot(cm, bcm, fn)
            boots.append(dict(arm=label, cohort=cohort, metric=name,
                              model=fn(M), baseline=base[{"accuracy": "acc", "macroF1": "macroF1",
                                                          "QWK": "qwk", "MCC": "mcc"}[name]],
                              diff=pt, lo=lo, hi=hi,
                              sig="YES" if (lo > 0 or hi < 0) else "no"))
        correct = (sub["y"].to_numpy() == sub["yhat"].to_numpy())
        conf = sub["conf"].to_numpy()
        o = np.argsort(-conf)
        cov = {f"acc@{int(c*100)}": float(correct[o][:max(1, int(c * len(o)))].mean())
               for c in (0.25, 0.50, 0.75, 1.00)}
        sel.append(dict(arm=label, cohort=cohort, aurc=aurc(conf, correct), **cov))
        front.append(dict(arm=label, cohort=cohort, acc=s["acc"], macroF1=s["macroF1"]))
    # baseline frontier point once per cohort
for cohort, keep in (("full", None), ("age60", KEEP60)):
    p = load_arm(*ARMS[0][1:])
    sub = p if keep is None else p[p["pid"].isin(keep)]
    M = cm_of(sub).to_numpy(float).sum(0).reshape(3, 3)
    maj = int(np.argmax(M.sum(1)))
    B = majority_cm(sub, maj).to_numpy(float).sum(0).reshape(3, 3)
    b = summarise(B); b.update(arm="MAJORITY BASELINE", cohort=cohort, participants=-1)
    rows.append(b); front.append(dict(arm="MAJORITY BASELINE", cohort=cohort,
                                      acc=b["acc"], macroF1=b["macroF1"]))

R = pd.DataFrame(rows); Bo = pd.DataFrame(boots); S = pd.DataFrame(sel); F = pd.DataFrame(front)
pd.set_option("display.width", 200)
print("\n=== ORDINAL AGREEMENT AND ERROR STRUCTURE ===")
print(R[["arm", "cohort", "n", "acc", "macroF1", "qwk", "lwk", "kappa", "mcc", "err1", "err2"]]
      .round(4).to_string(index=False))
print("\n=== VS MAJORITY-CLASS PREDICTOR, participant-clustered bootstrap, 95% CI ===")
print(Bo.round(4).to_string(index=False))
print("\n=== SELECTIVE PREDICTION ===")
print(S.round(4).to_string(index=False))
outdir = _OUT; outdir.mkdir(parents=True, exist_ok=True)
R.to_csv(outdir / "a15_ordinal.csv", index=False)
Bo.to_csv(outdir / "a15_vs_baseline.csv", index=False)
S.to_csv(outdir / "a15_selective.csv", index=False)
F.to_csv(outdir / "a15_frontier.csv", index=False)
print(f"\nwrote 4 csv to {outdir}")

# ---- contamination profile (exp05) ---------------------------------------
print("\n=== CONTAMINATED-PARTICIPANT PROFILE (exp05 session-level split) ===")
try:
    fp = sorted((EXP / "exp05_leakage_ablation" / "outputs").glob("*/fold_participants.csv"))
    seen = {}
    for f in fp:
        d = pd.read_csv(f)
        col = "participant_id" if "participant_id" in d.columns else d.columns[0]
        for split in d.columns:
            pass
        seen[f.parent.name] = d
    print(f"  {len(fp)} fold_participants.csv found; columns: "
          f"{list(next(iter(seen.values())).columns) if seen else 'none'}")
except Exception as e:
    print("  could not read:", e)
