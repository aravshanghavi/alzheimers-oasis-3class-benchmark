#!/usr/bin/env python3
"""a16: the full judge-mandated remediation battery. ZERO GPU. CPU only.

Answers, in order:
  1  participant-level evaluation (mean-prob and majority-vote), both cohorts
  2  threshold-free discrimination: macro AUROC, macro AP, balanced accuracy, MCC
  3  a proper baseline panel: constant-majority, stratified-random, demographics-only
  4  participant-clustered bootstrap CI on the Spearman rho itself
  5  paired CIs on AURC and on the two-step error share, plus a marginal-preserving null
  6  age-cutoff sensitivity curve
  7  the confound-free core sub-task, CDR 0.5 vs CDR >= 1
  8  inter-arm prediction agreement
  9  the three separated restrictions (CDR-blank only / under-60 only / prior-matched)
 10  slice-index profile
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent
if not (ROOT / "experiments").exists():
    ROOT = Path.cwd()
EXP = ROOT / "experiments"
OUT = ROOT / "analysis" / "outputs"; OUT.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(20260823)
NB, NB_RHO = 2000, 1000
K = 3

ARMS = [("Weighted CE","exp01_main_sweep","WCE"), ("LDAM","exp01_main_sweep","LDAM"),
        ("Focal g3.0","exp01_main_sweep","FocalLoss"), ("FA-FL","exp01_main_sweep","FA_FL"),
        ("Class-Balanced","exp02_class_balanced","ClassBalanced"),
        ("Focal g1.37","exp07_focal_gamma137","FocalLoss"),
        ("Focal g2.0","exp08_focal_gamma200","FocalLoss")]

# --------------------------------------------------------------- load
def load_arm(exp, loss):
    parts = []
    for mf in sorted((EXP/exp/"outputs").glob("*/manifest.json")):
        m = json.loads(mf.read_text(encoding="utf-8"))
        if m.get("status") != "COMPLETE": continue
        if m.get("loss",{}).get("type") != loss: continue
        if m.get("split",{}).get("group_by") != "participant": continue
        d = np.load(mf.parent/"predictions_test.npz", allow_pickle=False)
        P = d["probabilities"].astype(np.float64)
        f = pd.DataFrame({"pid": d["participant_id"].astype(str),
                          "y": d["labels"].astype(int),
                          "img": d["image_path"].astype(str),
                          "fold": int(m["split"]["fold"])})
        for c in range(K): f[f"p{c}"] = P[:, c]
        parts.append(f)
    if not parts: raise SystemExit(f"no runs for {exp}/{loss}")
    p = pd.concat(parts, ignore_index=True)
    if (p.groupby("pid")["fold"].nunique() > 1).any(): raise SystemExit(f"{exp}/{loss}: folds not disjoint")
    if p["fold"].nunique() != 5: raise SystemExit(f"{exp}/{loss}: {p['fold'].nunique()} folds")
    return p.sort_values(["pid","img"]).reset_index(drop=True)

print("loading seven arms ...", flush=True)
D = {lab: load_arm(e, l) for lab, e, l in ARMS}
BASE = D["FA-FL"][["pid","y","img","fold"]].copy()
for lab, f in D.items():
    a = f.sort_values(["pid","img"]).reset_index(drop=True)
    if not (a["pid"].values == BASE["pid"].values).all() or not (a["img"].values == BASE["img"].values).all():
        raise SystemExit(f"{lab}: image order does not match FA-FL; cannot align arms")
PROB = {lab: D[lab].sort_values(["pid","img"])[["p0","p1","p2"]].to_numpy() for lab in D}
Y = BASE["y"].to_numpy(); PID = BASE["pid"].to_numpy()
print(f"  aligned: {len(BASE)} images, {BASE['pid'].nunique()} participants", flush=True)

meta = pd.read_excel(ROOT/"oasis_cross-sectional.xlsx")
meta["pid"] = meta["ID"].str.replace(r"_MR\d+$","",regex=True)
meta = meta.drop_duplicates("pid").set_index("pid")
AGE = meta["Age"].reindex(pd.unique(PID))
CDR = meta["CDR"].reindex(pd.unique(PID))
PLAB = pd.Series(Y, index=PID).groupby(level=0).first().reindex(pd.unique(PID))

def cohort(name):
    pids = pd.Index(pd.unique(PID))
    if name == "full":       return set(pids)
    if name == "age60":      return set(pids[(CDR.notna() & (AGE >= 60)).values])
    if name == "cdronly":    return set(pids[CDR.notna().values])
    if name == "age60only":  return set(pids[(AGE >= 60).values])
    if name == "dementiaonly": return set(pids[PLAB.isin([1,2]).values])
    m = re.match(r"age(\d+)$", name)
    if m: return set(pids[(CDR.notna() & (AGE >= int(m.group(1)))).values])
    raise KeyError(name)

# ------------------------------------------------------- metric kernels
W_Q = np.array([[(i-j)**2 for j in range(K)] for i in range(K)], float)/((K-1)**2)
def kappa(M, W):
    n = M.sum()
    if n == 0: return np.nan
    po = (M*W).sum()/n; r, c = M.sum(1), M.sum(0)
    pe = (np.outer(r,c)*W).sum()/(n*n)
    return np.nan if pe == 0 else 1.0 - po/pe
def macro_f1(M):
    out = []
    for k in range(K):
        tp = M[k,k]; fp = M[:,k].sum()-tp; fn = M[k,:].sum()-tp
        out.append(0.0 if 2*tp+fp+fn == 0 else 2*tp/(2*tp+fp+fn))
    return float(np.mean(out))
def bal_acc(M):
    r = M.sum(1); return float(np.mean([M[k,k]/r[k] if r[k] else np.nan for k in range(K)]))
def mcc(M):
    n = M.sum(); t = M.sum(1); p = M.sum(0); c = np.trace(M).astype(float)
    den = np.sqrt(max(n*n-(p*p).sum(),0))*np.sqrt(max(n*n-(t*t).sum(),0))
    return np.nan if den == 0 else (c*n-(t*p).sum())/den
def acc(M): return np.trace(M)/M.sum()
def twostep(M):
    off = M.copy().astype(float); np.fill_diagonal(off,0)
    two = off[0,2]+off[2,0]; return two/max(off.sum(),1)
def confmat(y, yhat, w=None):
    M = np.zeros((K,K))
    if w is None: w = np.ones(len(y))
    np.add.at(M, (y, yhat), w)
    return M
def auroc_bin(score, pos):
    n1 = pos.sum(); n0 = len(pos)-n1
    if n1 == 0 or n0 == 0: return np.nan
    r = pd.Series(score).rank().to_numpy()
    return (r[pos].sum() - n1*(n1+1)/2)/(n1*n0)
def macro_auroc(P, y):
    return float(np.nanmean([auroc_bin(P[:,k], y == k) for k in range(K)]))
def ap_bin(score, pos):
    o = np.argsort(-score); p = pos[o].astype(float)
    if p.sum() == 0: return np.nan
    prec = np.cumsum(p)/np.arange(1, len(p)+1)
    return float((prec*p).sum()/p.sum())
def macro_ap(P, y):
    return float(np.nanmean([ap_bin(P[:,k], y == k) for k in range(K)]))
def aurc(conf, correct):
    o = np.argsort(-conf); c = correct[o].astype(float)
    return float((1.0-np.cumsum(c)/np.arange(1, len(c)+1)).mean())

def metrics(P, y, tag=""):
    yh = P.argmax(1); M = confmat(y, yh)
    conf = P.max(1); corr = (y == yh)
    return dict(n=len(y), acc=acc(M), macroF1=macro_f1(M), balAcc=bal_acc(M),
                qwk=kappa(M,W_Q), kappa=kappa(M,1-np.eye(K)), mcc=mcc(M),
                auroc=macro_auroc(P,y), ap=macro_ap(P,y),
                twostep=twostep(M), aurc=aurc(conf,corr))

# ---------------------------------------------- 1. participant level
print("\n[1] participant-level evaluation", flush=True)
def agg(lab, keep, how="mean"):
    m = np.isin(PID, list(keep))
    df = pd.DataFrame(PROB[lab][m], columns=["p0","p1","p2"])
    df["pid"] = PID[m]; df["y"] = Y[m]
    if how == "mean":
        g = df.groupby("pid")[["p0","p1","p2"]].mean()
    else:
        v = df.assign(yh=PROB[lab][m].argmax(1)).groupby(["pid","yh"]).size().unstack(fill_value=0)
        v = v.reindex(columns=range(K), fill_value=0)
        g = (v.T/v.sum(1)).T; g.columns = ["p0","p1","p2"]
    yy = df.groupby("pid")["y"].first().reindex(g.index).to_numpy()
    return g.to_numpy(), yy

rows = []
for coh in ("full","age60"):
    keep = cohort(coh)
    for how in ("mean","vote"):
        for lab in D:
            P, yy = agg(lab, keep, how)
            r = metrics(P, yy); r.update(arm=lab, cohort=coh, unit=f"participant-{how}")
            rows.append(r)
        # baselines at participant level
        _, yy = agg("FA-FL", keep, how)
        cnt = np.bincount(yy, minlength=K); maj = int(cnt.argmax()); pri = cnt/cnt.sum()
        M = confmat(yy, np.full(len(yy), maj))
        rows.append(dict(n=len(yy), acc=acc(M), macroF1=macro_f1(M), balAcc=bal_acc(M),
                         qwk=kappa(M,W_Q), kappa=kappa(M,1-np.eye(K)), mcc=mcc(M),
                         auroc=np.nan, ap=np.nan, twostep=twostep(M), aurc=np.nan,
                         arm="BASELINE constant", cohort=coh, unit=f"participant-{how}"))
        rows.append(dict(n=len(yy), acc=float((pri**2).sum()), macroF1=1.0/K, balAcc=1.0/K,
                         qwk=0.0, kappa=0.0, mcc=0.0, auroc=0.5, ap=float(pri.mean()),
                         twostep=float(2*pri[0]*pri[2]/(1-(pri**2).sum())), aurc=np.nan,
                         arm="BASELINE stratified-random (analytic)", cohort=coh,
                         unit=f"participant-{how}"))
# image level for comparison
for coh in ("full","age60"):
    keep = cohort(coh); m = np.isin(PID, list(keep))
    for lab in D:
        r = metrics(PROB[lab][m], Y[m]); r.update(arm=lab, cohort=coh, unit="image")
        rows.append(r)
    yy = Y[m]; cnt = np.bincount(yy, minlength=K); maj = int(cnt.argmax()); pri = cnt/cnt.sum()
    M = confmat(yy, np.full(len(yy), maj))
    rows.append(dict(n=len(yy), acc=acc(M), macroF1=macro_f1(M), balAcc=bal_acc(M),
                     qwk=kappa(M,W_Q), kappa=kappa(M,1-np.eye(K)), mcc=mcc(M), auroc=np.nan,
                     ap=np.nan, twostep=twostep(M), aurc=np.nan,
                     arm="BASELINE constant", cohort=coh, unit="image"))
    rows.append(dict(n=len(yy), acc=float((pri**2).sum()), macroF1=1.0/K, balAcc=1.0/K, qwk=0.0,
                     kappa=0.0, mcc=0.0, auroc=0.5, ap=float(pri.mean()),
                     twostep=float(2*pri[0]*pri[2]/(1-(pri**2).sum())), aurc=np.nan,
                     arm="BASELINE stratified-random (analytic)", cohort=coh, unit="image"))
T1 = pd.DataFrame(rows)
T1.to_csv(OUT/"a16_units.csv", index=False)
pd.set_option("display.width", 250)
for u in ("participant-mean","participant-vote","image"):
    print(f"\n--- unit = {u}")
    print(T1[T1.unit == u][["arm","cohort","n","acc","macroF1","balAcc","qwk","mcc","auroc","ap","twostep"]]
          .round(4).to_string(index=False))

# ------------------------------------- 3. demographics-only baseline
print("\n[3] demographics-only baseline (age, sex, eTIV, nWBV), same grouped folds", flush=True)
def softmax(z): 
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e/e.sum(1, keepdims=True)
def fit_mnlogit(X, y, iters=400, lr=0.5, l2=1e-3):
    n, d = X.shape; Wm = np.zeros((d, K))
    for _ in range(iters):
        P = softmax(X@Wm); G = X.T@(P - np.eye(K)[y])/n + l2*Wm
        Wm -= lr*G
    return Wm
foldof = BASE.groupby("pid")["fold"].first()
feat = meta.reindex(pd.unique(PID))[["Age","M/F","eTIV","nWBV"]].copy()
feat["M/F"] = (feat["M/F"] == "M").astype(float)
feat = feat.apply(lambda c: c.fillna(c.median()))
Xd = np.column_stack([np.ones(len(feat)), (feat.to_numpy() - feat.to_numpy().mean(0))/feat.to_numpy().std(0)])
yd = PLAB.to_numpy().astype(int); fd = foldof.reindex(pd.unique(PID)).to_numpy()
Pdem = np.zeros((len(yd), K))
for f in range(5):
    tr, te = fd != f, fd == f
    Wm = fit_mnlogit(Xd[tr], yd[tr]); Pdem[te] = softmax(Xd[te]@Wm)
demrows = []
for coh in ("full","age60"):
    keep = cohort(coh); m = np.isin(pd.unique(PID), list(keep))
    r = metrics(Pdem[m], yd[m]); r.update(arm="BASELINE demographics-only", cohort=coh,
                                          unit="participant-mean"); demrows.append(r)
    r2 = metrics(Pdem[m][:, :], yd[m]); 
    # age-only variant
    Xa = Xd[:, [0,1]]; Pa = np.zeros((len(yd), K))
    for f in range(5):
        tr, te = fd != f, fd == f
        Wm = fit_mnlogit(Xa[tr], yd[tr]); Pa[te] = softmax(Xa[te]@Wm)
    r3 = metrics(Pa[m], yd[m]); r3.update(arm="BASELINE age-only", cohort=coh,
                                          unit="participant-mean"); demrows.append(r3)
TD = pd.DataFrame(demrows)
print(TD[["arm","cohort","n","acc","macroF1","balAcc","qwk","mcc","auroc","ap"]].round(4).to_string(index=False))
TD.to_csv(OUT/"a16_demographics.csv", index=False)

# ------------------------------- 4. bootstrap CI on the Spearman rho
print("\n[4] participant-clustered bootstrap CI on Spearman rho across the seven arms", flush=True)
def spear(a, b):
    ra = pd.Series(a).rank().to_numpy(); rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0,1])
def suff(lab, keep, unit):
    """per-participant additive stats: 3x3 confusion, for one arm."""
    m = np.isin(PID, list(keep))
    if unit == "image":
        yh = PROB[lab][m].argmax(1); pid = PID[m]; y = Y[m]
    else:
        P, y = agg(lab, keep, "mean"); yh = P.argmax(1)
        pid = np.array(sorted(keep))
    df = pd.DataFrame({"pid": pid, "y": y, "yh": yh})
    c = df.groupby(["pid","y","yh"]).size().unstack(fill_value=0).reindex(columns=range(K), fill_value=0)
    full = c.unstack(fill_value=0).reindex(
        columns=pd.MultiIndex.from_product([range(K), range(K)]), fill_value=0)
    return full
rho_rows = []
for coh in ("full","age60"):
    for unit in ("image","participant-mean"):
        keep = cohort(coh)
        S = {lab: suff(lab, keep, unit) for lab in D}
        order = list(D)
        idx0 = S[order[0]].index
        A = np.stack([S[l].reindex(idx0).to_numpy(float) for l in order])   # arms x pids x 9
        pt = {}
        Ms = [A[i].sum(0).reshape(K,K) for i in range(len(order))]
        pt["acc_f1"] = spear([acc(m) for m in Ms], [macro_f1(m) for m in Ms])
        pt["acc_qwk"] = spear([acc(m) for m in Ms], [kappa(m,W_Q) for m in Ms])
        npid = A.shape[1]
        bi = RNG.integers(0, npid, size=(NB_RHO, npid))
        r1 = np.empty(NB_RHO); r2 = np.empty(NB_RHO)
        for b in range(NB_RHO):
            Mb = [A[i][bi[b]].sum(0).reshape(K,K) for i in range(len(order))]
            aa = [acc(m) for m in Mb]
            r1[b] = spear(aa, [macro_f1(m) for m in Mb]); r2[b] = spear(aa, [kappa(m,W_Q) for m in Mb])
        rho_rows.append(dict(cohort=coh, unit=unit, pair="acc~macroF1", rho=pt["acc_f1"],
                             lo=np.nanpercentile(r1,2.5), hi=np.nanpercentile(r1,97.5),
                             p_neg=float(np.mean(r1 >= 0))))
        rho_rows.append(dict(cohort=coh, unit=unit, pair="acc~QWK", rho=pt["acc_qwk"],
                             lo=np.nanpercentile(r2,2.5), hi=np.nanpercentile(r2,97.5),
                             p_neg=float(np.mean(r2 >= 0))))
TR = pd.DataFrame(rho_rows); print(TR.round(4).to_string(index=False))
TR.to_csv(OUT/"a16_rho_ci.csv", index=False)

# ------------- 5. paired CIs on AURC and two-step, + marginal-preserving null
print("\n[5] paired participant-clustered CIs: AURC and two-step share", flush=True)
def boot_pairs(keep, fn, unit="image"):
    """fn maps (probs, y) -> scalar. returns per-arm point + paired diffs vs best."""
    m = np.isin(PID, list(keep)); pid = PID[m]
    upid = np.array(sorted(set(pid))); pos = {p:i for i,p in enumerate(upid)}
    gi = np.array([pos[p] for p in pid])
    order = list(D)
    pts = {l: fn(PROB[l][m], Y[m]) for l in order}
    bi = RNG.integers(0, len(upid), size=(NB//4, len(upid)))
    # index lists per participant
    byp = [np.where(gi == i)[0] for i in range(len(upid))]
    vals = {l: np.empty(NB//4) for l in order}
    for b in range(NB//4):
        sel = np.concatenate([byp[j] for j in bi[b]])
        yb = Y[m][sel]
        for l in order: vals[l][b] = fn(PROB[l][m][sel], yb)
    out = []
    for l in order:
        for l2 in order:
            if l == l2: continue
            d = vals[l] - vals[l2]
            out.append(dict(arm=l, vs=l2, metric=fn.__name__, point=pts[l]-pts[l2],
                            lo=np.nanpercentile(d,2.5), hi=np.nanpercentile(d,97.5),
                            sig="YES" if (np.nanpercentile(d,2.5) > 0 or np.nanpercentile(d,97.5) < 0) else "no"))
    return pts, pd.DataFrame(out)
def f_aurc(P, y): return aurc(P.max(1), y == P.argmax(1))
f_aurc.__name__ = "AURC"
def f_two(P, y): return twostep(confmat(y, P.argmax(1)))
f_two.__name__ = "twostep"
pair_frames = []
for coh in ("full","age60"):
    for fn in (f_aurc, f_two):
        pts, dfp = boot_pairs(cohort(coh), fn); dfp["cohort"] = coh
        pair_frames.append(dfp)
TP = pd.concat(pair_frames, ignore_index=True)
TP.to_csv(OUT/"a16_paired_aurc_twostep.csv", index=False)
for coh in ("full","age60"):
    for met in ("AURC","twostep"):
        s = TP[(TP.cohort==coh)&(TP.metric==met)&(TP.arm=="FA-FL")]
        print(f"  FA-FL vs others, {met}, {coh}: "
              f"{int((s.sig=='YES').sum())}/{len(s)} paired differences with CI excluding zero")
# marginal-preserving null for two-step
print("\n  marginal-preserving permutation null for the two-step share:", flush=True)
nullrows = []
for coh in ("full","age60"):
    keep = cohort(coh); m = np.isin(PID, list(keep)); y = Y[m]
    pri = np.bincount(y, minlength=K)/len(y)
    for l in D:
        yh = PROB[l][m].argmax(1); obs = twostep(confmat(y, yh))
        q = np.bincount(yh, minlength=K)/len(yh)
        Mn = np.outer(pri, q)*len(y)                      # independent, same marginals
        nullrows.append(dict(cohort=coh, arm=l, observed=obs, null_same_marginals=twostep(Mn),
                             prior_random=twostep(np.outer(pri,pri)*len(y))))
TN = pd.DataFrame(nullrows); print(TN.round(4).to_string(index=False))
TN.to_csv(OUT/"a16_twostep_null.csv", index=False)

# --------------------------------------- 6. age-cutoff sensitivity
print("\n[6] age-cutoff sensitivity", flush=True)
srows = []
for cut in ("full","cdronly","age50","age55","age60","age65","age70"):
    keep = cohort(cut)
    if len(keep) < 30: continue
    m = np.isin(PID, list(keep)); yy = Y[m]
    cnt = np.bincount(yy, minlength=K); maj = int(cnt.argmax())
    b = confmat(yy, np.full(len(yy), maj))
    srows.append(dict(cut=cut, participants=len(keep), images=int(m.sum()),
                      arm="BASELINE constant", acc=acc(b), macroF1=macro_f1(b), qwk=0.0, auroc=np.nan))
    for l in D:
        r = metrics(PROB[l][m], yy)
        srows.append(dict(cut=cut, participants=len(keep), images=int(m.sum()), arm=l,
                          acc=r["acc"], macroF1=r["macroF1"], qwk=r["qwk"], auroc=r["auroc"]))
TS = pd.DataFrame(srows); TS.to_csv(OUT/"a16_agecurve.csv", index=False)
print(TS.pivot_table(index="cut", columns="arm", values="acc").round(4).to_string())
print()
print(TS.pivot_table(index="cut", columns="arm", values="macroF1").round(4).to_string())

# ------------------------- 7. confound-free core sub-task CDR .5 vs >=1
print("\n[7] core sub-task: dementia classes only (CDR 0.5 vs CDR >= 1)", flush=True)
keep = cohort("dementiaonly"); m = np.isin(PID, list(keep)); yy = Y[m]
print(f"  {len(keep)} participants, {int(m.sum())} images, class counts {np.bincount(yy, minlength=K)}")
crows = []
cnt = np.bincount(yy, minlength=K); maj = int(cnt.argmax())
b = confmat(yy, np.full(len(yy), maj))
crows.append(dict(arm="BASELINE constant", acc=acc(b), macroF1=macro_f1(b), balAcc=bal_acc(b), auroc=np.nan))
for l in D:
    P2 = PROB[l][m][:, [1,2]]; P2 = P2/P2.sum(1, keepdims=True)
    yb = (yy == 2).astype(int)
    a = auroc_bin(P2[:,1], yb == 1)
    yh = np.where(P2[:,1] > 0.5, 2, 1)
    M = confmat(yy, yh)
    crows.append(dict(arm=l, acc=(yh == yy).mean(), macroF1=macro_f1(M[1:,1:]),
                      balAcc=bal_acc(M[1:,1:]), auroc=a))
TC = pd.DataFrame(crows); print(TC.round(4).to_string(index=False)); TC.to_csv(OUT/"a16_coretask.csv", index=False)

# --------------------------------------- 8. inter-arm agreement
print("\n[8] inter-arm prediction agreement (image level, full cohort)", flush=True)
YH = {l: PROB[l].argmax(1) for l in D}
labs = list(D); Ag = pd.DataFrame(index=labs, columns=labs, dtype=float)
for i in labs:
    for j in labs:
        Ag.loc[i,j] = kappa(confmat(YH[i], YH[j]), 1-np.eye(K))
print(Ag.round(3).to_string())
allsame = np.all(np.stack([YH[l] for l in labs]) == YH[labs[0]], axis=0).mean()
print(f"  all seven arms identical on {allsame:.1%} of images")
Ag.to_csv(OUT/"a16_interarm.csv")

# -------------------------- 9. three separated restrictions
print("\n[9] separated restrictions: CDR-blank only / under-60 only / prior-matched", flush=True)
pids = pd.Index(pd.unique(PID))
young0 = pids[(PLAB.values == 0) & (AGE.values < 60)]
target = None
keep60 = cohort("age60"); m60 = np.isin(PID, list(keep60))
target = np.bincount(Y[m60], minlength=K)/m60.sum()
rrows = []
for nm, keep in (("full", cohort("full")), ("CDR-assessed only", cohort("cdronly")),
                 ("age>=60 only", cohort("age60only")), ("CDR & age>=60", cohort("age60"))):
    m = np.isin(PID, list(keep)); yy = Y[m]
    pri = np.bincount(yy, minlength=K)/len(yy)
    cnt = np.bincount(yy, minlength=K); b = confmat(yy, np.full(len(yy), int(cnt.argmax())))
    row = dict(restriction=nm, participants=len(keep), p0=pri[0], base=acc(b))
    for l in D: row[l] = metrics(PROB[l][m], yy)["macroF1"]
    row["rho_acc_f1"] = spear([metrics(PROB[l][m], yy)["acc"] for l in D],
                              [metrics(PROB[l][m], yy)["macroF1"] for l in D])
    rrows.append(row)
# prior-matched: keep all non-class-0, subsample class-0 participants to hit p0 = target[0]
rng2 = np.random.default_rng(7)
n_non0 = int((~np.isin(PID, list(pids[PLAB.values == 0]))).sum())
best = None
allp0 = list(pids[PLAB.values == 0])
for trial in range(40):
    k = rng2.permutation(allp0)
    for take in range(5, len(allp0)):
        keep = set(pids[PLAB.values != 0]) | set(k[:take])
        m = np.isin(PID, list(keep)); p0 = (Y[m] == 0).mean()
        if p0 >= target[0]:
            if best is None or abs(p0-target[0]) < best[0]: best = (abs(p0-target[0]), keep, p0)
            break
_, keepPM, p0PM = best
m = np.isin(PID, list(keepPM)); yy = Y[m]
cnt = np.bincount(yy, minlength=K); b = confmat(yy, np.full(len(yy), int(cnt.argmax())))
row = dict(restriction="prior-matched (age structure kept)", participants=len(keepPM), p0=p0PM, base=acc(b))
for l in D: row[l] = metrics(PROB[l][m], yy)["macroF1"]
row["rho_acc_f1"] = spear([metrics(PROB[l][m], yy)["acc"] for l in D],
                          [metrics(PROB[l][m], yy)["macroF1"] for l in D])
rrows.append(row)
TRS = pd.DataFrame(rrows); print(TRS.round(4).to_string(index=False)); TRS.to_csv(OUT/"a16_restrictions.csv", index=False)

# --------------------------------------------- 10. slice-index profile
print("\n[10] slice-index profile", flush=True)
sl = BASE["img"].str.extract(r"(\d+)(?!.*\d)")[0].astype(float)
if sl.notna().mean() > 0.9:
    prof = pd.DataFrame({"slice": sl.astype(int), "y": Y,
                         "corr": (PROB["FA-FL"].argmax(1) == Y).astype(int),
                         "conf": PROB["FA-FL"].max(1)})
    g = prof.groupby("slice").agg(n=("corr","size"), acc=("corr","mean"), conf=("conf","mean"))
    print(g.describe().round(4).to_string())
    print(g.head(3).round(4).to_string()); print(g.tail(3).round(4).to_string())
    g.to_csv(OUT/"a16_sliceprofile.csv")
else:
    print("  could not parse a slice index from image_path; skipped")
print("\nDONE. csv written to", OUT, flush=True)
