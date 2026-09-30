#!/usr/bin/env python3
"""a17: the decisive contrast, at the PARTICIPANT level (one row per person).

Seven CNN arms, a constant-majority predictor, and cross-validated metadata
regressions (age, sex, eTIV, nWBV / age only / nWBV only), on the same
participants and the same grouped folds, with paired nonparametric bootstrap CIs.
Reads analysis/outputs/_cache.npz built by analysis/_cache_build.py.
"""
import numpy as np, pandas as pd
from pathlib import Path
C=np.load("analysis/outputs/_cache.npz",allow_pickle=False)
labs=[str(x) for x in C["labs"]]; PART=C["PART"]; VOTE=C["VOTE"]
upid=C["upid"].astype(str); py=C["py"].astype(int); pf=C["pf"].astype(int)
K=3; RNG=np.random.default_rng(31337); NB=1500
meta=pd.read_excel("oasis_cross-sectional.xlsx")
meta["pid"]=meta["ID"].str.replace(r"_MR\d+$","",regex=True)
meta=meta.drop_duplicates("pid").set_index("pid").reindex(upid)
AGE=meta["Age"].to_numpy(float); CDR=meta["CDR"].to_numpy(float)
COH={"full":np.ones(len(upid),bool),"age60":(~np.isnan(CDR))&(AGE>=60)}
def sm(z): z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
def fit(X,yy,it=1200,lr=0.5,l2=1e-3):
    W=np.zeros((X.shape[1],K))
    for _ in range(it): W-=lr*(X.T@(sm(X@W)-np.eye(K)[yy])/len(yy)+l2*W)
    return W
def cvfit(cols):
    F=meta[cols].copy()
    if "M/F" in cols: F["M/F"]=(F["M/F"]=="M").astype(float)
    V=F.apply(pd.to_numeric,errors="coerce").apply(lambda c:c.fillna(c.median())).to_numpy(float)
    V=(V-V.mean(0))/V.std(0); X=np.column_stack([np.ones(len(V)),V]); Q=np.zeros((len(V),K))
    for f in range(5):
        tr,te=pf!=f,pf==f; Q[te]=sm(X[te]@fit(X[tr],py[tr]))
    return Q
MODELS={l:PART[i] for i,l in enumerate(labs)}
VOTEONLY={f"{l} (vote)":VOTE[i] for i,l in enumerate(labs)}
MODELS["META age+sex+eTIV+nWBV"]=cvfit(["Age","M/F","eTIV","nWBV"])
MODELS["META age only"]=cvfit(["Age"]); MODELS["META nWBV only"]=cvfit(["nWBV"])
MODELS["META age+nWBV"]=cvfit(["Age","nWBV"])
WQ=np.array([[(i-j)**2 for j in range(K)] for i in range(K)],float)/4.0
def met(P,yy):
    yh=P.argmax(1); M=np.zeros((K,K)); np.add.at(M,(yy,yh),1); n=M.sum()
    po=(M*WQ).sum()/n; pe=(np.outer(M.sum(1),M.sum(0))*WQ).sum()/(n*n)
    f1=[]
    for k in range(K):
        tp=M[k,k]; fp=M[:,k].sum()-tp; fn=M[k,:].sum()-tp
        f1.append(0.0 if 2*tp+fp+fn==0 else 2*tp/(2*tp+fp+fn))
    au=[]
    for k in range(K):
        s=P[:,k]; pos=(yy==k); n1=pos.sum(); n0=len(yy)-n1
        if n1==0 or n0==0: au.append(np.nan); continue
        o=np.argsort(s,kind="stable"); r=np.empty(len(s)); r[o]=np.arange(1,len(s)+1)
        au.append((r[pos].sum()-n1*(n1+1)/2)/(n1*n0))
    return np.array([np.trace(M)/n, float(np.mean(f1)), (np.nan if pe==0 else 1-po/pe),
                     float(np.nanmean(au))])
MN=["accuracy","macroF1","QWK","macroAUROC"]
rows=[]
for coh,mask in COH.items():
    yy=py[mask]; c=np.bincount(yy,minlength=K); maj=int(c.argmax())
    Pc=np.zeros((mask.sum(),K)); Pc[:,maj]=1.0
    M2=dict({k:v[mask] for k,v in MODELS.items()}); M2["BASELINE constant"]=Pc
    keys=list(M2); pts={k:met(M2[k],yy) for k in keys}
    for k in keys:
        rows.append(dict(model=k,cohort=coh,n=int(mask.sum()),kind="point",
                         **dict(zip(MN,pts[k]))))
    n=mask.sum(); bi=RNG.integers(0,n,size=(NB,n))
    cache={k:np.empty((NB,4)) for k in keys}
    for b in range(NB):
        s=bi[b]; yb=yy[s]
        for k in keys: cache[k][b]=met(M2[k][s],yb)
    for ref in ("BASELINE constant","META age+sex+eTIV+nWBV"):
        for k in keys:
            if k==ref: continue
            d=cache[k]-cache[ref]
            lo=np.nanpercentile(d,2.5,axis=0); hi=np.nanpercentile(d,97.5,axis=0)
            pt=pts[k]-pts[ref]
            for i,mn in enumerate(MN):
                rows.append(dict(model=k,cohort=coh,n=int(mask.sum()),kind=f"vs {ref}",
                                 metric=mn,point=pt[i],lo=lo[i],hi=hi[i],
                                 sig="YES" if (lo[i]>0 or hi[i]<0) else "no"))
T=pd.DataFrame(rows); T.to_csv("analysis/outputs/a17_participant_contrasts.csv",index=False)
pd.set_option("display.width",250)
print("=== PARTICIPANT-LEVEL POINT ESTIMATES ===")
print(T[T.kind=="point"][["model","cohort","n"]+MN].round(4).to_string(index=False))
for ref in ("vs BASELINE constant","vs META age+sex+eTIV+nWBV"):
    print(f"\n=== {ref}, paired participant bootstrap, {NB} resamples ===")
    s=T[T.kind==ref].copy(); s["cell"]=s.apply(lambda r:f"{r.point:+.4f} [{r.lo:+.4f},{r.hi:+.4f}] {r.sig}",axis=1)
    print(s.pivot_table(index=["cohort","model"],columns="metric",values="cell",aggfunc="first").to_string())
