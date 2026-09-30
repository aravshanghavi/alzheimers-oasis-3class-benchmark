#!/usr/bin/env python3
"""a18: incremental value of the CNN over nWBV, inter-arm agreement, age-cutoff
sensitivity, the confound-free core sub-task, and the corrected two-step null.
All from analysis/outputs/_cache.npz. Zero GPU."""
import numpy as np, pandas as pd
C=np.load("analysis/outputs/_cache.npz",allow_pickle=False)
labs=[str(x) for x in C["labs"]]; PART=C["PART"]; IMG=C["IMG"]; gi=C["gi"]
upid=C["upid"].astype(str); py=C["py"].astype(int); pf=C["pf"].astype(int); y=C["y"].astype(int)
K=3; RNG=np.random.default_rng(99); NB=1500
meta=pd.read_excel("oasis_cross-sectional.xlsx")
meta["pid"]=meta["ID"].str.replace(r"_MR\d+$","",regex=True)
meta=meta.drop_duplicates("pid").set_index("pid").reindex(upid)
AGE=meta["Age"].to_numpy(float); CDR=meta["CDR"].to_numpy(float)
NWBV=pd.to_numeric(meta["nWBV"],errors="coerce"); NWBV=NWBV.fillna(NWBV.median()).to_numpy(float)
COH={"full":np.ones(len(upid),bool),"age60":(~np.isnan(CDR))&(AGE>=60)}
WQ=np.array([[(i-j)**2 for j in range(K)] for i in range(K)],float)/4.0
def sm(z): z=z-z.max(1,keepdims=True); e=np.exp(z); return e/e.sum(1,keepdims=True)
def fit(X,yy,it=1500,lr=0.5,l2=1e-3):
    W=np.zeros((X.shape[1],K))
    for _ in range(it): W-=lr*(X.T@(sm(X@W)-np.eye(K)[yy])/len(yy)+l2*W)
    return W
def cv(X):
    Q=np.zeros((len(X),K))
    for f in range(5):
        tr,te=pf!=f,pf==f; Q[te]=sm(X[te]@fit(X[tr],py[tr]))
    return Q
def met(P,yy):
    yh=P.argmax(1); M=np.zeros((K,K)); np.add.at(M,(yy,yh),1); n=M.sum()
    po=(M*WQ).sum()/n; pe=(np.outer(M.sum(1),M.sum(0))*WQ).sum()/(n*n)
    f1=[]
    for k in range(K):
        tp=M[k,k]; fp=M[:,k].sum()-tp; fn=M[k,:].sum()-tp
        f1.append(0.0 if 2*tp+fp+fn==0 else 2*tp/(2*tp+fp+fn))
    au=[]
    for k in range(K):
        s=P[:,k]; pos=(yy==k); n1=pos.sum()
        if n1==0 or n1==len(yy): au.append(np.nan); continue
        o=np.argsort(s,kind="stable"); r=np.empty(len(s)); r[o]=np.arange(1,len(s)+1)
        au.append((r[pos].sum()-n1*(n1+1)/2)/(n1*(len(yy)-n1)))
    return np.array([np.trace(M)/n,float(np.mean(f1)),(np.nan if pe==0 else 1-po/pe),float(np.nanmean(au))])
MN=["accuracy","macroF1","QWK","macroAUROC"]
z=lambda v:(v-v.mean())/v.std()
print("=== [A] INCREMENTAL VALUE: does the CNN add anything on top of nWBV? ===")
print("    stacked logistic regression, same grouped folds, cross-validated")
rows=[]
Xn=np.column_stack([np.ones(len(upid)),z(NWBV)])
Pn=cv(Xn)
for i,l in enumerate(labs):
    lg=np.log(np.clip(PART[i],1e-6,1))
    Pc=cv(np.column_stack([np.ones(len(upid)),lg[:,1]-lg[:,0],lg[:,2]-lg[:,0]]))
    Pb=cv(np.column_stack([np.ones(len(upid)),z(NWBV),lg[:,1]-lg[:,0],lg[:,2]-lg[:,0]]))
    for coh,m in COH.items():
        yy=py[m]; a=met(Pn[m],yy); b=met(Pc[m],yy); c=met(Pb[m],yy)
        n=m.sum(); bi=RNG.integers(0,n,size=(NB,n))
        d1=np.empty((NB,4)); d2=np.empty((NB,4))
        for bb in range(NB):
            s=bi[bb]; yb=yy[s]
            mn=met(Pn[m][s],yb); mb=met(Pb[m][s],yb); mc=met(Pc[m][s],yb)
            d1[bb]=mb-mn; d2[bb]=mb-mc
        for j,nm in enumerate(MN):
            rows.append(dict(arm=l,cohort=coh,metric=nm,nwbv=a[j],cnn=b[j],both=c[j],
                add_cnn_over_nwbv=c[j]-a[j],lo1=np.nanpercentile(d1[:,j],2.5),hi1=np.nanpercentile(d1[:,j],97.5),
                add_nwbv_over_cnn=c[j]-b[j],lo2=np.nanpercentile(d2[:,j],2.5),hi2=np.nanpercentile(d2[:,j],97.5)))
TA=pd.DataFrame(rows); TA.to_csv("analysis/outputs/a18_increment.csv",index=False)
pd.set_option("display.width",260)
for coh in COH:
    s=TA[(TA.cohort==coh)&(TA.metric.isin(["QWK","macroAUROC","macroF1"]))].copy()
    s["CNN adds over nWBV"]=s.apply(lambda r:f"{r.add_cnn_over_nwbv:+.4f} [{r.lo1:+.4f},{r.hi1:+.4f}]"+(" SIG" if (r.lo1>0 or r.hi1<0) else ""),axis=1)
    s["nWBV adds over CNN"]=s.apply(lambda r:f"{r.add_nwbv_over_cnn:+.4f} [{r.lo2:+.4f},{r.hi2:+.4f}]"+(" SIG" if (r.lo2>0 or r.hi2<0) else ""),axis=1)
    print(f"\n-- cohort {coh}")
    print(s[["arm","metric","nwbv","cnn","both","CNN adds over nWBV","nWBV adds over CNN"]].round(4).to_string(index=False))
print("\n=== [B] INTER-ARM AGREEMENT (image level, full cohort) ===")
YH=IMG.argmax(2)
def kapn(a,b):
    M=np.zeros((K,K)); np.add.at(M,(a,b),1); n=M.sum()
    po=np.trace(M)/n; pe=(M.sum(1)*M.sum(0)).sum()/(n*n); return (po-pe)/(1-pe)
A=pd.DataFrame(index=labs,columns=labs,dtype=float)
for i,li in enumerate(labs):
    for j,lj in enumerate(labs): A.loc[li,lj]=kapn(YH[i],YH[j])
print(A.round(3).to_string()); A.to_csv("analysis/outputs/a18_interarm.csv")
print(f"  all seven arms agree on {np.all(YH==YH[0],axis=0).mean():.1%} of the 86,437 images")
print("\n=== [C] AGE-CUTOFF SENSITIVITY (image level) ===")
rows=[]
for nm,mask in [("no restriction",np.ones(len(upid),bool)),("CDR recorded",~np.isnan(CDR))]+[
        (f"CDR & age>={c}",(~np.isnan(CDR))&(AGE>=c)) for c in (50,55,60,65,70,75)]:
    if mask.sum()<25: continue
    im=np.isin(gi,np.where(mask)[0]); yy=y[im]
    cnt=np.bincount(yy,minlength=K); M=np.zeros((K,K)); np.add.at(M,(yy,np.full(len(yy),int(cnt.argmax()))),1)
    f1=[2*M[k,k]/(2*M[k,k]+M[:,k].sum()-M[k,k]+M[k,:].sum()-M[k,k]) if (2*M[k,k]+M[:,k].sum()-M[k,k]+M[k,:].sum()-M[k,k])>0 else 0 for k in range(K)]
    r=dict(cut=nm,participants=int(mask.sum()),images=int(im.sum()),p_major=cnt.max()/cnt.sum(),
           base_acc=np.trace(M)/M.sum(),base_f1=float(np.mean(f1)))
    for i,l in enumerate(labs):
        mm=met(IMG[i][im],yy); r[f"{l}|acc"]=mm[0]; r[f"{l}|f1"]=mm[1]; r[f"{l}|qwk"]=mm[2]
    rows.append(r)
TC=pd.DataFrame(rows); TC.to_csv("analysis/outputs/a18_agecurve.csv",index=False)
print(TC[["cut","participants","images","p_major","base_acc","base_f1"]+[f"{l}|acc" for l in labs]].round(4).to_string(index=False))
print()
print(TC[["cut","participants"]+[f"{l}|qwk" for l in labs]].round(4).to_string(index=False))
print("\n=== [D] CONFOUND-FREE CORE SUB-TASK: CDR 0.5 vs CDR >= 1 (participant level) ===")
m=np.isin(py,[1,2]); yy=(py[m]==2).astype(int)
print(f"  {m.sum()} participants, {int(yy.sum())} Demented / {int((1-yy).sum())} Very Mild;"
      f" median age {np.nanmedian(AGE[m&(py==1)]):.0f} vs {np.nanmedian(AGE[m&(py==2)]):.0f}")
def auc1(s,pos):
    n1=pos.sum(); o=np.argsort(s,kind="stable"); r=np.empty(len(s)); r[o]=np.arange(1,len(s)+1)
    return (r[pos].sum()-n1*(n1+1)/2)/(n1*(len(s)-n1))
rows=[]
for i,l in enumerate(labs):
    p2=PART[i][m][:,2]/(PART[i][m][:,1]+PART[i][m][:,2])
    a=auc1(p2,yy==1); n=m.sum(); bi=RNG.integers(0,n,size=(NB,n)); bs=np.array([auc1(p2[s],(yy[s]==1)) if 0<(yy[s]==1).sum()<n else np.nan for s in bi])
    rows.append(dict(model=l,auroc=a,lo=np.nanpercentile(bs,2.5),hi=np.nanpercentile(bs,97.5)))
a=auc1(-NWBV[m],yy==1); bi=RNG.integers(0,m.sum(),size=(NB,m.sum()))
bs=np.array([auc1(-NWBV[m][s],(yy[s]==1)) if 0<(yy[s]==1).sum()<m.sum() else np.nan for s in bi])
rows.append(dict(model="nWBV alone (lower = more atrophy)",auroc=a,lo=np.nanpercentile(bs,2.5),hi=np.nanpercentile(bs,97.5)))
a=auc1(AGE[m],yy==1); bs=np.array([auc1(AGE[m][s],(yy[s]==1)) if 0<(yy[s]==1).sum()<m.sum() else np.nan for s in bi])
rows.append(dict(model="age alone",auroc=a,lo=np.nanpercentile(bs,2.5),hi=np.nanpercentile(bs,97.5)))
TD=pd.DataFrame(rows); print(TD.round(4).to_string(index=False)); TD.to_csv("analysis/outputs/a18_coretask.csv",index=False)
print("\n=== [E] CORRECTED TWO-STEP ERROR NULL ===")
rows=[]
for coh,mask in COH.items():
    im=np.isin(gi,np.where(mask)[0]); yy=y[im]; pri=np.bincount(yy,minlength=K)/len(yy)
    for i,l in enumerate(labs):
        yh=IMG[i][im].argmax(1); M=np.zeros((K,K)); np.add.at(M,(yy,yh),1)
        off=M.copy(); np.fill_diagonal(off,0); obs=(off[0,2]+off[2,0])/off.sum()
        q=np.bincount(yh,minlength=K)/len(yh); N=np.outer(pri,q); np.fill_diagonal(N,0)
        pr=np.outer(pri,pri); np.fill_diagonal(pr,0)
        rows.append(dict(cohort=coh,arm=l,observed=obs,null_own_marginal=(N[0,2]+N[2,0])/N.sum(),
                         null_prior_random=(pr[0,2]+pr[2,0])/pr.sum(),
                         wrong_value_used_in_v2=0.2857))
TE=pd.DataFrame(rows); print(TE.round(4).to_string(index=False)); TE.to_csv("analysis/outputs/a18_twostep.csv",index=False)
print("\nDONE")
