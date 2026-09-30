import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
INK="#0b0b0b"; MUTED="#52514e"; GRID="#e6e5e0"; S1="#2a78d6"; S2="#eb6834"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"axes.edgecolor":MUTED,"axes.labelcolor":INK,"xtick.color":MUTED,"ytick.color":INK})
# verified values: t23_protocol_audit_long.csv (R1 participant accuracy vs constant, R4 macro-F1 vs META age+sex+eTIV+nWBV), B=2000, nominal 95%
full = {  # arm: (acc_diff, lo, hi, f1_diff, lo, hi)
 "WCE":(2.0173,-2.3055,6.9164, 0.0890,-0.0645,0.2684),
 "LDAM":(1.1527,-1.4409,3.7464,-0.1210,-0.2411,0.0238),
 "CB":(2.5937,-0.8646,6.6282, 0.0300,-0.1334,0.1992),
 "FL (γ=1.37)":(2.3055,-2.5937,7.4928, 0.1034,-0.0499,0.2421),
 "FL (γ=2.0)":(2.3055,-2.3055,7.2046, 0.0828,-0.0829,0.2321),
 "FL (γ=3.0)":(1.1527,-3.4582,6.3401, 0.0690,-0.0944,0.2220),
 "FA-FL":(2.0173,-2.0173,6.6282, 0.0220,-0.1380,0.1808)}
restr = {
 "WCE":(5.4217,-4.2169,15.6627, 0.0006,-0.1151,0.1217),
 "LDAM":(4.2169,-4.2169,12.6506,-0.1110,-0.2164,-0.0018),
 "CB":(6.6265,-2.4096,16.2651,-0.0020,-0.1207,0.1176),
 "FL (γ=1.37)":(4.8193,-5.4217,15.0602,-0.0070,-0.1144,0.1043),
 "FL (γ=2.0)":(4.8193,-6.0241,15.0602, 0.0012,-0.1073,0.1190),
 "FL (γ=2.69)":(7.2289,-3.6145,17.4699, 0.0055,-0.0982,0.1131),
 "FL (γ=3.0)":(6.6265,-4.2169,16.8675, 0.0172,-0.0954,0.1373),
 "FA-FL":(3.6145,-5.4217,12.6506,-0.0725,-0.1740,0.0291)}
fig,axes=plt.subplots(2,2,figsize=(7.2,5.0),gridspec_kw={"height_ratios":[7,8]})
titles=[("Full cohort (347 participants)","accuracy minus constant predictor (points)"),("","macro-F1 minus metadata model")]
for r,(name,d) in enumerate([("Full cohort, 347 participants",full),("Restricted cohort, 166 participants",restr)]):
    arms=list(d)[::-1]; y=np.arange(len(arms))
    for c,(k,xl) in enumerate([(0,"Accuracy minus constant predictor (percentage points)"),(3,"Macro-F1 minus age + sex + eTIV + nWBV model")]):
        ax=axes[r,c]
        v=np.array([d[a][k] for a in arms]); lo=np.array([d[a][k+1] for a in arms]); hi=np.array([d[a][k+2] for a in arms])
        ax.axvline(0,color=MUTED,lw=1,ls=(0,(3,2)))
        ax.hlines(y,lo,hi,color=S1,lw=2,capstyle="round")
        ax.plot(v,y,"o",ms=5,color=S1,mec="white",mew=1)
        ax.set_yticks(y); ax.set_yticklabels(arms if c==0 else []); ax.grid(axis="x",color=GRID,lw=0.6); ax.set_axisbelow(True)
        for s in ["top","right"]: ax.spines[s].set_visible(False)
        if r==1: ax.set_xlabel(xl)
        ax.set_title(("a" if (r,c)==(0,0) else "b" if (r,c)==(0,1) else "c" if (r,c)==(1,0) else "d")+"   "+name,loc="left",fontsize=8,color=INK)
    axes[r,0].set_xlim(-8,19); axes[r,1].set_xlim(-0.26,0.30)
fig.tight_layout()
fig.savefig("figures/fig_floors.pdf"); fig.savefig("figures/fig_floors.png",dpi=300)

# per-class F1 vs null range, restricted cohort (t23 R2 / md 8.7)
arms=["WCE","LDAM","CB","FL (γ=1.37)","FL (γ=2.0)","FL (γ=2.69)","FL (γ=3.0)","FA-FL"]
f1={"CDR 0":[0.647,0.674,0.670,0.634,0.617,0.650,0.634,0.648],
    "CDR 0.5":[0.508,0.478,0.508,0.535,0.548,0.578,0.571,0.488],
    "CDR ≥ 1":[0.400,0.069,0.368,0.364,0.391,0.341,0.400,0.200]}
null={"CDR 0":(0.380,0.578,0.507),"CDR 0.5":(0.288,0.479,0.384),"CDR ≥ 1":(0.045,0.158,0.115)}
fig,axes=plt.subplots(1,3,figsize=(7.2,2.8),sharey=True)
y=np.arange(len(arms))[::-1]
for ax,(cls,vals) in zip(axes,f1.items()):
    lo,hi,mu=null[cls]
    ax.axvspan(lo,hi,color=GRID,lw=0)
    ax.axvline(mu,color=MUTED,lw=1,ls=(0,(3,2)))
    ax.plot(vals,y,"o",ms=5,color=S1,mec="white",mew=1)
    ax.set_xlim(0,0.75); ax.set_title(cls,loc="left",fontsize=8,color=INK)
    ax.set_yticks(y); ax.set_yticklabels(arms); ax.grid(axis="x",color=GRID,lw=0.6); ax.set_axisbelow(True)
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    ax.set_xlabel("F1")
axes[0].text(0.39,-1.6,"",fontsize=7)
fig.tight_layout()
fig.savefig("figures/fig_null_perclass.pdf"); fig.savefig("figures/fig_null_perclass.png",dpi=300)
print("ok")
