import json, numpy as np, pandas as pd
from pathlib import Path
EXP=Path("experiments"); K=3
ARMS=[("Weighted CE","exp01_main_sweep","WCE"),("LDAM","exp01_main_sweep","LDAM"),
      ("Focal g3.0","exp01_main_sweep","FocalLoss"),("FA-FL","exp01_main_sweep","FA_FL"),
      ("Class-Balanced","exp02_class_balanced","ClassBalanced"),
      ("Focal g1.37","exp07_focal_gamma137","FocalLoss"),
      ("Focal g2.0","exp08_focal_gamma200","FocalLoss")]
def load(exp,loss):
    fr=[]
    for mf in sorted((EXP/exp/"outputs").glob("*/manifest.json")):
        m=json.loads(mf.read_text(encoding="utf-8"))
        if m.get("status")!="COMPLETE" or m.get("loss",{}).get("type")!=loss: continue
        if m.get("split",{}).get("group_by")!="participant": continue
        d=np.load(mf.parent/"predictions_test.npz",allow_pickle=False)
        f=pd.DataFrame(d["probabilities"].astype(np.float32),columns=["p0","p1","p2"])
        f["pid"]=d["participant_id"].astype(str); f["y"]=d["labels"].astype(np.int8)
        f["img"]=d["image_path"].astype(str); f["fold"]=np.int8(m["split"]["fold"]); fr.append(f)
    p=pd.concat(fr,ignore_index=True)
    assert (p.groupby("pid")["fold"].nunique()==1).all(), f"{exp}/{loss} folds not disjoint"
    return p.sort_values(["pid","img"]).reset_index(drop=True)
labs=[a[0] for a in ARMS]; F={l:load(e,ls) for l,e,ls in ARMS}
b=F[labs[0]]
for l in labs[1:]:
    assert (F[l]["pid"].values==b["pid"].values).all() and (F[l]["img"].values==b["img"].values).all()
IMG=np.stack([F[l][["p0","p1","p2"]].to_numpy(np.float32) for l in labs])   # arms x N x 3
pid=b["pid"].to_numpy(); y=b["y"].to_numpy(); img=b["img"].to_numpy(); fold=b["fold"].to_numpy()
g=b.groupby("pid")
upid=np.array(sorted(set(pid))); pos={p:i for i,p in enumerate(upid)}
gi=np.array([pos[p] for p in pid])
PART=np.zeros((len(labs),len(upid),3),np.float64); cnt=np.bincount(gi,minlength=len(upid))
VOTE=np.zeros((len(labs),len(upid),3),np.float64)
for a in range(len(labs)):
    for c in range(3): PART[a,:,c]=np.bincount(gi,weights=IMG[a][:,c],minlength=len(upid))/cnt
    yh=IMG[a].argmax(1)
    for c in range(3): VOTE[a,:,c]=np.bincount(gi,weights=(yh==c).astype(float),minlength=len(upid))/cnt
py=np.zeros(len(upid),np.int8); pf=np.zeros(len(upid),np.int8)
py[gi]=y; pf[gi]=fold
np.savez_compressed("analysis/outputs/_cache.npz",labs=np.array(labs),IMG=IMG,y=y,gi=gi,
                    upid=upid,PART=PART,VOTE=VOTE,py=py,pf=pf,img=img)
print("cached:",IMG.shape,PART.shape,len(upid),"participants")
