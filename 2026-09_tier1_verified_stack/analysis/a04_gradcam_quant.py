#!/usr/bin/env python3
"""Quantitative Grad-CAM over whole test folds.

Reviewer 1 comment 7: the submitted Grad-CAM evidence was three hand-picked
images with no numbers behind it. This computes, over a participant-stratified
sample of each test fold:

  * the fraction of CAM mass falling inside a brain mask rather than background
    or skull -- does the model look at the brain at all;
  * the spatial spread of the CAM (normalised second moment) per predicted class
    -- the manuscript claims activation becomes more distributed with severity,
    and this is the number that either supports that or does not;
  * the CAM centroid position per class.

Everything is aggregated with participants as the unit, so the spread comparison
is not driven by one person contributing hundreds of slices.

If the effect is not there, the claim in section 5.2.2 should be deleted. A null
result here is a finding, not a failure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib                                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                      # noqa: E402

from analysis._common import CLASS_NAMES, N_CLASSES, OUT, save_table  # noqa: E402
from analysis.discover import load_runs                              # noqa: E402


def brain_mask(img: np.ndarray) -> np.ndarray:
    """Otsu threshold, then keep the largest connected component. No atlas needed."""
    x = img.astype(np.float64)
    x = (x - x.min()) / max(x.max() - x.min(), 1e-9)
    hist, edges = np.histogram(x, bins=64, range=(0, 1))
    p = hist / max(hist.sum(), 1)
    omega = np.cumsum(p)
    mu = np.cumsum(p * ((edges[:-1] + edges[1:]) / 2))
    mu_t = mu[-1]
    denom = omega * (1 - omega)
    with np.errstate(invalid="ignore", divide="ignore"):
        sigma_b = np.where(denom > 0, (mu_t * omega - mu) ** 2 / denom, 0.0)
    thr = ((edges[:-1] + edges[1:]) / 2)[int(np.nanargmax(sigma_b))]
    m = x > thr
    try:
        from scipy import ndimage
        lab, n = ndimage.label(m)
        if n > 1:
            sizes = ndimage.sum(m, lab, range(1, n + 1))
            m = lab == (int(np.argmax(sizes)) + 1)
        m = ndimage.binary_fill_holes(m)
    except Exception:
        pass                       # scipy optional: plain threshold is still informative
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", default="FA_FL")
    ap.add_argument("--participants-per-fold", type=int, default=40)
    ap.add_argument("--slices-per-participant", type=int, default=15)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    print("[a04] quantitative Grad-CAM")
    try:
        import torch
        from pytorch_grad_cam import GradCAM
        from src.data import DementiaDataset, build_transforms, make_splits, scan_dataset
        from src.model import DementiaModel
        from src.seeding import resolve_device
        from src.config import _normalise_class_map
    except Exception as exc:
        print(f"  SKIPPED -- required package missing: {exc!r}")
        print("  install with:  pip install grad-cam")
        (OUT / "gradcam_SKIPPED.txt").write_text(repr(exc), encoding="utf-8")
        return 0

    import json
    runs = load_runs("exp01_main_sweep")
    runs = runs[runs["loss"] == args.loss].sort_values("fold")
    if runs.empty:
        print(f"  no runs for {args.loss}")
        return 0

    cfg = json.loads((Path(runs.iloc[0]["run_dir"]) / "manifest.json").read_text())["resolved_config"]
    _normalise_class_map(cfg)
    device = resolve_device(args.device or cfg["device"])
    df = scan_dataset(cfg["data"]["root"], cfg["data"]["original_classes"], cfg["data"]["class_map"])
    tf = build_transforms(cfg["data"]["image_size"])["test"]
    rng = np.random.default_rng(0)

    rows = []
    for _, r in runs.iterrows():
        fold = int(r["fold"])
        cfg["split"]["fold"] = fold
        splits = make_splits(df, cfg)
        test = df.iloc[splits["test"]]

        model = DementiaModel(N_CLASSES, pretrained=False,
                              head_hidden=cfg["model"]["head_hidden"],
                              dropout=cfg["model"]["dropout"])
        try:
            state = torch.load(Path(r["run_dir"]) / "checkpoint_best.pt",
                               map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(Path(r["run_dir"]) / "checkpoint_best.pt", map_location=device)
        model.load_state_dict(state); model.to(device).eval()
        cam = GradCAM(model=model, target_layers=model.gradcam_target_layers)

        picks = []
        for pid, grp in test.groupby("participant_id"):
            picks.append((pid, grp))
        rng.shuffle(picks)
        picks = picks[: args.participants_per_fold]
        print(f"  fold {fold}: {len(picks)} participants")

        for pid, grp in picks:
            sub = grp.sample(min(args.slices_per_participant, len(grp)), random_state=0)
            for _, row in sub.iterrows():
                from PIL import Image
                with Image.open(row["path"]) as im:
                    pil = im.convert("RGB").copy()
                x = tf(pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = int(model(x).argmax(1).item())
                g = cam(input_tensor=x, targets=None)[0]
                g = np.clip(g, 0, None)
                if g.sum() <= 0:
                    continue
                g = g / g.sum()
                gray = np.asarray(pil.convert("L").resize(g.shape[::-1]), dtype=np.float64)
                m = brain_mask(gray)
                H, W = g.shape
                yy, xx = np.mgrid[0:H, 0:W]
                cy, cx = float((g * yy).sum()), float((g * xx).sum())
                spread = float(np.sqrt((g * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum()) / np.hypot(H, W))
                rows.append({"fold": fold, "participant_id": pid, "true_class": int(row["class_3"]),
                             "pred_class": pred, "mass_in_brain": float(g[m].sum()),
                             "centroid_y": cy / H, "centroid_x": cx / W, "spread": spread})

    if not rows:
        print("  no CAMs computed")
        return 0
    per_image = pd.DataFrame(rows)
    per_image.to_csv(OUT / "gradcam_per_image.csv", index=False)

    per_part = (per_image.groupby(["participant_id", "true_class"])
                .agg(mass_in_brain=("mass_in_brain", "mean"), spread=("spread", "mean"),
                     centroid_y=("centroid_y", "mean"), centroid_x=("centroid_x", "mean"),
                     n_images=("spread", "size")).reset_index())
    save_table(per_part, "gradcam_per_participant")

    summary = (per_part.groupby("true_class")
               .agg(participants=("participant_id", "nunique"),
                    mass_in_brain_mean=("mass_in_brain", "mean"),
                    mass_in_brain_sd=("mass_in_brain", "std"),
                    spread_mean=("spread", "mean"), spread_sd=("spread", "std")).reset_index())
    summary["class"] = [CLASS_NAMES[c] for c in summary["true_class"]]
    save_table(summary, "gradcam_summary")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    from src.figures import SERIES, INK_MUTED, _style, _save
    for ax, col, lab in ((axes[0], "mass_in_brain", "fraction of CAM mass inside brain mask"),
                         (axes[1], "spread", "CAM spatial spread (normalised)")):
        for c in range(N_CLASSES):
            v = per_part[per_part["true_class"] == c][col].to_numpy()
            if not len(v):
                continue
            ax.scatter(np.full(len(v), c) + rng.normal(0, 0.05, len(v)), v,
                       color=SERIES[c], s=30, alpha=0.75)
            ax.plot([c - 0.25, c + 0.25], [v.mean()] * 2, color=SERIES[c], lw=2.5)
        ax.set_xticks(range(N_CLASSES), [n.replace(" ", "\n") for n in CLASS_NAMES], fontsize=8)
        ax.set_ylabel(lab); _style(ax)
    fig.suptitle("Grad-CAM, aggregated per participant (one point = one participant)",
                 fontsize=10, x=0.02, ha="left")
    _save(fig, OUT / "fig_gradcam_quant.png")
    print("    -> fig_gradcam_quant.png")
    print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
