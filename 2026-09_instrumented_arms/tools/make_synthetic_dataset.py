#!/usr/bin/env python3
"""Generate a synthetic dataset that mirrors the real OASIS folder's STRUCTURE.

Used only to exercise the pipeline where the real 86,437-image dataset is not
available. It reproduces the properties the code actually depends on:

  * filename schema  OAS1_XXXX_MR<n>_mpr-<a>_<slice>.jpg
  * 347 participants / 366 sessions, with 19 Non-Demented participants
    carrying two sessions (the OASIS-1 reliability subset)
  * participants per original class: 266 / 58 / 21 / 2
  * multiple MPRAGE acquisitions per session, contiguous slice indices
  * a weak class-dependent visual signal, so a smoke run is not pure noise

Slice count and acquisition count are reduced for speed; nothing else differs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]
N_PARTICIPANTS = [266, 58, 21, 2]      # real participant counts per original class
N_MULTI_SESSION = 19                   # all Non-Demented, per the reliability subset


def synth_image(rng: np.random.Generator, cls: int, slice_idx: int, size=(248, 496)) -> Image.Image:
    h, w = size
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = h / 2, w / 2
    skull = np.exp(-(((yy - cy) / (h * 0.42)) ** 2 + ((xx - cx) / (w * 0.42)) ** 2) ** 3)
    brain = 110 + 40 * skull + 8 * rng.standard_normal((h, w))
    # ventricles widen with severity: the signal the model can learn
    vr = 0.045 + 0.030 * cls + 0.004 * ((slice_idx % 7) - 3)
    for sign in (-1, 1):
        v = np.exp(-(((yy - cy) / (h * vr * 1.6)) ** 2 + ((xx - (cx + sign * w * 0.07)) / (w * vr)) ** 2))
        brain -= 95 * v
    brain *= skull > 0.02
    return Image.fromarray(np.clip(brain, 0, 255).astype(np.uint8)).convert("RGB")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--slices", type=int, default=6, help="slice positions per acquisition")
    ap.add_argument("--acquisitions", type=int, default=2)
    ap.add_argument("--slice-start", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    root = Path(args.out)
    for c in CLASSES:
        (root / c).mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    pid, total = 1, 0
    for cls, n_part in enumerate(N_PARTICIPANTS):
        for k in range(n_part):
            participant = f"OAS1_{pid:04d}"
            pid += 1
            n_sessions = 2 if (cls == 0 and k < N_MULTI_SESSION) else 1
            for session in range(1, n_sessions + 1):
                for acq in range(1, args.acquisitions + 1):
                    for s in range(args.slice_start, args.slice_start + args.slices):
                        img = synth_image(rng, cls, s)
                        img.save(root / CLASSES[cls] / f"{participant}_MR{session}_mpr-{acq}_{s}.jpg",
                                 quality=70)
                        total += 1
    print(f"wrote {total:,} images for {pid-1} participants to {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
