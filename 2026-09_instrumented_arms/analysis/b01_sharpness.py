#!/usr/bin/env python3
"""b01: is image sharpness a third image-readable shortcut, alongside age and sex?

    python analysis/b01_sharpness.py
    python analysis/b01_sharpness.py --quick      # 200 resamples
    python analysis/b01_sharpness.py --boot 10000

THE MECHANISM UNDER TEST
------------------------
Head motion during acquisition blurs an MR image. Head motion is commoner in
older people and commoner in people with cognitive impairment, for reasons that
have nothing to do with what the image shows about the brain. So blur is a
candidate shortcut: a quantity a convolutional network can read straight off the
pixels, correlated with the label, and carrying no information about
neurodegeneration at all.

Two such shortcuts are already documented in this project. Age is readable and
predicts CDR class. Sex is readable and the networks reproduce a sex-specific
CDR 0.5 base rate. If per-participant sharpness also separates the CDR classes,
that is a third one, and the paper has to name it in the limitations rather than
wait for a reviewer to ask.

The test is deliberately blunt and does not need a model. Compute a sharpness
number per image, average it per participant, and ask how well that single number
separates the classes by itself. A quantity that separates the classes on its own
is available to any network trained on these images, whether or not this
particular network used it.

WHAT IS MEASURED
----------------
The variance of the Laplacian of the grayscale image. It is the standard
no-reference focus measure: the Laplacian responds to intensity transitions, a
sharp image has many strong transitions and therefore a high variance, a blurred
one has few. It is a RELATIVE measure. It is sensitive to image content, to
contrast and to resolution, so its absolute value means nothing and only the
comparison between participants scanned on the same protocol is interpretable.
That is the comparison made here, and it is the reason this is an association
test and not a measurement of motion.

NO SCIPY, SO SPEARMAN IS BUILT FROM WHAT IS HERE
------------------------------------------------
Spearman's rank correlation is Pearson's correlation computed on ranks. This
environment has no scipy, so it is computed exactly that way, on the midranks
from _protocol_lib (tied values share the average of their positions). This is
the Spearman coefficient, not an approximation of it. No p-value comes from a
t-approximation; the interval comes from the participant bootstrap like
everything else in this file.

THE LAPLACIAN ITSELF
--------------------
OpenCV's cv2.Laplacian is used when OpenCV is importable, and a three-by-three
numpy convolution with the same four-neighbour kernel is used when it is not. No
dependency is added either way. The output header says which one ran, because the
two differ at the image border and the border is a few percent of a small image.

THIS SCRIPT NEEDS THE JPEG FILES, WHICH MAY NOT BE REACHABLE
------------------------------------------------------------
Every other analysis in this stack runs off the prediction archives and needs no
pixels. This one needs the 39,894 restricted-cohort JPEGs. Their location is
taken from resolved_config.data.root in the run MANIFESTS, not from
config/base.yaml, because the manifests record the root the results were actually
computed on and the live config may have been edited since.

If that root is not a reachable directory the script says so in one line and
exits BEFORE it opens anything. A loop that dies on image 12,000 after twenty
minutes is worse than no run at all, and on a deadline it is the difference
between a known gap and a broken night.

MULTIPLICITY
------------
No family in analysis/_registry.py covers an image-quality covariate. Every row
carries the literal string "undeclared", every interval is a nominal 95 percent
one, and nothing here is a pre-registered test. Nothing is written to the
registry: the changelog entry is PRINTED for the first author to paste.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import analysis.discover as dsc                                               # noqa: E402
from analysis._common import save_table                                       # noqa: E402
from analysis._paths import EXPERIMENTS_DIR, OUT, metadata_xlsx               # noqa: E402
from analysis._protocol_lib import set_include_new_arms                       # noqa: E402
from analysis._protocol_lib import (Cohort, RESTRICTED_EXPERIMENTS,           # noqa: E402
                                    changelog_entry, midrank_auc, midranks,
                                    percentile_interval,
                                    stratified_participant_bootstrap,
                                    unstratified_participant_bootstrap)
from analysis.a23_protocol_audit import ARM_ORDER                             # noqa: E402
from analysis.a25_increment_over_metadata import auc_draws, multiplicity      # noqa: E402

SOURCE = "analysis/b01_sharpness.py"
EXPERIMENT_ID = "b01"
TABLE = "t38_sharpness"
N_BOOT = 10000
QUICK_BOOT = 200
BASE_SEED = 12345
N_CLASSES = 3
UNDECLARED = "undeclared"

EXPECTED_IMAGES = 39894
EXPECTED_PARTICIPANTS = 166

MEASURE = ("variance of the Laplacian of the grayscale image, the standard no-reference "
           "focus measure. Relative only: higher means sharper within one protocol, and "
           "the absolute value carries no units")

# The three clinically readable separations, the same three a25 and a26 use, so the
# rows join.
PAIRWISE: List[Tuple[str, Optional[int], int]] = [
    ("AUC CDR>=1 vs CDR 0", 2, 0),
    ("AUC any impairment vs CDR 0", None, 0),
    ("AUC CDR>=1 vs CDR 0.5", 2, 1),
]

LAPLACIAN_KERNEL = np.array([[0.0, 1.0, 0.0],
                             [1.0, -4.0, 1.0],
                             [0.0, 1.0, 0.0]])


# ---------------------------------------------------------------------------
# The two things that can stop this script before it starts
# ---------------------------------------------------------------------------

def restricted_runs() -> pd.DataFrame:
    """The COMPLETE restricted-cohort runs, without building a whole Cohort.

    Cohort loads and pools every arm's predictions, which costs seconds this
    script may be about to throw away. The data root lives in the manifests, so
    it can be read from the run list alone and the reachability check can happen
    before any of that work.
    """
    prev = dsc.EXPERIMENTS
    try:
        dsc.EXPERIMENTS = Path(EXPERIMENTS_DIR)
        frames = []
        for exp in RESTRICTED_EXPERIMENTS:
            f = dsc.load_runs(exp)
            if not f.empty:
                frames.append(f)
    finally:
        dsc.EXPERIMENTS = prev
    if not frames:
        raise SystemExit(f"b01 CANNOT RUN. No COMPLETE restricted-cohort runs under "
                         f"{EXPERIMENTS_DIR}.")
    return pd.concat(frames, ignore_index=True)


def manifest_data_root(runs: pd.DataFrame) -> str:
    """resolved_config.data.root from the manifests, and it must be unanimous.

    _common.data_root_from_runs does the same thing and raises on disagreement.
    It is reimplemented here only so the failure message can name b01 and say
    what the consequence is, rather than exiting from a shared helper with a
    message about tables.
    """
    roots = set()
    for _, r in runs.iterrows():
        m = json.loads((Path(r["run_dir"]) / "manifest.json").read_text(encoding="utf-8"))
        roots.add(m["resolved_config"]["data"]["root"])
    if len(roots) != 1:
        raise SystemExit(f"b01 CANNOT RUN. The restricted-cohort runs record "
                         f"{len(roots)} different data roots: {sorted(roots)}. Sharpness "
                         f"measured across two roots would not be one measurement.")
    return roots.pop()


def remap(image_path: str, manifest_root: str, resolved_root: Path) -> Path:
    """The image's path under the root this process can actually reach.

    The archives store absolute paths from the machine that trained the models. If
    the same data is reachable at another mount point, only the prefix differs, so
    the recorded prefix is swapped for the resolved one and the rest is kept.
    Separators are normalised because the paths were written on Windows.
    """
    p = str(image_path).replace("\\", "/")
    root = str(manifest_root).replace("\\", "/").rstrip("/")
    if p.lower().startswith(root.lower()):
        return resolved_root / p[len(root):].lstrip("/")
    return Path(p)


# ---------------------------------------------------------------------------
# Sharpness
# ---------------------------------------------------------------------------

def load_opencv():
    try:
        import cv2                                                    # noqa: PLC0415
        return cv2
    except Exception:                                                 # noqa: BLE001
        return None


def laplacian_variance_numpy(img: np.ndarray) -> float:
    """Variance of the four-neighbour Laplacian over the interior pixels only.

    The border is dropped rather than padded. Padding invents intensity beyond the
    edge and the invented values land in the variance; on a 208 pixel image the
    border is under two percent of the pixels, so dropping it costs less than
    making something up.
    """
    a = np.asarray(img, dtype=np.float64)
    if a.ndim == 3:
        a = a.mean(axis=2)
    if a.shape[0] < 3 or a.shape[1] < 3:
        return float("nan")
    lap = (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
           - 4.0 * a[1:-1, 1:-1])
    return float(lap.var())


def sharpness_of_file(path: Path, cv2mod) -> float:
    if cv2mod is not None:
        img = cv2mod.imread(str(path), cv2mod.IMREAD_GRAYSCALE)
        if img is None:
            return float("nan")
        return float(cv2mod.Laplacian(img, cv2mod.CV_64F).var())
    from PIL import Image                                             # noqa: PLC0415
    with Image.open(path) as im:
        return laplacian_variance_numpy(np.asarray(im.convert("L")))


def spearman_on_midranks(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho, computed as Pearson on midranks. No scipy in this environment."""
    rx, ry = midranks(x), midranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def spearman_draws(x: np.ndarray, y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Spearman on each resample. Ranks are recomputed inside every draw.

    Ranking once on the full sample and then resampling the ranks would hold the
    rank structure fixed and understate the interval. The loop is over draws
    rather than vectorised because ranking is not a linear operation in the
    multiplicities, so there is no matrix form to use.
    """
    out = np.empty(idx.shape[0])
    for b in range(idx.shape[0]):
        take = idx[b]
        out[b] = spearman_on_midranks(x[take], y[take])
    return out


def main() -> int:
    t_start = time.time()
    ap = argparse.ArgumentParser(
        description="Per-participant image sharpness as a candidate shortcut.")
    ap.add_argument("--boot", type=int, default=N_BOOT,
                    help=f"participant-clustered resamples (default {N_BOOT})")
    ap.add_argument("--quick", action="store_true",
                    help=f"smoke test with {QUICK_BOOT} resamples; intervals are NOT reportable")
    ap.add_argument("--data-root", default=None, metavar="DIR",
                    help="override the data root recorded in the manifests")
    ap.add_argument("--include-new-arms", action="store_true",
                    help="also read exp15, exp16 and exp17 when they have COMPLETE runs. "
                         "OFF by default, which reproduces this script's table exactly. "
                         "Also settable with SREP_INCLUDE_NEW_ARMS=1")
    args = ap.parse_args()
    set_include_new_arms(args.include_new_arms)
    n_boot = QUICK_BOOT if args.quick else int(args.boot)
    if args.quick:
        print(f"QUICK MODE. {n_boot} resamples. Intervals are a smoke test, not results.")

    print(f"b01  image sharpness as a shortcut, {n_boot:,} resamples, base seed {BASE_SEED}")
    print(f"     restricted runs from {EXPERIMENTS_DIR}")
    print(f"     measure: {MEASURE}")

    # -- the reachability gate, before any pixel is opened -----------------------
    runs = restricted_runs()
    manifest_root = manifest_data_root(runs)
    resolved = Path(args.data_root).expanduser() if args.data_root else Path(manifest_root)
    print(f"     data root recorded in the manifests: {manifest_root}")
    if args.data_root:
        print(f"     overridden on the command line to: {resolved}")
    if not resolved.is_dir():
        print(f"\nb01 CANNOT RUN. The image data root '{resolved}' is not a reachable "
              f"directory from this machine, so the 39,894 restricted-cohort JPEGs cannot "
              f"be read and no sharpness number can be computed. Nothing was written. "
              f"Re-run where that path exists, or pass --data-root pointing at the folder "
              f"holding the class subdirectories.")
        print(f"  wall time {time.time() - t_start:.1f} s")
        return 2
    print(f"     data root is reachable")

    cv2mod = load_opencv()
    backend = ("cv2.Laplacian, OpenCV " + cv2mod.__version__) if cv2mod is not None else \
        "numpy three-by-three convolution, interior pixels only, OpenCV not importable"
    print(f"     Laplacian backend: {backend}")

    coh = Cohort("restricted", RESTRICTED_EXPERIMENTS, EXPERIMENTS_DIR)
    y = coh.labels
    cls = np.bincount(y, minlength=N_CLASSES)
    print(f"     restricted: {len(coh.arms)} arms, {coh.n_participants} participants, "
          f"{coh.n_images:,} images, CDR 0 / 0.5 / >=1 = {cls[0]} / {cls[1]} / {cls[2]}")
    if coh.n_images != EXPECTED_IMAGES or coh.n_participants != EXPECTED_PARTICIPANTS:
        print(f"  !! expected {EXPECTED_IMAGES:,} images over {EXPECTED_PARTICIPANTS} "
              f"participants. Got {coh.n_images:,} over {coh.n_participants}. The subset "
              f"below is not the one the paper names.")

    # -- one image list, from one arm's five folds -------------------------------
    arm = next(a for a in ARM_ORDER if a in coh.arms)
    sub = runs[runs["loss"] == coh.runs[coh.runs["arm"] == arm]["loss"].iloc[0]]
    sub = sub[sub["run_dir"].isin(coh.runs[coh.runs["arm"] == arm]["run_dir"])]
    parts = []
    for _, r in sub.sort_values("fold").iterrows():
        z = np.load(Path(r["run_dir"]) / "predictions_test.npz", allow_pickle=False)
        parts.append(pd.DataFrame({"participant_id": z["participant_id"].astype(str),
                                   "label": z["labels"].astype(int),
                                   "image_path": z["image_path"].astype(str)}))
    images = pd.concat(parts, ignore_index=True).drop_duplicates("image_path")
    print(f"     image list from the {arm} arm's five test folds: {len(images):,} unique "
          f"files over {images['participant_id'].nunique()} participants")

    # -- the loop --------------------------------------------------------------
    print(f"     measuring sharpness on {len(images):,} files ...")
    vals = np.empty(len(images))
    missing: List[str] = []
    paths = [remap(p, manifest_root, resolved) for p in images["image_path"]]
    for i, p in enumerate(paths):
        vals[i] = sharpness_of_file(p, cv2mod)
        if not np.isfinite(vals[i]):
            missing.append(str(p))
        if (i + 1) % 5000 == 0:
            print(f"       {i + 1:,} of {len(paths):,}")
    if missing:
        print(f"  !! {len(missing)} of {len(paths)} files could not be read, first "
              f"{missing[0]}. Every participant average below is over the files that "
              f"could.")
    images["sharpness"] = vals

    per = images.groupby("participant_id", sort=True)["sharpness"].mean()
    per = per.reindex(coh.participants)
    if per.isna().any():
        raise SystemExit(f"b01 STOPPED. {int(per.isna().sum())} participant(s) have no "
                         f"readable image, so their sharpness average does not exist and "
                         f"the AUCs below would be computed on a different cohort.")
    s = per.to_numpy(dtype=float)
    n_img_per = images.groupby("participant_id", sort=True)["sharpness"].size() \
        .reindex(coh.participants).to_numpy()

    rows: List[Dict] = []
    for c in range(N_CLASSES):
        m = y == c
        rows.append({"kind": "descriptive", "group": f"CDR class {c}", "n": int(m.sum()),
                     "mean_sharpness": float(s[m].mean()),
                     "median_sharpness": float(np.median(s[m])),
                     "sd_sharpness": float(s[m].std(ddof=1)),
                     "min_sharpness": float(s[m].min()), "max_sharpness": float(s[m].max()),
                     "measure": MEASURE, "laplacian_backend": backend,
                     "n_images": int(n_img_per[m].sum()), "family": UNDECLARED,
                     "n_bootstrap": n_boot, "note": "per-participant mean over that "
                                                    "participant's images"})

    # -- the three AUCs --------------------------------------------------------
    print(f"     AUC of per-participant sharpness alone, {n_boot:,} stratified resamples ...")
    for name, higher, lower in PAIRWISE:
        mask = ((y > 0) if higher is None else (y == higher)) | (y == lower)
        pos = (y[mask] > 0) if higher is None else (y[mask] == higher)
        point = midrank_auc(s[mask], pos)
        idx = stratified_participant_bootstrap(pos.astype(int), n_boot, "b01:auc",
                                              BASE_SEED, contrast=name)
        d = auc_draws(s[mask], pos, multiplicity(idx, int(mask.sum())))
        lo, hi = percentile_interval(d)
        rows.append({"kind": "auc", "group": name, "n": int(mask.sum()),
                     "n_positive": int(pos.sum()), "n_negative": int((~pos).sum()),
                     "value": float(point), "ci_low": lo, "ci_high": hi, "ci_level": 0.95,
                     "bootstrap": "stratified participant, resampled within class",
                     "measure": MEASURE, "laplacian_backend": backend,
                     "n_bootstrap": n_boot, "family": UNDECLARED,
                     "note": "sharpness alone, no model. An AUC away from 0.5 here means "
                             "the quantity is available to any network trained on these "
                             "images"})

    # -- sharpness against age within CDR 0 ------------------------------------
    meta = pd.read_excel(metadata_xlsx())
    meta["pid"] = meta["ID"].astype(str).str.replace(r"_MR\d+$", "", regex=True)
    meta = meta.drop_duplicates("pid").set_index("pid").reindex(coh.participants)
    age = pd.to_numeric(meta["Age"], errors="coerce").to_numpy()
    for group, m in (("CDR 0 only", (y == 0) & np.isfinite(age)),
                     ("whole restricted cohort", np.isfinite(age))):
        rho = spearman_on_midranks(s[m], age[m])
        idx = unstratified_participant_bootstrap(int(m.sum()), n_boot, "b01:rho",
                                                 BASE_SEED, group=group)
        d = spearman_draws(s[m], age[m], idx)
        lo, hi = percentile_interval(d)
        rows.append({"kind": "rank_correlation", "group": f"sharpness vs age, {group}",
                     "n": int(m.sum()), "value": float(rho),
                     "ci_low": lo, "ci_high": hi, "ci_level": 0.95,
                     "bootstrap": "unstratified participant",
                     "measure": MEASURE, "laplacian_backend": backend,
                     "n_bootstrap": n_boot, "family": UNDECLARED,
                     "note": "Spearman computed as Pearson on midranks, because this "
                             "environment has no scipy. Ranks are recomputed inside every "
                             "resample"})

    table = pd.DataFrame(rows)
    lead = ["kind", "group", "n", "value", "ci_low", "ci_high", "ci_level", "bootstrap"]
    table = table[[c for c in lead if c in table.columns]
                  + [c for c in table.columns if c not in lead]]
    save_table(table, TABLE, float_fmt="%.6f")

    # -- one screen ------------------------------------------------------------
    elapsed = time.time() - t_start
    desc = table[table["kind"] == "descriptive"]
    auc = table[table["kind"] == "auc"]
    rho = table[table["kind"] == "rank_correlation"]
    print("\n" + "=" * 100)
    print("b01  IS SHARPNESS A THIRD IMAGE-READABLE SHORTCUT?")
    print("=" * 100)
    print(f"  {MEASURE}.")
    print(f"  Laplacian backend: {backend}.")
    print(f"\n  {'group':<22}{'n':>5}{'mean':>14}{'median':>14}{'sd':>14}")
    for _, r in desc.iterrows():
        print(f"  {r['group']:<22}{int(r['n']):>5}{r['mean_sharpness']:>14.2f}"
              f"{r['median_sharpness']:>14.2f}{r['sd_sharpness']:>14.2f}")
    print(f"\n  SHARPNESS ALONE, NO MODEL:")
    print(f"  {'contrast':<34}{'AUC':>8}{'95 percent interval':>26}")
    for _, r in auc.iterrows():
        flag = "  <-- separates" if (r["ci_low"] > 0.5 or r["ci_high"] < 0.5) else ""
        print(f"  {r['group']:<34}{r['value']:>8.3f}"
              f"{f'[{r.ci_low:.3f}, {r.ci_high:.3f}]':>26}{flag}")
    print(f"\n  SHARPNESS AGAINST AGE (Spearman as Pearson on midranks, no scipy):")
    for _, r in rho.iterrows():
        print(f"  {r['group']:<44}{r['value']:>+8.3f}"
              f"{f'[{r.ci_low:+.3f}, {r.ci_high:+.3f}]':>26}   n={int(r['n'])}")
    sep = int(((auc["ci_low"] > 0.5) | (auc["ci_high"] < 0.5)).sum())
    print(f"\n  VERDICT. {sep} of {len(auc)} contrasts have an interval excluding 0.5.")
    if sep:
        print(f"  Sharpness is readable off these images and separates the classes without "
              f"a model, so it")
        print(f"  belongs in the limitations beside age and sex. This is ASSOCIATION. It "
              f"does not show that")
        print(f"  any network used it, and it does not measure head motion.")
    else:
        print(f"  Sharpness does not separate the classes on its own in this cohort, which "
              f"removes it as a")
        print(f"  candidate shortcut. It does not rule out other image-quality effects.")
    print(f"\n  EVERY comparison here is UNDECLARED. No registry family covers an "
          f"image-quality covariate,")
    print(f"  and every interval is a nominal 95 percent one.")
    if args.quick:
        print(f"\n  QUICK MODE: {n_boot} resamples. Re-run without --quick before quoting.")
    print(f"\n  outputs in {OUT}")
    print(f"  wall time {elapsed:.1f} s")

    print("\n  PASTE THIS INTO analysis/REGISTRY_CHANGELOG.md (nothing was written for you):")
    print(changelog_entry(
        EXPERIMENT_ID,
        "per-participant image sharpness, the variance of the Laplacian averaged over each "
        "participant's images, as a standalone separator of the restricted cohort's CDR "
        "classes, and its rank correlation with age within CDR 0",
        "midrank AUC (three contrasts) and Spearman rank correlation",
        "participant"))
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        raise SystemExit(main())
