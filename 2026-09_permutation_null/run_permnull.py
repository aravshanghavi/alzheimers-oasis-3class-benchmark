#!/usr/bin/env python3
"""Queue the permuted-label null: N permutation seeds x 5 grouped folds, one arm.

Each run is a separate process (the same isolation run_all.py uses), so VRAM is
reclaimed by the operating system between runs rather than by the CUDA caching
allocator. The dataset is scanned once and cached to CSV, so 86k files are not
walked forty times.

    python run_permnull.py                    # 5 seeds x 5 folds = 25 runs
    python run_permnull.py --seeds 10         # 10 seeds x 5 folds = 50 runs
    python run_permnull.py --resume           # skip runs already COMPLETE
    python run_permnull.py --smoke            # 1 seed, 1 fold, 2 epochs, 64px

At the observed ~1,700 s per run, 25 runs is about 11.8 GPU-hours. A laptop 3060
will thermally throttle over that span, so budget 14 to 16 wall hours.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.runfilter import permuted_spec  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
CONFIG = "exp14_permnull_wce.yaml"
EXPERIMENT = "exp14_permnull_wce"
LOSS = "WCE"
INIT_SEED = 42


def completed(perm_seed: int, fold: int) -> bool:
    """True when a PRODUCTION run for this spec is already COMPLETE.

    Smoke runs live in the same directory and carry a real seed and fold, so they
    are excluded here. Without that, --resume would skip the real run for
    whichever spec the smoke run happened to use.
    """
    out = REPO_ROOT / "experiments" / EXPERIMENT / "outputs"
    if not out.is_dir():
        return False
    for d in out.iterdir():
        if not (d / "_COMPLETE").exists():
            continue
        mf = d / "manifest.json"
        if not mf.exists():
            continue
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            continue
        if permuted_spec(m) == (perm_seed, fold):
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Permuted-label null queue")
    ap.add_argument("--seeds", type=int, default=5, help="number of permutation seeds (default 5)")
    ap.add_argument("--first-seed", type=int, default=1)
    ap.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--resume", action="store_true", help="skip runs already COMPLETE")
    ap.add_argument("--smoke", action="store_true", help="1 seed, 1 fold, 2 epochs, 64px")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the queue and exit")
    args = ap.parse_args()

    seeds = list(range(args.first_seed, args.first_seed + args.seeds))
    folds = list(args.folds)
    extra: list[str] = []
    if args.smoke:
        seeds, folds = seeds[:1], folds[:1]
        extra = ["--epochs", "2", "--warmup-epochs", "1", "--image-size", "64", "--no-pretrained"]
        print("SMOKE MODE: the numbers are meaningless. This only proves the wiring executes.\n")

    queue = [(s, f) for s in seeds for f in folds]
    print(f"Permuted-label null: {len(seeds)} seed(s) x {len(folds)} fold(s) = {len(queue)} runs")
    print(f"  arm            : {LOSS} (deliberately not FA-FL)")
    print(f"  init seed      : {INIT_SEED}, held fixed across replicates")
    print(f"  permuted       : train + val, participant level; test keeps TRUE labels")
    if not args.smoke:
        est = timedelta(seconds=1700 * len(queue))
        print(f"  rough estimate : {est} of GPU time, longer with thermal throttling")
    print()

    if args.dry_run:
        for s, f in queue:
            print(f"  perm_seed={s} fold={f}" + ("   [COMPLETE, would skip]"
                                                 if args.resume and completed(s, f) else ""))
        return 0

    scan_cache = REPO_ROOT / "experiments" / EXPERIMENT / "scan_cache.csv"
    scan_cache.parent.mkdir(parents=True, exist_ok=True)

    done = failed = skipped = 0
    t_start = time.time()
    for i, (s, f) in enumerate(queue, 1):
        if args.resume and completed(s, f):
            print(f"[{i}/{len(queue)}] perm_seed={s} fold={f}  SKIP (already COMPLETE)")
            skipped += 1
            continue
        cmd = [sys.executable, str(REPO_ROOT / "run_one.py"),
               "--config", CONFIG, "--loss", LOSS, "--fold", str(f),
               "--init-seed", str(INIT_SEED), "--perm-seed", str(s),
               "--console", "progress"]
        if scan_cache.exists():
            cmd += ["--scan-cache", str(scan_cache)]
        if args.data_root:
            cmd += ["--data-root", args.data_root]
        if args.device:
            cmd += ["--device", args.device]
        cmd += extra

        t0 = time.time()
        print(f"[{i}/{len(queue)}] perm_seed={s} fold={f}  started {datetime.now():%H:%M:%S}",
              flush=True)
        rc = subprocess.run(cmd).returncode
        dt = time.time() - t0
        if rc == 0:
            done += 1
            print(f"           COMPLETE in {timedelta(seconds=int(dt))}")
        else:
            failed += 1
            print(f"           FAILED rc={rc} after {timedelta(seconds=int(dt))}. "
                  "Queue continues; see the run log.")
        elapsed = time.time() - t_start
        remaining = len(queue) - i
        if remaining and (done + failed):
            eta = timedelta(seconds=int(elapsed / (done + failed) * remaining))
            print(f"           {remaining} left, ETA {eta}", flush=True)

    print(f"\nFinished. COMPLETE={done} FAILED={failed} SKIPPED={skipped}")
    print("Next: python analysis/a22_permutation_null.py")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
