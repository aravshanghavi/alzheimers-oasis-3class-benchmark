#!/usr/bin/env python3
"""Pre-flight: print the exact participant composition of every fold.

Run this BEFORE committing the GPU to a long queue. It uses the same code path
as training, so what it prints is what the runs will use -- including your local
scikit-learn version, which matters: StratifiedGroupKFold's assignment changed
between releases, so fold composition is version-dependent.

    python tools/check_folds.py
    python tools/check_folds.py --config exp01_main_sweep.yaml --split-seed 42
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sklearn  # noqa: E402

from src.config import load_experiment_config  # noqa: E402
from src.data import (assert_group_disjoint, cohort_summary, make_splits,  # noqa: E402
                      scan_dataset, stratification_is_sound)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="exp01_main_sweep.yaml")
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--split-seed", type=int, default=None)
    ap.add_argument("--shuffle", action="store_true",
                    help="probe shuffled partitioning instead of the deterministic default")
    args = ap.parse_args()

    cfg = load_experiment_config(args.config)
    if args.data_root:
        cfg["data"]["root"] = args.data_root
    if args.split_seed is not None:
        cfg["split"]["split_seed"] = args.split_seed
    if args.shuffle:
        cfg["split"]["shuffle"] = True

    sound = stratification_is_sound()
    print(f"scikit-learn {sklearn.__version__}")
    print(f"shuffle      {cfg['split'].get('shuffle', False)}"
          + (f"  (split_seed {cfg['split']['split_seed']})" if cfg["split"].get("shuffle") else
             "  -- deterministic partition, no split seed"))
    print(f"shuffle=True stratification probe: {'SOUND' if sound else 'BROKEN in this build'}")
    if not sound:
        print("  !! StratifiedGroupKFold(shuffle=True) silently degrades to unstratified")
        print("     GroupKFold here. shuffle=False avoids it and is the configured default.")
    print(f"data root    {cfg['data']['root']}\n")

    df = scan_dataset(cfg["data"]["root"], cfg["data"]["original_classes"],
                      cfg["data"]["class_map"])
    cs = cohort_summary(df)
    print(f"COHORT  {cs['images']:,} images | {cs['participants']} participants | "
          f"{cs['sessions']} sessions | {cs['participants_multi_session']} with >1 session")
    print(f"        participants/class {cs['participants_per_class']}\n")

    names = cfg["data"]["new_classes"]
    hdr = f"{'fold':>4} | {'train ND/VM/D':>16} | {'val ND/VM/D':>14} | {'test ND/VM/D':>14} | " \
          f"{'test imgs':>10} | assert"
    print(hdr); print("-" * len(hdr))

    dem_test, dem_val, minima = [], [], []
    for fold in range(cfg["split"]["n_splits"]):
        cfg["split"]["fold"] = fold
        splits = make_splits(df, cfg)
        ok = assert_group_disjoint(df, splits, cfg["split"]["group_by"])
        cells = {}
        for name in ("train", "val", "test"):
            per = df.iloc[splits[name]].groupby("participant_id")["class_3"].first()
            c = Counter(per)
            cells[name] = "/".join(str(c.get(i, 0)) for i in range(len(names)))
            if name == "test":
                dem_test.append(c.get(2, 0))
            if name == "val":
                dem_val.append(c.get(2, 0))
        print(f"{fold:>4} | {cells['train']:>16} | {cells['val']:>14} | {cells['test']:>14} | "
              f"{len(splits['test']):>10,} | {ok}")

    print("-" * len(hdr))
    print(f"Demented participants per TEST fold: {dem_test}  (sum {sum(dem_test)})")
    print(f"Demented participants per VAL  fold: {dem_val}")
    print()
    worst = min(dem_test)
    print(f"The smallest test fold holds {worst} Demented participant(s). Per-fold minority")
    print(f"metrics rest on that many people; only the POOLED estimate across folds covers")
    print(f"all {sum(dem_test)}. Report per-fold counts in the paper and lead with the pooled result.")
    print()
    print("Do NOT try other split_seeds to find a prettier balance. Choosing a partition")
    print("by how its composition looks is the same class of error this manuscript exists")
    print("to criticise. With shuffle=False there is only one partition, which removes")
    print("the temptation entirely -- that is part of why it is the default.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
