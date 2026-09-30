#!/usr/bin/env python3
"""Preflight: decide whether the analysis layer can run, before it is run.

    python analysis/a00_preflight.py

Answers one question: if I start the pipeline now, what will work and what will
fail? Every check is cheap. Nothing here trains, loads a checkpoint, or
bootstraps anything.

This exists because the failure mode it prevents already happened: the analysis
code and the run artefacts now live in different folders, and without a check
the first sign of that is a stack trace several minutes into a run.
"""
from __future__ import annotations

import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analysis import _paths                                    # noqa: E402
from analysis._registry import FAMILIES                        # noqa: E402

OK, WARN, BAD = "  [ok]  ", "  [warn]", "  [FAIL]"


def main() -> int:
    problems, warnings = [], []
    print("=" * 78)
    print("PREFLIGHT")
    print("=" * 78)

    # -- 1. interpreter and packages ----------------------------------------
    print("\n1. environment")
    print(f"       interpreter: {sys.executable}")
    missing = []
    for mod in ("numpy", "pandas", "sklearn", "matplotlib", "yaml", "openpyxl"):
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        problems.append(f"missing package(s): {', '.join(missing)}")
        print(f"{BAD} missing: {', '.join(missing)}")
        print(f"         conda install {' '.join(missing)} \"scikit-learn=1.5.2\"")
        print("         scikit-learn is pinned in that command on purpose: StratifiedGroupKFold")
        print("         changed behaviour between releases, and letting the solver move it would")
        print("         silently change the fold assignments on any future training run.")
        print("         Prefer conda over pip inside a conda env -- conda's solver has no record")
        print("         of pip-installed files, so a later conda operation can clobber them.")
        print("         openpyxl is needed to read the OASIS metadata workbook.")
    else:
        print(f"{OK} numpy, pandas, sklearn, matplotlib, yaml, openpyxl present")
    # torch is only needed by a04 (Grad-CAM)
    try:
        importlib.import_module("torch")
        print(f"{OK} torch present (a04 Grad-CAM can run)")
    except ImportError:
        warnings.append("torch absent: a04 Grad-CAM will be skipped")
        print(f"{WARN} torch absent -- a04 Grad-CAM will be skipped, everything else is fine")

    # -- 2. paths ------------------------------------------------------------
    print("\n2. paths")
    for line in _paths.describe().splitlines():
        print("       " + line)
    if not _paths.EXPERIMENTS_DIR.is_dir():
        problems.append(f"experiments directory not found: {_paths.EXPERIMENTS_DIR}")
        print(f"{BAD} experiments directory does not exist")
        print(f"         set {_paths.ENV_EXPERIMENTS} to the folder holding the run outputs, e.g.")
        print(f"         set {_paths.ENV_EXPERIMENTS}=C:\\Users\\aravs\\PycharmProjects\\Alzheimers\\"
              f"2026-08_srep_revision_participant_level\\experiments")
    else:
        print(f"{OK} experiments directory resolves")
    try:
        _paths.metadata_xlsx()
        print(f"{OK} OASIS metadata workbook found")
    except SystemExit as exc:
        problems.append("oasis_cross-sectional.xlsx not found")
        print(f"{BAD} {str(exc).splitlines()[0]}")
        print("         needed by a13 and a15-a18 (age, sex, eTIV, nWBV, CDR)")

    # -- 3. run inventory ----------------------------------------------------
    print("\n3. runs on disk")
    manifests = sorted(_paths.EXPERIMENTS_DIR.glob("*/outputs/*/manifest.json")) \
        if _paths.EXPERIMENTS_DIR.is_dir() else []
    per_exp, fingerprints, dupes = defaultdict(int), set(), defaultdict(list)
    n_complete = 0
    for mf in manifests:
        if not (mf.parent / "_COMPLETE").exists():
            continue
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            continue
        n_complete += 1
        per_exp[m.get("experiment")] += 1
        fingerprints.add(m.get("code_fingerprint"))
        key = (m.get("experiment"), m.get("loss", {}).get("type"),
               m.get("split", {}).get("fold"), m.get("init_seed"), m.get("code_fingerprint"))
        dupes[key].append(mf.parent.name)

    if n_complete == 0:
        problems.append("no COMPLETE runs found")
        print(f"{BAD} no COMPLETE runs under {_paths.EXPERIMENTS_DIR}")
        print("         every analysis step will fail. Point at a tree that has run outputs,")
        print("         or run the experiments first.")
    else:
        print(f"{OK} {n_complete} COMPLETE run(s)")
        for exp in sorted(per_exp):
            print(f"         {exp:<28} {per_exp[exp]:>3}")

    if len(fingerprints) > 1:
        problems.append(f"{len(fingerprints)} distinct code fingerprints among COMPLETE runs")
        print(f"{BAD} runs span {len(fingerprints)} code fingerprints: "
              f"{sorted(f[:8] for f in fingerprints if f)}")
        print("         aggregating across differing code is how silent inconsistencies enter a paper")
    elif fingerprints:
        print(f"{OK} one code fingerprint: {list(fingerprints)[0][:8]}")

    real_dupes = {k: v for k, v in dupes.items() if len(v) > 1}
    if real_dupes:
        problems.append(f"{len(real_dupes)} duplicate-spec group(s)")
        print(f"{BAD} {len(real_dupes)} duplicate-spec group(s): the same "
              f"(experiment, loss, fold, seed, code) has more than one COMPLETE run")
        for k, v in list(real_dupes.items())[:5]:
            print(f"         {k[0]} {k[1]} fold{k[2]} init{k[3]}: {len(v)} runs")
        print("         pooling these double-counts that fold's participants and the")
        print("         fold-disjointness check does NOT catch it. Resolve before any table.")
    elif n_complete:
        print(f"{OK} no duplicate specs")

    # -- 4. derived artefacts ------------------------------------------------
    print("\n4. derived artefacts")
    if _paths.CACHE_NPZ.exists():
        print(f"{OK} _cache.npz present ({_paths.CACHE_NPZ.stat().st_size/1e6:.1f} MB)")
    else:
        warnings.append("_cache.npz absent; a17 and a18 need it")
        print(f"{WARN} _cache.npz absent -- a17 and a18 cannot run until _cache_build.py has run")
        print("         run_analysis.py builds it automatically as its second step")

    # -- 5. the pre-registration --------------------------------------------
    print("\n5. pre-registered comparison families")
    primary = [f for f in FAMILIES.values() if f.primary]
    print(f"{OK} {len(FAMILIES)} declared, {len(primary)} primary")
    for f in sorted(primary, key=lambda x: x.key):
        print(f"         {f.key:<28} n={f.n_declared:<3} alpha={f.alpha_corrected():.5f}")

    # -- verdict -------------------------------------------------------------
    print("\n" + "=" * 78)
    if problems:
        print(f"NOT READY -- {len(problems)} blocking problem(s):")
        for p in problems:
            print(f"  - {p}")
        if warnings:
            print(f"plus {len(warnings)} warning(s):")
            for w in warnings:
                print(f"  - {w}")
        print("=" * 78)
        return 1
    print("READY" + (f" -- with {len(warnings)} warning(s):" if warnings else ""))
    for w in warnings:
        print(f"  - {w}")
    print("\nnext:  python analysis/run_analysis.py")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
