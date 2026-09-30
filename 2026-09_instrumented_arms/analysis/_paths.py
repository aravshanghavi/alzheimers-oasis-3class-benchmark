"""Single source of truth for where the analysis layer reads and writes.

Five different path conventions had accumulated across a01-a18: REPO_ROOT-relative
in discover/a12/a13/a14, analysis-dir-relative-with-cwd-fallback in a15/a16, and
bare cwd-relative strings in a11, a17, a18 and _cache_build. They agree only when
you happen to launch from the repository root, and they give no way at all to
read run artefacts that live somewhere else.

That last point is the one that matters now. This stack holds the analysis code
but not the 55 run folders, which stay in the 2026-08 tree, and the ADNI work
will need to point at a third location again. So the experiments directory is
resolved once, here, from the first of these that answers:

    1. the SREP_EXPERIMENTS_DIR environment variable   (one-off override)
    2. experiments_dir.txt at the repository root      (the persistent default)
    3. experiments/ inside this repository             (the ordinary case)

The pointer file exists because `set VAR=...` in cmd.exe lives only as long as
that console window. Relying on it meant every new terminal silently fell back
to the empty local experiments/ folder, and the run died at preflight with a
message that looked like missing data rather than a missing variable.

Outputs always land under THIS repository's analysis/outputs, so reading someone
else's runs can never write into their tree.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

OUT = REPO_ROOT / "analysis" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
CACHE_NPZ = OUT / "_cache.npz"

ENV_EXPERIMENTS = "SREP_EXPERIMENTS_DIR"
ENV_METADATA = "SREP_METADATA_XLSX"


POINTER_FILE = REPO_ROOT / "experiments_dir.txt"


def _read_pointer() -> Optional[Path]:
    """First non-comment line of experiments_dir.txt, if it exists.

    An environment variable set with `set` in cmd.exe lasts only for that
    console window, so every new terminal silently reverts to the local
    (empty) experiments/ folder. This file makes the location a property of
    the checkout instead of the shell session.
    """
    if not POINTER_FILE.is_file():
        return None
    for line in POINTER_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            return Path(line).expanduser()
    return None


def _resolve_experiments() -> Tuple[Path, str]:
    env = os.environ.get(ENV_EXPERIMENTS)
    if env:
        p = Path(env).expanduser()
        if not p.is_dir():
            raise SystemExit(
                f"{ENV_EXPERIMENTS} is set to '{env}' but that is not a directory.\n"
                f"Point it at the 'experiments' folder holding the run outputs, or unset it "
                f"to fall back to {POINTER_FILE.name} or {REPO_ROOT / 'experiments'}.")
        return p.resolve(), f"environment variable {ENV_EXPERIMENTS}"

    pointed = _read_pointer()
    if pointed is not None:
        if not pointed.is_dir():
            raise SystemExit(
                f"{POINTER_FILE} points at '{pointed}', which is not a directory.\n"
                f"Edit that file to the 'experiments' folder holding the run outputs, or "
                f"delete it to use {REPO_ROOT / 'experiments'}.")
        return pointed.resolve(), POINTER_FILE.name

    return REPO_ROOT / "experiments", "this repository"


EXPERIMENTS_DIR, EXPERIMENTS_SOURCE = _resolve_experiments()
LOCAL_EXPERIMENTS = REPO_ROOT / "experiments"
USING_EXTERNAL_EXPERIMENTS = EXPERIMENTS_DIR != LOCAL_EXPERIMENTS


def count_complete(root: Optional[Path] = None) -> int:
    """How many COMPLETE runs sit under a tree. Used by the preflight guard.

    The dangerous case is not a missing path, which fails loudly. It is this
    repository having its own finished runs while the resolved experiments
    directory points somewhere else, because then every table is built from the
    other tree and nothing says so except one line of preflight output.
    """
    root = Path(root) if root is not None else EXPERIMENTS_DIR
    if not root.is_dir():
        return 0
    return sum(1 for mf in root.glob("*/outputs/*/manifest.json")
               if (mf.parent / "_COMPLETE").exists())


def metadata_xlsx() -> Path:
    """The OASIS cross-sectional metadata table (age, sex, eTIV, nWBV, CDR).

    Needed by a13 and by a15-a18. Searched in order: the environment override,
    this repository, then sibling project folders, so the file does not have to
    be duplicated into every stack.
    """
    env = os.environ.get(ENV_METADATA)
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p.resolve()
        raise SystemExit(f"{ENV_METADATA} is set to '{env}' but that file does not exist.")

    candidates = [REPO_ROOT / "oasis_cross-sectional.xlsx"]
    candidates += sorted(REPO_ROOT.parent.glob("*/oasis_cross-sectional.xlsx"))
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise SystemExit(
        "Could not find oasis_cross-sectional.xlsx.\n"
        "It carries age, sex, eTIV, nWBV and CDR, without which the metadata baselines, "
        "the increment analysis and the age-cohort work cannot run.\n"
        f"Looked in: {REPO_ROOT}, and one level up under {REPO_ROOT.parent}.\n"
        f"Set {ENV_METADATA} to its full path, or copy it into {REPO_ROOT}.")


def describe() -> str:
    lines = [f"repo root       {REPO_ROOT}",
             f"experiments     {EXPERIMENTS_DIR}"
             + (f"   (external, via {EXPERIMENTS_SOURCE})" if USING_EXTERNAL_EXPERIMENTS else ""),
             f"outputs         {OUT}"]
    try:
        lines.append(f"metadata xlsx   {metadata_xlsx()}")
    except SystemExit:
        lines.append("metadata xlsx   NOT FOUND")
    return "\n".join(lines)
