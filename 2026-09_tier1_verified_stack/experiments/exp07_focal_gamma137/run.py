#!/usr/bin/env python3
"""Driver for exp07_focal_gamma137. All shared logic lives in src/runner.py.

Usage
-----
    python experiments/exp07_focal_gamma137/run.py --dry-run
    python experiments/exp07_focal_gamma137/run.py
    python experiments/exp07_focal_gamma137/run.py --resume
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main("exp07_focal_gamma137.yaml", __file__))
