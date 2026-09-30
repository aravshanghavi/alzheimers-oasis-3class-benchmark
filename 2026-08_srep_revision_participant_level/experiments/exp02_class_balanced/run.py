#!/usr/bin/env python3
"""Driver for exp02_class_balanced. All shared logic lives in src/runner.py.

Usage
-----
    python experiments/exp02_class_balanced/run.py --dry-run
    python experiments/exp02_class_balanced/run.py
    python experiments/exp02_class_balanced/run.py --resume
    python experiments/exp02_class_balanced/run.py --only FA_FL:fold3
"""
import os
# Must be set before torch initialises CUDA. Reduces allocator fragmentation
# across a long sequence of runs; the log warns when it is unset.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.runner import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main("exp02_class_balanced.yaml", __file__))
