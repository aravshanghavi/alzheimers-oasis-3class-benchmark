#!/usr/bin/env python3
"""Driver for exp15_age60_unweighted. All shared logic lives in src/runner.py.

Usage
-----
    python experiments/exp15_age60_unweighted/run.py --dry-run
    python experiments/exp15_age60_unweighted/run.py --isolate
    python experiments/exp15_age60_unweighted/run.py --resume --isolate
    python experiments/exp15_age60_unweighted/run.py --only WCE:fold3
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
    raise SystemExit(main("exp15_age60_unweighted.yaml", __file__))
