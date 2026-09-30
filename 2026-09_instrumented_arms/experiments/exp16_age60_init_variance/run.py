#!/usr/bin/env python3
"""Driver for exp16_age60_init_variance. All shared logic lives in src/runner.py.

Usage
-----
    python experiments/exp16_age60_init_variance/run.py --dry-run
    python experiments/exp16_age60_init_variance/run.py --isolate
    python experiments/exp16_age60_init_variance/run.py --resume --isolate
    python experiments/exp16_age60_init_variance/run.py --only WCE:fold3
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
    raise SystemExit(main("exp16_age60_init_variance.yaml", __file__))
