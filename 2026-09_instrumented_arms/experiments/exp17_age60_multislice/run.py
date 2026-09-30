#!/usr/bin/env python3
"""Driver for exp17_age60_multislice. All shared logic lives in src/runner.py.

Usage
-----
    python experiments/exp17_age60_multislice/run.py --dry-run
    python experiments/exp17_age60_multislice/run.py --isolate
    python experiments/exp17_age60_multislice/run.py --resume --isolate
    python experiments/exp17_age60_multislice/run.py --only WCE:fold3
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
    raise SystemExit(main("exp17_age60_multislice.yaml", __file__))
