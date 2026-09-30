"""Reproducibility controls.

The submitted codebase seeded nothing: RANDOM_STATE fed only train_test_split,
so parameter initialisation, batch ordering and augmentation were uncontrolled
and no run could be reproduced. Everything below exists to fix that.
"""
from __future__ import annotations

import os
import random
from typing import Callable, Tuple

import numpy as np
import torch


def set_all_seeds(seed: int, deterministic: bool = True, strict: bool = False) -> None:
    """Seed every RNG that can influence a run.

    deterministic : set cuDNN to deterministic mode and disable autotuning.
    strict        : additionally require deterministic kernels for every op.
                    Raises at runtime if an op has no deterministic implementation,
                    which is informative but can abort a run.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)

    if strict:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=False)


def seed_worker(worker_id: int) -> None:
    """Reseed numpy and random inside a DataLoader worker.

    MUST be a module-level function, not a closure.

    On Linux, DataLoader workers are forked and inherit the parent's objects, so
    a nested function works. On Windows (and macOS spawn) the worker_init_fn is
    PICKLED and sent to a fresh interpreter -- and closures are not picklable.
    The symptom is an opaque `EOFError: Ran out of input` from
    multiprocessing/spawn.py, with no mention of the actual cause.

    It needs no closure state: torch seeds each worker deterministically from the
    DataLoader generator, so torch.initial_seed() already differs per worker and
    per epoch in a reproducible way.
    """
    worker_seed = (torch.initial_seed() + worker_id) % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_dataloader_rng(seed: int) -> Tuple[torch.Generator, Callable[[int], None]]:
    """Generator and worker_init_fn so shuffling and augmentation are reproducible.

    Without both of these, DataLoader worker processes reseed themselves from
    entropy and the augmentation stream differs between otherwise identical runs.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator, seed_worker


def resolve_device(spec: str = "auto") -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)
