"""Order-independent seeded random generators.

THE PROBLEM THIS FIXES
----------------------
a15, a16, a17 and a18 each draw from ONE module-level generator, consumed
sequentially as the script runs:

    RNG = np.random.default_rng(99)        # a18
    ...
    bi = RNG.integers(...)                 # section A, inside a loop over arms
    ...
    bi = RNG.integers(...)                 # section D

Every draw depends on how many draws came before it. Insert one comparison in
section A, add an arm to the ARMS list, or run section D on its own, and every
confidence interval downstream of the change moves. The script is reproducible
only if it is run end to end, unmodified, in exactly the same order -- which is
not a property you want underneath a paper that is being revised.

THE FIX
-------
Each call site asks for its own generator by name:

    rng = seeded_rng("a18:increment", arm=label, cohort=coh)

The seed is a stable hash of the base seed and the tag, so the same tag always
gives the same stream regardless of what else ran, and different tags give
independent streams. Python's built-in hash() is NOT usable here: it is salted
per process unless PYTHONHASHSEED is fixed, so it would make runs differ from
each other rather than agree.

NOTE ON NUMBERS CHANGING
------------------------
Switching to per-site seeding changes which resamples are drawn, so bootstrap
intervals from a15-a18 will differ from the August figures in roughly the fourth
decimal. That is expected. The new numbers are the reproducible ones; the old
ones were a function of execution order.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np


def seed_for(base_seed: int, tag: str, **parts: Any) -> int:
    """A stable 64-bit seed from a base seed, a tag and any keyword parts."""
    key = f"{base_seed}|{tag}|" + "|".join(f"{k}={parts[k]}" for k in sorted(parts))
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def seeded_rng(tag: str, base_seed: int, **parts: Any) -> np.random.Generator:
    """A generator whose stream depends only on (base_seed, tag, parts).

    Independent of how many draws any other call site has taken, so sections can
    be added, removed or run alone without moving each other's numbers.
    """
    return np.random.default_rng(seed_for(base_seed, tag, **parts))
