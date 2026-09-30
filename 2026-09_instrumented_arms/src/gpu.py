"""GPU memory hygiene between runs in a long unattended queue.

What actually reclaims memory, and what does not
------------------------------------------------
``torch.cuda.empty_cache()`` returns *cached* blocks to the driver. It does not
free memory still referenced by live Python objects, and it does not defragment
the allocator. So calling it alone between runs is close to useless: the model,
optimiser, criterion, DataLoaders and their worker processes must be dropped
first, and a gc pass has to run so their CUDA storages are actually released.

The order that works is: drop references -> gc.collect() -> synchronize ->
empty_cache() -> reset_peak_memory_stats(). ``release_between_runs`` does that.

Two things worth knowing beyond this module:

* Fragmentation across many sequential runs is best handled by the allocator
  itself. Set ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` before the
  process starts (``preflight_env_hint`` prints the exact command).
* The only way to guarantee a completely clean allocator is to end the process.
  That is what ``--isolate`` does: each run executes in its own subprocess, so
  reclamation is enforced by the operating system rather than hoped for.
"""
from __future__ import annotations

import gc
import os
from typing import Dict, Iterable, Optional

import torch

ALLOC_CONF_RECOMMENDED = "expandable_segments:True"


def cuda_available() -> bool:
    return torch.cuda.is_available()


def memory_snapshot() -> Optional[Dict[str, float]]:
    """Current allocator state in GB, plus true device-level free/total."""
    if not cuda_available():
        return None
    free_b, total_b = torch.cuda.mem_get_info()
    return {
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        "device_free_gb": free_b / 1e9,
        "device_total_gb": total_b / 1e9,
        "device_used_pct": 100.0 * (1.0 - free_b / total_b),
    }


def format_snapshot(snap: Optional[Dict[str, float]]) -> str:
    if snap is None:
        return "cpu (no CUDA)"
    return (f"allocated {snap['allocated_gb']:.2f} GB | reserved {snap['reserved_gb']:.2f} GB | "
            f"peak {snap['peak_allocated_gb']:.2f} GB | device {snap['device_used_pct']:.1f}% used "
            f"({snap['device_free_gb']:.2f} of {snap['device_total_gb']:.2f} GB free)")


def shutdown_dataloaders(loaders: Optional[Iterable]) -> int:
    """Terminate persistent DataLoader worker processes.

    With persistent_workers=True the workers outlive one epoch by design; if they
    also outlive the run they hold pinned host memory and file handles for the
    rest of the queue. Dropping the iterator triggers DataLoader's own shutdown.
    """
    if not loaders:
        return 0
    closed = 0
    for loader in (loaders.values() if isinstance(loaders, dict) else loaders):
        it = getattr(loader, "_iterator", None)
        if it is not None:
            try:
                it._shutdown_workers()
            except Exception:
                pass
            loader._iterator = None
            closed += 1
    return closed


def release_between_runs(log=None) -> Optional[Dict[str, float]]:
    """Collect and return allocator caches between two runs in the same process.

    IMPORTANT CONTRACT: the caller must already have dropped every reference to
    the model, criterion, optimiser and dataloaders before calling this. Passing
    objects in as arguments would NOT work -- that only rebinds this function's
    parameters, while the caller's frame still holds the originals, so the
    gc.collect() below would free nothing. Null the locals first, then call.

    Call shutdown_dataloaders() beforehand, while the loaders still exist.
    """
    before = memory_snapshot()

    n_objects = gc.collect()
    if cuda_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

    after = memory_snapshot()
    if log is not None:
        if before is None:
            log.info("  teardown: CPU run, gc collected %d object(s), no VRAM to release", n_objects)
        else:
            log.info("  VRAM before teardown: %s", format_snapshot(before))
            log.info("  VRAM after  teardown: %s", format_snapshot(after))
            log.info("  reclaimed %.2f GB reserved / %.2f GB allocated; gc collected %d object(s)",
                     before["reserved_gb"] - after["reserved_gb"],
                     before["allocated_gb"] - after["allocated_gb"], n_objects)
            if after["allocated_gb"] > 0.05:
                log.warning("  %.2f GB still allocated after teardown -- something is holding a "
                            "reference. Use --isolate if this grows across runs.",
                            after["allocated_gb"])
    return after


def assert_headroom(min_free_gb: float, log=None) -> bool:
    """Warn (do not abort) when free VRAM is below what a run is likely to need."""
    snap = memory_snapshot()
    if snap is None:
        return True
    if snap["device_free_gb"] < min_free_gb:
        msg = (f"only {snap['device_free_gb']:.2f} GB free on the device "
               f"({snap['device_used_pct']:.1f}% used); this run wants ~{min_free_gb:.1f} GB. "
               f"Another process may be holding VRAM.")
        if log is not None:
            log.warning("  MEMORY WARNING: %s", msg)
        return False
    return True


def preflight_env_hint() -> str:
    current = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if ALLOC_CONF_RECOMMENDED in current:
        return f"PYTORCH_CUDA_ALLOC_CONF={current}  (expandable segments active)"
    return (f"PYTORCH_CUDA_ALLOC_CONF={current or '<unset>'}  -- consider "
            f"`set PYTORCH_CUDA_ALLOC_CONF={ALLOC_CONF_RECOMMENDED}` before a long queue "
            f"to reduce allocator fragmentation across runs")
