"""Tell a production permuted run apart from a smoke run.

`run_permnull.py --smoke` writes a real, COMPLETE run into the same outputs
directory as the production runs: two epochs, 64-pixel images, no pretrained
weights. Its manifest carries permutation.enabled with a real seed and fold, so
anything that selects runs by seed and fold alone will pick it up.

Two things would then go wrong, both silently:

  - `--resume` would see seed 1 fold 0 as already COMPLETE and skip the real run.
  - The analysis would count six runs for seed 1, fail its "exactly five folds"
    check, and drop that replicate entirely.

So production-ness is decided here, from the run's own resolved config, and both
the queue and the analysis use this one function.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

PROD_IMAGE_SIZE = [224, 224]
PROD_MIN_EPOCHS = 5


def production_reason(manifest: Dict) -> Optional[str]:
    """Return None if this is a production run, else why it is not."""
    rc = manifest.get("resolved_config") or {}
    data = rc.get("data") or {}
    train = rc.get("train") or {}
    model = rc.get("model") or {}

    size = data.get("image_size")
    if size is not None and list(size) != PROD_IMAGE_SIZE:
        return f"image_size {list(size)} != {PROD_IMAGE_SIZE} (smoke run)"

    epochs = train.get("total_epochs")
    if epochs is not None and int(epochs) < PROD_MIN_EPOCHS:
        return f"total_epochs {epochs} < {PROD_MIN_EPOCHS} (smoke run)"

    pretrained = model.get("pretrained")
    if pretrained is False:
        return "pretrained=False (smoke run)"

    return None


def permuted_spec(manifest: Dict) -> Optional[Tuple[int, int]]:
    """Return (perm_seed, fold) for a PRODUCTION permuted run, else None."""
    perm = manifest.get("permutation") or {}
    if not perm.get("enabled"):
        return None
    if production_reason(manifest) is not None:
        return None
    seed = perm.get("seed")
    fold = (manifest.get("split") or {}).get("fold")
    if seed is None or fold is None:
        return None
    return int(seed), int(fold)
