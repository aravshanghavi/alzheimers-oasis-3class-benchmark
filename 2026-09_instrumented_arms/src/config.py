"""Configuration loading, merging and validation.

A Config is a plain nested dict wrapped in a small accessor. It is deliberately
not a dataclass tree: the resolved dict is what gets hashed into the code
fingerprint and written verbatim into manifest.json, so a single canonical
JSON-serialisable representation keeps provenance simple.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"

VALID_LOSSES = {"WCE", "FocalLoss", "LDAM", "FA_FL", "ClassBalanced"}
VALID_STRATEGIES = {"grouped_kfold", "grouped_holdout"}
VALID_GROUP_BY = {"participant", "session"}


class ConfigError(ValueError):
    """Raised when a configuration is internally inconsistent or unusable."""


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _normalise_class_map(cfg: Dict[str, Any]) -> None:
    """YAML may parse integer keys as strings depending on quoting. Force int keys."""
    raw = cfg["data"]["class_map"]
    cfg["data"]["class_map"] = {int(k): int(v) for k, v in raw.items()}


def load_config(experiment_config: str | Path | None = None,
                overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Load base.yaml, merge an experiment config, then merge explicit overrides."""
    with open(CONFIG_DIR / "base.yaml", "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if experiment_config is not None:
        path = Path(experiment_config)
        if not path.is_absolute():
            path = CONFIG_DIR / path
        if not path.exists():
            raise ConfigError(f"Experiment config not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})
        cfg["_config_file"] = str(path.relative_to(REPO_ROOT)).replace("\\", "/")

    if overrides:
        cfg = _deep_merge(cfg, overrides)

    _normalise_class_map(cfg)
    validate_config(cfg)
    return cfg


def load_experiment_config(config_name: str) -> Dict[str, Any]:
    """Load base + experiment WITHOUT validating (so --dry-run works with no data)."""
    with open(CONFIG_DIR / "base.yaml", "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    path = CONFIG_DIR / config_name
    if not path.exists():
        raise ConfigError(f"Experiment config not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = _deep_merge(cfg, yaml.safe_load(fh) or {})
    cfg["_config_file"] = f"config/{config_name}"
    _normalise_class_map(cfg)
    return cfg


def validate_config(cfg: Dict[str, Any], require_data_root: bool = True) -> None:
    """Fail at second zero rather than epoch forty."""
    loss_type = cfg["loss"]["type"]
    if loss_type not in VALID_LOSSES:
        raise ConfigError(f"Unknown loss type {loss_type!r}; expected one of {sorted(VALID_LOSSES)}")
    if loss_type not in cfg["loss"]["params"]:
        raise ConfigError(f"loss.params has no entry for {loss_type!r}")

    split = cfg["split"]
    if split["strategy"] not in VALID_STRATEGIES:
        raise ConfigError(f"Unknown split.strategy {split['strategy']!r}")
    if split["group_by"] not in VALID_GROUP_BY:
        raise ConfigError(f"Unknown split.group_by {split['group_by']!r}")
    if split["strategy"] == "grouped_kfold":
        if split["n_splits"] < 2:
            raise ConfigError("split.n_splits must be >= 2")
        if not 0 <= split["fold"] < split["n_splits"]:
            raise ConfigError(f"split.fold {split['fold']} out of range for n_splits={split['n_splits']}")
        if split["val_n_splits"] < 2:
            raise ConfigError("split.val_n_splits must be >= 2")

    ctx = cfg["data"]["context_slices"]
    if ctx not in (1, 3):
        raise ConfigError(f"data.context_slices must be 1 or 3, got {ctx}")

    train = cfg["train"]
    if train["warmup_epochs"] >= train["total_epochs"]:
        raise ConfigError("train.warmup_epochs must be < train.total_epochs")
    if train["grad_accumulation_steps"] < 1:
        raise ConfigError("train.grad_accumulation_steps must be >= 1")
    if train["early_stop_metric"] not in ("val_loss", "val_acc"):
        raise ConfigError("train.early_stop_metric must be val_loss or val_acc")
    if train["checkpoint_metric"] not in ("val_loss", "val_acc"):
        raise ConfigError("train.checkpoint_metric must be val_loss or val_acc")

    if require_data_root and not Path(cfg["data"]["root"]).is_dir():
        raise ConfigError(
            f"data.root does not exist: {cfg['data']['root']}\n"
            "Set it in config/base.yaml or override with --data-root."
        )


def resolved_json(cfg: Dict[str, Any]) -> str:
    """Canonical JSON for hashing and logging. Sorted keys so it is stable."""
    return json.dumps(cfg, sort_keys=True, indent=2, default=str)
