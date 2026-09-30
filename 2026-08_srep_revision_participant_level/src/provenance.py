"""Provenance: code fingerprinting, environment capture, run manifests.

Every run records enough to answer, months later, exactly which code and which
configuration produced a number in the paper.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import platform
import socket
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def src_file_hashes() -> Dict[str, str]:
    out = {}
    for path in sorted(SRC_DIR.glob("*.py")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        out[f"src/{path.name}"] = digest
    return out


def code_fingerprint() -> str:
    """SHA-256 over every src/*.py. Code only -- deliberately NOT the config.

    Two runs of different experiments legitimately differ in configuration; they
    must not differ in code. The analysis layer warns when aggregated runs
    disagree here, which is the case that silently corrupts a results table.
    """
    h = hashlib.sha256()
    for name, digest in src_file_hashes().items():
        h.update(name.encode("utf-8"))
        h.update(digest.encode("utf-8"))
    return h.hexdigest()


def config_fingerprint(resolved_config_json: str) -> str:
    """SHA-256 over the resolved configuration. Identifies a run's exact settings
    and is what --resume matches on, so an edited config forces a re-run."""
    return hashlib.sha256(resolved_config_json.encode("utf-8")).hexdigest()


def git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def capture_environment() -> Dict[str, object]:
    env: Dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
        "cpu_count": None,
        "git_commit": git_commit(),
    }
    try:
        import os
        env["cpu_count"] = os.cpu_count()
    except Exception:
        pass
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
        env["cudnn"] = torch.backends.cudnn.version()
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        else:
            env["gpu_name"] = "cpu"
    except Exception as exc:
        env["torch_error"] = repr(exc)
    try:
        import torchvision
        env["torchvision"] = torchvision.__version__
    except Exception:
        pass
    for mod in ("numpy", "pandas", "sklearn", "PIL"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            pass
    return env


def write_environment_txt(run_dir: Path, env: Dict[str, object]) -> None:
    lines = [f"{k:<18} {v}" for k, v in sorted(env.items())]
    (run_dir / "environment.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def snapshot_code(run_dir: Path, resolved_config_json: str) -> None:
    """Zip src/ and the resolved config into the run folder (~30 KB).

    Cheap insurance: any result can be traced to its exact source without
    depending on the current state of the repository.
    """
    with zipfile.ZipFile(run_dir / "code_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(SRC_DIR.glob("*.py")):
            zf.write(path, f"src/{path.name}")
        zf.writestr("resolved_config.json", resolved_config_json)


def build_run_id(*, timestamp: str, experiment: str, loss: str, fold: int,
                 split_seed, init_seed: int, fingerprint: str) -> str:
    """split_seed is None when the partition is deterministic (shuffle=False),
    in which case the name says so rather than quoting a seed that had no effect."""
    short_exp = experiment.split("_")[0]
    split_tag = "fixed" if split_seed is None else str(split_seed)
    return (f"{timestamp}__{short_exp}__{loss}__fold{fold}"
            f"__split{split_tag}-init{init_seed}__{fingerprint[:8]}")


def set_status(run_dir: Path, status: str) -> None:
    for existing in run_dir.glob("_*"):
        if existing.name in ("_RUNNING", "_COMPLETE", "_FAILED"):
            existing.unlink()
    (run_dir / f"_{status}").touch()


def write_manifest(run_dir: Path, manifest: Dict) -> None:
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)


def list_artifacts(run_dir: Path) -> List[str]:
    return sorted(p.name for p in run_dir.iterdir() if p.is_file())
