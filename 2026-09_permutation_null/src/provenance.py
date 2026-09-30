"""Provenance: code fingerprinting, environment capture, run manifests.

Every run records enough to answer, months later, exactly which code and which
configuration produced a number in the paper.
"""
from __future__ import annotations

import getpass
import hashlib
import json
import platform
import shutil
import socket
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


# ---------------------------------------------------------------------------
# Batch / rerun bookkeeping
#
# Added 2026-09: run_id already makes every attempt unique (timestamp + code
# fingerprint), so re-running an unchanged spec without --resume does not
# overwrite anything -- it adds a sibling folder. That is correct for
# provenance (nothing is ever silently lost) but it creates a real hazard one
# level up: analysis/discover.py's load_runs() globs every COMPLETE manifest
# under experiments/*/outputs/*/ with no per-spec dedup. Two COMPLETE runs of
# the same (experiment, loss, fold, init_seed) under the same code fingerprint
# -- the exact situation an unflagged rerun produces -- both get pooled. The
# existing "no participant in >1 fold" check does NOT catch this: the two
# runs report the SAME fold number, so every participant in that fold is
# silently double-counted rather than flagged as appearing in two folds.
# This has to be closed in two places: detection (used by both runner.py
# after a queue finishes and discover.py before any table is built) and,
# optionally, cleanup (supersede_duplicate_specs, opt-in, never destructive).
# ---------------------------------------------------------------------------

DuplicateKey = Tuple[Optional[str], Optional[str], Optional[int], Optional[int], Optional[str]]


def find_duplicate_specs(search_root: Path,
                         pattern: str = "*/outputs/*/manifest.json"
                         ) -> Dict[DuplicateKey, List[Path]]:
    """COMPLETE run folders under search_root/pattern that share
    (experiment, loss, fold, init_seed, code_fingerprint).

    More than one such folder means the same spec was executed twice under
    unchanged code without the earlier run being cleared out first. Runs with
    DIFFERENT code fingerprints are never flagged here -- that is a legitimate
    before/after-a-fix comparison and is already caught separately by the
    code-fingerprint check in analysis/discover.py. Only FAILED or still-
    _RUNNING folders (no _COMPLETE marker) are ignored; those cannot pollute
    a pooled table because discover.load_runs() already filters on status.
    """
    groups: Dict[DuplicateKey, List[Path]] = {}
    for manifest_path in sorted(Path(search_root).glob(pattern)):
        run_dir = manifest_path.parent
        if not (run_dir / "_COMPLETE").exists():
            continue
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        key: DuplicateKey = (m.get("experiment"), m.get("loss", {}).get("type"),
                             m.get("split", {}).get("fold"), m.get("init_seed"),
                             m.get("code_fingerprint"))
        groups.setdefault(key, []).append(run_dir)
    return {k: v for k, v in groups.items() if len(v) > 1}


def supersede_duplicate_specs(outputs_dir: Path, keep_run_id: str,
                              spec_key: DuplicateKey) -> List[Path]:
    """Move every OTHER COMPLETE run sharing spec_key into outputs_dir/_superseded/.

    Moved, never deleted. The point of run_id embedding a code fingerprint is
    that a result can always be traced back to exactly the code that produced
    it; silently deleting a prior COMPLETE run destroys that guarantee the
    moment a fresh run turns out to have a problem of its own (OOM-degraded
    eval, a bad seed, a crash partway through evaluation that still wrote a
    _COMPLETE marker some other way). A move is nearly free and fully
    reversible -- an actual permanent delete of _superseded/ is a separate,
    manual, explicit action, not something an unattended queue does to itself.
    """
    outputs_dir = Path(outputs_dir)
    dupes = find_duplicate_specs(outputs_dir, pattern="*/manifest.json")
    group = dupes.get(spec_key, [])
    if not group:
        return []
    superseded_dir = outputs_dir / "_superseded"
    moved: List[Path] = []
    for run_dir in group:
        if run_dir.name == keep_run_id:
            continue
        superseded_dir.mkdir(exist_ok=True)
        dest = superseded_dir / run_dir.name
        if dest.exists():
            continue
        shutil.move(str(run_dir), str(dest))
        moved.append(dest)
    return moved


def write_batch_manifest(outputs_dir: Path, results: List[Dict],
                         queue_log: Optional[Path] = None) -> Dict:
    """Write outputs_dir/CURRENT_BATCH.json: the canonical "what to trust right
    now" pointer for one experiment, plus a zero-extra-compute delta against
    the batch it replaces.

    This does not replace manifest.json -- every run still keeps its own
    permanent record regardless of how many times the queue has been re-run.
    CURRENT_BATCH.json only answers two questions a bare directory listing
    cannot: which run_ids, among however many COMPLETE folders have
    accumulated, were produced together by the most recent full pass, and how
    their headline numbers moved relative to the batch before that -- both
    read straight from data the runs already wrote.
    """
    outputs_dir = Path(outputs_dir)
    batch_path = outputs_dir / "CURRENT_BATCH.json"
    previous = None
    if batch_path.exists():
        try:
            previous = json.loads(batch_path.read_text(encoding="utf-8"))
        except Exception:
            previous = None

    runs: Dict[str, Dict] = {}
    for r in results:
        if r.get("status") != "COMPLETE":
            continue
        runs[r["spec"]] = {
            "run_id": r["run_id"], "run_dir": r["run_dir"],
            "test_accuracy": r.get("test_accuracy"), "test_macro_f1": r.get("test_macro_f1"),
            "test_ece": r.get("test_ece"), "test_brier": r.get("test_brier"),
        }

    delta: Dict[str, Dict] = {}
    if previous:
        prev_runs = previous.get("runs", {})
        for spec, cur in runs.items():
            prev = prev_runs.get(spec)
            if prev is None:
                delta[spec] = {"status": "new"}
                continue
            d: Dict[str, object] = {}
            for k in ("test_accuracy", "test_macro_f1", "test_ece", "test_brier"):
                if cur.get(k) is not None and prev.get(k) is not None:
                    d[k] = round(cur[k] - prev[k], 6)
            d["previous_run_id"] = prev.get("run_id")
            delta[spec] = d
        for spec, prev in prev_runs.items():
            if spec not in runs:
                delta[spec] = {"status": "missing_from_new_batch", "previous_run_id": prev.get("run_id")}

    dupes = find_duplicate_specs(outputs_dir, pattern="*/manifest.json")
    batch = {
        "written_utc": utc_now_iso(),
        "n_runs": len(runs),
        "n_failed": sum(1 for r in results if r.get("status") == "FAILED"),
        "n_skipped": sum(1 for r in results if r.get("status") == "SKIPPED"),
        "runs": runs,
        "delta_vs_previous_batch": delta,
        "previous_batch_written_utc": previous.get("written_utc") if previous else None,
        "duplicate_specs_present": {"__".join(str(x) for x in k): [str(p) for p in v]
                                    for k, v in dupes.items()},
        "queue_log": str(queue_log) if queue_log else None,
    }
    batch_path.write_text(json.dumps(batch, indent=2, default=str), encoding="utf-8")
    return batch
