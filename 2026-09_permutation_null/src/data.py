"""Dataset scanning, participant-level partitioning, and torch Datasets.

THE CORRECTION THIS MODULE EXISTS FOR
-------------------------------------
The submitted code grouped images with

    re.compile(r'(OAS\\d_\\d{4}_MR\\d)')

which includes the MR **session** number. OASIS-1 contains a reliability subset
of nondemented participants rescanned within 90 days, so 19 of the 347
participants in this dataset are represented by two sessions. Treating those
sessions as independent subjects put the same person in train and test:
5.5-8.5% of every test partition, measured across the five submitted seeds.

Here the grouping key is the participant, ``OAS1_XXXX``, and
``assert_group_disjoint`` is called unconditionally on every run.
"""
from __future__ import annotations

import os
import pickle
import re
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# OAS1_0061_MR2_mpr-3_142.jpg
#  group(1) participant  -> OAS1_0061      <- the grouping key
#  group(2) session      -> 2
#  group(3) acquisition  -> 3
#  group(4) slice index  -> 142
FILENAME_RE = re.compile(r"^(OAS\d_\d{4})_MR(\d+)_mpr-(\d+)_(\d+)\.(?:jpg|jpeg|png)$", re.IGNORECASE)

SPLITS = ("train", "val", "test")


class DataError(RuntimeError):
    pass


class GroupLeakageError(AssertionError):
    """Raised when one group appears in more than one partition."""


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def parse_filename(name: str) -> Optional[Dict[str, object]]:
    m = FILENAME_RE.match(name)
    if m is None:
        return None
    return {
        "participant_id": m.group(1),
        "session_id": f"{m.group(1)}_MR{m.group(2)}",
        "acquisition": int(m.group(3)),
        "slice_idx": int(m.group(4)),
    }


def scan_dataset(root: str | Path, original_classes: List[str],
                 class_map: Dict[int, int]) -> pd.DataFrame:
    """One row per image. Sorted deterministically so runs do not depend on
    filesystem enumeration order (the submitted code did, via os.listdir)."""
    root = Path(root)
    if not root.is_dir():
        raise DataError(f"data root not found: {root}")

    records, unparsed = [], []
    for label_4, class_name in enumerate(original_classes):
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise DataError(f"class folder missing: {class_dir}")
        for fname in os.listdir(class_dir):
            parsed = parse_filename(fname)
            if parsed is None:
                unparsed.append(f"{class_name}/{fname}")
                continue
            parsed.update({
                "path": str(class_dir / fname),
                "filename": fname,
                "class_4": label_4,
                "class_4_name": class_name,
                "class_3": class_map[label_4],
            })
            records.append(parsed)

    if not records:
        raise DataError(f"no parseable images under {root}")
    if unparsed:
        raise DataError(
            f"{len(unparsed)} file(s) did not match the expected filename schema, "
            f"e.g. {unparsed[:3]}. Refusing to proceed: silently skipping files "
            f"would change the cohort without telling you."
        )

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(["participant_id", "session_id", "acquisition", "slice_idx"],
                        kind="mergesort").reset_index(drop=True)

    # A participant must carry exactly one label.
    conflicts = df.groupby("participant_id")["class_3"].nunique()
    bad = conflicts[conflicts > 1]
    if len(bad):
        raise DataError(f"{len(bad)} participant(s) span more than one class: {list(bad.index[:5])}")
    return df


def cohort_summary(df: pd.DataFrame) -> Dict[str, object]:
    per_participant = df.groupby("participant_id").agg(
        sessions=("session_id", "nunique"), images=("path", "size"), cls=("class_3", "first"))
    multi = per_participant[per_participant["sessions"] > 1]
    return {
        "images": int(len(df)),
        "participants": int(df["participant_id"].nunique()),
        "sessions": int(df["session_id"].nunique()),
        "participants_multi_session": int(len(multi)),
        "multi_session_ids": sorted(multi.index.tolist()),
        "participants_per_class": per_participant["cls"].value_counts().sort_index().to_dict(),
        "sessions_per_class": df.groupby("session_id")["class_3"].first().value_counts().sort_index().to_dict(),
        "images_per_class": df["class_3"].value_counts().sort_index().to_dict(),
        "slice_positions": int(df["slice_idx"].nunique()),
        "slice_range": (int(df["slice_idx"].min()), int(df["slice_idx"].max())),
        "acquisitions_per_session": df.groupby("session_id")["acquisition"].nunique().value_counts().sort_index().to_dict(),
    }


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------

def _group_frame(df: pd.DataFrame, group_by: str) -> pd.DataFrame:
    """One row per group, with its label. Groups are the unit of splitting."""
    key = "participant_id" if group_by == "participant" else "session_id"
    g = df.groupby(key)["class_3"].first().reset_index()
    g.columns = ["group", "label"]
    return g.sort_values("group", kind="mergesort").reset_index(drop=True)


def stratification_is_sound(n_splits: int = 5, seed: int = 0) -> bool:
    """Probe whether StratifiedGroupKFold(shuffle=True) actually stratifies here.

    Several scikit-learn releases shuffle the per-group class-count matrix without
    re-mapping groups_inv, so the greedy balancing is applied to one permutation
    and membership is read from another. Stratification is silently lost and the
    splitter behaves like plain GroupKFold -- no error, no warning.

    Rather than maintain a table of affected versions, this constructs a case with
    a known-perfect answer (100 groups, 20 of them positive, 5 folds -> exactly 4
    positives per fold) and checks that the splitter finds it. Costs about a
    millisecond and is correct for any release, past or future.
    """
    groups = np.arange(100)
    y = np.zeros(100, dtype=int)
    y[::5] = 1                                     # 20 positives, evenly spaced
    kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    per_fold = [int(y[te].sum()) for _, te in kf.split(groups, y, groups=groups)]
    return max(per_fold) - min(per_fold) <= 1


def make_splits(df: pd.DataFrame, cfg: Dict) -> Dict[str, np.ndarray]:
    """Return {'train','val','test'} -> integer row indices into df."""
    sp = cfg["split"]
    key = "participant_id" if sp["group_by"] == "participant" else "session_id"
    groups = _group_frame(df, sp["group_by"])
    g_names, g_labels = groups["group"].to_numpy(), groups["label"].to_numpy()

    if sp["strategy"] == "grouped_kfold":
        shuffle = bool(sp.get("shuffle", False))
        if shuffle and not stratification_is_sound():
            raise DataError(
                "split.shuffle is true, but StratifiedGroupKFold(shuffle=True) fails a "
                "known-answer stratification probe in this scikit-learn build: it is "
                "silently degrading to unstratified GroupKFold. Set split.shuffle: false "
                "(deterministic, correctly stratified, identical across versions) or "
                "upgrade scikit-learn."
            )
        # sklearn rejects random_state when shuffle is False.
        rs = sp["split_seed"] if shuffle else None
        outer = StratifiedGroupKFold(n_splits=sp["n_splits"], shuffle=shuffle, random_state=rs)
        folds = list(outer.split(g_names, g_labels, groups=g_names))
        dev_idx, test_idx = folds[sp["fold"]]

        inner = StratifiedGroupKFold(n_splits=sp["val_n_splits"], shuffle=shuffle,
                                     random_state=rs)
        dev_names, dev_labels = g_names[dev_idx], g_labels[dev_idx]
        tr_rel, va_rel = next(iter(inner.split(dev_names, dev_labels, groups=dev_names)))
        train_groups, val_groups = dev_names[tr_rel], dev_names[va_rel]
        test_groups = g_names[test_idx]

    elif sp["strategy"] == "grouped_holdout":
        gss = GroupShuffleSplit(n_splits=1, test_size=sp["test_fraction"],
                                random_state=sp["split_seed"])
        dev_idx, test_idx = next(iter(gss.split(g_names, g_labels, groups=g_names)))
        dev_names, dev_labels = g_names[dev_idx], g_labels[dev_idx]
        gss2 = GroupShuffleSplit(n_splits=1, test_size=sp["val_fraction_of_dev"],
                                 random_state=sp["split_seed"])
        tr_rel, va_rel = next(iter(gss2.split(dev_names, dev_labels, groups=dev_names)))
        train_groups, val_groups, test_groups = dev_names[tr_rel], dev_names[va_rel], g_names[test_idx]
    else:
        raise DataError(f"unknown split strategy {sp['strategy']!r}")

    member = {"train": set(train_groups), "val": set(val_groups), "test": set(test_groups)}
    col = df[key].to_numpy()
    return {name: np.flatnonzero(np.isin(col, list(ids))) for name, ids in member.items()}


def assert_group_disjoint(df: pd.DataFrame, splits: Dict[str, np.ndarray],
                          group_by: str = "participant",
                          allow_leakage: bool = False) -> str:
    """Hard guard. Called on every run; its result is logged every run.

    Deliberately checks the PARTICIPANT column regardless of what was used to
    split, so that splitting by session is detected as leakage rather than
    silently accepted.
    """
    sets = {name: set(df.iloc[idx]["participant_id"]) for name, idx in splits.items()}
    offenders = []
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = sets[a] & sets[b]
        if shared:
            offenders.append(f"{a} n {b}: {len(shared)} participant(s) e.g. {sorted(shared)[:5]}")

    total = sum(len(v) for v in splits.values())
    if total != len(df):
        offenders.append(f"partitions cover {total} rows but the frame has {len(df)}")

    if offenders:
        detail = "participant-level leakage between partitions:\n  " + "\n  ".join(offenders)
        if not allow_leakage:
            raise GroupLeakageError(detail)
        # Only reachable when a config explicitly opts in, which exactly one
        # experiment does: the ablation that measures what the flawed protocol
        # was worth. Recorded verbatim in the log and the manifest.
        n_leaked = len(set(df.iloc[splits["train"]]["participant_id"]) &
                       set(df.iloc[splits["test"]]["participant_id"]))
        return f"LEAKAGE PERMITTED BY CONFIG ({n_leaked} participant(s) in train and test) -- {detail}"
    return "PASS"


def split_summary(df: pd.DataFrame, splits: Dict[str, np.ndarray], n_classes: int = 3) -> pd.DataFrame:
    rows = []
    for name in SPLITS:
        sub = df.iloc[splits[name]]
        per_part = sub.groupby("participant_id")["class_3"].first()
        counts = per_part.value_counts()
        rows.append({
            "split": name,
            "participants": int(sub["participant_id"].nunique()),
            "sessions": int(sub["session_id"].nunique()),
            "images": int(len(sub)),
            **{f"participants_c{c}": int(counts.get(c, 0)) for c in range(n_classes)},
            **{f"images_c{c}": int((sub["class_3"] == c).sum()) for c in range(n_classes)},
        })
    return pd.DataFrame(rows)


def compute_class_weights(train_labels: np.ndarray, n_classes: int = 3) -> np.ndarray:
    """Normalised inverse frequency, identical to the submitted implementation:
    w_c = N / n_c, then normalised to sum to 1. Reproduces alpha = [0.055, 0.273, 0.671]."""
    counts = np.array([(train_labels == c).sum() for c in range(n_classes)], dtype=np.float64)
    total = counts.sum()
    w = np.where(counts > 0, total / np.maximum(counts, 1), 0.0)
    return (w / w.sum()).astype(np.float32)


def class_counts(train_labels: np.ndarray, n_classes: int = 3) -> List[int]:
    return [int((train_labels == c).sum()) for c in range(n_classes)]


# ---------------------------------------------------------------------------
# Transforms and Dataset
# ---------------------------------------------------------------------------

def build_transforms(image_size) -> Dict[str, transforms.Compose]:
    """Unchanged from the submitted protocol so results stay comparable.

    Source images are 496x248; Resize to a square does not preserve aspect ratio.
    That is the submitted behaviour and is documented in Methods rather than
    silently altered here.
    """
    size = tuple(image_size)
    norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    train = transforms.Compose([
        transforms.Resize(size),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        norm,
    ])
    eval_tf = transforms.Compose([transforms.Resize(size), transforms.ToTensor(), norm])
    return {"train": train, "val": eval_tf, "test": eval_tf}


class DementiaDataset(Dataset):
    """2D by default. With context_slices=3, packs slices i-1, i, i+1 into the
    three input channels (2.5D), clamping at the edges of the 100-160 band."""

    def __init__(self, frame: pd.DataFrame, transform, context_slices: int = 1):
        if context_slices not in (1, 3):
            raise DataError("context_slices must be 1 or 3")
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.context_slices = context_slices
        self.paths = self.frame["path"].tolist()
        self.labels = self.frame["class_3"].to_numpy()
        self.participants = self.frame["participant_id"].tolist()
        self.sessions = self.frame["session_id"].tolist()
        self._neighbour_index = self._build_neighbour_index() if context_slices == 3 else None

    def _build_neighbour_index(self) -> Dict[Tuple[str, int, int], str]:
        return {(r.session_id, r.acquisition, r.slice_idx): r.path
                for r in self.frame.itertuples(index=False)}

    def __len__(self) -> int:
        return len(self.frame)

    def _load_gray(self, path: str) -> Image.Image:
        with Image.open(path) as im:
            return im.convert("L").copy()

    def __getitem__(self, i: int):
        if self.context_slices == 1:
            with Image.open(self.paths[i]) as im:
                img = im.convert("RGB").copy()
        else:
            row = self.frame.iloc[i]
            key = (row.session_id, int(row.acquisition))
            centre = int(row.slice_idx)
            planes = []
            for offset in (-1, 0, 1):
                path = self._neighbour_index.get(key + (centre + offset,))
                planes.append(self._load_gray(path if path is not None else self.paths[i]))
            img = Image.merge("RGB", planes)
        return self.transform(img), int(self.labels[i])


def assert_spawn_safe(dataset: "DementiaDataset", worker_init_fn) -> None:
    """Verify everything DataLoader must ship to a worker can actually be pickled.

    On Windows and macOS, DataLoader workers are spawned, not forked: the dataset,
    its transform and worker_init_fn are pickled into a fresh interpreter. A
    closure or lambda anywhere in that graph fails with
    `EOFError: Ran out of input` from multiprocessing/spawn.py -- an error that
    names nothing useful and costs an hour to trace. This check fails in
    milliseconds and says which object is at fault.

    A one-row copy of the dataset is used so the check costs nothing on an
    86,000-image frame.
    """
    probe = DementiaDataset(dataset.frame.head(1), dataset.transform,
                            context_slices=dataset.context_slices)
    for name, obj in (("worker_init_fn", worker_init_fn),
                      ("transform", dataset.transform),
                      ("dataset", probe)):
        try:
            pickle.loads(pickle.dumps(obj))
        except Exception as exc:
            raise DataError(
                f"{name} cannot be pickled, so DataLoader workers cannot be spawned "
                f"on this platform ({sys.platform}): {type(exc).__name__}: {exc}\n"
                f"Closures and lambdas are the usual cause -- move the offending "
                f"callable to module level. Alternatively run with --num-workers 0."
            ) from exc


def make_eval_loader(df: pd.DataFrame, idx: np.ndarray, cfg: Dict,
                     num_workers: Optional[int] = None, pin_memory: Optional[bool] = None,
                     batch_size: Optional[int] = None, persistent: bool = False) -> DataLoader:
    """A single-use loader for inference.

    persistent defaults to False so eval workers die when the pass ends rather
    than accumulating across train -> val -> test. The validation loader is the
    one exception: it is consumed once per epoch, and on Windows the spawn cost
    of restarting workers every epoch is far worse than keeping two of them.
    Defaults come from train.eval_*; the caller may override any of them to
    degrade after an out-of-memory failure.
    """
    tcfg = cfg["train"]
    nw = tcfg["eval_num_workers"] if num_workers is None else num_workers
    pin = tcfg["eval_pin_memory"] if pin_memory is None else pin_memory
    bs = tcfg["eval_batch_size"] if batch_size is None else batch_size
    ds = DementiaDataset(df.iloc[idx], build_transforms(cfg["data"]["image_size"])["test"],
                         context_slices=cfg["data"]["context_slices"])
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw,
                      pin_memory=bool(pin) and torch.cuda.is_available(),
                      drop_last=False, persistent_workers=bool(persistent) and nw > 0)


def build_dataloaders(df: pd.DataFrame, splits: Dict[str, np.ndarray], cfg: Dict,
                      generator, worker_init_fn) -> Dict[str, DataLoader]:
    """Training-time loaders ONLY: train and val.

    The test loader is deliberately absent. Building all three up front kept
    three sets of persistent workers alive simultaneously -- twelve processes and
    three pin-memory threads on a 6 GB card -- which is what caused an
    out-of-memory failure in the pin thread at test time, after training had
    already succeeded. Test data is loaded on demand via make_eval_loader.

    num_workers is a deliberate config value; torch's advisory about exceeding the
    CPU count repeats once per loader per run and is suppressed here. The value
    used is recorded in every manifest.
    """
    warnings.filterwarnings("ignore", message=".*This DataLoader will create.*")
    tfs = build_transforms(cfg["data"]["image_size"])
    ctx = cfg["data"]["context_slices"]
    tcfg = cfg["train"]

    train_ds = DementiaDataset(df.iloc[splits["train"]], tfs["train"], context_slices=ctx)
    if tcfg["num_workers"] > 0:
        assert_spawn_safe(train_ds, worker_init_fn)

    loaders = {
        "train": DataLoader(
            train_ds, batch_size=tcfg["batch_size"], shuffle=True,
            num_workers=tcfg["num_workers"], pin_memory=torch.cuda.is_available(),
            drop_last=False, generator=generator,
            worker_init_fn=worker_init_fn if tcfg["num_workers"] > 0 else None,
            persistent_workers=(tcfg["num_workers"] > 0)),
        # Validation is consumed once per epoch, so its workers stay alive -- but
        # only two of them, and with pinning off.
        "val": make_eval_loader(df, splits["val"], cfg, persistent=True),
    }
    return loaders
