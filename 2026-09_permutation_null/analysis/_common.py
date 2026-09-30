"""Shared machinery for the analysis layer.

Two ideas do most of the work here.

POOLING. Participant-level grouped 5-fold CV tests every participant exactly
once, so concatenating the five test folds gives predictions for the whole
cohort -- all 347 participants -- with no participant counted twice. That pooled
set is the estimate the manuscript should lead with, because any single fold's
minority-class metric rests on four or five people.

SUFFICIENT STATISTICS. Every metric used here (accuracy, per-class F1, macro-F1,
ECE, Brier) is a function of counts that are ADDITIVE over participants. So each
participant is reduced once to a small fixed-size summary -- a 3x3 confusion
contribution, per-bin calibration counts, a squared-error sum -- and the
bootstrap then resamples those summaries instead of 86,000 image rows. The
result is numerically identical to the naive bootstrap and roughly two orders of
magnitude faster, which is what makes 2000 participant-clustered resamples
practical for every metric and every loss.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.discover import load_predictions, load_runs  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "analysis" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Non Demented", "Very Mild Demented", "Demented"]
SHORT = ["NonDem", "VeryMild", "Dem"]
LOSS_ORDER = ["WCE", "LDAM", "FocalLoss", "ClassBalanced", "FA_FL"]
LOSS_LABEL = {"WCE": "Weighted CE", "LDAM": "LDAM", "FocalLoss": "Focal",
              "ClassBalanced": "Class-Balanced", "FA_FL": "FA-FL (proposed)"}
N_CLASSES = 3
ECE_BINS = 15


# ---------------------------------------------------------------------------
# Loading and pooling
# ---------------------------------------------------------------------------

def runs_frame(experiments: Sequence[str] = ("exp01_main_sweep", "exp02_class_balanced")) -> pd.DataFrame:
    df = pd.concat([load_runs(e) for e in experiments], ignore_index=True)
    if df.empty:
        raise SystemExit("no completed runs found -- run the experiments first")
    fps = df["code_fingerprint"].dropna().unique()
    if len(fps) > 1:
        raise SystemExit(
            f"refusing to build tables from {len(fps)} different code fingerprints: "
            f"{[f[:8] for f in fps]}. Delete or re-run the odd ones out.")
    return df


def data_root_from_runs(runs: pd.DataFrame) -> str:
    """The data root RECORDED IN THE MANIFESTS, not whatever config/base.yaml says now.

    Table 1 must describe the cohort the results were actually computed on. Reading
    the live config would silently describe a different dataset if the config were
    edited after the runs.
    """
    roots = set()
    for _, r in runs.iterrows():
        m = json.loads((Path(r["run_dir"]) / "manifest.json").read_text(encoding="utf-8"))
        roots.add(m["resolved_config"]["data"]["root"])
    if len(roots) > 1:
        raise SystemExit(f"runs used {len(roots)} different data roots: {sorted(roots)}")
    return roots.pop()


def cohort_config_from_runs(runs: pd.DataFrame) -> Dict:
    """Resolved config of the first run, with class_map keys restored to ints."""
    from src.config import _normalise_class_map
    m = json.loads((Path(runs.iloc[0]["run_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    cfg = m["resolved_config"]
    _normalise_class_map(cfg)
    return cfg


def pooled_predictions(runs: pd.DataFrame, loss: str, temperature_scaled: bool = False) -> pd.DataFrame:
    """Concatenate the five test folds for one loss -> one row per image, whole cohort."""
    sub = runs[runs["loss"] == loss].sort_values("fold")
    if sub.empty:
        raise SystemExit(f"no runs for loss {loss!r}")
    parts = []
    for _, r in sub.iterrows():
        if temperature_scaled:
            f = Path(r["run_dir"]) / "predictions_test_temperature_scaled.npz"
            if not f.exists():
                raise SystemExit(f"missing temperature-scaled predictions in {r['run_dir']}")
            d = np.load(f, allow_pickle=False)
            probs = d["probabilities"]
            frame = pd.DataFrame({"participant_id": d["participant_id"].astype(str),
                                  "label": d["labels"].astype(int)})
        else:
            frame = load_predictions(r["run_dir"], "test")
            probs = frame[[f"p{c}" for c in range(N_CLASSES)]].to_numpy()
            frame = frame[["participant_id", "label"]].copy()
        frame["pred"] = probs.argmax(axis=1)
        for c in range(N_CLASSES):
            frame[f"p{c}"] = probs[:, c]
        frame["fold"] = int(r["fold"])
        parts.append(frame)
    pooled = pd.concat(parts, ignore_index=True)

    dupes = pooled.groupby("participant_id")["fold"].nunique()
    if (dupes > 1).any():
        raise SystemExit(f"{int((dupes > 1).sum())} participant(s) appear in more than one test "
                         f"fold -- the folds are not disjoint, refusing to pool")
    return pooled


# ---------------------------------------------------------------------------
# Per-participant sufficient statistics
# ---------------------------------------------------------------------------

class SuffStats:
    """Additive per-participant summaries: everything the metrics need."""

    def __init__(self, pooled: pd.DataFrame, n_bins: int = ECE_BINS):
        self.participants = pooled["participant_id"].to_numpy()
        self.uniq, inv = np.unique(self.participants, return_inverse=True)
        self.n_p = len(self.uniq)
        self.n_bins = n_bins

        y = pooled["label"].to_numpy()
        yhat = pooled["pred"].to_numpy()
        probs = pooled[[f"p{c}" for c in range(N_CLASSES)]].to_numpy(dtype=np.float64)

        # 3x3 confusion contribution per participant
        self.cm = np.zeros((self.n_p, N_CLASSES, N_CLASSES))
        np.add.at(self.cm, (inv, y, yhat), 1.0)

        # calibration: per participant x bin -> (count, sum confidence, sum correct)
        conf = probs.max(axis=1)
        correct = (yhat == y).astype(np.float64)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        b = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
        self.bin_n = np.zeros((self.n_p, n_bins))
        self.bin_conf = np.zeros((self.n_p, n_bins))
        self.bin_ok = np.zeros((self.n_p, n_bins))
        np.add.at(self.bin_n, (inv, b), 1.0)
        np.add.at(self.bin_conf, (inv, b), conf)
        np.add.at(self.bin_ok, (inv, b), correct)

        # Brier
        onehot = np.eye(N_CLASSES)[y]
        self.brier_sum = np.zeros(self.n_p)
        np.add.at(self.brier_sum, inv, ((probs - onehot) ** 2).sum(axis=1))
        self.n_img = self.cm.sum(axis=(1, 2))

    # -- metrics from summed statistics -------------------------------------

    @staticmethod
    def _from_cm(cm: np.ndarray) -> Dict[str, object]:
        tp = np.diag(cm)
        support, predicted = cm.sum(axis=1), cm.sum(axis=0)
        prec = np.divide(tp, predicted, out=np.zeros(N_CLASSES), where=predicted > 0)
        rec = np.divide(tp, support, out=np.zeros(N_CLASSES), where=support > 0)
        denom = prec + rec
        f1 = np.divide(2 * prec * rec, denom, out=np.zeros(N_CLASSES), where=denom > 0)
        total = cm.sum()
        return {"accuracy": 100.0 * tp.sum() / total if total else 0.0,
                "macro_f1": float(f1.mean()), "per_class_f1": f1,
                "per_class_precision": prec, "per_class_recall": rec,
                "weighted_f1": float((f1 * support).sum() / total) if total else 0.0}

    def metrics(self, idx: Optional[np.ndarray] = None) -> Dict[str, object]:
        sel = slice(None) if idx is None else idx
        cm = self.cm[sel].sum(axis=0)
        out = self._from_cm(cm)
        n = self.bin_n[sel].sum(axis=0)
        c = self.bin_conf[sel].sum(axis=0)
        k = self.bin_ok[sel].sum(axis=0)
        total = n.sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            gap = np.abs(np.where(n > 0, k / n, 0.0) - np.where(n > 0, c / n, 0.0))
        out["ece"] = float((n / total * gap).sum()) if total else 0.0
        out["mce"] = float(gap[n > 0].max()) if (n > 0).any() else 0.0
        out["brier"] = float(self.brier_sum[sel].sum() / total) if total else 0.0
        out["n_images"] = int(total)
        out["n_participants"] = int(self.n_p if idx is None else len(idx))
        out["confusion_matrix"] = cm
        return out


def clustered_bootstrap(stats: Dict[str, SuffStats], metric: str, n_boot: int = 2000,
                        seed: int = 12345, per_class: Optional[int] = None) -> Dict[str, Dict]:
    """Resample PARTICIPANTS with replacement; recompute the metric for every loss
    on the SAME resample, so between-loss differences stay paired.

    Resampling images instead would treat ~86,000 correlated rows as independent
    -- the test set holds 3-4 near-duplicate acquisitions of each slice (r = 0.988)
    and only ~347 independent people -- and would understate every interval by
    more than an order of magnitude.
    """
    losses = list(stats)
    n_p = stats[losses[0]].n_p
    for L in losses:
        if stats[L].n_p != n_p:
            raise SystemExit("losses cover different participant counts; cannot pair")

    rng = np.random.default_rng(seed)
    draws = {L: np.empty(n_boot) for L in losses}
    pull = lambda m: (m["per_class_f1"][per_class] if per_class is not None else m[metric])

    for b in range(n_boot):
        idx = rng.integers(0, n_p, size=n_p)
        for L in losses:
            draws[L][b] = pull(stats[L].metrics(idx))

    out = {}
    for L in losses:
        point = pull(stats[L].metrics())
        lo, hi = np.percentile(draws[L], [2.5, 97.5])
        out[L] = {"point": float(point), "ci_low": float(lo), "ci_high": float(hi),
                  "boot_mean": float(draws[L].mean()), "boot_sd": float(draws[L].std(ddof=1)),
                  "_draws": draws[L]}
    return out


def paired_difference(boot: Dict[str, Dict], a: str, b: str,
                      family: Optional[str] = None) -> Dict[str, float]:
    """CI on (a - b) from the paired bootstrap draws, plus a two-sided bootstrap p.

    The p-value is FLOORED at 1/n_boot. A bootstrap over B resamples cannot
    resolve a probability below 1/B, so the raw formula's ability to return
    exactly 0.0 is an artefact of resolution, not a finding. A literal
    "p = 0.0000" in a table is both wrong and the first thing a reviewer circles;
    report it as "< 1/B".

    When `family` names a declared comparison family, the interval is computed at
    that family's MULTIPLICITY-CORRECTED level rather than at a nominal 95%, so
    that the interval and the p-threshold test the same hypothesis at the same
    level. Pairing a 95% interval with a Bonferroni-corrected p-threshold is an
    inconsistent test, and the two will eventually disagree in print.
    """
    d = boot[a]["_draws"] - boot[b]["_draws"]
    n_boot = len(d)

    if family is not None:
        from analysis._registry import REGISTRY
        lo_pct, hi_pct = REGISTRY.ci_percentiles(family)
        ci_level = 1.0 - REGISTRY.alpha_corrected(family)
    else:
        lo_pct, hi_pct, ci_level = 2.5, 97.5, 0.95

    lo, hi = np.percentile(d, [lo_pct, hi_pct])
    p_raw = 2.0 * min((d <= 0).mean(), (d >= 0).mean())
    p = min(max(p_raw, 1.0 / n_boot), 1.0)

    return {"diff": float(boot[a]["point"] - boot[b]["point"]),
            "ci_low": float(lo), "ci_high": float(hi),
            "ci_level": float(ci_level),
            "p_bootstrap": float(p),
            "p_at_resolution_floor": bool(p_raw < 1.0 / n_boot),
            "n_bootstrap": int(n_boot),
            "excludes_zero": bool(lo > 0 or hi < 0)}


# ---------------------------------------------------------------------------
# Fold-level helpers
# ---------------------------------------------------------------------------

def fold_metrics(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per run, read from metrics.json (no recomputation)."""
    rows = []
    for _, r in runs.iterrows():
        m = json.loads((Path(r["run_dir"]) / "metrics.json").read_text(encoding="utf-8"))
        t = m["test"]
        ts = t.get("temperature_scaling", {})
        rows.append({"loss": m["loss"], "fold": m["fold"], "init_seed": m["init_seed"],
                     "accuracy": t["accuracy"], "macro_f1": t["macro_f1"],
                     **{f"f1_{SHORT[c]}": t["per_class_f1"][c] for c in range(N_CLASSES)},
                     "ece": t["ece"], "brier": t["brier_sum"],
                     "ece_temp_scaled": ts.get("ece_after"), "temperature": ts.get("temperature"),
                     "epochs_run": m["training"]["epochs_run"],
                     "run_dir": r["run_dir"]})
    cols = ["loss", "fold", "init_seed", "accuracy", "macro_f1",
            *[f"f1_{s}" for s in SHORT], "ece", "brier", "ece_temp_scaled",
            "temperature", "epochs_run", "run_dir"]
    if not rows:                      # empty, but with the right columns, so callers
        return pd.DataFrame(columns=cols)   # can filter and test .empty as usual
    return pd.DataFrame(rows).sort_values(["loss", "fold", "init_seed"]).reset_index(drop=True)


def wilcoxon_signed_rank(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """Exact two-sided Wilcoxon for the small n used here, plus rank-biserial size.

    With five folds the smallest attainable two-sided p is 0.0625, so this test
    CANNOT reach 0.05 no matter how large the effect. It is reported as a
    conservative companion to the participant-clustered bootstrap, never as the
    primary evidence -- a point worth stating explicitly in the manuscript.
    """
    from itertools import product
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return {"n": 0, "W": float("nan"), "p": 1.0, "rank_biserial": 0.0,
                "min_attainable_p": float("nan")}
    ranks = pd.Series(np.abs(d)).rank().to_numpy()
    w_pos = ranks[d > 0].sum()
    w_neg = ranks[d < 0].sum()
    W = min(w_pos, w_neg)
    count = sum(1 for signs in product([1, -1], repeat=n)
                if min(sum(r for r, s in zip(ranks, signs) if s > 0),
                       sum(r for r, s in zip(ranks, signs) if s < 0)) <= W)
    return {"n": int(n), "W": float(W), "p": float(min(count / 2 ** n, 1.0)),
            "rank_biserial": float((w_pos - w_neg) / (n * (n + 1) / 2)),
            "min_attainable_p": float(2.0 / 2 ** n)}


def _to_markdown(df: pd.DataFrame, decimals: int = 4) -> str:
    """Render a markdown table without pandas.to_markdown.

    to_markdown requires the optional `tabulate` package. Making the analysis
    layer depend on an extra install -- and fail after a ten-minute bootstrap
    because of a formatting helper -- is not a trade worth making, so the few
    lines of formatting are done here instead.
    """
    def cell(v) -> str:
        if isinstance(v, (bool, np.bool_)):
            return str(bool(v))
        if isinstance(v, (int, np.integer)):
            return f"{int(v):,}"
        if isinstance(v, (float, np.floating)):
            if np.isnan(v):
                return ""
            return f"{v:,.{decimals}f}"
        return str(v)

    cols = [str(c) for c in df.columns]
    body = [[cell(v) for v in row] for row in df.itertuples(index=False, name=None)]
    widths = [max(len(cols[i]), *(len(r[i]) for r in body)) if body else len(cols[i])
              for i in range(len(cols))]
    numeric = [pd.api.types.is_numeric_dtype(df[c]) and df[c].dtype != bool for c in df.columns]

    def line(vals):
        return "| " + " | ".join(
            v.rjust(widths[i]) if numeric[i] else v.ljust(widths[i])
            for i, v in enumerate(vals)) + " |"

    rule = "|" + "|".join(("-" * (w + 1) + ":") if numeric[i] else (":" + "-" * (w + 1))
                          for i, w in enumerate(widths)) + "|"
    return "\n".join([line(cols), rule, *(line(r) for r in body)]) + "\n"


def save_table(df: pd.DataFrame, stem: str, float_fmt: str = "%.4f") -> None:
    df.to_csv(OUT / f"{stem}.csv", index=False, float_format=float_fmt)
    decimals = 0 if float_fmt == "%.0f" else 4
    (OUT / f"{stem}.md").write_text(_to_markdown(df, decimals), encoding="utf-8")
    print(f"    -> {stem}.csv / .md")
