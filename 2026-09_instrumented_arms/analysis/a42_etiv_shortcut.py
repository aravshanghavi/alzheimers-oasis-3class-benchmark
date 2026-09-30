#!/usr/bin/env python3
"""a42: is the sex shortcut really head size, and is head size a shortcut of its own?

    python analysis/a42_etiv_shortcut.py
    python analysis/a42_etiv_shortcut.py --quick      # 200 resamples
    python analysis/a42_etiv_shortcut.py --boot 10000

WHERE THIS COMES FROM
---------------------
Two image-readable shortcuts are already on the record in this project. Age
predicts CDR class and is readable off a brain image. Sex is readable, and among
the cognitively normal participants every one of the eight arms calls men impaired
more often than women, by 0.317 to 0.438 in absolute risk.

Estimated total intracranial volume is the obvious next candidate. It is a head
size measurement, it is strongly associated with sex, and it is about as readable
off a structural MR image as a quantity can be. That makes two questions, and they
are different:

    (1) Does eTIV separate the CDR classes on its own? If yes, head size is a
        third shortcut and belongs in the limitations beside age and sex.
    (2) Is the sex disparity actually head size wearing a sex label? If the
        male minus female gap survives inside narrow bands of eTIV, the answer is
        no and the sex finding stands on its own. If the gap collapses inside those
        bands, the paper has been describing head size.

ASSOCIATION, NOT MECHANISM. STATED ONCE HERE AND AGAIN IN THE OUTPUT
-------------------------------------------------------------------
Nothing in this file shows that any network read eTIV, or read head size, or read
anything at all. eTIV is not even an input: it is a number from the OASIS
spreadsheet, computed by the standard segmentation. What the file measures is
whether a quantity correlated with the label and with the errors exists in the
data. That is enough to require a limitation and nowhere near enough to support a
mechanism. Any sentence of the form "the network is using head size" is not
supported by anything below.

The conditioning is also observational. Adjusting for sex by stratifying on sex
removes the part of the eTIV association that sex accounts for, and leaves
everything sex does not measure, including whatever else differs between the
groups. It is not a causal adjustment and the residual is not a direct effect.

WHAT IS COMPUTED
----------------
eTIV ALONE. The midrank AUC of eTIV for the three named contrasts, the same three
a25, a26 and b01 use so that the rows join. Plus, as a check on the premise, the
AUC of eTIV for separating men from women, which is how much of a sex proxy eTIV
is in this cohort.

ERRORS AGAINST eTIV, CONDITIONING ON SEX. For each arm, an error flag per
participant, then the AUC of eTIV for separating the arm's errors from its
successes computed SEPARATELY WITHIN each sex and then pooled over the two strata
by the number of error-by-success pairs each contributes. That pooled statistic is
the Mann-Whitney statistic conditional on sex, so the part of the association that
sex alone explains is gone from it. Each stratum is also reported on its own,
because a pooled number that hides two opposite-signed strata would be worse than
no number.

THE SEX GAP INSIDE eTIV BANDS. Among the participants whose true CDR is 0, the
share predicted impaired is computed for men and for women within each eTIV
tertile, and the difference is taken inside each tertile. The tertiles are cut
within the CDR 0 group itself, because that is the group the disparity is measured
on. A weighted average of the three within-tertile differences is the head size
adjusted sex gap, and it is printed beside the unadjusted one. Those two numbers
side by side are the answer to question (2).

THE CELLS GET SMALL, AND THAT IS REPORTED RATHER THAN SMOOTHED
--------------------------------------------------------------
On the restricted cohort there are 85 CDR 0 participants, and cutting them by sex
and by eTIV tertile makes six cells of which the male ones hold single digits. The
count in every cell is printed, any cell below five is flagged, and the bootstrap
resamples WITHIN those cells so that no draw empties one. A within-tertile
difference estimated on seven men is a real estimate of a real quantity and its
interval will be very wide; the interval is the honest part and it is not narrowed
by wishing.

Both cohorts are run. The full cohort has 266 CDR 0 participants, which is what
makes the tertile cut informative at all, and the restricted cohort is the paper's
primary one, so neither can substitute for the other.

MULTIPLICITY
------------
No family in analysis/_registry.py covers eTIV, a sex disparity, or an error
association. Every row carries the literal string "undeclared", every interval is
a nominal 95 percent one, and nothing here is a pre-registered test. Nothing is
written to the registry: the changelog entry is PRINTED for the first author to
paste.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from analysis._common import save_table                                       # noqa: E402
from analysis._paths import EXPERIMENTS_DIR, OUT, metadata_xlsx               # noqa: E402
from analysis._protocol_lib import set_include_new_arms                       # noqa: E402
from analysis._protocol_lib import (Cohort, FULL_EXPERIMENTS,                 # noqa: E402
                                    RESTRICTED_EXPERIMENTS, bootstrap_p,
                                    changelog_entry, find_full_tree,
                                    midrank_auc, percentile_interval,
                                    stratified_participant_bootstrap)
from analysis.a23_protocol_audit import ARM_ORDER, warn                       # noqa: E402
from analysis.a25_increment_over_metadata import auc_draws, multiplicity      # noqa: E402

SOURCE = "analysis/a42_etiv_shortcut.py"
EXPERIMENT_ID = "a42"
TABLE = "t42_etiv_shortcut"
N_BOOT = 10000
QUICK_BOOT = 200
BASE_SEED = 12345
N_CLASSES = 3
UNDECLARED = "undeclared"

ASSOCIATION_ONLY = ("ASSOCIATION, NOT MECHANISM. eTIV is a spreadsheet column, not a network "
                    "input. Nothing here shows any network read head size")

PAIRWISE: List[Tuple[str, Optional[int], int]] = [
    ("AUC CDR>=1 vs CDR 0", 2, 0),
    ("AUC any impairment vs CDR 0", None, 0),
    ("AUC CDR>=1 vs CDR 0.5", 2, 1),
]

N_TERTILES = 3
MIN_CELL = 5
STRATIFIED_CELLS = "stratified participant, resampled within sex by eTIV tertile cells"
STRATIFIED_CLASS = "stratified participant, resampled within class"
UNSTRATIFIED_COHORT = "unstratified participant over the whole cohort, one shared draw"


def sex_vector(participants: np.ndarray, meta: pd.DataFrame) -> np.ndarray:
    """1 for male, 0 for female, NaN when absent. a34's convention, kept identical."""
    raw = meta.reindex(participants)["M/F"]
    v = np.where(raw.astype(str).str.upper().str.startswith("M"), 1.0, 0.0)
    v[raw.isna().to_numpy()] = np.nan
    return v


def tertiles(x: np.ndarray) -> np.ndarray:
    """Tertile index 0, 1, 2 by the empirical 33rd and 67th percentiles of x.

    Cut on the group the comparison is made within, not on the whole cohort. A
    tertile boundary imported from a wider population would put most of one sex in
    one band and defeat the point of banding.
    """
    q1, q2 = np.percentile(x, [100.0 / 3.0, 200.0 / 3.0])
    return np.digitize(x, [q1, q2]).astype(int)


def pooled_conditional_auc(score: np.ndarray, positive: np.ndarray,
                           strata: np.ndarray, W: np.ndarray
                           ) -> Tuple[float, np.ndarray, Dict[int, Tuple[float, int, int]]]:
    """AUC of `score` for `positive`, computed within strata then pooled by pair count.

    Pooling by the number of positive-by-negative pairs each stratum contributes is
    what makes this the Mann-Whitney statistic conditional on the stratum, so the
    part of the association the stratum itself explains is not in the result. A
    stratum with no positives or no negatives contributes no pairs and therefore
    drops out with weight zero rather than being averaged in as a 0.5.
    """
    num_p = 0.0
    den_p = 0.0
    num_d = np.zeros(W.shape[0])
    den_d = np.zeros(W.shape[0])
    per: Dict[int, Tuple[float, int, int]] = {}
    for s in np.unique(strata):
        m = strata == s
        pos = positive[m]
        n1, n0 = int(pos.sum()), int((~pos).sum())
        if n1 == 0 or n0 == 0:
            per[int(s)] = (float("nan"), n1, n0)
            continue
        a_point = midrank_auc(score[m], pos)
        per[int(s)] = (float(a_point), n1, n0)
        num_p += a_point * n1 * n0
        den_p += n1 * n0
        Wm = W[:, m]
        pairs = Wm[:, pos].sum(axis=1) * Wm[:, ~pos].sum(axis=1)
        a_draw = auc_draws(score[m], pos, Wm)
        ok = np.isfinite(a_draw) & (pairs > 0)
        num_d[ok] += a_draw[ok] * pairs[ok]
        den_d[ok] += pairs[ok]
    point = (num_p / den_p) if den_p > 0 else float("nan")
    draws = np.divide(num_d, den_d, out=np.full(W.shape[0], np.nan), where=den_d > 0)
    return point, draws, per


def main() -> int:
    t_start = time.time()
    ap = argparse.ArgumentParser(
        description="eTIV as a shortcut, and the sex gap inside eTIV bands.")
    ap.add_argument("--boot", type=int, default=N_BOOT,
                    help=f"participant-clustered resamples (default {N_BOOT})")
    ap.add_argument("--quick", action="store_true",
                    help=f"smoke test with {QUICK_BOOT} resamples; intervals are NOT reportable")
    ap.add_argument("--full-from", default=None, metavar="EXPERIMENTS_DIR")
    ap.add_argument("--include-new-arms", action="store_true",
                    help="also read exp15, exp16 and exp17 when they have COMPLETE runs. "
                         "OFF by default, which reproduces this script's table exactly. "
                         "Also settable with SREP_INCLUDE_NEW_ARMS=1")
    args = ap.parse_args()
    set_include_new_arms(args.include_new_arms)
    n_boot = QUICK_BOOT if args.quick else int(args.boot)
    if args.quick:
        print(f"QUICK MODE. {n_boot} resamples. Intervals are a smoke test, not results.")

    print(f"a42  eTIV as a shortcut, {n_boot:,} resamples, base seed {BASE_SEED}")
    print(f"     {ASSOCIATION_ONLY}")

    cohorts: List[Cohort] = [Cohort("restricted", RESTRICTED_EXPERIMENTS, EXPERIMENTS_DIR)]
    full_dir = find_full_tree(args.full_from)
    if full_dir is None:
        warn("no full-cohort experiments folder found, so only the restricted cohort is "
             "reported. Pass --full-from to include the full cohort.")
    else:
        try:
            cohorts.insert(0, Cohort("full", FULL_EXPERIMENTS, full_dir))
        except SystemExit as exc:                                   # noqa: PERF203
            warn(f"full cohort unusable ({exc}); reporting the restricted cohort alone")

    meta_raw = pd.read_excel(metadata_xlsx())
    meta_raw["pid"] = meta_raw["ID"].astype(str).str.replace(r"_MR\d+$", "", regex=True)
    meta_raw = meta_raw.drop_duplicates("pid").set_index("pid")

    rows: List[Dict] = []
    headline: Dict[str, Dict[str, object]] = {}

    for coh in cohorts:
        y = coh.labels
        n = coh.n_participants
        arms = [a for a in ARM_ORDER if a in coh.arms] + \
               [a for a in sorted(coh.arms) if a not in ARM_ORDER]
        meta = meta_raw.reindex(coh.participants)
        etiv = pd.to_numeric(meta["eTIV"], errors="coerce").to_numpy()
        male = sex_vector(coh.participants, meta_raw)
        have = np.isfinite(etiv) & np.isfinite(male)
        print(f"\n     == {coh.name}: {len(arms)} arms, {n} participants. eTIV missing for "
              f"{int((~np.isfinite(etiv)).sum())}, sex missing for "
              f"{int((~np.isfinite(male)).sum())}.")
        print(f"        {int(have.sum())} participants carry both and are the only ones used "
              f"below. Nothing is imputed: an imputed eTIV would measure the median, not the "
              f"head.")
        if int(have.sum()) < 30:
            warn(f"{coh.name}: only {int(have.sum())} participants carry eTIV and sex; "
                 f"skipping this cohort")
            continue

        # -- descriptive, and the premise check --------------------------------
        for lab, m in (("male", have & (male == 1)), ("female", have & (male == 0))):
            rows.append({"kind": "descriptive", "cohort": coh.name, "group": f"sex {lab}",
                         "n": int(m.sum()), "mean_etiv": float(etiv[m].mean()),
                         "median_etiv": float(np.median(etiv[m])),
                         "sd_etiv": float(etiv[m].std(ddof=1)),
                         "family": UNDECLARED, "n_bootstrap": n_boot,
                         "note": ASSOCIATION_ONLY})
        for c in range(N_CLASSES):
            m = have & (y == c)
            if not m.any():
                continue
            rows.append({"kind": "descriptive", "cohort": coh.name,
                         "group": f"CDR class {c}", "n": int(m.sum()),
                         "mean_etiv": float(etiv[m].mean()),
                         "median_etiv": float(np.median(etiv[m])),
                         "sd_etiv": float(etiv[m].std(ddof=1)),
                         "family": UNDECLARED, "n_bootstrap": n_boot,
                         "note": ASSOCIATION_ONLY})

        sex_auc = midrank_auc(etiv[have], male[have] == 1)
        rows.append({"kind": "etiv_auc", "cohort": coh.name,
                     "group": "AUC male vs female (premise check)",
                     "n": int(have.sum()), "value": float(sex_auc),
                     "bootstrap": "point estimate only, premise check",
                     "family": UNDECLARED, "n_bootstrap": n_boot,
                     "note": "how much of a sex proxy eTIV is in this cohort. An AUC near 1 "
                             "means the two questions in this file cannot be fully separated"})

        # -- (1) eTIV alone on the three contrasts -----------------------------
        for name, higher, lower in PAIRWISE:
            mask = have & (((y > 0) if higher is None else (y == higher)) | (y == lower))
            pos = (y[mask] > 0) if higher is None else (y[mask] == higher)
            if pos.sum() == 0 or (~pos).sum() == 0:
                continue
            point = midrank_auc(etiv[mask], pos)
            idx = stratified_participant_bootstrap(pos.astype(int), n_boot, "a42:auc",
                                                  BASE_SEED, cohort=coh.name, contrast=name)
            d = auc_draws(etiv[mask], pos, multiplicity(idx, int(mask.sum())))
            lo, hi = percentile_interval(d)
            rows.append({"kind": "etiv_auc", "cohort": coh.name, "group": name,
                         "n": int(mask.sum()), "n_positive": int(pos.sum()),
                         "n_negative": int((~pos).sum()),
                         "value": float(point), "ci_low": lo, "ci_high": hi,
                         "ci_level": 0.95, "bootstrap": STRATIFIED_CLASS,
                         "separates": bool(lo > 0.5 or hi < 0.5),
                         "family": UNDECLARED, "n_bootstrap": n_boot,
                         "note": "eTIV alone, no model. " + ASSOCIATION_ONLY})

        # -- (2) errors against eTIV, conditioning on sex ----------------------
        idx_all = stratified_participant_bootstrap(
            np.where(have, male, -1).astype(int), n_boot, "a42:errors", BASE_SEED,
            cohort=coh.name)
        W_all = multiplicity(idx_all, n)
        sub = np.flatnonzero(have)
        W_sub = W_all[:, sub]
        for arm in arms:
            pred = coh.part[arm]["pred"].to_numpy().astype(int)
            err = (pred != y)[sub]
            if err.sum() == 0 or (~err).sum() == 0:
                warn(f"{coh.name}/{arm}: {int(err.sum())} errors, cannot form an AUC")
                continue
            point, draws, per = pooled_conditional_auc(etiv[sub], err, male[sub].astype(int),
                                                      W_sub)
            lo, hi = percentile_interval(draws)
            marginal = midrank_auc(etiv[sub], err)
            rows.append({
                "kind": "error_vs_etiv", "cohort": coh.name, "arm": arm,
                "group": "pooled within sex, Mann-Whitney conditional on sex",
                "n": int(have.sum()), "n_errors": int(err.sum()),
                "n_correct": int((~err).sum()),
                "value": float(point), "ci_low": lo, "ci_high": hi, "ci_level": 0.95,
                "marginal_auc_not_conditioned": float(marginal),
                "auc_female_stratum": per.get(0, (float("nan"), 0, 0))[0],
                "n_errors_female": per.get(0, (float("nan"), 0, 0))[1],
                "auc_male_stratum": per.get(1, (float("nan"), 0, 0))[0],
                "n_errors_male": per.get(1, (float("nan"), 0, 0))[1],
                "associated_after_conditioning": bool(lo > 0.5 or hi < 0.5),
                "bootstrap": STRATIFIED_CELLS.replace(" by eTIV tertile cells", ""),
                "n_undefined_draws": int(np.isnan(draws).sum()),
                "family": UNDECLARED, "n_bootstrap": n_boot,
                "code_fingerprint": coh.fingerprint.get(arm, ""),
                "note": "conditioning on sex is observational. The residual is not a direct "
                        "effect of head size. " + ASSOCIATION_ONLY})

        # -- (3) the sex gap inside eTIV tertiles, among true CDR 0 ------------
        base = have & (y == 0)
        b_idx = np.flatnonzero(base)
        if len(b_idx) < 12:
            warn(f"{coh.name}: only {len(b_idx)} CDR 0 participants carry eTIV and sex; "
                 f"the tertile analysis is skipped")
            continue
        tert = tertiles(etiv[b_idx])
        cell = tert * 2 + male[b_idx].astype(int)
        cell_counts = {int(c): int((cell == c).sum()) for c in np.unique(cell)}
        thin = [c for c, k in cell_counts.items() if k < MIN_CELL]
        print(f"        CDR 0 with eTIV and sex: {len(b_idx)}. sex by tertile cell sizes "
              f"(tertile, male) {cell_counts}")
        if thin:
            print(f"  !!    {len(thin)} of {len(cell_counts)} cells hold fewer than "
                  f"{MIN_CELL} participants. The within-tertile differences below are real "
                  f"estimates with very wide intervals, not precise ones.")
        b_cell_idx = stratified_participant_bootstrap(cell, n_boot, "a42:tertile", BASE_SEED,
                                                     cohort=coh.name)
        Wb = multiplicity(b_cell_idx, len(b_idx))

        for arm in arms:
            pred = coh.part[arm]["pred"].to_numpy().astype(int)
            impaired = (pred[b_idx] > 0).astype(float)
            m_flag = male[b_idx] == 1
            f_flag = ~m_flag

            def gap(sel: np.ndarray) -> Tuple[float, np.ndarray, int, int]:
                mm = sel & m_flag
                ff = sel & f_flag
                nm, nf = int(mm.sum()), int(ff.sum())
                if nm == 0 or nf == 0:
                    return float("nan"), np.full(n_boot, np.nan), nm, nf
                pt = float(impaired[mm].mean() - impaired[ff].mean())
                dm = (Wb[:, mm] @ impaired[mm]) / np.maximum(Wb[:, mm].sum(axis=1), 1e-12)
                df = (Wb[:, ff] @ impaired[ff]) / np.maximum(Wb[:, ff].sum(axis=1), 1e-12)
                return pt, dm - df, nm, nf

            everything = np.ones(len(b_idx), dtype=bool)
            pt_all, d_all, nm_all, nf_all = gap(everything)
            lo, hi = percentile_interval(d_all)
            st = bootstrap_p(d_all)
            rows.append({
                "kind": "sex_gap", "cohort": coh.name, "arm": arm,
                "group": "all eTIV, unadjusted", "etiv_tertile": "all",
                "n_male": nm_all, "n_female": nf_all,
                "value": pt_all, "ci_low": lo, "ci_high": hi, "ci_level": 0.95,
                "p_bootstrap": st["p_bootstrap"],
                "p_at_resolution_floor": st["p_at_resolution_floor"],
                "excludes_zero": bool(lo > 0 or hi < 0),
                "bootstrap": STRATIFIED_CELLS, "n_bootstrap": n_boot,
                "family": UNDECLARED, "code_fingerprint": coh.fingerprint.get(arm, ""),
                "note": "share of true CDR 0 participants predicted impaired, male minus "
                        "female. a34's quantity, recomputed here so the adjusted and "
                        "unadjusted numbers come off one code path"})

            pooled_num = 0.0
            pooled_den = 0.0
            pooled_num_d = np.zeros(n_boot)
            pooled_den_d = np.zeros(n_boot)
            for t in range(N_TERTILES):
                sel = tert == t
                pt, d, nm, nf = gap(sel)
                lo_t, hi_t = percentile_interval(d)
                st_t = bootstrap_p(d)
                rows.append({
                    "kind": "sex_gap", "cohort": coh.name, "arm": arm,
                    "group": f"eTIV tertile {t + 1} of {N_TERTILES}",
                    "etiv_tertile": t + 1,
                    "etiv_low": float(etiv[b_idx][sel].min()) if sel.any() else np.nan,
                    "etiv_high": float(etiv[b_idx][sel].max()) if sel.any() else np.nan,
                    "n_male": nm, "n_female": nf,
                    "value": pt, "ci_low": lo_t, "ci_high": hi_t, "ci_level": 0.95,
                    "p_bootstrap": st_t["p_bootstrap"],
                    "p_at_resolution_floor": st_t["p_at_resolution_floor"],
                    "excludes_zero": bool(lo_t > 0 or hi_t < 0),
                    "cell_below_minimum": bool(min(nm, nf) < MIN_CELL),
                    "bootstrap": STRATIFIED_CELLS, "n_bootstrap": n_boot,
                    "family": UNDECLARED, "code_fingerprint": coh.fingerprint.get(arm, ""),
                    "note": "within one band of head size, so a sex gap surviving here is "
                            "not head size relabelled. " + ASSOCIATION_ONLY})
                if np.isfinite(pt):
                    w = float(min(nm, nf))
                    pooled_num += pt * w
                    pooled_den += w
                    ok = np.isfinite(d)
                    pooled_num_d[ok] += d[ok] * w
                    pooled_den_d[ok] += w
            adj = (pooled_num / pooled_den) if pooled_den > 0 else float("nan")
            d_adj = np.divide(pooled_num_d, pooled_den_d,
                              out=np.full(n_boot, np.nan), where=pooled_den_d > 0)
            lo_a, hi_a = percentile_interval(d_adj)
            st_a = bootstrap_p(d_adj)
            rows.append({
                "kind": "sex_gap", "cohort": coh.name, "arm": arm,
                "group": "pooled within eTIV tertiles, head size adjusted",
                "etiv_tertile": "pooled",
                "value": adj, "ci_low": lo_a, "ci_high": hi_a, "ci_level": 0.95,
                "p_bootstrap": st_a["p_bootstrap"],
                "p_at_resolution_floor": st_a["p_at_resolution_floor"],
                "excludes_zero": bool(lo_a > 0 or hi_a < 0),
                "unadjusted_value": pt_all,
                "attenuation": pt_all - adj,
                "bootstrap": STRATIFIED_CELLS, "n_bootstrap": n_boot,
                "family": UNDECLARED, "code_fingerprint": coh.fingerprint.get(arm, ""),
                "note": "weighted average of the three within-tertile gaps, weights the "
                        "smaller sex count in each tertile. Compare against the unadjusted "
                        "gap on the same row"})

        gaps = [r for r in rows if r["kind"] == "sex_gap" and r["cohort"] == coh.name]
        un = [r for r in gaps if r["etiv_tertile"] == "all"]
        po = [r for r in gaps if r["etiv_tertile"] == "pooled"]
        headline[coh.name] = {
            "n_arms": len(arms), "n_cdr0": len(b_idx),
            "unadj_min": min(r["value"] for r in un), "unadj_max": max(r["value"] for r in un),
            "adj_min": min(r["value"] for r in po), "adj_max": max(r["value"] for r in po),
            "adj_sig": sum(r["excludes_zero"] for r in po), "n_po": len(po),
            "sex_auc": float(sex_auc)}

    table = pd.DataFrame(rows)
    lead = ["kind", "cohort", "arm", "group", "etiv_tertile", "value", "ci_low", "ci_high",
            "ci_level", "excludes_zero", "bootstrap", "family"]
    table = table[[c for c in lead if c in table.columns]
                  + [c for c in table.columns if c not in lead]]
    save_table(table, TABLE, float_fmt="%.6f")

    # -- one screen ----------------------------------------------------------------
    elapsed = time.time() - t_start
    print("\n" + "=" * 104)
    print("a42  IS HEAD SIZE A SHORTCUT, AND IS THE SEX GAP HEAD SIZE IN DISGUISE?")
    print("=" * 104)
    print(f"  {ASSOCIATION_ONLY}.")
    for coh in cohorts:
        if coh.name not in headline:
            continue
        h = headline[coh.name]
        t = table[table["cohort"] == coh.name]
        print(f"\n  == cohort: {coh.name}, {h['n_arms']} arms, {h['n_cdr0']} CDR 0 "
              f"participants with eTIV and sex")
        print(f"     eTIV separates men from women at AUC {h['sex_auc']:.3f}, so eTIV and "
              f"sex are not independent questions here.")
        print(f"\n     (1) eTIV ALONE, no model:")
        named = [name for name, _h, _l in PAIRWISE]
        for _, r in t[(t["kind"] == "etiv_auc")
                      & t["group"].isin(named)].iterrows():
            flag = "  <-- separates" if bool(r.get("separates")) else ""
            print(f"        {r['group']:<34}{r['value']:>8.3f}"
                  f"{f'[{r.ci_low:.3f}, {r.ci_high:.3f}]':>22}{flag}")
        ev = t[t["kind"] == "error_vs_etiv"]
        print(f"\n     (2) ARM ERRORS AGAINST eTIV, CONDITIONED ON SEX:")
        print(f"        {'arm':<16}{'conditional AUC':>26}{'marginal':>11}"
              f"{'female':>9}{'male':>9}")
        for _, r in ev.iterrows():
            print(f"        {r['arm']:<16}"
                  f"{f'{r.value:.3f} [{r.ci_low:.3f}, {r.ci_high:.3f}]':>26}"
                  f"{r['marginal_auc_not_conditioned']:>11.3f}"
                  f"{r['auc_female_stratum']:>9.3f}{r['auc_male_stratum']:>9.3f}")
        print(f"        {int(ev['associated_after_conditioning'].sum())} of {len(ev)} arms "
              f"have an interval excluding 0.5 after conditioning on sex.")
        sg = t[t["kind"] == "sex_gap"]
        print(f"\n     (3) MALE MINUS FEMALE RISK DIFFERENCE AMONG TRUE CDR 0, BY eTIV "
              f"TERTILE:")
        tert_labels = ["all"] + [str(i + 1) for i in range(N_TERTILES)] + ["pooled"]
        print(f"        {'arm':<16}" + "".join(
            f"{('unadj' if x == 'all' else 'tertile ' + x if x.isdigit() else 'adjusted'):>20}"
            for x in tert_labels))
        for arm in sorted(sg["arm"].unique(), key=lambda a: ARM_ORDER.index(a)
                          if a in ARM_ORDER else 99):
            cells = []
            for x in tert_labels:
                r = sg[(sg["arm"] == arm) & (sg["etiv_tertile"].astype(str) == x)]
                if r.empty:
                    cells.append(f"{'':>20}")
                else:
                    rr = r.iloc[0]
                    cells.append(f"{rr['value']:>+7.3f} [{rr['ci_low']:+.2f},"
                                 f"{rr['ci_high']:+.2f}]".rjust(20))
            print(f"        {arm:<16}" + "".join(cells))
        print(f"        unadjusted gap ranges {h['unadj_min']:+.3f} to {h['unadj_max']:+.3f} "
              f"across arms.")
        print(f"        head size adjusted gap ranges {h['adj_min']:+.3f} to "
              f"{h['adj_max']:+.3f}, interval excludes zero for "
              f"{h['adj_sig']} of {h['n_po']} arms.")
        survives = h["adj_min"] > 0 and h["adj_sig"] > 0
        print(f"        VERDICT: the sex gap {'SURVIVES' if survives else 'does NOT clearly survive'} "
              f"inside bands of head size, so head size "
              f"{'does not explain it away' if survives else 'may account for part of it'}.")

    print(f"\n  HOW TO WRITE THIS UP. If eTIV separates the classes on its own, it joins age "
          f"and sex in the")
    print(f"  limitations as a quantity a network could read instead of pathology. If the "
          f"sex gap survives inside")
    print(f"  eTIV bands, the sex finding is not a head size artefact and should be reported "
          f"as its own result.")
    print(f"  Neither sentence may say a network used either quantity. {ASSOCIATION_ONLY}.")
    print(f"\n  EVERY comparison here is UNDECLARED. No registry family covers eTIV, a sex "
          f"disparity or an error")
    print(f"  association, no p-value here is corrected, and every interval is a nominal 95 "
          f"percent one.")
    if args.quick:
        print(f"\n  QUICK MODE: {n_boot} resamples. Re-run without --quick before quoting.")
    print(f"\n  outputs in {OUT}")
    print(f"  wall time {elapsed:.1f} s")

    print("\n  PASTE THIS INTO analysis/REGISTRY_CHANGELOG.md (nothing was written for you):")
    print(changelog_entry(
        EXPERIMENT_ID,
        "eTIV alone as a separator of the CDR classes on both cohorts, each arm's errors "
        "against eTIV conditioned on sex, and the male minus female risk difference among "
        "true CDR 0 participants recomputed within eTIV tertiles",
        "midrank AUC, conditional Mann-Whitney AUC, risk difference",
        "participant"))
    return 0


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        raise SystemExit(main())
