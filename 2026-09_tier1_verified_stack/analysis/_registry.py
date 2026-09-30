"""Pre-registered multiplicity registry.

WHY THIS EXISTS
---------------
The audit of the 2026-08 analysis layer found ~100 significance flags emitted
across a16 section 5, a17 and a18 with no multiplicity correction anywhere in
code. a03 and a10 mention Bonferroni in their prose notes and never compute it.
That is the single largest statistical exposure in the paper: 42 pairwise
comparisons per metric per cohort, each tested at a nominal 0.05, will produce
roughly two "significant" results per metric under a true null.

Adding more analyses, which is the current direction of travel, makes this
strictly worse. More analyses on the same 347 participants do not add
information; they add opportunities for a false positive. The determinism of a
computation does not protect against the non-determinism of which computation
gets reported.

WHAT THIS MODULE ENFORCES
-------------------------
1. Every comparison that receives a significance verdict must belong to a
   family DECLARED IN THIS FILE, before the data is touched. Asking for a
   verdict on an unregistered family raises. You cannot add a comparison after
   seeing its result and have it silently acquire a verdict.

2. Family size is DECLARED, not counted at runtime. If the size were counted,
   adding a comparison would silently move every threshold in that family, and
   dropping an inconvenient one would silently loosen it. The declared count is
   the pre-registration. Runtime drift from it is reported by audit() and is a
   protocol violation, not a correction.

3. There is exactly one source of truth for a corrected threshold. Scripts do
   not compute alpha/n themselves.

4. Bootstrap intervals get percentiles at the CORRECTED level. A 95% interval
   compared against a Bonferroni-corrected p threshold is an inconsistent test:
   the interval and the p-value must be corrected together or the figure and
   the table will disagree.

HOW TO CHANGE A DECLARATION
---------------------------
Editing a family after its results have been seen defeats the purpose of the
file. If a declaration genuinely needs to change, change it, and record the
change and its reason in REGISTRY_CHANGELOG.md in this directory. A reviewer
who asks "was this pre-specified?" is entitled to an auditable answer, and
"the file says so" is only an answer if the file has a history.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

OUT = Path(__file__).resolve().parents[1] / "analysis" / "outputs"


# ---------------------------------------------------------------------------
# Declarations
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Family:
    """One pre-registered family of comparisons.

    n_declared is the number of comparisons this family will contain, fixed in
    advance. alpha is the family-wise error rate, not the per-comparison rate.
    """
    key: str
    description: str
    n_declared: int
    alpha: float = 0.05
    method: str = "bonferroni"
    primary: bool = False
    note: str = ""

    def alpha_corrected(self) -> float:
        if self.method == "none":
            return self.alpha
        if self.method == "bonferroni":
            return self.alpha / max(self.n_declared, 1)
        raise ValueError(f"unknown correction method {self.method!r} for family {self.key!r}")

    def ci_percentiles(self) -> Tuple[float, float]:
        """Percentiles for a bootstrap interval at the corrected level."""
        a = self.alpha_corrected()
        return (100.0 * a / 2.0, 100.0 * (1.0 - a / 2.0))


# The declared families. Sizes are counted from the analysis design, not from
# whatever the code happens to emit.
FAMILIES: Dict[str, Family] = {f.key: f for f in [

    # -- Objective comparisons -------------------------------------------------
    # Seven objectives. The paper's comparisons of interest are each arm against
    # FA-FL, so six per metric, not the full 21 pairwise.
    Family("objective_ece", "Each objective against FA-FL on expected calibration error.",
           n_declared=6, primary=True,
           note="The surviving FA-FL claim lives here. Declared at the image level AND the "
                "participant level separately; see objective_ece_participant."),
    Family("objective_ece_participant", "As objective_ece, at the participant level.",
           n_declared=6, primary=True,
           note="The paper declares participant as the primary unit. FA-FL ranks fourth here. "
                "Reporting only the image-level family would be unit-shopping."),
    Family("objective_accuracy", "Each objective against FA-FL on accuracy.", n_declared=6),
    Family("objective_macro_f1", "Each objective against FA-FL on macro-F1.", n_declared=6),

    # -- The gamma control ------------------------------------------------------
    # This is the family that decides whether FA-FL is distinguishable from focal
    # loss at a well-chosen gamma. Three comparisons, so the threshold is 0.0167.
    Family("gamma_control_ece",
           "FA-FL against focal loss at gamma 1.37, 2.0 and 3.0 on ECE.",
           n_declared=3, primary=True,
           note="FA-FL's frequency-weighted effective gamma is 1.88, adjacent to canonical 2.0. "
                "This family exists because a reviewer will derive that number and ask whether "
                "FA-FL is focal loss at a better-chosen gamma. The FA-FL vs focal-2.0 margin "
                "reported in the 2026-08 documents is p = 0.0175 against this family's corrected "
                "threshold of 0.016667. It does not pass. That result is the reason this registry "
                "enforces thresholds in code rather than in prose."),

    # -- Baselines --------------------------------------------------------------
    Family("arm_vs_constant_accuracy",
           "Every model against a constant majority-class predictor, accuracy.", n_declared=11,
           primary=True),
    Family("arm_vs_constant_macro_f1",
           "Every model against a constant majority-class predictor, macro-F1.", n_declared=11,
           primary=True),
    Family("arm_vs_constant_qwk",
           "Every model against a constant majority-class predictor, QWK.", n_declared=11),
    Family("arm_vs_constant_auroc",
           "Every model against a constant majority-class predictor, macro-AUROC.", n_declared=11),
    Family("arm_vs_metadata",
           "Every CNN arm against the age+sex+eTIV+nWBV metadata regression, all metrics.",
           n_declared=28),

    # -- The increment result ---------------------------------------------------
    # The strongest finding in the corpus. Seven arms, one direction each.
    Family("cnn_increment_over_nwbv",
           "Does adding CNN log-odds to nWBV improve on nWBV alone? One test per arm per metric.",
           n_declared=28, primary=True,
           note="Paired with nwbv_increment_over_cnn. The asymmetry between these two families is "
                "the claim: if 7/7 arms improve when the scalar is added and 0/7 scalars improve "
                "when an arm is added, the network's information is a subset of the scalar's. "
                "Both directions must be declared and reported, or the asymmetry is unfalsifiable."),
    Family("nwbv_increment_over_cnn",
           "Does adding nWBV to CNN log-odds improve on the CNN alone? One test per arm per metric.",
           n_declared=28, primary=True),

    # -- Leakage ----------------------------------------------------------------
    Family("leakage_contrasts",
           "Naive contaminated-vs-clean contrast, the negative control, and the "
           "difference-in-differences.",
           n_declared=3, primary=True,
           note="The DiD is underpowered by construction (treated headroom 0.58pp). Its null "
                "result must be reported alongside the arithmetic bound, not as evidence of "
                "absence on its own."),

    # -- Secondary --------------------------------------------------------------
    Family("multislice", "2.5D multi-slice input against the 2D baseline.", n_declared=4,
           note="Reported as a bounded null."),
    Family("age_cohort_shift",
           "Full cohort against the CDR-assessed age-60+ cohort, per arm, on accuracy.",
           n_declared=7),
]}


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Verdict:
    family: str
    comparison_id: str
    p: Optional[float]
    ci: Optional[Tuple[float, float]]
    alpha_corrected: float
    n_declared: int
    method: str
    significant: bool
    basis: str          # "p", "ci", or "p+ci"
    disagreement: bool  # p and CI reach different conclusions

    def sentence(self) -> str:
        """The verdict in words. Generated here so no caller writes it by hand."""
        verb = "is" if self.significant else "is NOT"
        s = (f"{verb} distinguishable from zero at the family-corrected level "
             f"(alpha = {self.alpha_corrected:.6g}, {self.method}, "
             f"{self.n_declared} declared comparisons in family '{self.family}')")
        if self.disagreement:
            s += (". NOTE: the p-value and the interval disagree at this threshold; both are "
                  "reported above and the disagreement is not resolved in the model's favour")
        return s


class Registry:
    def __init__(self, families: Dict[str, Family]):
        self._families = families
        self._seen: Dict[str, List[str]] = {k: [] for k in families}
        self._verdicts: List[Verdict] = []

    # -- lookups ------------------------------------------------------------
    def family(self, key: str) -> Family:
        if key not in self._families:
            raise KeyError(
                f"'{key}' is not a declared comparison family. Declare it in "
                f"analysis/_registry.py BEFORE computing it, or this comparison gets no "
                f"significance verdict. Declared families: {sorted(self._families)}")
        return self._families[key]

    def alpha_corrected(self, key: str) -> float:
        return self.family(key).alpha_corrected()

    def ci_percentiles(self, key: str) -> Tuple[float, float]:
        """Percentiles for a bootstrap interval at this family's corrected level.

        Use these instead of (2.5, 97.5) whenever the interval will be used to
        make a significance claim, so that the interval and the p-value test the
        same hypothesis at the same level.
        """
        return self.family(key).ci_percentiles()

    # -- the main entry point -----------------------------------------------
    def verdict(self, family: str, comparison_id: str, *,
                p: Optional[float] = None,
                ci: Optional[Tuple[float, float]] = None) -> Verdict:
        """Record a comparison and return its family-corrected verdict.

        At least one of p or ci must be given. When both are given and they
        disagree, the verdict takes the CONSERVATIVE branch (not significant)
        and flags the disagreement, because a disagreement between the interval
        and the p-value is exactly what a reviewer looks for and hiding it is
        how it gets found.
        """
        fam = self.family(family)
        if p is None and ci is None:
            raise ValueError("verdict() needs a p-value, an interval, or both")

        a = fam.alpha_corrected()
        sig_p = (p is not None and p < a)
        sig_ci = (ci is not None and (ci[0] > 0 or ci[1] < 0))

        if p is not None and ci is not None:
            basis, disagreement = "p+ci", (sig_p != sig_ci)
            significant = sig_p and sig_ci          # conservative
        elif p is not None:
            basis, disagreement, significant = "p", False, sig_p
        else:
            basis, disagreement, significant = "ci", False, sig_ci

        if comparison_id in self._seen[family]:
            raise ValueError(
                f"comparison '{comparison_id}' registered twice in family '{family}'. "
                f"Duplicate registration corrupts the family count.")
        self._seen[family].append(comparison_id)

        v = Verdict(family=family, comparison_id=comparison_id, p=p, ci=ci,
                    alpha_corrected=a, n_declared=fam.n_declared, method=fam.method,
                    significant=significant, basis=basis, disagreement=disagreement)
        self._verdicts.append(v)
        return v

    # -- audit ---------------------------------------------------------------
    def audit(self) -> Dict[str, Dict]:
        """Declared size against realised size, per family."""
        out = {}
        for key, fam in self._families.items():
            seen = len(self._seen[key])
            out[key] = {
                "declared": fam.n_declared, "seen": seen,
                "drift": seen - fam.n_declared,
                "status": ("ok" if seen == fam.n_declared
                           else "NOT RUN" if seen == 0
                           else "PROTOCOL VIOLATION"),
                "alpha_corrected": fam.alpha_corrected(),
                "primary": fam.primary,
            }
        return out

    def write_audit(self, path: Optional[Path] = None) -> Path:
        path = Path(path) if path else OUT / "multiplicity_audit.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        a = self.audit()
        lines = [
            "# Multiplicity audit",
            "",
            "Declared family sizes against the number of comparisons actually registered.",
            "A family showing PROTOCOL VIOLATION ran a different number of comparisons than",
            "it declared, which means its corrected threshold does not describe what was done.",
            "Fix the code or amend the declaration and record why in REGISTRY_CHANGELOG.md.",
            "",
            "| family | primary | declared | seen | status | corrected alpha |",
            "|---|---|---:|---:|---|---:|",
        ]
        for key in sorted(a, key=lambda k: (not a[k]["primary"], k)):
            r = a[key]
            lines.append(f"| {key} | {'yes' if r['primary'] else ''} | {r['declared']} | "
                         f"{r['seen']} | {r['status']} | {r['alpha_corrected']:.6g} |")

        viol = [k for k, r in a.items() if r["status"] == "PROTOCOL VIOLATION"]
        notrun = [k for k, r in a.items() if r["status"] == "NOT RUN"]
        lines += ["", f"**Protocol violations: {len(viol)}**"]
        if viol:
            lines += [f"- {k}: declared {a[k]['declared']}, ran {a[k]['seen']}" for k in viol]
        lines += ["", f"**Declared but not run this pass: {len(notrun)}**"]
        if notrun:
            lines += [f"- {k}" for k in notrun]

        if self._verdicts:
            lines += ["", "## Every registered comparison", "",
                      "| family | comparison | p | interval | corrected alpha | verdict |",
                      "|---|---|---:|---|---:|---|"]
            for v in self._verdicts:
                pstr = "" if v.p is None else f"{v.p:.4g}"
                cistr = "" if v.ci is None else f"[{v.ci[0]:+.4g}, {v.ci[1]:+.4g}]"
                verdict = "significant" if v.significant else "not significant"
                if v.disagreement:
                    verdict += " (p/CI disagree)"
                lines.append(f"| {v.family} | {v.comparison_id} | {pstr} | {cistr} | "
                             f"{v.alpha_corrected:.6g} | {verdict} |")

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        (path.parent / "multiplicity_audit.json").write_text(
            json.dumps({"families": a,
                        "verdicts": [v.__dict__ for v in self._verdicts]},
                       indent=2, default=str), encoding="utf-8")
        return path

    def reset(self) -> None:
        """Only for tests. Clears recorded comparisons."""
        self._seen = {k: [] for k in self._families}
        self._verdicts = []


REGISTRY = Registry(FAMILIES)


def declared_families_markdown() -> str:
    """The pre-registration, as a table for the supplement."""
    lines = ["| family | primary | n | alpha | corrected alpha | description |",
             "|---|---|---:|---:|---:|---|"]
    for key in sorted(FAMILIES, key=lambda k: (not FAMILIES[k].primary, k)):
        f = FAMILIES[key]
        lines.append(f"| {f.key} | {'yes' if f.primary else ''} | {f.n_declared} | {f.alpha} | "
                     f"{f.alpha_corrected():.6g} | {f.description} |")
    return "\n".join(lines) + "\n"
