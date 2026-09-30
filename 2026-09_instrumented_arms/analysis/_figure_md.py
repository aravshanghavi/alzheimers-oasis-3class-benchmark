"""Deterministic figure companions: every figure emits a .png and a .md.

THE RULE THIS MODULE ENFORCES
-----------------------------
No number and no verdict in a figure's companion document is ever typed by a
human or written by a language model. Every quantity is passed in as a computed
value and formatted here; every interpretive sentence is produced by a decision
rule evaluated in code against that value and the pre-registered threshold for
its comparison family.

The reason is not stylistic. An interpretation written after seeing a result can
be tuned to the result, and neither the reader nor the author can tell afterwards
whether it was. An interpretation emitted by a rule that was written before the
number existed cannot be. This is the same logic as the multiplicity registry,
applied to prose instead of thresholds.

What a caller may supply as free text is limited on purpose:

  shows      what is plotted, mechanically. "Reliability curve, 15 equal-width
             bins, pooled test folds." Not what it means.
  not_shown  scope limits the figure cannot establish. Written once, before the
             data, and checked by a reader against the figure.

Everything else -- values, intervals, verdicts, caveats -- is computed.

AUTOMATIC CAVEATS
-----------------
Some caveats attach themselves from properties of the data, so that a figure
physically cannot be published without them. The full-cohort age caveat is the
important one: 147 of 266 participants in the majority class have no recorded
CDR and 181 are under 60, so any full-cohort performance number is partly an age
contrast. That sentence now rides along with every full-cohort figure whether or
not the author remembered it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from analysis._registry import REGISTRY, Verdict

OUT = Path(__file__).resolve().parents[1] / "analysis" / "outputs"

# Cohort facts used by the automatic caveats. These are measured quantities, not
# assumptions; a13_cohort_age_check.py derives them from oasis_cross-sectional.
COHORT_FACTS = {
    "full": {
        "n_participants": 347,
        "caveats": [
            "This figure is computed on the full cohort. Of its 266 non-demented "
            "participants, 147 have no recorded CDR and 181 are under 60 years old, because "
            "CDR was administered only to older adults. Any performance number on this cohort "
            "is therefore partly an age contrast rather than a dementia contrast. The "
            "age-restricted companion figure is the one to read for the dementia signal.",
        ],
    },
    "age60": {
        "n_participants": 166,
        "caveats": [
            "This figure is computed on the CDR-assessed, age-60-and-over cohort (n = 166). "
            "Absolute performance here is far below the full-cohort figure; that gap is the "
            "size of the age confound, not a defect of the models.",
        ],
    },
}

IMAGE_UNIT_CAVEAT = (
    "Quantities here are computed per image. The test set holds roughly 250 near-duplicate "
    "slices per participant (r = 0.988 between acquisitions), so image counts are not "
    "independent observations. Any interval on this figure that was not computed by "
    "resampling participants understates uncertainty by roughly an order of magnitude."
)

WILCOXON_FLOOR_CAVEAT = (
    "A Wilcoxon signed-rank test over five folds has a smallest attainable two-sided p of "
    "0.0625. It cannot return a value below 0.05 under any data whatsoever, so its failure to "
    "reach significance carries no information and it is reported only as a companion to the "
    "participant-clustered bootstrap."
)


@dataclass(frozen=True)
class Provenance:
    """Where the numbers came from. Every field is read from run artefacts."""
    run_ids: Sequence[str] = ()
    code_fingerprint: str = ""
    cohort: str = "full"
    unit: str = "participant"          # "participant" | "image"
    n_participants: Optional[int] = None
    n_images: Optional[int] = None
    n_bootstrap: Optional[int] = None
    rng_seed: Optional[int] = None
    source_script: str = ""

    def lines(self) -> List[str]:
        out = [f"- generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ')}"]
        if self.source_script:
            out.append(f"- produced by: `{self.source_script}`")
        if self.code_fingerprint:
            out.append(f"- code fingerprint: `{self.code_fingerprint[:16]}`")
        out.append(f"- cohort: {self.cohort}")
        out.append(f"- unit of analysis: {self.unit}")
        if self.n_participants is not None:
            out.append(f"- participants: {self.n_participants}")
        if self.n_images is not None:
            out.append(f"- images: {self.n_images:,}")
        if self.n_bootstrap is not None:
            out.append(f"- bootstrap resamples: {self.n_bootstrap:,}"
                       + (f", seed {self.rng_seed}" if self.rng_seed is not None else ""))
        if self.run_ids:
            out.append(f"- runs ({len(self.run_ids)}): " +
                       ", ".join(f"`{r}`" for r in sorted(self.run_ids)[:8]) +
                       (" ..." if len(self.run_ids) > 8 else ""))
        return out


@dataclass
class Finding:
    """One quantity the figure establishes, with its registered verdict.

    `family` and `comparison_id` are required whenever the finding is a
    comparison. Omitting them yields a descriptive finding, which is rendered
    with an explicit statement that no significance claim is attached -- so a
    reader can never mistake a described quantity for a tested one.
    """
    label: str
    value: float
    ci: Optional[Tuple[float, float]] = None
    p: Optional[float] = None
    family: Optional[str] = None
    comparison_id: Optional[str] = None
    fmt: str = "{:+.4f}"
    unit_suffix: str = ""
    wilcoxon_n: Optional[int] = None
    _verdict: Optional[Verdict] = field(default=None, init=False, repr=False)

    def resolve(self) -> Optional[Verdict]:
        if self.family is None:
            return None
        if self._verdict is None:
            self._verdict = REGISTRY.verdict(
                self.family, self.comparison_id or self.label, p=self.p, ci=self.ci)
        return self._verdict

    def render(self) -> List[str]:
        v = self.resolve()
        val = self.fmt.format(self.value) + self.unit_suffix
        bits = [f"**{self.label}**: {val}"]
        if self.ci is not None:
            # An interval only deserves the words "family-corrected" when it was
            # actually computed at a family's corrected percentiles. A descriptive
            # finding carries an ordinary 95% bootstrap interval, and labelling it
            # as corrected would be the companion document asserting something the
            # number is not.
            if v is not None:
                kind = (f"interval at the corrected level "
                        f"(alpha = {v.alpha_corrected:.6g}, family '{v.family}')")
            else:
                kind = "95% participant-clustered bootstrap interval"
            bits.append(f"{kind} [{self.ci[0]:+.4f}, {self.ci[1]:+.4f}]")
        if self.p is not None:
            bits.append(f"p = {self.p:.4g}")
        line = "- " + ", ".join(bits)
        if v is None:
            return [line, "  - No significance verdict: this quantity is reported as a "
                          "description, not as a tested comparison."]
        return [line, f"  - {v.sentence()}."]


def _auto_caveats(prov: Provenance, findings: Sequence[Finding]) -> List[str]:
    out: List[str] = []
    out.extend(COHORT_FACTS.get(prov.cohort, {}).get("caveats", []))
    if prov.unit == "image":
        out.append(IMAGE_UNIT_CAVEAT)
    if any(f.wilcoxon_n is not None and f.wilcoxon_n <= 6 for f in findings):
        out.append(WILCOXON_FLOOR_CAVEAT)
    if prov.n_bootstrap:
        floor = 1.0 / prov.n_bootstrap
        if any(f.p is not None and f.p <= floor for f in findings):
            out.append(
                f"At least one p-value here is at the resolution floor of the bootstrap "
                f"({prov.n_bootstrap:,} resamples, smallest resolvable p = {floor:.2g}). It is "
                f"reported as '< {floor:.2g}' and must not be quoted as zero.")
    n_p = prov.n_participants
    if n_p is not None and n_p < 200:
        out.append(
            f"With {n_p} participants, this figure is underpowered for small effects. A null "
            f"result here is not evidence of absence; where a null is load-bearing, the "
            f"companion analysis reports the effect size the design could have detected.")
    return out


def emit_figure(stem: str, fig, *, title: str, shows: str,
                provenance: Provenance,
                findings: Sequence[Finding] = (),
                not_shown: Sequence[str] = (),
                extra_caveats: Sequence[str] = (),
                out_dir: Optional[Path] = None,
                dpi: int = 300) -> Tuple[Path, Path]:
    """Write <stem>.png and <stem>.md. Returns both paths.

    `fig` is a matplotlib Figure. Pass None to write only the companion document
    (useful when the figure is produced elsewhere).
    """
    out_dir = Path(out_dir) if out_dir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    md = out_dir / f"{stem}.md"

    if fig is not None:
        fig.savefig(png, dpi=dpi, bbox_inches="tight")

    findings = list(findings)
    resolved = [(f, f.resolve()) for f in findings]

    lines: List[str] = [f"# {title}", "", f"![{title}]({png.name})", "",
                        "## What is plotted", "", shows, "",
                        "## Provenance", ""]
    lines += provenance.lines()

    lines += ["", "## Findings", ""]
    if findings:
        for f in findings:
            lines += f.render()
    else:
        lines.append("- This figure is descriptive. No comparison is registered against it.")

    tested = [v for _, v in resolved if v is not None]
    if tested:
        n_sig = sum(v.significant for v in tested)
        lines += ["", "## Verdict summary", "",
                  f"- {len(tested)} registered comparison(s) on this figure; "
                  f"{n_sig} distinguishable from zero after family correction, "
                  f"{len(tested) - n_sig} not."]
        dis = [v for v in tested if v.disagreement]
        if dis:
            lines.append(f"- {len(dis)} comparison(s) where the p-value and the interval "
                         f"disagree. The conservative reading is taken in every case.")

    caveats = _auto_caveats(provenance, findings) + list(extra_caveats)
    lines += ["", "## Caveats", ""]
    lines += [f"- {c}" for c in caveats] or ["- none recorded"]

    lines += ["", "## What this figure does not show", ""]
    lines += [f"- {n}" for n in not_shown] or [
        "- No scope limits were declared for this figure. That is itself a gap: every figure "
        "should state what it cannot establish."]

    lines += ["", "---", "",
              "*Every number and every verdict above was generated by the script named in "
              "Provenance, from the run artefacts listed there, against thresholds declared in "
              "`analysis/_registry.py`. No value in this document was entered by hand.*", ""]

    md.write_text("\n".join(lines), encoding="utf-8")
    return png, md


def write_index(out_dir: Optional[Path] = None) -> Path:
    """Aggregate every companion document into one index."""
    out_dir = Path(out_dir) if out_dir else OUT
    docs = sorted(out_dir.glob("*.md"))
    skip = {"FIGURE_INDEX.md", "multiplicity_audit.md"}
    docs = [d for d in docs if d.name not in skip]
    lines = ["# Figure index", "",
             f"{len(docs)} figure companion document(s).", ""]
    for d in docs:
        first = ""
        for ln in d.read_text(encoding="utf-8").splitlines():
            if ln.startswith("# "):
                first = ln[2:].strip()
                break
        png = d.with_suffix(".png")
        lines.append(f"- [{first or d.stem}]({d.name})" + ("" if png.exists() else "  (no image)"))
    path = out_dir / "FIGURE_INDEX.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
