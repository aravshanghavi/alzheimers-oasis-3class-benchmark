# 2026-09 tier-1 verified stack

## How to run

This folder holds the analysis code but **not** the 55 run folders, which stay
in the 2026-08 tree. `experiments_dir.txt` at the repository root points at
them. Nothing is copied, and nothing is ever written into that tree — outputs
always land in this repository's `analysis\outputs\`.

```bat
cd C:\Users\aravs\PycharmProjects\Alzheimers\2026-09_tier1_verified_stack
conda activate alzheimers
python analysis\a00_preflight.py
```

The experiments location resolves from the first of these that answers:

1. the `SREP_EXPERIMENTS_DIR` environment variable — a one-off override
2. `experiments_dir.txt` — the persistent default, and what you normally want
3. `experiments\` inside this repository

The pointer file exists because `set VAR=...` in cmd.exe lasts only for that
console window. Depending on it meant every new terminal silently fell back to
the empty local `experiments\`, and preflight reported it as missing data rather
than as a missing variable. Edit the file to change the default; use the
environment variable only to aim a single run somewhere else.

Preflight is cheap and trains nothing. It reports the interpreter, the resolved
paths, how many COMPLETE runs it can see and in which experiments, whether they
share one code fingerprint, whether any duplicate specs are present, and the
corrected threshold for every primary comparison family. Run it first, every
time. If it says READY:

```bat
python analysis\run_analysis.py
```

Outputs land in `analysis\outputs\` of **this** folder. A few notes:

- `openpyxl` is required (it reads the OASIS metadata workbook). If preflight
  flags it, `pip install openpyxl`.
- `torch` is needed only by a04 Grad-CAM. Without it everything else still runs.
- **Bootstrap intervals from a15-a18 will differ from the August numbers in
  roughly the fourth decimal.** That is the determinism fix landing, not a
  regression: those scripts previously drew from one generator consumed in
  execution order, so their intervals moved whenever anything upstream changed.
  See `analysis/_determinism.py`.



This folder is **not** the Scientific Reports resubmission. That is
`2026-09_srep_revision_round2`, it is due 30 Sep 2026, and it should ship on the
evidence that already exists.

This folder is the rebuild: every number regenerated from raw predictions,
every comparison corrected for multiplicity, every figure carrying a
machine-generated companion document, aimed at a cross-cohort paper once ADNI
lands.

## Why two folders

The two goals are in conflict for the next two weeks and pretending otherwise
costs the deadline. ADNI is still downloading. There is no version of this where
external validation is acquired, verified and written to tier-1 standard by
30 Sep. So:

| | `2026-09_srep_revision_round2` | this folder |
|---|---|---|
| deadline | 30 Sep 2026 | none |
| scope | fix what R1/R2 asked, no new claims | full regeneration, cross-cohort |
| cohorts | OASIS-1 | OASIS-1 + ADNI |
| status of FA-FL | one hedged paragraph | one arm of seven |

The SR paper becomes the OASIS-specific methodological foundation the
cross-cohort paper cites. That is not salami-slicing as long as the second
paper's contribution is the generalisation rather than a re-report.

## What is new here

### `analysis/_registry.py` — pre-registered multiplicity

The audit of the 2026-08 stack found roughly 100 significance flags emitted
across a16 §5, a17 and a18 with no correction anywhere in code. a03 and a10
mention Bonferroni in prose and never compute it. At 42 comparisons per metric
per cohort tested at a nominal 0.05, roughly two "significant" results per metric
are expected under a true null.

The registry declares every family of comparisons, with its size, **before** the
data is touched. Asking for a verdict on an undeclared family raises. Family
sizes are declared, not counted at runtime, so adding a comparison cannot
silently move a threshold and dropping one cannot silently loosen it.

The family that matters most is `gamma_control_ece`: FA-FL against focal loss at
gamma 1.37, 2.0 and 3.0. Three comparisons, corrected threshold 0.016667. The
FA-FL vs focal-2.0 margin reported in the 2026-08 documents is p = 0.0175. It
does not pass. The 2026-08 documents knew this and wrote it in prose; the
registry enforces it in code, which is the difference between knowing and
not shipping it.

### `analysis/_figure_md.py` — deterministic figure companions

Every figure emits `<stem>.png` and `<stem>.md`. No number and no verdict in the
companion is typed by a human or written by a language model. Values are passed
in as computed quantities; interpretive sentences are produced by decision rules
evaluated against the pre-registered threshold for that comparison family.

Callers may supply free text in exactly two places: `shows` (what is plotted,
mechanically) and `not_shown` (declared scope limits). Everything else is
computed.

Caveats attach themselves from properties of the data, so a figure cannot be
published without them:

- any **full-cohort** figure carries the age-composition caveat (147 of 266
  non-demented participants have no recorded CDR, 181 are under 60)
- any **image-unit** figure carries the correlated-slice caveat (r = 0.988)
- any figure with a **five-fold Wilcoxon** carries the p-floor-of-0.0625 caveat
- any **p at the bootstrap resolution floor** is reported as `< 1/B`, never as zero
- any cohort under 200 participants carries a power caveat

### Statistical fixes applied

| fix | sites |
|---|---|
| bootstrap p floored at `1/n_boot` | `_common.paired_difference`, `a08`, `a09` |
| intervals computed at the family-corrected level | `_common.paired_difference(family=...)` |
| unpaired a08 contrast documented as conservative, not merely "wider" | `a08` |

## Still outstanding

Carried from the audit, not yet done in this folder:

1. `a11`–`a18` are unreachable from `run_analysis.py`; `a17`/`a18` additionally
   require `_cache_build.py` to have been run by hand. Roughly half the
   revision's evidence is produced by scripts nothing orchestrates and nothing
   verifies.
2. `a15`–`a18` draw from one module-level RNG consumed sequentially. Inserting a
   comparison or running a section alone changes every downstream interval in
   that file. Needs per-call seeding.
3. `a07_verify.py` covers only `a01`'s Table 2 and Table 4 — 25 of 55 runs. It
   never touches `a03`'s intervals, `a06`'s AUCs, or anything in `a11`–`a18`.
4. `a06` reports AUC and AP with no interval, computed over ~86,000 correlated
   images as if independent.
5. Full-sample median-fill and z-scoring leak into the cross-validated metadata
   baselines (`a16:215`, `a17:28`, `a18:14`). Conservative for the CNN claim,
   but a reviewer will name it.
6. Contradictions between the 2026-08 documents are unresolved: run counts
   (45/50/55/60), the 78.19% arm labelled Class-Balanced in one document and
   unweighted CE in another, a Spearman rho that reverses sign, two-step error
   rankings that reverse, contamination counts of 18 and 19.

## The claim this folder is being built to test

Not that FA-FL is better. The strongest result in the corpus is in
`a18_increment_and_structure.py`, which nothing currently runs:

> Seven of seven CNN arms improve when a single 2007 morphometric scalar (nWBV)
> is added. Zero of seven scalars improve when a CNN is added. Macro-AUROC
> 0.8833 for the scalar alone against 0.8828 for the best of seven networks.

On one cohort that is a curiosity about OASIS-1. On two independent cohorts it
is a claim about what structural-MRI dementia classification actually measures.
ADNI is what makes the difference.

**The objection to budget for:** the networks see 2D slices; nWBV is volumetric.
A reviewer will call the comparison rigged and demand a 3D arm. Either run one,
or narrow the claim to the 2D-slice paradigm that dominates this literature.
