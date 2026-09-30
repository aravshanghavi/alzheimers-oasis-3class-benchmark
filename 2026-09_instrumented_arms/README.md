# 2026-09_instrumented_arms

Train-matched age-60 arms, with per-class training diagnostics.

Nothing in `2026-08_srep_revision_participant_level`, `2026-09_srep_revision_round2`
or `2026-09_tier1_verified_stack` is touched by anything here.

---

## Why this folder exists

Two problems with the current evidence, one scientific and one mechanical.

**The scientific one.** Every age-restricted number reported so far came from a
model trained on the full cohort and merely evaluated on the clean subset. The
full cohort contains 266 non-demented participants, 147 of whom have no recorded
CDR and 181 of whom are under 60, because CDR was only administered to older
adults. A model trained on that cohort can reach high accuracy by learning an
age contrast, and evaluating it on older participants does not remove what it
learned. Training on the restricted cohort is the honest comparison and it has
never been run.

**The mechanical one.** The per-epoch log recorded aggregate loss and accuracy
only. The FA-FL claim is a claim about where the loss sits across classes, and
nothing in the pipeline measured that. So when two arms ended up in different
places, there was no way to say when they diverged or which class drove it.

---

## The pre-registered prediction

Written before any run in this folder existed. The registry entries are dated in
`analysis/REGISTRY_CHANGELOG.md`.

FA-FL sets a per-class focusing exponent from the class frequencies:

    gamma_k = gamma_base + lambda * alpha_k        gamma_base = 1.0, lambda = 7.0

with alpha the normalised inverse-frequency weight. So the exponents are a
function of the imbalance, and restricting the cohort changes the imbalance.

On the full cohort the participant split is roughly 266 / 58 / 23, the majority
class is about 77 percent, and the frequency-weighted effective exponent is
about 1.9, adjacent to canonical focal loss at gamma 2.0.

Restricting to CDR-assessed participants aged 60 and over leaves 85 / 58 / 23
and the majority falls to 51.2 percent. The restricted exponent is **measured,
not estimated**: fold 0's training split gives class counts
`[13420, 8540, 3355]`, so `alpha = [0.152174, 0.239130, 0.608696]`,
`gamma_t = [2.065, 2.674, 5.261]`, and the frequency-weighted effective exponent
is **2.694** image-weighted and **2.700** participant-weighted. An earlier
version of this file said "roughly 2.8"; that was an arithmetic error, corrected
from the run manifests and recorded with its reason in
`analysis/REGISTRY_CHANGELOG.md`. Every run recomputes the value from its own
training fold and writes it to its manifest as `criterion.effective_gamma`.

**The prediction.** If FA-FL is focal loss at its effective exponent, then the
constant-gamma arm it most closely resembles must MOVE with the cohort: nearest
to 2.0 on the full cohort, nearest to 2.69 here with 3.0 second. That migration
is the signature of the null hypothesis, and it is a prediction that can fail.

The ladder is 1.37 / 2.00 / **2.69** / 3.00. The 2.69 arm exists because without
it one outcome would be unreadable. If FA-FL came out distinguishable from 1.37,
2.00 and 3.00, there would be no way to separate "frequency adaptation does
something no scalar can" from "the right scalar was never tested" — the nearest
declared arm would have been 0.31 away from the value the whole argument turns
on. The 2.69 arm sits 0.004 away.

**What each outcome means.**

| outcome | reading |
| --- | --- |
| FA-FL tracks focal 2.69 here, having tracked 2.0 on the full cohort | The effective-exponent explanation holds. Report FA-FL's calibration descriptively, state that a single constant gamma reproduces it, drop the mechanistic claim. |
| FA-FL tracks neither, or keeps an advantage concentrated in the rare classes that no constant gamma reproduces | Frequency adaptation is doing something a scalar cannot. The mechanistic claim survives its hardest test. |
| Every arm collapses toward the majority-class baseline | The full-cohort headline was substantially an age contrast. That is the primary finding of this folder regardless of what FA-FL does, and it is the one a reviewer will care about most. |

A stated-in-advance power caveat: 166 participants with 23 Demented across five
folds leaves four or five Demented per test fold. Intervals will be wide and
minority-class F1 will be unstable. That is a property of the cohort, not of the
design, and it belongs in the paper rather than in a footnote discovered during
review.

---

## What is new in the code

| file | what it does |
| --- | --- |
| `src/cohort.py` | Restricts the scanned frame after scanning and before splitting, so folds, class weights and therefore FA-FL's exponents all follow the restricted cohort. Raises rather than silently dropping a participant missing from the metadata table. |
| `src/diagnostics.py` | Per-class accumulators for loss share, recall, confidence, true-class probability, prediction share and ECE, for train and validation, every epoch. |
| `src/train.py` | Accumulates the above from the same forward pass, under `no_grad` on detached outputs after the optimizer step. Training numerics are unchanged with diagnostics on or off. |
| `src/runner.py` | Writes the realized per-class exponents and the effective exponent into each manifest, so the mechanism number is structured data rather than a line in a log. |
| `analysis/a19_training_dynamics.py` | Turns the new columns into figures with deterministic companions. |
| `analysis/_registry.py` | Seven new `age60_*` families, declared before the runs; amendments logged in `analysis/REGISTRY_CHANGELOG.md`. |

`src/data.py` and `src/losses.py` are **byte-identical** to the 2026-08 versions
(md5 confirmed). Any difference in results is therefore attributable to the
cohort and the configuration, not to a quietly edited data or loss path.

### The mirror, and why it is checked rather than trusted

Every loss in `src/losses.py` reduces internally to a scalar, so per-sample
values cannot be recovered from what the criterion returns. Adding a `reduction`
argument to each loss would have edited the one file whose byte-for-byte
stability makes these results comparable with the previous runs. So
`src/diagnostics.py` re-derives the per-sample vector instead, and then checks
itself: on the first batch of every epoch it reduces its own vector exactly the
way the criterion does and compares against the criterion's own output. Two of
the five losses reduce by a weighted mean rather than a plain mean, and a naive
`.mean()` comparison for weighted cross-entropy gives 0.277 where the true value
is 2.010, so the check is not cosmetic. The maximum drift per run is written to
`training_curve.csv` as `mirror_abs_diff`, and `a19` refuses to plot any run
whose drift exceeds tolerance.

---

## How to run it

    cd C:\Users\aravs\PycharmProjects\Alzheimers\2026-09_instrumented_arms
    conda activate alzheimers
    run_all.bat

That is 40 runs: exp10 (five objectives x five folds), then exp11, exp12 and
exp13 (focal at 1.37, 2.00 and 2.69, five folds each). On roughly 46 percent of
the data it should be about nine GPU-hours. It is an overnight job. Logs land in
`console_logs/`, each run keeps its own `run.log`, and the terminal shows one
heartbeat line per run.

`run_all.bat --resume` skips anything already COMPLETE, which is what to use
after an interruption.

A two-minute wiring check first, if you want one:

    run_all.bat --smoke

Two epochs, 64-pixel images, one run per experiment. The numbers are meaningless
on purpose; it only proves the pipeline executes.

**This folder's default run order is exp10, exp11, exp12, exp13 and nothing
else.** The
older configs are kept so the comparison is reproducible from one checkout, but
running them here would duplicate work that is already COMPLETE in the 2026-08
tree under a different code fingerprint. To run one deliberately:

    python run_all.py --experiments exp01_main_sweep

### Then the analysis

    cd C:\Users\aravs\PycharmProjects\Alzheimers\2026-09_instrumented_arms
    conda activate alzheimers
    python analysis\a00_preflight.py
    python analysis\a19_training_dynamics.py
    python analysis\a20_age60_registered.py

Preflight has to say READY first. If it does not, nothing downstream is worth
reading.

One thing to watch: if `SREP_EXPERIMENTS_DIR` is set in that console from an
earlier session, the analysis will read that tree instead of this one. Preflight
now fails loudly when this repository has COMPLETE runs of its own and the
variable points somewhere else, because a table silently built from the wrong
runs is the worst failure mode available. To clear it:

    set SREP_EXPERIMENTS_DIR=

---

## What is still missing

Stated here rather than discovered later.

- **The permuted-label null has not been built.** Shuffling labels at the
  participant level and re-running one arm would bound how much apparent signal
  the pipeline produces from nothing. Until it exists, the floor is assumed
  rather than measured.
- **`a11_ece_robustness.py` still excludes exp07 and exp08.** Extending it to the
  full gamma ladder costs no GPU time and is the cheapest remaining test of the
  mechanism claim.
- **`_registry.py` and `_figure_md.py` are not yet wired into a03, a10, a14, a16,
  a17 and a18.** Those scripts still emit uncorrected significance flags. The
  machinery exists and is tested, and a19 and a20 use it, but the full-cohort
  scripts do not.
- **a01 through a18 cannot run in this folder at all.** They are written against
  the full-cohort experiment names: a01, a02, a03 and a06 call `runs_frame()`,
  whose default is exp01 and exp02, and the rest name exp03, exp04, exp05, exp07
  or exp08 explicitly. Only a00, a19 and a20 apply here. Do not run
  `analysis\run_analysis.bat` expecting the whole layer to work.
- **The 61-slice sub-volume arm has not been built.** The data is a Kaggle JPEG
  derivative with 61 axial slices per participant and no volumes, so true 3D
  would need OASIS-1 raw.
- **The age-confound decomposition is not yet registered.** `a21_age_confound.py`
  separates the full-cohort headline into a test-set composition effect and a
  training-set effect. The registered half needs the full-cohort arms rebuilt in
  this tree (25 runs, about 13 GPU-hours). Before spending that, run the
  half-hour equivalence check: train one full-cohort run here
  (`python experiments\exp01_main_sweep\run.py --only FA_FL:fold0`) and run
  `python tools\compare_equivalence.py`. If this tree reproduces the 2026-08 run
  exactly, the cross-tree comparison is justified by demonstration and the other
  24 runs are unnecessary.
- `_to_delete/` holds figures generated from fabricated data during a wiring
  check. Delete that folder.
