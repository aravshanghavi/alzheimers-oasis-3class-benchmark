# 2026-09_permutation_null

Measures the pipeline's noise floor by permuting participant-level labels.

Nothing in `2026-08_srep_revision_participant_level`, `2026-09_srep_revision_round2`,
`2026-09_instrumented_arms` or `2026-09_tier1_verified_stack` is touched by anything here.
This is a separate tree on purpose: adding `src/permute.py` and editing
`src/runner.py` changes the code fingerprint, and the instrumented_arms tree is
the evidence base for the current results. It stays exactly as it is.

## Why this exists

The headline is now a null. On the CDR-assessed age-60 cohort no objective's
accuracy is distinguishable from a constant majority-class predictor, while
macro-F1 sits well above it. A null result needs a measured floor. Without one
the paper asserts how much apparent signal the pipeline produces from nothing
rather than measuring it, and that is the largest remaining hole a reviewer can
put a finger through.

Permuting the model's output predictions is a different and weaker test. It
holds the trained model fixed and cannot capture the optimism the pipeline
itself injects: five-fold CV, early stopping, checkpoint selection on validation
accuracy, and temperature fitting on that same validation split. This permutes
the labels the model learns from and then runs the complete unmodified pipeline
on top, so the resulting distribution absorbs all of it.

## The design, and the three decisions that matter

1. **Participant level, never image level.** Slices from one participant are
   near-duplicates (r = 0.988 between acquisitions). Permuting per image would
   hand the model contradictory labels for visually identical inputs, which is a
   different and much easier task to fail at.
2. **Train and val only. The test fold keeps its true labels**, so the metric is
   computed against reality and is directly comparable to the real runs. The
   code asserts this and refuses to continue if a test label moved.
3. **Train and val are permuted separately, each within itself**, so the
   PARTICIPANT-level class marginal of each split is preserved exactly. A joint
   permutation would preserve only their combined marginal and let the
   train-only marginal drift.

### What is not preserved, and why that is correct

The IMAGE-level class marginal shifts by a few percent. It has to. Participants
here contribute 183, 244 or 366 images, so exchanging the labels of a 183-image
and a 244-image participant moves 61 images between classes while leaving the
participant counts untouched. No participant-level permutation can avoid this.

One consequence, stated plainly because an earlier version of this file got it
wrong: `compute_class_weights` runs on IMAGE counts in the training fold, so a
permuted run does **not** see class weights identical to its real counterpart.
They differ by roughly the same few percent. For a noise floor that is
immaterial, and it is written to every manifest
(`permutation.per_split.<split>.image_counts_before` / `_after` and
`image_drift_max_rel`) rather than left for a reader to discover.

Forcing the image marginal to hold exactly would mean permuting only within
blocks of equal image count, which would tie a participant's label to its
acquisition count and stop the result being a permutation null. It is not done.
The verifier reports the realised drift and gates it loosely, to catch a bug
rather than to police the expected behaviour.

The permutation is a derangement only by chance. Some participants keep their
true label at the rate the marginal implies, which is correct: forcing a
derangement would make the null harder than chance rather than equal to it. The
realised unchanged fraction is written to each manifest next to the value the
marginal predicts, so the reader can check it.

**The arm is weighted cross-entropy, deliberately not FA-FL.** WCE is the
conventional baseline, it reproduces the manuscript's Table 4, and choosing it
means the measured floor cannot be read as having been selected to flatter the
proposed method.

## What each outcome means, written before the run

| outcome | reading |
| --- | --- |
| Null pooled macro-F1 near 0.33, upper envelope below about 0.42 | The real arms at 0.41 to 0.54 sit outside it. "Real discriminative signal exists" survives with a measured rather than an assumed floor. |
| Null upper envelope above about 0.42 | LDAM (0.4068) and FA-FL (0.4453) fall inside the null. That sentence comes out of the paper for those arms. The composition decomposition, the label defect, the nWBV comparison, the calibration-criterion finding and the mechanism null all stand without it. |
| Null accuracy materially above the 51.2 percent majority share | The permutation is not doing what it claims. Discard the run rather than reporting it. |

Five seeds give five replicates. That supports a statement about where the
replicates fall and a permutation p-value with a floor of 1/6 = 0.167. It does
not support a fitted tail or a quantile estimate. Ten seeds halve the floor and
cost twice the GPU time.

## How to run it

**Pre-flight first. Do not skip this.** No GPU, about a minute. It exercises the
real scan, cohort filter, splitter and permutation module and asserts every
invariant the null depends on.

    cd C:\Users\aravs\PycharmProjects\Alzheimers\2026-09_permutation_null
    conda activate alzheimers
    python tools\verify_permutation.py

It must print READY. If it does not, the twelve-hour job would produce numbers
that mean nothing.

Then a two-minute wiring check, if you want one. Two epochs at 64 pixels, one
run. The numbers are meaningless on purpose.

    python run_permnull.py --smoke

Do not paste a trailing `# comment` after these commands. Windows `cmd` does not
strip it and argparse will reject it as an unrecognised argument.

Then the real job. 5 seeds x 5 folds = 25 runs, about 11.8 GPU-hours of pure
compute. A laptop 3060 throttles over that span, so budget 14 to 16 wall hours.

    python run_permnull.py

`--resume` skips anything already COMPLETE, which is what to use after an
interruption. `--seeds 10` doubles the replicates and the time. `--dry-run`
prints the queue without running it.

### About the smoke run

`--smoke` writes a real COMPLETE run into the same outputs directory, carrying a
real seed and fold in its manifest but only two epochs at 64 pixels with no
pretrained weights. Anything selecting runs by seed and fold alone would pick it
up, and two things would then go wrong silently: `--resume` would skip the real
run for that spec, and the analysis would count six runs for that replicate and
drop it.

`src/runfilter.py` decides production-ness from each run's own resolved config,
and both the queue and the analysis use it. The smoke run is excluded by name
and reason in the analysis output. You do not have to delete it, though you can.

### Then the analysis

    python analysis\a22_permutation_null.py

Reads the permuted runs from this tree and the real age-60 arms from
`2026-09_instrumented_arms`, pools each replicate's five test folds to one
prediction per participant, and reports where each real arm falls in the null.
Writes `analysis/outputs/t22_permutation_null.json`.

Unit of analysis is the participant throughout, using the mean of the predicted
distributions over a participant's images. That is the manuscript's Table 4 rule
and it is the unit the paper should report everywhere.

## What this does not do

- It does not permute the test fold. That would measure nothing.
- It does not cover every arm. One arm bounds the floor for the pipeline; it
  does not give a per-arm null. A per-arm null is eight times the compute.
- It does not fix the checkpoint-selection or estimator questions. Those are
  separate and are recorded elsewhere.
- Five replicates cannot produce a p-value below 0.167. The strongest honest
  statement available is "outside all five replicates".
