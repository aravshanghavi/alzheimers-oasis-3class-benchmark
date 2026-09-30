# RUN_PROTOCOL

How to run the protocol audit (a23), the Murphy resolution bootstrap (a24) and the
patched participant baseline (a17). Written to be followed without reading any of
the code.

Open a normal `cmd.exe` and change to the repository root:

```
cd C:\Users\aravs\PycharmProjects\Alzheimers\2026-09_instrumented_arms
```

Do **not** set `SREP_EXPERIMENTS_DIR`. This repository has no `experiments_dir.txt`,
so the analysis layer reads its own `experiments\` folder, which holds the restricted
age-60 runs (exp10 to exp13). That is what a23 and a24 expect. The full-cohort runs are
found automatically at `..\2026-08_srep_revision_participant_level\experiments`.

Steps 1 to 4 write only into `analysis\outputs\`, and only to files whose name starts
with `t23_` or `t24_`. Step 5 is the exception: it creates two more files in that same
folder, `_cache.npz` and `a17_participant_contrasts.csv`, both new in this repository.
Nothing anywhere else in the tree is touched by any step.

---

## Step 1. Smoke test a23, about 25 seconds

```
python analysis\a23_protocol_audit.py --quick
```

Runs the whole audit at 200 bootstrap resamples instead of 2000. The point is to find
a missing folder or a broken path in under a minute, before committing to the real run.
The intervals it prints are not reportable.

Produces `analysis\outputs\t23_protocol_audit.csv`, `.md` and
`t23_protocol_audit_long.csv`, which the full run then overwrites.

**Good looks like:** the header lines report `restricted: 8 arms, 166 participants,
39,894 images` and `full: 7 arms, 347 participants, 86,437 images`, followed by a
two-fingerprint warning naming 8378f120 for the full arms and cb7ee123 for the
restricted ones, then the permuted-label line reading `5 replicates from 25 production
runs`, then the PASS and FAIL grid.

Two more lines belong in that header and should be there on every run. The metadata block
now says the imputation median and the standardising mean and SD are taken from the training
folds of each split only, and it flags `META MMSE only` and `META age+sex+eTIV+nWBV+MMSE` on
the full cohort as NOT INTERPRETABLE at 147 of 347 imputed. After the permuted-label line
there is a warning that 2 of the 7 shared arms, `Focal g=1.37` and `Focal g=2.00`, have no row
in `t21_age_confound.csv` and so fall back to a nominal 95 percent interval in R5.

**Bad looks like:** `full cohort unusable` or `no full-cohort experiments folder found`.
The audit still finishes, on the restricted cohort alone, with R5 reported as NA for
every arm. That is a degraded run, not a result.

---

## Step 2. Smoke test a24, about 15 seconds

```
python analysis\a24_resolution_bootstrap.py --quick
```

Same idea for the Murphy decomposition.

**Good looks like:** eight `cohort scaling unit` progress lines, then an identity check
whose worst leftover is about 1 percent of the Brier score, then two 24 row verdict
panels, one per comparator.

**Bad looks like:** a warning that `predictions_test_temperature_scaled.npz` is missing
for some arm. The temperature-scaled half is then skipped for that cohort and the table
comes out with 12 verdict rows instead of 24. The unscaled half is still valid.

---

## Step 3. The real a23, about 90 seconds

```
python analysis\a23_protocol_audit.py
```

2000 participant-clustered resamples, which is the number a20 uses, so the intervals are
comparable with `t20_comparisons.csv`.

Produces:

* `analysis\outputs\t23_protocol_audit.csv` and `.md`, one row per arm per cohort. R1
  appears as four separate cells, accuracy and macro-F1 at each of the two units, plus
  the combined `R1_beats_constant` column that ANDs them. The printed grid shows the four
  cells, not the combined one, because R1 splits on this data and a single collapsed cell
  hides which half failed.
* `analysis\outputs\t23_protocol_audit_long.csv`, every underlying comparison, 554 rows,
  with its family, its corrected level and its code fingerprint. Four columns carry the
  protocol bookkeeping. `fold_honest_preprocessing` is true on every metadata row.
  `reference_interpretable` and `counted_in_R4_verdict` say whether a metadata reference was
  allowed to decide the R4 verdict. `r5_interval_basis` names, on every R5 row, whether the
  interval came from the declared `age60_train_matched_vs_evaluated` family, from the nominal
  95 percent fallback, or from a cell where no family was ever declared.

**Good looks like:** the same grid the smoke test printed. Verdicts should not move
between 200 and 2000 resamples. If one does, that comparison sits on the edge of its
interval and must be described that way in the paper. The expected pattern is accuracy
FAIL in all 15 arm rows at both units and macro-F1 PASS in all 15 at both units. That
split is the finding, not a bug.

**Bad looks like:** a verdict that flips between the smoke run and the full run, or a
`PROTOCOL VIOLATION` line. a23 does not register comparisons and does not write the
multiplicity audit, so a violation here would mean something else in the session did.

Two rules in that table are worth knowing before reading it.

The metadata reference models are fold honest. The median used to fill a missing value, and
the mean and standard deviation used to standardise a predictor, come from the training folds
of each split and are then applied to the held-out fold. a17 still does this the other way,
once over the whole cohort, so a23 and a17 can differ slightly on the same model. On the
current data that difference is small. Predicted probabilities move by at most 0.004, and the
only macro-F1 that moves at all is `META sex only` on the restricted cohort, from 0.303112 to
0.333523, because a sex-only model's three class probabilities sit nearly tied and a small
shift flips ten participants. Every other metadata macro-F1 is unchanged to six decimals, and
the full-cohort MMSE median is 29.0 in the whole cohort and in all five training splits, so
the imputation itself never differed. The leak was real and it was tiny. The fix removes the
argument, not a large number.

R4 no longer lets a reference the script refuses to interpret decide a verdict. A metadata
model with more than 20 percent of any one predictor imputed prints as NOT INTERPRETABLE, and
on the full cohort that is `META MMSE only` and `META age+sex+eTIV+nWBV+MMSE`, both at 147 of
347 imputed. Those two stay in the long table with `reference_interpretable` false and are
left out of the R4 pass or fail. The console names them under the R4 legend, so the exclusion
cannot be read as a silent drop. The R4 column does not move on this data. Every arm still
fails R4 on both cohorts, because `META age+sex+eTIV+nWBV` on its own already beats every arm
on macro-F1, and that reference is fully observed and fully interpretable.

What the numbers should reproduce, as a cross-check against work already done:

| quantity | value | where to read it |
|---|---|---|
| restricted constant predictor, participant level | 51.204819 percent | `reference_value`, R1 rows, unit participant |
| restricted constant predictor, image level | 51.834862 percent | `reference_value`, R1 rows, unit image |
| full cohort constant predictor, image level | 77.769936 percent | `reference_value`, R1 rows, unit image |
| null macro-F1 mean over 5 replicates | 0.335355 | `reference_value`, R2 rows |
| null macro-F1 standard deviation | 0.059798 | `null_sd`, R2 rows |
| standard error of that standard deviation | 0.021142 | `null_sd_standard_error`, R2 rows |
| permuted-label replicates and production runs | 5 and 25 | the console line, and `n_null_replicates` |

All of those are columns of `t23_protocol_audit_long.csv`. The image-level restricted
figure also appears in `t20_pooled_metrics.csv`, and the null figures are recomputed here
from the same runs that produced
`..\2026-09_permutation_null\analysis\outputs\t22_permutation_null.json`.

a23 does NOT report the null ACCURACY mean of 40.963855 percent. It scores the arms
against the null on macro-F1 and on each class's F1 only. That figure lives in
`t22_permutation_null.json` and is checked there, not here.

Read `null_sd_standard_error` next to every `z_against_null_sd`. The standard deviation
comes from five numbers, so its own standard error is about a third of its size, and a z
quoted without that is the most misreadable number in the package.

If any of these differs, stop and find out why before reading anything else in the table.

---

## Step 4. The real a24, about 20 seconds

```
python analysis\a24_resolution_bootstrap.py
```

Produces `analysis\outputs\t24_resolution.csv` and `.md` with the decomposition and every
FA-FL minus other-arm difference, and `t24_resolution_verdict.csv` with one row per
cohort, unit, scaling and bin count.

Two comparators are printed, both against FA-FL's resolution, in separate labelled
panels. The PRIMARY one is the arm with the highest resolution other than FA-FL, which is
the hardest available test. The SECONDARY one is the arm nearest to FA-FL in resolution.
The secondary exists so you can see whether the verdict depends on which comparator was
chosen. The `BOTTOM LINE` counts both and says in how many cells they disagree.

**Good looks like:** a `BOTTOM LINE` counting how many of the 24 cells show an FA-FL
resolution advantage whose interval excludes zero, under each comparator, and a stability
block saying whether FA-FL's rank holds across 10, 15 and 20 bins.

**Bad looks like:** an identity leftover above 2 percent of the Brier score. The table
flags that per row in `decomposition_describes_score`. A large leftover means the binned
decomposition is not describing that arm's score and its components should not be quoted.
The leftover is usually negative, because the binned form drops both the within-bin spread
of the predicted probabilities and its covariance with the outcome. A negative leftover is
not an error. The worst one in the current table is 1.19 percent of the Brier score.

The Brier column is the cross-check: for the restricted cohort at the image level with
raw probabilities it must match `t20_pooled_metrics.csv` to four decimals, for example
Weighted CE 0.6767 and FA-FL 0.5869.

---

## Step 5, optional. The patched a17

a17's metadata preprocessing is still transductive, fitted once over the whole cohort. That
was left alone on purpose, so every number a17 produced before stays reproducible. a23 is the
fold-honest one, and the two will not agree to the last decimal on the same model.

a17 now fits two extra reference models, `META sex only` and `META age+sex`. Nothing else
about it changed. Every number it produced before is bit identical. That was checked by
running the patched file and comparing all 200 pre-existing rows against
`..\2026-09_tier1_verified_stack\analysis\outputs\a17_participant_contrasts.csv`. The
accuracy, macro-F1, QWK, macro-AUROC, point, lo and hi columns agree exactly, to zero
difference.

a17 reads `analysis\outputs\_cache.npz`, **which does not exist in this repository**. It
holds the seven full-cohort arms and is built from the 2026-08 runs. Build it here.

Do not run the patched a17 out of `..\2026-09_tier1_verified_stack` instead. That tree has
the cache, but its copy of a17 is the PRE-PATCH file, not an identical one, so taking that
route means copying the patched script into a second repository and then overwriting that
repository's own `a17_participant_contrasts.csv`, which is a result file belonging to the
tier-1 stack. Building the cache here touches nothing outside this tree.

Run these four lines in one `cmd.exe` window, in this order:

```
set SREP_EXPERIMENTS_DIR=C:\Users\aravs\PycharmProjects\Alzheimers\2026-08_srep_revision_participant_level\experiments
python analysis\_cache_build.py
python analysis\a17_participant_baseline.py
set SREP_EXPERIMENTS_DIR=
```

The first line points the analysis layer at the full-cohort runs, which is what the cache
is made of. The last line clears the variable again, so a later a23 or a24 in the same
console does not read the wrong tree. Closing the window clears it too.

`_cache_build.py` takes about 5 seconds and writes `analysis\outputs\_cache.npz`, roughly
7 MB. a17 takes about 8 seconds and writes `analysis\outputs\a17_participant_contrasts.csv`.
Both files are new in this repository. Neither is read by a23 or a24, so the order of
step 5 against steps 1 to 4 does not matter.

**Good looks like:** `cached: (7, 86437, 3) (7, 347, 3) 347 participants` from the cache
build, then `META sex only` landing exactly on the constant predictor, which is what the
joint demographic floor question was asking. On the full cohort it scores accuracy 0.7666,
macro-F1 0.2893, QWK 0.0000 and macro-AUROC 0.5140, identical to `BASELINE constant` on the
first three. `META age+sex` comes out at macro-F1 0.3640, a little above `META age only` at
0.3233.

**Bad looks like:** `META sex only` scoring well above the constant predictor. Sex carries
almost no dementia signal in OASIS cross-sectional, so a high score there means the column
was read wrong. Also bad: `_cache_build.py` failing an assertion about folds not being
disjoint, which would mean the 2026-08 runs it read are not the ones a17 expects.

**One thing to settle before quoting a17 in the paper.** The patch takes the number of
comparisons a17 makes against `BASELINE constant` from 11 to 13 per cohort. The registry
declares `arm_vs_constant_accuracy` and `arm_vs_constant_macro_f1` at n = 11, which is the
pre-patch count. Either amend both declarations to 13 and record the change and its reason
in `analysis\REGISTRY_CHANGELOG.md`, or state in writing that the two new reference models
sit outside those families. Leaving the mismatch unstated is the one option that is not
available, because n = 11 would then no longer describe what was run.

---

## If something fails, check this first

**The experiments folder.** Almost every failure in this stack is the analysis layer
reading the wrong tree, and it always announces itself in the first three lines of output.
Run:

```
python -c "from analysis._paths import describe; print(describe())"
```

It prints the repository root, the experiments directory actually in use, the outputs
directory and the metadata workbook. The experiments line must end in
`2026-09_instrumented_arms\experiments`. If it does not, an `SREP_EXPERIMENTS_DIR` left
over from an earlier command is the cause. Clear it with `set SREP_EXPERIMENTS_DIR=` and
run again.

Two other failures worth naming:

* `oasis_cross-sectional.xlsx` not found stops a23 at the metadata step. It lives in the
  repository root and a23 also looks one level up.
* A missing `..\2026-09_permutation_null` tree makes R2 report NA for every arm. a23 says
  so and carries on rather than stopping.

Neither script reads a single image file. There are no JPEGs anywhere in this tree and
none is needed.
