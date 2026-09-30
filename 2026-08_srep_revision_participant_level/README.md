# Scientific Reports revision — corrected participant-level protocol

Codebase for the revision of submission `c2393356-e281-4bd5-83a5-d1ff9e275007`
("A Methodological Re-Evaluation of Alzheimer's Classification on OASIS").

**Nothing outside this folder is modified.** The scripts under
`review_experiments/` and `new_strategy/` remain as the record of the submitted
version. Everything here is the corrected protocol.

---

## Why this folder exists

The submitted code grouped images with

```python
re.compile(r'(OAS\d_\d{4}_MR\d)')     # includes the MR SESSION number
```

OASIS-1 contains a reliability subset of nondemented participants rescanned
within 90 days. In this dataset **19 of 347 participants have two sessions**, so
the split was at the session level, not the subject level: measured across the
five submitted seeds, **5.5–8.5% of every test partition** came from
participants who were also in training.

Here the grouping key is `OAS1_XXXX`, and `assert_group_disjoint` runs on every
single run with its result written into the log and the manifest.

---

## Pre-flight

```bat
conda activate alzheimers
python tools/check_folds.py        :: what the partition actually looks like
python run_all.py --smoke          :: ~12 min wiring check on the real data
```

`check_folds.py` prints the participant composition of all five folds using your
local scikit-learn. Read it before spending a day of GPU time.

**scikit-learn is pinned to 1.5.2 and that is deliberate.** `StratifiedGroupKFold`
changed its fold assignment between releases: the same data and the same seed give
a different partition on 1.8.0 than on 1.5.2. For a paper whose contribution is a
reproducible evaluation protocol, the split cannot depend on a minor upgrade. The
version is recorded in every run's `environment.txt`.

---

## Before anything: activate the environment

```bat
conda activate alzheimers
```

Bare `python` on this machine has `torch` but **not** `torchvision`, so the run
fails at import. `run_all.py` now preflights every dependency and the data path
before launching anything, and refuses with an explicit message rather than
three subprocesses that each exit 1.

---

## Quick start

### Unattended: everything, one command

```bat
run_all.bat                 :: all experiments back to back, then close the window
run_all.bat --resume        :: skip anything already COMPLETE
run_all.bat --isolate       :: each run in its own subprocess (safest for VRAM)
```

or directly:

```bash
python run_all.py --with-optional --resume
python run_all.py --smoke              # 2 epochs, 64px, one run each: wiring check
```

The terminal shows **one heartbeat line per run** and nothing else — a full
four-experiment batch printed 59 lines. Everything else goes to files. Close the
window, come back tomorrow, read the logs.

### One experiment at a time

```bash
python experiments/exp01_main_sweep/run.py --dry-run      # list the 20 planned runs
python experiments/exp01_main_sweep/run.py                # execute
python experiments/exp01_main_sweep/run.py --resume       # skip completed specs
python experiments/exp01_main_sweep/run.py --only FA_FL:fold3
python run_one.py --config exp01_main_sweep.yaml --loss FA_FL --fold 3 --init-seed 42
python analysis/discover.py --check                       # integrity + completeness
python analysis/discover.py --summary                     # results table
```

Set `data.root` in `config/base.yaml`, or pass `--data-root`.

### Experiments

| Folder | Runs | Purpose |
|---|---:|---|
| `exp01_main_sweep` | 20 | 4 losses × 5 grouped CV folds. The main result |
| `exp02_class_balanced` | 5 | Class-Balanced loss baseline (Reviewer 1 #6) |
| `exp03_init_variance` | 10 | Fold fixed, `init_seed` varies. Variance decomposition (Reviewer 3 #6) |
| `exp04_multislice_25d` | 5 | Optional 2.5D arm (Reviewer 1 #3, Reviewer 3 #1) |

---

## Reading an output folder

```
experiments/exp01_main_sweep/outputs/
  20260816-021304__exp01__FA_FL__fold3__split42-init42__7c64ab91/
  ^timestamp      ^exp   ^loss  ^fold  ^seeds          ^code fingerprint
```

The suffix is a SHA-256 over every `src/*.py`. **All runs in one results table
must share it** — if they don't, the code changed mid-sweep.

| File | What it is |
|---|---|
| `manifest.json` | Source of truth: identity, config, environment, split sizes, headline metrics |
| `run.log` | Complete log, identical to console output |
| `metrics.json` | All scalar results |
| `predictions_{test,val}.npz` | `probabilities`, `labels`, **`participant_id`**, `session_id`, `image_path` |
| `confusion_matrix_{test,val}.csv` | Raw counts |
| `classification_report_{test,val}.txt` | sklearn report |
| `training_curve.csv` | Per-epoch loss, accuracy, lr, duration |
| `fold_participants.csv` | Which participant went to which split |
| `checkpoint_best.pt` | Best-validation weights |
| `code_snapshot.zip` | The exact `src/` and resolved config that produced this run (~27 KB) |
| `environment.txt` | torch / CUDA / GPU / OS versions |
| `_COMPLETE` | Status marker: `_RUNNING`, `_COMPLETE` or `_FAILED` |
| `figures/` | Per-run diagnostics: confusion matrix, ROC, PR, reliability (test + val) and the training curve |

`figures/` exists so a run can be eyeballed without writing code. **The paper's
figures are not these** — those come from the analysis layer, pooled across folds,
because a figure from one fold overstates what one fold supports. Nothing is lost
either way: every figure is reconstructible from `predictions_*.npz`, and Grad-CAM
from `checkpoint_best.pt`, long after the run.

`participant_id` in the `.npz` is not optional: the test set holds ~17,000
images but only ~70 independent participants, with 3–4 near-duplicate
acquisitions of each anatomical slice (r = 0.988). Bootstrap intervals must
resample **participants**. This cannot be added retroactively.

---

## Logging — built for unattended batches

Nothing you need is ever only on the terminal. Three tiers, written concurrently:

| Destination | Contents |
|---|---|
| `console_logs/<ts>__run_all.log` | Master log for the batch, plus every child process's stderr |
| `console_logs/<ts>__<experiment>.log` | Complete queue log for one experiment: every run's full detail |
| `<run_dir>/run.log` | That single run's complete log |

`--console` controls the terminal only:

- `progress` (default) — one heartbeat line per run, plus warnings. This is what
  makes an overnight batch readable the next morning
- `full` — everything, as before
- `silent` — nothing

Child-process stderr (torch warnings, tracebacks, library chatter) is appended to
the master log rather than the console, so the unattended terminal stays clean.

Each run logs: banner (run id, code fingerprint, host/GPU, versions, determinism
flags), the fully resolved config, a cohort summary, a per-split participant/
session/image table with the disjointness assertion result, class weights and
effective FA-FL gammas, GPU state at start, per-epoch metrics with duration and
ETA, a rendered confusion matrix with recall and precision margins, VRAM before
and after teardown, and a footer with total duration and artifacts written.

A failed run logs its traceback, writes `_FAILED`, and **the queue continues** —
verified by corrupting an input mid-batch.

---

## GPU memory across a long queue

`torch.cuda.empty_cache()` on its own does almost nothing: it returns *cached*
blocks, not memory still referenced by live objects, and it does not defragment.
`src/gpu.py` does the whole sequence, in the order that actually works:

1. shut down persistent DataLoader workers **while the loaders still exist**
2. null every local reference in the caller (model, criterion, loaders)
3. `gc.collect()`
4. `synchronize()` → `empty_cache()` → `ipc_collect()`
5. `reset_peak_memory_stats()`

Step 2 is the one that is easy to get wrong: passing objects *into* a cleanup
function only rebinds that function's parameters while the caller's frame still
holds them, so the collect frees nothing.

Every run logs VRAM before and after teardown, and warns if more than 0.05 GB is
still allocated afterwards — a leak that grows across runs shows up immediately.
Before each run the queue also checks free VRAM against `--min-free-gb`
(default 3.0) and warns if another process is holding the card.

**Two stronger measures when a long batch matters:**

- `set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before launching.
  `run_all.bat` does this for you. It is the documented fix for allocator
  fragmentation across many sequential allocations, which teardown cannot undo.
- `--isolate` runs every spec in its own subprocess, so reclamation is enforced
  by the OS rather than hoped for. Costs ~20 s per run (the parent caches the
  dataset scan to CSV so children skip re-scanning 86,437 files), which is 0.7%
  of a 45-minute run. **For a 35-run overnight batch this is the safer default.**

---

## Layout

```
config/      base.yaml + one yaml per experiment
src/         importable library, no entry points
experiments/ thin drivers, each with its own outputs/
analysis/    discovery layer and analysis scripts
tools/       make_synthetic_dataset.py (pipeline testing without the real data)
```

`src/` has no `__main__`. That is what prevents the previous codebase's problem
of five near-identical 39 KB scripts that drifted apart.

---

## Key configuration

`config/base.yaml`:

- `split.strategy` — `grouped_kfold` (default) or `grouped_holdout`
- `split.group_by` — `participant` (default). `session` reproduces the submitted
  bug and exists only so the leakage assertion can be tested against it
- `data.context_slices` — `1` for 2D, `3` for 2.5D
- `train.early_stop_metric` / `checkpoint_metric` — the submitted code stopped on
  validation **loss** while checkpointing on validation **accuracy**. That
  behaviour is preserved for comparability and both are now explicit and logged
- `deterministic` — cuDNN deterministic mode, autotuning off
- `strict_deterministic` — additionally `torch.use_deterministic_algorithms(True)`

### Determinism

Measure the cost before launching the full sweep:

```bash
python experiments/exp01_main_sweep/run.py --only FA_FL:fold0 --epochs 3
# run twice, diff the two metrics.json
```

Under a 30% penalty, keep determinism on. Over 30%, set `deterministic: false`
and state in Methods exactly which nondeterminism remains.

---

## Verified in a sandbox (15 Aug 2026)

Against a synthetic dataset mirroring the real structure (347 participants, 366
sessions, 19 two-session participants, 266/58/23 per class):

- `scan_dataset` recovers the cohort exactly
- grouped 5-fold gives 69–70 test participants per fold, 4–5 Demented, every
  participant tested exactly once
- the leakage assertion **fires** on the old session-level key and passes on the new one
- two independent process launches produce **bitwise-identical** predictions
- a corrupt input fails one run and the queue continues through the rest
- `--resume` skips completed specs; 2.5D packs distinct neighbouring slices and
  clamps at the 100/160 band edges
- model parameter count is 4,664,959 — the 4.66M the manuscript reports
- `run_all.py` completed a four-experiment batch with 59 lines of terminal
  output; all detail in five log files plus four per-run logs
- `--isolate` completed a two-run queue via subprocesses, with different
  `init_seed` values producing different results (ECE 0.4160 vs 0.3936),
  confirming the split of `SPLIT_SEED` from `INIT_SEED` takes effect

Sandbox runs used `--no-pretrained` (ImageNet weights are not downloadable
there) and reduced image size. **Production runs use neither flag.**

**Untested in the sandbox, because it has no GPU:** every CUDA branch in
`src/gpu.py`. The logic is straightforward and the CPU path exercises the same
call sequence, but the first real run should be watched to confirm the
"VRAM before/after teardown" lines report sensible numbers.
