# sept_images

Upload `figures/` to Overleaf as your `figures/` folder. Everything in it is PNG, 300 dpi.

Regenerate everything (about 2.5 minutes, CPU only, reads the runs and the 29 Sep audit tables, writes only into figures/):

    python code/make_all_figures.py

`_old/` holds earlier versions from 30 Sep. Do not use them.

## Data behind each figure
- Restricted cohort (166): September runs, 2026-09_instrumented_arms (exp10, 11, 12, 13, 15), one run per fold.
- Full cohort (347): August runs, 2026-08_srep_revision_participant_level. These are the only complete five-fold
  full-cohort runs; the September tree holds fold-0 reproducibility reruns only.
- fig_floors, fig_null_perclass: t23_protocol_audit_long.csv. fig_gamma_ladder: t30_ece_estimator_robustness.csv.
- fig_leakage_ablation, fig_multislice: 2026-09_tier1_verified_stack leakage_ablation.csv, multislice_comparison.csv.
- fig_augmentation_*: review_experiments/data_vvisualizations (copied). fig_train_loss_share_restricted: copy of
  2026-09_instrumented_arms f19a (old class names in its panel titles).
