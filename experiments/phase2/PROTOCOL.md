# Phase 2: baseline adequacy and training replication

Drafted on 2026-10-02 and registered on 2026-10-03 (Asia/Shanghai), after examining phase-one test results.
This is a timestamped project protocol, not an external preregistration.

## Scope and disclosure

Phase one (commit a3b6bc4, 35 locked conditions) is preserved. Its test set has
already been examined. Phase two is a hypothesis-informed extension using the
same PTB-XL fold 10, not an independent confirmatory study or external validation.
No original source, model, threshold, result, or test lock is overwritten.
The phase-one Git tree is archived locally and its tracked files and 35 selected
models are checked by SHA256. New manifests, models and tables use phase2 paths.

Only steps 1 and 2 of the publication-readiness plan are authorized here.
Total-label-budget experiments, external datasets and other foundation models
are deferred. The full labeled validation cohort remains in use and must be
counted separately from the reduced training-label budget.

## Questions and comparisons

1. Do 100% results remain stable after adding seeds 52 and 62 for scratch,
   pretrained full fine-tuning, and pretrained linear probing? Retain the exact
   phase-one configuration (30 epochs, patience 7; full LR 1e-4, probe LR 1e-3).
   This adds six runs. Seed 42 is reused, explicitly identified as phase one.
2. How does pretrained frozen Net1D compare with random frozen Net1D at
   5%, 10%, and 100%, each at seeds 42/52/62? Add nine random-probe runs.
   Use the exact same 5-label head initialization, patient manifests, optimizer,
   batch order, LR 1e-3, 30-epoch cap and patience 7 as the pretrained probe.
   Random encoder realization is determined by the seed. No feature scaling is
   fitted. Report this as a fixed-head-training-budget comparison, not optimal
   linear separability; an optimizer grid for frozen features is out of scope.
3. Does the initialization advantage survive an equal candidate-search budget?
   Compare scratch Net1D, pretrained Net1D and supervised xResNet1D-101 at
   1% and 100%. For each model/fraction, train seed 42 at LR
   {1e-5,1e-4,1e-3}; AdamW weight decay .01, BCE logits, max 60 epochs,
   patience 12, effective batch 32, gradient norm clip 1, BF16.
   Select the candidate with maximum best validation Macro AUROC; ties choose
   the smaller LR. Fix that LR and repeat at seeds 52/62. Do not reselect using
   their results. This adds 18 search and 12 repeat runs. Three candidate trials
   and equal epoch caps are equal search opportunities, not equal FLOPs/time or
   proof of globally optimal hyperparameters. Report cap hits and exposures.

Total NEW training runs: 6 + 9 + 18 + 12 = 45. Search seed 42 is included in
three-seed descriptive summaries but its validation selection optimism is
disclosed; seeds 52/62 are additional training/subsampling repetitions, not
independent test cohorts. Low-budget sampling and initialization remain coupled
by seed, so their variance components cannot be separated.

## Architecture and data

Use unchanged phase-one ECGFounder Net1D and preprocessing. xResNet1D-101
comes from helme/ecg_ptbxl_benchmarking, commit
cdbf4e66d7e57d9b6a2657b6024716212b8d0afa, GPL-3.0. Load its original
xresnet1d.py with a small compatible PyTorch implementation of its default
FastAI-v1 classification head (concatenated max/average pooling, flatten,
BatchNorm, dropout .5, linear). Architecture: bottleneck [3,4,23,3], 12 inputs,
five outputs, default kernels 5, default widths. All models receive identical
500-Hz 10-second preprocessed signals. This is the published architecture under
our protocol, not reproduction of the original paper's training recipe/scores.

Official train folds 1–8, validation 9, test 10; no new split or label mapping.
Copy phase-one train/validation manifests and add full-training seed52/62
manifests containing the identical complete cohort. Frozen feature caches are
keyed by encoder tensor hash, manifest, preprocessing, precision and source.
No test features may be used to fit heads or choose parameters.

## Endpoints, selection, and access

Primary descriptive endpoint: Macro AUROC. Secondary: Macro AUPRC, Macro/Micro
F1, sensitivity, specificity, classwise AUROC/AUPRC, Brier and ECE. Per-label
thresholds maximize validation F1 on the unchanged grid. No test selection.
Preserve all search histories including losing candidates and failures.
Nonfinite training is a recorded failed candidate, never silently replaced;
if no candidate succeeds, stop that comparison and report the failure.
Infrastructure fixes may be documented before resuming; no metric-driven grid
extension after inspecting tests. At least one successful candidate is required
per family/fraction, and chosen configurations must complete all three seeds.

After all 45 attempts/repeats are accounted for, verify phase-one integrity and
freeze a second lock containing protocol, sources, manifests, selected model
weights, thresholds and candidate selection. Only then evaluate selected new
models once on fold 10; failed/unselected trials do not receive test predictions.
Reuse existing phase-one test predictions without rerunning/reselecting them.
Summaries report mean and sample SD, paired per-seed differences and patient
paired bootstrap confidence intervals conditional on trained models. Do not
treat seed × ECG predictions as independent cases or claim equivalence from
non-significance. No SOTA, clinical efficacy, external generalization, or total
annotation-cost claim will be made from this stage.

## Sources

- https://github.com/helme/ecg_ptbxl_benchmarking
- https://arxiv.org/abs/2004.13701
- https://proceedings.mlr.press/v306/berger26a.html
- https://arxiv.org/abs/2509.25095
