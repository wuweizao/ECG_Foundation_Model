# Phase 3: explicitly accounting for validation annotations

Registered 2026-10-03, Asia/Shanghai, after inspecting phases one and two.
This is a retrospective, hypothesis-informed benchmark extension. It is not an
independent confirmation, prospective annotation-cost study, or external-domain
validation. Prior experiments used all validation labels. Their informational
influence cannot be erased by constructing new subsets; the budget below counts
labels consumed by this stage's fixed training/selection algorithm, not the
historical project-wide annotation cost or foundation-model pretraining labels.

## Primary question

With identical training patients, how much does reducing the validation-label
budget affect scratch and pretrained Net1D? Does the initialization advantage
persist under an explicit combined training-plus-validation patient budget?

## Allocation and pairing

Use unchanged PTB-XL 1.0.3 labels, preprocessing and official folds 1–8/9/10.
Eligible development patients: N_train=14823, N_val=1917, N_dev=16740.
For f in {1%,5%,10%,25%,100%}, let B=ceil(f*N_dev),
n_val=floor(B*N_val/N_dev), n_train=B-n_val. This preserves the official
development split ratio rather than optimizing budget allocation on outcomes.

For each seed 42/52/62, use a fixed seeded permutation of sorted training
patients and a separate permutation of sorted validation patients (seed+10000).
Take prefixes and all eligible ECGs from those patients. Subsets are nested
within each pool and identical across models. Never redraw based on labels.
Audit all class counts, both patient and ECG counts, and missing classes.

Two validation policies:
- budgeted: only n_val patients' validation labels may be used;
- full_validation_control: the same n_train training patients, but all 1917
  validation patients. This is deliberately a larger total-label-budget control.
At 100% the policies are identical; run once per model/seed and reuse explicitly.
The shared full-budget endpoint is not an independent replicate of both policies.

Expected allocations (training/validation): 1%=149/19; 5%=742/95;
10%=1483/191; 25%=3706/479; 100%=14823/1917.
The budget is a patient count, not number of ECGs, annotation minutes, or cost.
All five labels of each included ECG are assumed available.

## Fixed training and model selection

Models: scratch and pretrained Net1D only. Identical architecture/head seed and
preprocessing. AdamW LR=1e-4 for BOTH models, weight decay .01, physical batch16,
gradient accumulation2, BCEWithLogitsLoss, gradient clipping1, BF16,
maximum60 epochs, patience12. No LR or architecture search in this phase.
This fixed controlled recipe is not claimed to be optimal and does not reuse
the model-specific winning learning rates from phase two.

Select the earliest epoch attaining minimum mean validation BCE (averaged over
ECGs and five labels), with no minimum improvement delta. Use the SAME BCE rule
in both policies: AUROC is undefined if the small validation subset has no
positive or no negative instance of a class. Do not drop missing classes from
the macro selection metric, enlarge the subset, or inspect unused validation
labels to choose a checkpoint. Validation AUROC may be logged as null, not used.
No fitted feature normalization, class weighting, augmentation or calibration.

For each class containing both outcomes in its allowed validation subset, tune
F1 threshold on .01:.01:.99, choosing the lowest maximizer as in phase one.
If either outcome is absent, use threshold .5 and record that fallback. Also
report fixed-.5 test F1 to separate discrimination from threshold-selection noise.

New runs: 4 low fractions × 2 validation policies × 2 models × 3 seeds +
1 full fraction × 2 models × 3 seeds = 54. Earlier models cannot be reused as
these controls: they used validation-AUROC selection rather than validation BCE.
All new conditions are independently trained from their registered seed.

## Evaluation and reporting

Freeze protocol, manifests, training code and all 54 selected checkpoints and
thresholds before any phase-three test prediction. Refuse incomplete/failed
conditions; record failures and stop rather than replace seeds or widen grids.
Retain all previous results, locks, source and weights with verified hashes.
No test-driven retraining. PTB-XL fold10 was examined in earlier phases and is
not a fresh confirmatory sample. External validation is a separate future step.

Primary endpoint: Macro AUROC. Secondary: Macro average precision, classwise
AUROC/AP, F1 (validation-selected and .5), sensitivity, specificity, Brier/ECE.
Report mean±sample SD across three seeds and individual seed results. Plot both
performance against actual train+validation patient count and matched-training
comparisons across policies, with explicit control annotation costs.
Report selected epochs, actual exposures, cap hits and threshold fallbacks.

Predeclared descriptive patient-paired bootstrap (2000 replicates/contrast):
pretrained minus scratch at all fractions/policies/seeds (count the shared100%
endpoint once:27 pairs), and budgeted minus full-validation at the four low
fractions for each model/seed (24 pairs). Intervals are conditional on fitted
models, unadjusted, and not a replacement for training variance. No familywise
significance, noninferiority/equivalence or newly discovered optimum claims.
