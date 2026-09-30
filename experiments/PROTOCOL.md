# Frozen experimental protocol

## Question
Does ECGFounder initialization improve multi-label PTB-XL classification under limited patient labels, under the same architecture, preprocessing and training budget?

## Data and labels
Use PTB-XL 1.0.3, native 500 Hz, 10 seconds, 12 leads. Map SCP codes with diagnostic=1 to NORM, MI, STTC, CD, HYP. Membership uses presence of diagnostic codes (including likelihood 0), matching the conventional PTB-XL superclass benchmark. Exclude records without any mapped diagnostic label; never treat them as normal or all-negative training examples. Do not remove poor-quality records based on held-out performance. Nonfinite/malformed waveforms stop preparation for an explicit audit.

Validate patient isolation in the original metadata before exclusions. Train folds 1–8, validation 9, test 10. Use a seeded permutation of training patients, take ceil(fraction × patients), include all their eligible ECGs. Within each seed, fractions are nested. Models use identical manifests. Seeds 42/52/62 at 1/5/10/25%; seed 42 at 100%. Report requested patient fraction plus actual patient and ECG counts. No label-balanced redraws.

## Preprocessing and model
Pinned official source: PKUDigitalHealth/ECGFounder @ 04edac702b61c91face519774ddcc0cd712fef23. Official checkpoint SHA-256: ee199f3781f4ae1f732973267f003da0a759ea12bddb0dd28a77faa60aca7997.

Lead order I, II, III, aVR, aVL, aVF, V1–V6, resolved by WFDB header names. Follow README-mandated dataset.py/util.py downstream preprocessing: 50 Hz notch Q30, fourth-order 0.67–40 Hz Butterworth (zero-phase), subtract 201-sample median baseline, global per-record z-score over all leads/time. No dataset-wide statistics or test-fitted transformations. The official ptbxl_eval.py differs (omits filtering and unnecessarily resamples); that 150-class evaluator is not used here. No resampling is needed for native records500.

Identical official Net1D, BN/dropout disabled, 5-logit dense head. Only dense weights are discarded from pretrained checkpoint; all other keys must match. Both models instantiate the head under the same seed. BCEWithLogitsLoss, AdamW, LR 1e-4, weight decay .01, gradient clipping 1, effective batch 32, BF16, maximum 30 epochs, early stopping patience 7 on validation macro AUROC. Fixed budget comparisons do not claim asymptotic best scratch performance. Linear probe only at 5/10/100%, LR 1e-3, same seed convention, frozen encoder.

## Model selection and test lock
Train and validation only until all 33 runs finish. Select highest validation macro AUROC checkpoint. Set each class threshold by maximizing validation F1 on 0.01–0.99 grid; ties choose lowest threshold. Freeze checkpoint/config/threshold hashes and test manifest before test access. Evaluate the full frozen set once, allowing interrupted evaluation to reuse saved predictions. No test-driven hyperparameter changes. Test outputs report AUROC/AUPRC, macro/micro F1, macro sensitivity/specificity, per-class values. Undefined class metrics are null rather than silently omitted.

## Statistics

Frozen linear probes cache per-record encoder features once (same BF16 inference, no batch normalization or dropout). The cache computes no statistics across records. Only selected patient rows and their labels enter head optimization. The full model state remains the exported checkpoint and is used for final waveform inference.
Mean and sample SD (ddof=1) across 3 seeds. At 100%, report the single result with no invented error bar. Patient-cluster paired bootstrap resamples test patients with replacement and keeps all their recordings; compare pretrained minus scratch within the same seed. This captures test-sampling uncertainty, not training uncertainty. Label-efficiency gain uses smallest observed fraction meeting scratch 100% mean AUROC, without interpolation or a claim of statistical equivalence.

## Predeclared extensions

### Pre-test budget-sensitivity amendment (2026-09-30)

Validation-only inspection found several 1% scratch runs still improving at the 30-epoch cap. Before any test inference, add a separate 1% / seed 42 scratch vs pretrained pair with max 100 epochs and patience 15; all other settings and patient subsets stay fixed. Both models receive the same expanded budget. These two runs are locked alongside the 33 core/probe models and evaluated in the same test pass. They do not replace the main curves. Report their results regardless of direction. This narrow audit does not establish convergence for all fractions or seeds.
At fraction 10%, seed 42, compare scratch/full pretrained under additive Gaussian noise with 20/10/5 dB SNR, 0.2/0.5 Hz baseline wander, and 1/3/6 lead dropout. Perturb the normalized model input, with identical per-record perturbations for both models; this is an input corruption stress test, not a simulation of pre-filter device noise. Use original clean-validation thresholds.

Calibration: per-label binary Brier score, mean 10-bin equal-width ECE, and per-class reliability plots (event frequency vs probability). Multi-label binary predictive entropy is descriptive uncertainty. It does not prove OOD detection or clinical reliability. PCA/UMAP use the same validation records in both models and multi-label panels, never forced single-label colors. Optional external datasets require a separate pretraining-overlap audit. ECGFounder uses clinical-label supervision; do not describe it as purely self-supervised. ECG-text models trained on these disease names would be closed-label alignment, not evidence of unseen-disease zero-shot generalization.
