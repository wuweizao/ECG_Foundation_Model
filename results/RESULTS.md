# Held-out results

All thresholds selected on fold 9. Test fold 10 was locked before evaluation.

| Model | Patients | Seeds | Macro AUROC | Macro AUPRC | Macro F1 |
|---|---:|---:|---:|---:|---:|
| Linear probe | 5% | 3 | 0.8986 ± 0.0036 | 0.7604 ± 0.0067 | 0.7046 ± 0.0085 |
| Linear probe | 10% | 3 | 0.9070 ± 0.0020 | 0.7750 ± 0.0054 | 0.7117 ± 0.0011 |
| Linear probe | 100% | 1 | 0.9207 (one seed) | 0.8053 (one seed) | 0.7361 (one seed) |
| Pretrained | 1% | 3 | 0.8537 ± 0.0085 | 0.6648 ± 0.0178 | 0.6331 ± 0.0107 |
| Pretrained | 5% | 3 | 0.8877 ± 0.0057 | 0.7383 ± 0.0112 | 0.6947 ± 0.0048 |
| Pretrained | 10% | 3 | 0.9014 ± 0.0024 | 0.7698 ± 0.0054 | 0.7137 ± 0.0020 |
| Pretrained | 25% | 3 | 0.9157 ± 0.0002 | 0.8005 ± 0.0015 | 0.7315 ± 0.0024 |
| Pretrained | 100% | 1 | 0.9274 (one seed) | 0.8211 (one seed) | 0.7464 (one seed) |
| Scratch | 1% | 3 | 0.8117 ± 0.0136 | 0.5858 ± 0.0248 | 0.5827 ± 0.0149 |
| Scratch | 5% | 3 | 0.8531 ± 0.0076 | 0.6675 ± 0.0145 | 0.6305 ± 0.0140 |
| Scratch | 10% | 3 | 0.8676 ± 0.0052 | 0.7007 ± 0.0136 | 0.6560 ± 0.0068 |
| Scratch | 25% | 3 | 0.8861 ± 0.0007 | 0.7341 ± 0.0045 | 0.6842 ± 0.0015 |
| Scratch | 100% | 1 | 0.9077 (one seed) | 0.7756 (one seed) | 0.7115 (one seed) |

Label efficiency gain is a discrete descriptive comparison, not an equivalence test.
The fixed 30-epoch maximum does not establish performance after unlimited training or model-specific tuning.
Seed variability combines patient subsampling and optimization. One full-data seed cannot estimate training variance.
