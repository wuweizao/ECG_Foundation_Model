# Study execution log

All dates below are 2026-09-30, Asia/Shanghai. This is an engineering/research execution record, not a registered clinical protocol.

- Started from the user's empty GitHub repository. Located existing PTB-XL 1.0.3 data and the official 12-lead checkpoint locally. The available GPU was RTX 5070 Ti 16 GB, rather than the 4060 Ti 8 GB in the original proposal.
- Verified official checkpoint SHA-256 and pinned Net1D source. Verified all 42,778 retained metadata, waveform and header files against PhysioNet's release checksums.
- Audited original patient folds before any label exclusion. Retained 21,388 labeled ECGs; excluded 411 records without a diagnostic superclass. Train / validation / test: 17,084 / 2,146 / 2,158 ECGs, with no patient overlap.
- Published the initial 26-run plus 7-probe protocol and implementation in commit `b4e759c`. No test prediction or test-driven model selection had occurred.
- Reproduced the 1%, seed 42 pretrained run from a fresh training process. Its selected checkpoint tensors and validation predictions were bitwise identical on the same hardware/environment. Also checked exact agreement of cached frozen features plus head against full forward inference on a training example.
- Validation histories motivated a pre-test amendment: a matched scratch/pretrained 1%, seed 42 pair with 100-epoch cap and patience 15. Commit `4f86ee9` records this decision. The pair is reported separately and cannot replace the primary curve. The selected checkpoints remained the same as the corresponding 30-epoch main runs; the longer runs stopped at 45 and 29 epochs.
- PCA and UMAP use the same validation records, independent positive-label panels and fixed seeds. The saliency example is the first MI-positive validation record by ECG ID, selected independently of model predictions.
- Independent full-data conditions run concurrently when memory permits. A per-run process lock prevents duplicate writes when the sequential queue reaches a concurrently running condition. Runtime is not a controlled speed comparison.
- Test access is gated on completion of all 35 registered conditions. The local active lock is excluded from Git; its release audit copy is written to `results/test_lock.json` with checkpoint, threshold, manifest and source hashes.
