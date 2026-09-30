# Data access

PTB-XL **1.0.3**: https://physionet.org/content/ptb-xl/1.0.3/ (CC BY 4.0).
Download `ptbxl_database.csv`, `scp_statements.csv`, and the complete `records500` tree. Native 500 Hz signals are required. Do not use 100 Hz files upsampled to 500 Hz for this protocol.

Copy `configs/local.example.json` to `configs/local.json` and set the dataset root, official weight file and cache path. Local paths are ignored by Git. The existing local dataset is read-only; processed arrays are written under this project's cache directory.

Prepare with `python -m src.prepare --workers 8`. It records source hashes, exclusions, split counts, class counts, and subset hashes in `results/data_audit.json`. Patient-level manifests and waveforms are not committed. Re-running preparation reproduces manifests exactly under the pinned NumPy version.

The 411 records with no mapped diagnostic superclass are excluded before patient sampling. No quality-based held-out exclusions are made. Code membership includes zero-likelihood diagnostic codes, consistent with the conventional PTB-XL superclass aggregation. Five labels are independent binary targets; NORM is not artificially made mutually exclusive.

Citation: Wagner et al., *PTB-XL, a large publicly available electrocardiography dataset*, Scientific Data (2020), DOI: 10.1038/s41597-020-0495-6. Dataset v1.0.3 DOI: 10.13026/kfzx-aw45.
