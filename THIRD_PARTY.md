# Third-party sources

ECGFounder official implementation: https://github.com/PKUDigitalHealth/ECGFounder

Pinned commit: `04edac702b61c91face519774ddcc0cd712fef23`.
Its MIT license is retained in the fetched repository and copied to `licenses/ECGFounder-MIT.txt`. The adapter loads the official Net1D implementation without changing its architecture. Filtering behavior in `src/dataset.py` follows the official `dataset.py` and `util.py` downstream protocol.

Official model checkpoint: https://huggingface.co/PKUDigitalHealth/ECGFounder/blob/main/12_lead_ECGFounder.pth
SHA-256 `ee199f3781f4ae1f732973267f003da0a759ea12bddb0dd28a77faa60aca7997`. Model weights are not redistributed by this repository; review the source's current model terms before reuse.

PTB-XL 1.0.3: https://physionet.org/content/ptb-xl/1.0.3/, CC BY 4.0. Waveforms are not redistributed. Diagnostic aggregation and patient folds follow the published data schema and benchmark conventions.
