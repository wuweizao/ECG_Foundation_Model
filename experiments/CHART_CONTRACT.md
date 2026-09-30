# Research figure contract

Static Matplotlib PNG/PDF exports requested for repository and paper use. Actual locked-test predictions only. No placeholders or simulated performance. Palette: blue pretrained, orange scratch; square/dashed vs circle/solid distinguishes grayscale. Validation/test identifiers and per-seed metrics retained in source artifacts.

1. Label efficiency: ordered numeric fraction vs macro AUROC; five prescribed fractions on log-x, markers and sample-SD error bars. Not a temporal series. No added artificial points. Missing results prevent final report generation.
2. AUPRC: same comparison and scale conventions; intentionally same family to compare two complementary primary metrics.
3. Per-class: two faceted horizontal dot-and-interval plots, five disease classes, 10% labels, mean ± SD across three seeds.
4. Reliability: five binary-class panels, same bins and identity line, counts stored with every bin.

All figures label metric, patient fraction, PTB-XL fold, seed count. Main metric axes span 0–1. SD undefined for n=1 and is not rendered as zero. Final PNGs must be visually inspected for overlap/cropping. UMAP, if available, is qualitative only.
