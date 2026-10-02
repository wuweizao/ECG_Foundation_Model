# Implementation correction before any baseline training

2026-10-03, Asia/Shanghai. The initial registration remains in
registration.v1.json and Git commit 204ebf8. The active registration advances
only the baseline_head.py hash; all data, candidate specifications, model
selection rules, budgets and scientific questions remain unchanged.

The default compatibility head initially used tensor amax/mean for concatenated
pooling. Its forward output matches adaptive max/average pooling, but tied maxima
can have different subgradient allocation. Replace these reductions with
global first-index max and nn.AdaptiveAvgPool1d(1), matching the upstream head's
pooling and tie-gradient semantics. PyTorch's native AdaptiveMaxPool1d CUDA
backward fails under strict deterministic algorithms (observed in preflight),
so max(dim=-1).values implements the same global first-index maximum without
disabling determinism. CPU tied-input forward/gradient equivalence and strict
CUDA BF16 backward are tested. This is source-fidelity correction, not
validation-metric tuning.

No xResNet baseline run has started. Only two full-data scratch Net1D replication
runs are active; neither uses the affected code. All original phase-one files
and the phase-two Net1D training implementation are unchanged. Re-run head and
baseline forward/backward checks before any baseline run is dispatched.
