# Phase-three execution

This phase is authorized by the user's 2026-10-03 instruction to proceed after
phase two completes. Prior phases remain sealed and their test exposure is
explicitly disclosed. Read PROTOCOL.md before interpreting label savings.

```powershell
python -m unittest discover -s tests
python -m src.phase3.register
python -m src.phase3.suite --parallel 2 --finalize
```

Progress: results/phase3/status.json. Individual logs:
runs/phase3/logs/. The queue runs 54 models and stops on a failure rather than
changing a seed, expanding a label subset, or silently dropping a condition.
Interrupted unfinished runs restart from their fixed seed; optimizer resumption
is not implemented. Successful completed runs are preserved.

The finalization pipeline checks prior-stage integrity, freezes the 54 models,
evaluates on the previously examined test fold, independently recomputes metrics,
and writes aggregate tables, plots and 51 patient-paired bootstrap comparisons.
Per-record manifests, labels, predictions and checkpoints remain local under runs/.

Do not treat the full-validation control as having the same total annotation
cost. Its training subset is matched, but it uses all validation labels.
The 100% endpoint is shared, giving 60 display rows from 54 distinct models.
These experiments do not erase labels used in previous project stages.
