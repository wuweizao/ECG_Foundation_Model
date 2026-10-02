# Phase-two execution

See PROTOCOL.md for the pre-training design. The original phase-one files are
sealed and must not be edited to bypass the first test lock.

Use the existing Python environment and prepared cache. Fetch the official
baseline into vendor/ptbxl_benchmark at commit
cdbf4e66d7e57d9b6a2657b6024716212b8d0afa. Upstream GPL-3.0 is copied to
licenses/PTBXL-benchmark-GPL-3.0.txt. The compatibility head and adapter follow
that license; no original benchmark weights are used. Phase-one ECGFounder
MIT attribution remains in THIRD_PARTY.md.

```powershell
git archive --format=zip --output=runs/phase2/phase1-a3b6bc4.zip a3b6bc4
python -m unittest discover -s tests
python -m src.phase2.register
python -m src.phase2.suite --parallel 2
```

Registration is immutable. The queue skips verified completed models and logs
failures. Interrupted runs restart from the seed, not an optimizer checkpoint.
It never reads the test cohort or test predictions. Completion here is training
completion, not permission to select models based on phase-one test outcomes.
The local zip, per-patient manifests, predictions and checkpoints remain under
ignored runs/. Public registration contains only hashes, specifications and
aggregate counts, not individual ECG/patient data.
