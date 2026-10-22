# Response to codex-job-smk1-1.md (DO NOT SUBMIT)

- The blocking items (acceptance binding, summary-trusting control predicates, resource certificate) are **offline**
  steps run after collection (`hscore.py`, `smoke_accept.py`, `resources.py`); none changes what the smoke job computes
  or writes. They are fixed before the smoke is accepted and before any production staging (staging refuses production
  without an accepted smoke and a valid certificate).
- In-job item fixed before submission: per-phase `nvidia-smi memory.used` sampling (A3.5 fallback) in `Job.phase`.
- The smoke is therefore submitted; its acceptance is decided only by the fixed offline scripts, re-audited.
