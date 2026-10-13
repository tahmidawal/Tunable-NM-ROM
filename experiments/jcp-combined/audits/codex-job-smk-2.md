Both MAJOR findings are **RESOLVED for smoke submission**.

1. **RESOLVED — escalation coverage.** [c3run.py:306](experiments/jcp-combined/c3run.py#L306) preserves `converged=None` and `status='unresolved'`; [line 311](experiments/jcp-combined/c3run.py#L311) records the fallback with `diagnostic_only=True`. `conv_` saves the original value and [line 326](experiments/jcp-combined/c3run.py#L326) restores it. Isolated control-flow checks passed for `None`, `gl56`, and `gl64`: all ladder rules execute, at most baseline plus one ladder prep remain resident, and panel labels/seeds are distinct for the staged configuration. Accuracy hashes exist for the fallback baseline.

2. **RESOLVED — resource-evidence collection.** [c2grid.py:178](experiments/jcp-combined/c2grid.py#L178), [line 204](experiments/jcp-combined/c2grid.py#L204), and [line 246](experiments/jcp-combined/c2grid.py#L246) time individual FOM order/calibration/evaluation calls through `block_until_ready`, preserving separate case records and first-call costs. Scheme, timestep and tolerances are recoverable from records plus pinned source/config. Phase marks and [the final peak sample](experiments/jcp-combined/c2grid.py#L334) cover the later phases and both timing panels. This supplies the missing evidence; certification still requires the deferred, fail-closed resource-gate rework. Missing/negative peaks must reject certification.

**No new BLOCKER/MAJOR found.** All manifests verify—46/33/33 entries—and staged Python sources match HEAD `c6b4c338c`. The earlier descriptor-hash concern also clears: isolated regeneration using the target cluster venv matches the staged `dev6` reference hash exactly.

Read-only review; no project imports, file modifications, or GPU workloads. Approval covers these smoke jobs, not production certification.

**SUBMIT**