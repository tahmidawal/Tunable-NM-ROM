**NOT RESOLVED.** Audited `6b40c6716` against `401cd4e42`. The actual `smoke_accept.py` returned **`PASS []`—151 gates passed** on fabricated artifacts, with both `SLURM_JOB_ID` and `SOURCE_COMMIT` set. No collector functions were mocked.

Reproducer: [/tmp/audit_speed3d_r5_mutations.py](/tmp/audit_speed3d_r5_mutations.py). Fabricated bundle and verdict: [/tmp/speed3d-r5-vr7hm7uf/ACCEPTANCE.json](/tmp/speed3d-r5-vr7hm7uf/ACCEPTANCE.json).

Blocking fixes:

- **Validate exact timing membership and durations.** Every panel passed with one invented subject, duplicate `(case=999, rep=99)` entries, and negative durations. PS warm timing passed with `[{}, {}]`; a separate mutation with infinite durations also passed. Derive expected subjects/cases/repetitions from configuration, enforce uniqueness, and require finite positive timings. See [panel validation](experiments/jcp-speed3d/smoke_accept.py:45) and [warm timing](experiments/jcp-speed3d/smoke_accept.py:152).
- **Require complete, finite, internally consistent check evidence.** Standalone QN passed with `valid=[]`; `[0, NaN, 0]` distances also passed. G6v passed with `rel=123`, K8 with `diff=0`, and both standalone artifacts with wrong setting/rule metadata. Profile MF accepted `adt_valid=[]`, convergence evidence `{}`, and mesh G5 accepted false individual gates under `G5_all=true`. Validate the underlying evidence and recompute applicable predicates. See [standalone checks](experiments/jcp-speed3d/smoke_accept.py:88), [profile checks](experiments/jcp-speed3d/smoke_accept.py:186), and [mesh checks](experiments/jcp-speed3d/smoke_accept.py:203).

Two narrow repairs are **RESOLVED**: wrong job/source identity now causes `FAIL`; empty matched-result cases now cause `FAIL`. Matched timing coverage remains affected by the first blocker.

CPU-only probes; writes confined to `/tmp`. No repository changes, submissions, or GPU use.

**NOT READY FOR CLUSTER SMOKE.**