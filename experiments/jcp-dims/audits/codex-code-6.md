**READY-FOR-SMOKE.** Both round-5 fixes are correct. I found no additional reproducible blocker in `smk2` → `dacc`, `smk2t` → `dtime`, or `smk3` → `d3run`.

Paths below are relative to `experiments/jcp-dims/`.

| Round-5 finding | Verdict | File:line evidence |
|---|---|---|
| Smoke misses the heavy three-table rounds path | **CORRECT** | `configs/smk3.json:328–334` adds `W1024|lat65536|fs|CN|0.01|K1`, with lat131072 and lat262144 successors. `d3run.py:518–520` invokes `q3run.member`; `q3run.py:53–65` exercises the simultaneous three-table path. Block peak and the 120 GB rejection are recorded at `d3run.py:555–560`. |
| Rounds control retains obsolete tables | **CORRECT** | `d3run.py:527–533` drops the control’s main and successor rules after `member` returns, then collects garbage. `s3common.py:130–131` removes their cached data. This occurs before the subsequent fixgate and timing blocks. |

**Validation:** 102/102 CPU regression checks passed, with bytecode disabled and the file-writing test entry point bypassed. Static tracing covered the smoke configurations, certification links, table lifetimes, timing inventory and staging environment. Actual GPU memory peaks, numerical controls and runtime remain for the cluster smokes to establish.

No files changed, jobs submitted, GPU used, or scratch created.

**Overall: READY-FOR-SMOKE. Ranked remaining blockers: none.**