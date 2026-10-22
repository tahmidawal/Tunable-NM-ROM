**smk1: SUBMIT as a diagnostic smoke.**  
**smk4096: SUBMIT as a memory/runtime pilot.**

Neither verdict authorizes production or establishes that all controls pass. The staged jobs match audit 8’s smoke-ready code; the remaining acceptance/resource-certificate defects still block production.

| Item | Verdict | Finding |
|---|---|---|
| Committed provenance | **CORRECT** | Both stages pin `011f38036cf0f4bbd256f26bfb1666c5e26d0a1c`. All 38 smk1 and 35 smk4096 provenance entries match their committed blobs, byte counts and SHA256 digests. |
| Manifest completeness | **CORRECT** | All 43 smk1 and 40 smk4096 manifest entries verify. No unlisted or missing files. Generated submission wrappers match `stage.py`. |
| Slurm resources/environment | **CORRECT** | Each requests one H200, `gpu`, 200G host memory and eight CPUs. Limits are 01:00:00 / 00:30:00. Absolute cluster venv, x64, highest precision, GPU preflight with exit 42, and correct `JCPREF_STORE` are present. |
| Exit handling | **CORRECT** | Driver status propagates; missing `COMPLETE` converts success to failure. Output checksums are generated. Actual driver allowances are **56 / 26 minutes**, with four minutes reserved for shutdown. `COMPLETE` explicitly means execution only. |
| Submission/admission mechanics | **CORRECT** | Local lane lock, durable intent, campaign lock, account cap eight, lane cap two, unique directories and queue checks are implemented. Shell syntax checks pass. Live queue/storage state was not inspected. |
| E0 pins | **CORRECT** | Both configs pin the staged E0 manifest hash and prescribed metric. Scoring requests ST and checks case/time/reference identity. |
| smk1 coverage | **CORRECT** | Bank ranks 128/384, registered rules/successors, all applicable bank arms, dev6[0,1] + val32[0], controls enabled, POD ladder/native picks and three timing repetitions. |
| smk4096 coverage | **CORRECT** | Rank 384, gauss128→gauss160, all six bank arms, dev6[0], no POD, controls disabled and one timing repetition. |
| Timing anchor | **CORRECT** | `L-CNR|384` exists in both jobs. smk1 also supplies the rank-128 incumbent needed for its evolve comparisons. |
| POD inputs/pins | **CORRECT** | All three staged GECSW files match committed blobs and collected ECSW artifacts. POD basis, native artifacts and tuning JSON have runtime SHA256 assertions. Remote files were not independently accessed. |
| Claimed POD inventory | **WRONG** | The staged rules yield **30 distinct ETDRK4 members + 2 native picks**, hence **96 POD case-runs**. All four κ rules are present and distinct at each rank. The stated 26-member inventory understates this job. |
| Complete control demonstration | **NEEDS-RESTATEMENT** | smk1 schedules the real-size in-job witnesses, including two distinct dev cases and a validation case. Actual firing remains an outcome. C9 comes from timing; collected ECSW JSON reports `accepted=true`, `c11_ok=true`. Offline controls require a subsequent checker run. |
| C19 acceptance coverage | **WRONG — existing production blocker** | The checker exercises successor changes using synthetic rules but does **not** replay qualification labels after checkpoint removal. Therefore “these outputs establish every A2.7/A5.10 control” is not supported. This is the already-disclosed audit-8 gap, not a reason to prevent evidence collection. |
| Memory/runtime instrumentation | **CORRECT for pilot evidence** | Results record device bytes in use, peak bytes and limit, plus setup, arm, rank, control, POD and timing durations. Final peak counters include the timing phase if execution completes. |
| Production cost certificate | **WRONG — existing production blocker** | The current estimator double-counts some setup/control work, does not enforce GPU compatibility, and lacks a justified bound for missing rank-128 measurements at 4096² and conditional refinement work. The smokes supply inputs; they do not repair those defects. |

**Time-limit assessment: NEEDS-RESTATEMENT, plausible pilot limits rather than certified completion bounds.**

- **smk1:** Besides 33 primary bank case-runs and 96 POD case-runs, it executes successors, certificates, anchors, twins, controls and possible refinements. Timing contains **66 evolve + 258 screening triplets**, plus up to **78 direct triplets**, before retries. The corrected inventory does not demonstrate that 56 minutes is insufficient, but the old 26-member estimate cannot justify completion.
- **smk4096:** Six primary bank case-runs still entail bank construction, successors, anchors, order work, possible refinements and timing. A rank-384 full-grid bank alone is approximately **48 GiB**. The 26-minute allowance is reasonable for discovering actual peak/runtime; completion and production memory headroom remain unverified.

**GPU-hour admission: CORRECT.** The recorded ledger has one completed job, **0.1692 GPU-hours**, no reservations or unresolved intents:

\[
0.1692 + 1.0 + 0.5 = 1.6692 < 4.
\]

Both full reservations leave **2.3308 GPU-hours**. Production must be repriced from collected evidence and re-audited.

No files were changed, no scratch was created, and no SSH, cluster commands or submissions were run.