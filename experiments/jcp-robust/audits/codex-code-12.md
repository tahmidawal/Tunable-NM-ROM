**Overall verdict: READY — acceptance code may be used.** Audited HEAD `14dbfe818`. No remaining BLOCKING defect established.

Paths below are relative to `experiments/jcp-robust/`.

1. **Round 11, item 3 — RESOLVED — CORRECT — formerly BLOCKING.**

   `e2analyze.py:220–226` now rejects inputs with the wrong registered attempt, mesh, or generating selection-file hash. Lines `227–233` require each selected/deployed row’s actual rule to match its role and bind the selection artifact to the selection job.

   Population enforcement remains in `e1analyze.py:39–42,76–82,120–126`: verdict attempts require all38 or the registered 4096² subset.

   **CPU evidence:** The actual role-check statements accepted matching synthetic evidence and rejected seven mutations: wrong mesh, attempt, selection hash, selected rule, deployed rule, extra-job mesh, and selection-job ID.

   **Fix:** None required.

2. **Adversarial re-check — CORRECT — no new BLOCKING finding established.**

   The examined paths retain the relevant protections:

   - Missing evidence and malformed populations cannot qualify through `e1stats.draw_qualified`; failed runs remain failures.
   - Bank identities and decoder caches are handled by `e2bank.py:29–49` and `e2score3.py:21–45`.
   - Score provenance and exact coverage remain checked in `e1analyze3d.py:48–73` and `e2analyze3d.py:90–115`.
   - Every contributing 2D accuracy job, including the extra factor job, must be complete and GPU-backed (`e2analyze.py:234–238`).
   - Production staging binds certificates to code, job specifications, inventories, configurations, time limits, and GPU counts (`cluster/stage.py:73–93`). Collection retains checksum and completion checks before cleanup.

   **Fix:** None established in this round.

3. **Smoke acceptance wiring — CORRECT — no BLOCKING finding established.**

   For the registered jobs, **ACCEPT is reachable when all required controls, gates, and supporting evidence pass; a failed or missing required control/gate prevents successful acceptance.** Missing files may terminate with an exception rather than produce an `ALL=false` report; that still prevents acceptance.

   | Smoke | Assessment and evidence |
   |---|---|
   | **smk1b** | Requires both dimensions’ E1 controls, reproduction, target certification, integrity, and the resource contract. `smoke_accept.py:77–130`. A7’s A100 move is reflected in the resource-contract argument. |
   | **smk2** | Requires incumbent replay, trained-bank identity, analysis integrity, and the registered selection-chain outcome. `smoke_accept.py:137–194`. A6.3 explicitly permits “verdict not reached”; successful qualification is not itself required. |
   | **smk3** | Requires memory gates, registered rung populations and validity, hash-verified pilot-bank decoding, finite saved evidence, certificate coverage, a defined own-family selection result, and incumbent replay. `smoke_accept.py:197–303`. Requires the fetched `--pilot-bank` and accounting/archive evidence. |
   | **smk4** | Requires the CPU control tests and E5’s acceptance conjunction on A100. `smoke_accept.py:318–327`. |
   | **smk5** | Requires E5’s acceptance conjunction on H200. `smoke_accept.py:306–315`. The shared CPU drift test is exercised through smk4. |

   `e5analyze.py:722–764` explicitly conjoins job status, populations, doubled-work control, injected/null fixtures, reproduction, and P-cross gates. G-E5.2 remains correctly descriptive.

   **CPU evidence:** The actual E5 conjunction accepted a positive synthetic case and rejected ten individual veto mutations, with estimator functions stubbed. The actual drift/retry implementation rejected persistent drift and successfully retried transient drift.

   **Fix:** None required.

4. **Carried-forward CLI documentation discrepancy — NEEDS-FIX — MINOR.**

   `e2verdicts.py:13` advertises `--seeds`, but `:102` uses the fixed registered seeds and ignores additional arguments. The implemented three-seed population is correct.

   **Fix:** Remove the advertised option or reject unsupported arguments. This does not block acceptance.

All 17 requested Python files parsed successfully. This was a source and in-memory predicate audit, not acceptance of actual smoke results. No smoke outputs were read, files changed, scratch created, jobs submitted, or SSH/cluster commands executed.