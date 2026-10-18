# Common brief for every lane agent in the Burgers 2D/3D campaign (2026-10-17)

You are a **lane agent** in the viscous Burgers 2D/3D JCP experiment campaign. A coordinator session
launched you and reads your hand-back. This file is binding for every lane; your lane brief
(`briefs/<lane>.md`, same folder) adds the items you own. Where they conflict, your lane brief wins.

Repository root: `/home/tahmid/Dev/Tunable-NM-ROM-Claude` (main). Campaign plan folder:
`reports/2026-10-17-full-experiment-plan/` (call it PLAN-DIR).

## 0. Read first
1. `CLAUDE.md` on main (compute rules, venvs, cluster mechanics, worktrees, code-only mirror, report rules).
2. `reports/2026-10-08-jcp-lane-agent-rules.md` — **follow it exactly**. It is your operating manual.
3. The plan: `PLAN-DIR/PLAN-v2.md`, then `PLAN-v3-amendments.md`, then `PLAN-v4-amendments.md`
   (later files supersede earlier ones), and the three Codex audits `codex-plan-audit-{1,2,3}.md` (known traps).
4. `LAB-LOG.md` top block and the entries from 2026-10-08 onwards (read-only for you).
5. The finished lanes you build on (their `report.md` and `DESIGN.md`) as named in your lane brief.

## 1. Sequence (no skipping; lane rules)
1. `DESIGN.md` (pre-registration) → 2. Codex design audit → 3. code → 4. Codex code audit →
5. real-size smoke where **every control is shown to fire** → 6. Codex audit of each job, then submit →
7. collect → 8. Codex results audit → 9. `make_report.py` → `report.md` (never hand-type a number) →
10. Codex report audit → 11. hand back.

Codex calls, always:
```
timeout 3600 codex exec -C <worktree> -o <lane-dir>/audits/codex-<step>-<n>.md "<adversarial prompt>" < /dev/null
```
then check the output file exists and is non-empty. Every Codex prompt must say: **read-only — make
no file changes, submit no jobs, never write to LAB-LOG.md or to main**; try to break the item; give a
per-item CORRECT / WRONG / NEEDS-RESTATEMENT verdict. Fix every WRONG and re-audit until clean. Codex's
report is a claim to check, not a result. Prompts that ask Codex to "fake a PASS" can trip its content
filter; ask about honest failure modes instead. If a Codex call dies or times out, re-run it once in the
foreground; if it dies twice, record that in the audit folder and in your heartbeat.

## 2. Scoring contract (plan §3 as amended by V3-1, V4-2)
- Selection on **dev/validation cohorts only** (2D dev6 + val32; 3D 923801). Historical test cohorts
  (2D test64, 3D 923901) are replication-only. **Never open sealed or reserved-OOD cohorts.** Never
  touch `sealed/` or `custody/` in the E0 store.
- **E0 ST is the primary reference** (loader `refstore.py` in `worktrees/2026-10-06-jcp-references`,
  store `JCPREF_STORE=/cluster/tufts/paralab/tawal01/jcp-ref/store`).
- The S reference is used **only for BE at physical Δt = 0.005**. In 3D the deployed BE step is 0.01,
  so **3D S is never used**. Any other S comparison is an "unmatched diagnostic".
- **Eligibility is read per case-kind from the E0 store**, never applied as a fixed count.
- Two arms rank on a case only if they differ by more than **2 ε_ref** (ties otherwise). ε_ref is a
  conditional Richardson estimate: label it so.
- **Post-hoc labels carry forward unchanged:** E0 A10; combined A10, A11 and the retracted A15.
- Cases are the statistical unit; do not bootstrap time snapshots as independent. Report failures.

## 3. Q1 contract (V4-2), three separate labels
An arm/configuration is **numerically qualified (Q1)** only if, certified **in-job at every step with a
fresh Jacobian**:
- every run valid (finite, no non-finite step, no damping saturation);
- stationarity ratio $\gamma = \|J^\top r\|/(\|J\|_F\|r\| + 10^{-300})$; a step is non-stationary if
  $\gamma > 10^{-3}$ or non-finite; non-stationary steps ≤ 1 % **pooled over steps**;
- distance to the adaptive tight solve ≤ 2.5e-4 (scoring-lattice relative distance) on **every** case;
- tight/tighter anchor agreement ≤ 2.5e-5 on the prescribed subset.
Q2 (quadrature) = agreement between successive rules on the arm's **own reached states**, on both the
ρ and the rollout-distance bars. Combined's finalist labels are prior evidence only, not a certificate.
Keep three labels separate: (1) **numerical qualification** (Q1+Q2), (2) **role qualification**
(meets a pre-declared role/cost criterion, e.g. "equal cost"), (3) **frontier usefulness** (not dominated).

## 4. Frozen mode lists (V4-1)
2D test-mode sets are built by sorting mesh-dependent discrete Laplacian eigenvalues
(`hops.py:35–41`); they **differ across meshes at M ≥ 1536** (4 modes at 1536, 8 at 2048, 256² vs 1024²).
3D shell completion makes M mesh-dependent. Any **fixed-mode** panel uses one explicit, hashed mode list
per (dimension, M): 2D taken at 1024², 3D at 65³, reused at every mesh (these are new configurations
and need Q1/Q2). **Nominal-policy** panels keep per-mesh lists, labelled "deployment policy (mode set
varies with mesh)", and report the symmetric difference against the frozen list.

## 5. Compute
- Tufts first (`ssh -o ConnectTimeout=20 tufts-login`, `sbatch`, `gpu` partition only, never `preempt`).
  Venv `/cluster/tufts/paralab/tawal01/ae-research/venv`; `JAX_DEFAULT_MATMUL_PRECISION=highest`; f64;
  GPU preflight (exit 42 if not gpu); grep `jax_backend=gpu` in every log. One job per directory under
  your namespace `/cluster/tufts/paralab/tawal01/<namespace>/<job>/`. `squeue -u tawal01` before AND
  after every submit. Pull results, then delete the cluster job dir.
- **Campaign-wide cap: ≤ 8 concurrent cluster jobs across all lanes.** Your lane's own cap is in your
  brief; never exceed it. Before each submit, also run `squeue -u tawal01 -h | wc -l` and do not submit
  if it would push the account above 8.
- Local GB10: only via `jaxrun`, **at most one local jaxrun per lane**, CPU preferred for analysis
  (`JAX_PLATFORMS=cpu`). The GB10 GPU has corrupted results inside one process before ("Red zone",
  a wrong condition number): **every GPU number comes from the cluster.** Local disk is ~94 % full
  (≈228 GB free): check `df -h /home` before any large write; keep big arrays on the cluster.
- Memory: large quadrature tables OOM (Gauss 112³ at R′=1024 needed 42.9 GiB; the 4096² R512 2D bank
  alone is ≈64 GiB). Memory pilot first, using actual device bytes (143,771 MiB on H200).
- Retrains are not bitwise reproducible: record seeds and hashes.

## 6. GPU-hour ledger
Keep `experiments/<lane>/runs/GPU-LEDGER.json`: every job's id, GPU type, measured elapsed GPU-hours
(from `sacct`), and cumulative total. Submit a new job only if `spent + certified projection ≤ cap`.
Projections come from a smoke-derived cost certificate (pattern: `worktrees/2026-10-14-jcp-speed3d/
experiments/jcp-speed3d/resources.py`). Never exceed your cap; defer items in your lane's defer order
and record them.

## 7. Lessons (each cost a lane before)
- Every background wait has an **overall deadline**, a `timeout` on each remote call, and checks that
  the awaited **process is alive** (`ps -p <pid>`), not only that a file is missing. Never write an
  `until`/`sleep` loop without a deadline. One agent waited 18 h on a Codex file that was never written.
- Every Codex call has `timeout` and `< /dev/null`; check the output file afterwards.
- Acceptance checkers recompute every predicate from raw outputs. Forgery-only findings are "noted",
  not blocking.
- Analysis and scoring on CPU.
- If you hit a usage limit and are resumed, re-read your HEARTBEAT and last commits before acting.

## 8. Heartbeat (mandatory)
Write `<worktree>/experiments/<lane>/HEARTBEAT.json` at least every 20 minutes while working, and
immediately before and after every long wait (job queue, Codex audit, local run). Keep it uncommitted
or commit it — either is fine — but keep it current. Schema:
```json
{"lane": "...", "utc": "2026-10-17T23:40:00Z", "step": "6: codex job audit for p1c",
 "waiting_on": {"slurm_jobs": ["5301xxx"], "codex_output": "audits/codex-job-p1c-1.md",
                "codex_pid": 12345, "local_pid": null},
 "expected_finish_utc": "2026-10-18T01:00:00Z", "last_commit": "abc1234",
 "gpu_hours_spent": 3.2, "gpu_hours_cap": 20, "notes": "..."}
```
Use `date -u +%Y-%m-%dT%H:%M:%SZ`. Read-only Codex watchdogs check these files every ~10 minutes; a
heartbeat older than 60 minutes with no live wait gets you resumed by the coordinator.

## 9. Writing
Only inside your own worktree. Never write to main, `LAB-LOG.md`, `notebook/`, another worktree, or
another lane's cluster namespace. Commit small and often; after every commit run your lane's
`sync_github.sh` (never `git push` an exp/ branch). Files > 50 MB: SHA256 manifest. Reports: level-1
title, status line, LaTeX maths, mermaid diagrams, provisional labels inline, closing plain-language
glossary, every number generated. Never mention or include the heat POD-128 comparison. Never quote a
pre-September Burgers speed-up. Retractions matter more than successes: if something contradicts an
earlier report, say so plainly.

## 10. Hand-back (final message to the coordinator)
- what ran (job ids, GPUs, namespace, GPU-hours from the ledger);
- the key numbers, read from the generated report, with each hypothesis verdict (pass/fail/inconclusive);
- what is provisional and why;
- anything wrong or retracted;
- every audit file with its verdict;
- paths of report, plots (PNG), DESIGN, ledger;
- the last lane commit and its mirror commit;
- deferred items and open questions.

## 11. Disk hygiene for Codex audits (added 2026-10-18 02:50 UTC by the coordinator)
Codex audits have been copying ~10 GB of lane data into `/tmp/<lane>-roundN-audit/` per round (speed3d:
89 GB in rounds 5–12, local disk fell to 95 GB free). From now on: every Codex prompt must say "do not copy
large arrays; if you need scratch, use at most 2 GB under `/tmp/<lane>-audit-scratch/` and reuse it";
after each audit round the lane agent deletes that round's scratch dir once the audit file is read
(check `pgrep -af <dir>` first). Check `df -h /home` before each audit; do not start one below 60 GB free.

## 12. Waiting on a process (added 2026-10-18 03:10 UTC)
Never wait with `pgrep -f <pattern>` where the pattern also appears in your own waiter's command line — it
matches itself and never ends (happened in jcp-p1-scaling). Wait on the **PID** (`while kill -0 $PID`)
captured when you launched the process, or run the Codex call in the foreground with `timeout`.
