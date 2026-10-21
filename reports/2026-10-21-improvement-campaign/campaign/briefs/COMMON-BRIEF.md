# Common brief for every lane agent in the improvement campaign (2026-10-21)

You are a **lane agent** in the improvement campaign for the JCP paper (tunable ordered coordinate-network
bank + Hari's off-mesh classical quadrature, viscous Burgers 2D/3D). A coordinator session launched you and
reads your hand-back. This file is binding for every lane; your lane brief (`briefs/<lane>.md`, same folder)
adds the items you own. Where they conflict, your lane brief wins.

- Repository root (main): `/home/tahmid/Dev/Tunable-NM-ROM-Claude`. CAMPAIGN-DIR =
  `reports/2026-10-21-improvement-campaign/` on main (the campaign prompt is `AGENT-PROMPT.md` there).
- Your worktree: `worktrees/2026-10-21-<lane>/`, branch `exp/2026-10-21-<lane>`, forked from
  `exp/2026-10-20-jcp-consolidated` @ `93f399faa` (all 19 JCP lanes merged). Your lane directory is
  `experiments/<lane>/` (already holds `sync_github.sh`, preconfigured; run it after **every** commit).
- **Sparse checkout:** the heavy `runs/` folders of jcp-combined, jcp-speed3d, jcp-hr-baselines, jcp-robust,
  jcp-budget, jcp-wide-bank, jcp-time2, jcp-regimes, quadrature-burgers3d, separable-decoder (and ns2d/ns3d,
  mr-heat2d runs) are **not checked out** in your worktree. Read them **read-only** from the full checkout
  `worktrees/2026-10-20-jcp-consolidated/experiments/<lane>/runs/...` (same commit). Copy a small file into
  your lane if you need it; never write into the consolidated worktree. Never re-sparse in cone mode.

## 0. Read first
1. `CLAUDE.md` on main (compute rules, venvs, cluster mechanics, worktrees, code-only mirror, report rules).
2. `reports/2026-10-08-jcp-lane-agent-rules.md` — **follow it exactly**. It is your operating manual. (Its
   "references are first-order" paragraph is superseded: E0 second-order references exist; see §2.)
3. `LAB-LOG.md` top block and the entries from 2026-10-17 on (read-only for you).
4. The findings this campaign responds to (read the parts your brief names):
   `reports/2026-10-17-full-experiment-plan/final/report.md` (+ `FINAL-EXPORT.json`);
   `reports/2026-10-20-improvement-ideation/report.md` (fixes F1–F18, wave W1–W4);
   `reports/2026-10-20-spectral-investigation/why-spectral/report.md` and `.../where-we-win/report.md`;
   the pilot `worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/` (`report.md`, `DESIGN.md`).
5. The finished lanes your brief names (their `DESIGN.md`, `report.md`, code).

## 1. Sequence (no skipping; lane rules)
1. `DESIGN.md` (pre-registration: question, arms, must-fail controls, metrics, pass/fail and **stop rules
   copied verbatim from your brief**, cohorts, GPU-h estimate, deliverables) → 2. Codex design audit →
3. code → 4. Codex code audit → 5. real-size smoke where **every must-fail control is shown to fire** →
6. Codex audit of each job, then submit → 7. collect → 8. Codex results audit → 9. `make_report.py` →
`report.md` (never hand-type a number) → 10. Codex report audit → 11. hand back.

Codex calls, always:
```
timeout 3600 codex exec -C <worktree> -o experiments/<lane>/audits/codex-<step>-<n>.md "<adversarial prompt>" < /dev/null
```
then check the output file exists and is non-empty. Every Codex prompt must say: **read-only — make no file
changes, submit no jobs, never write to LAB-LOG.md or to main; do not copy large arrays (scratch ≤ 2 GB
under `/tmp/<lane>-audit-scratch/`)**; try to break the item; give a per-item CORRECT / WRONG /
NEEDS-RESTATEMENT verdict. Fix every WRONG and re-audit until clean. Codex's report is a claim to check, not
a result. Prompts that ask Codex to "fake a PASS" trip its content filter; ask about honest failure modes.
If a Codex call dies or times out, re-run it once in the foreground; if it dies twice, record that in
`audits/` and in your heartbeat. Delete the audit scratch dir after reading each audit.

## 2. Scoring contract (binding)
- **Selection on dev/validation cohorts only** (2D dev6 + val32; 3D 923801). Historical test cohorts (2D
  test64, 3D 923901) are replication-only. **Never open sealed or reserved-OOD cohorts**; never touch `sealed/`
  or `custody/` in the E0 store. New cohorts you generate yourself (new geometries, new families) are split
  dev / val / test-once by seed **before** any run, and the test split is scored once at the end.
- **E0 ST is the primary reference** (loader `refstore.py` in `experiments/jcp-references/`; store
  `JCPREF_STORE=/cluster/tufts/paralab/tawal01/jcp-ref/store`). For new problems with no E0 entry, build a
  reference with the same recipe (second-order, Richardson ε_ref, labelled conditional).
- **S** (space-only reference) is used **only for BE at physical Δt = 0.005**. 3D deploys BE at 0.01, so
  **3D S is never used**. Any other S comparison is an "unmatched diagnostic".
- **Eligibility per case-kind from the E0 store**, never a fixed count.
- Two arms rank on a case only if they differ by more than **2 ε_ref** (otherwise a tie). ε_ref is a
  conditional Richardson estimate: label it so.
- **Post-hoc labels carry forward unchanged:** E0 A10; combined A10, A11 and the retracted A15.
- Cases are the statistical unit; never bootstrap time snapshots as independent. Report failures.

## 3. Q1 contract and three labels
An arm/configuration is **numerically qualified (Q1)** only if, certified **in-job at every step with a
fresh Jacobian** (for non-Newton steppers: the analogous in-job residual/stationarity certificate stated in
your DESIGN and audited):
- every run valid (finite, no non-finite step, no damping saturation);
- stationarity ratio $\gamma = \|J^\top r\|/(\|J\|_F\|r\| + 10^{-300})$; a step is non-stationary if
  $\gamma > 10^{-3}$ or non-finite; non-stationary steps ≤ 1 % **pooled over steps**;
- distance to the adaptive tight solve ≤ 2.5e-4 (scoring-lattice relative distance) on **every** case;
- tight/tighter anchor agreement ≤ 2.5e-5 on the prescribed subset.
**Q2** (quadrature) = agreement between successive rules on the arm's **own reached states**, on both the ρ
bar (0.116) and the rollout-distance bar (2.5e-4, every case). Earlier lanes' labels are prior evidence only.
Keep three labels separate: (1) **numerical** qualification (Q1+Q2), (2) **role** qualification (meets a
pre-declared role/cost criterion, e.g. "equal cost"), (3) **frontier** usefulness (not dominated).

## 4. Frozen mode lists
2D test-mode sets are built by sorting mesh-dependent discrete Laplacian eigenvalues (`hops.py:35–41`); they
**differ across meshes at M ≥ 1536** (4 modes at 1536, 8 at 2048, 256² vs 1024²). 3D shell completion makes
M mesh-dependent. Any **fixed-mode** comparison uses one explicit, hashed mode list per (dimension, M): 2D
taken at 1024², 3D at 65³, reused at every mesh (new configurations, need Q1/Q2). **Nominal-policy** panels
keep per-mesh lists, labelled "deployment policy (mode set varies with mesh)", with the symmetric difference
reported.

## 5. Fair competitors (binding)
Every competitor (FOM or reduced) is **tuned** over its own knobs (resolution, order, step, tolerance, rule
size, r), **accuracy-matched** on the same error metric and cases, and **timed on the same device in the same
job** as our arm (paired A–B–A, warm JIT, burn-in, f64). A finite competitor ladder is an **upper bound on our
speed-up**: say so next to every S. If a strong competitor cannot be built within budget, say so, use the
strongest available, and label the speed-up "against the strongest tested competitor". Never compare times
across GPU types.

## 6. Compute
- Tufts first (`ssh -o ConnectTimeout=20 tufts-login`, `sbatch`, `gpu` partition only, never `preempt`).
  Venv `/cluster/tufts/paralab/tawal01/ae-research/venv`; `JAX_DEFAULT_MATMUL_PRECISION=highest`; f64;
  GPU preflight (exit 42 if not gpu); grep `jax_backend=gpu` in every log. One job per directory under your
  namespace `/cluster/tufts/paralab/tawal01/<lane>/<job>/`. `squeue -u tawal01` before AND after every submit.
- **Campaign-wide cap: ≤ 8 queued+running cluster jobs across all lanes**, enforced by one shared lock.
  Every `sbatch` goes through it:
  ```
  timeout 700 ssh -o ConnectTimeout=20 tufts-login 'flock -w 600 /cluster/tufts/paralab/tawal01/.campaign-sbatch.lock \
    bash -c "n=\$(squeue -u tawal01 -h | wc -l); [ \$n -lt 8 ] || exit 75; cd <jobdir> && sbatch run.sbatch"'
  ```
  Exit 75 = account at cap; retry later. Your lane's own concurrent-job cap is in your brief; never exceed it.
- Waiting on a Slurm job: break on `sacct -j <id> -X -n -o State` reaching a terminal state (COMPLETED,
  FAILED, CANCELLED, TIMEOUT, OUT_OF_MEMORY, NODE_FAIL), `timeout 60` on every ssh, and an overall deadline.
  `squeue -j` fails once a job has left the queue. **Collect a finished job within 1 h** (one lane once sat
  44 h on finished, uncollected jobs).
- Local GB10: only via `jaxrun`, and only through the shared slot guard (3 slots for the whole box):
  `bash /home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-21-improvement-campaign/campaign/jaxslot.sh 3600 jaxrun /home/tahmid/Dev/.venv/bin/python <script> ...`
  (exit 75 = no slot). CPU-only analysis: `JAX_PLATFORMS=cpu`, still under the slot guard if it imports JAX.
  **Every GPU number comes from the cluster** with `jax_backend=gpu` (the GB10 GPU has corrupted results
  inside a process). Analysis and scoring run on CPU.
- Memory: large quadrature tables OOM (Gauss 112³ at R′ = 1024; the 4096² 2D bank alone is ≈ 64 GiB). Run a
  memory pilot first, using actual device bytes (143,771 MiB usable on an H200).
- Disk: ≈ 600 GB free locally on 2026-10-21, shared by all lanes. Check `df -h /home` before big writes and
  before each Codex audit (do not start one below 60 GB free). Raw arrays > 100 MB live in
  `/cluster/tufts/paralab/tawal01/burgers-archive/<lane>/<job>/`; pull only JSONs, logs, manifests and files
  < 100 MB; commit a sha256 listing of the archive. `campaign/offload.sh <lane> <job> <dir>` offloads an
  already-pulled local archive with verification. Delete cluster job dirs after collection.
- Retrains are not bitwise reproducible: record seeds and hashes.

## 7. GPU-hour ledger
Keep `experiments/<lane>/runs/GPU-LEDGER.json`: every job's id, GPU type, measured elapsed GPU-hours (from
`sacct`), cumulative total. Submit only if `spent + certified projection ≤ cap`. Projections come from a
smoke-derived cost certificate (pattern: `experiments/jcp-speed3d/resources.py`). Never exceed your cap;
defer items in your brief's defer order and record them.

## 8. Lessons (each cost a lane before)
- Every wait has an **overall deadline**, a `timeout` on each remote call, and a check that the awaited
  **process is alive** (`kill -0 <pid>` on the PID you captured), never only "the file is missing", and
  never `pgrep -f <pattern>` that matches your own waiter.
- Every Codex call has `timeout` and `< /dev/null`; check the output file afterwards.
- Acceptance checkers recompute every predicate from raw outputs. Forgery-only findings are "noted", not
  blocking.
- **Usage limits.** Keep your context lean: do not cat huge files, use `head`/`grep`/`sed -n`. Record state
  in files (HEARTBEAT, DESIGN amendments, ledger, commits) so a fresh agent can resume. If you are resumed
  after a stop, re-read your HEARTBEAT and last commits before acting.
- Never mention or include the heat POD-128 comparison. Never quote a pre-September Burgers speed-up.
- Never `git worktree remove` anything. Never merge branches.

## 9. Heartbeat (mandatory)
Write `<worktree>/experiments/<lane>/HEARTBEAT.json` at least every 20 minutes while working, and immediately
before and after every wait (job queue, Codex audit, local run). Schema:
```json
{"lane": "...", "utc": "2026-10-21T23:40:00Z", "step": "6: codex job audit for x",
 "waiting_on": {"slurm_jobs": ["53xxxxx"], "codex_output": "audits/codex-job-x-1.md",
                "codex_pid": 12345, "local_pid": null},
 "expected_finish_utc": "2026-10-22T01:00:00Z", "last_commit": "abc1234",
 "gpu_hours_spent": 0.0, "gpu_hours_cap": 4, "notes": "..."}
```
Use `date -u +%Y-%m-%dT%H:%M:%SZ`. When finished: `"step": "DONE / hand-back"`. Read-only Codex watchdogs
check these files every ~25 minutes; a heartbeat older than 60 minutes with no live wait gets you resumed.
Long waits: run them as foreground commands with `timeout` (≤ 10 min each) in a loop that rewrites the
heartbeat between polls, or as a background waiter with a deadline whose PID you record.

## 10. Writing
Only inside your own worktree. Never write to main, `LAB-LOG.md`, `notebook/`, another worktree, or another
lane's cluster namespace. Commit small and often; after every commit run `bash experiments/<lane>/sync_github.sh`
(never `git push` an exp/ branch). Files > 50 MB: SHA256 manifest. Reports: level-1 title, status line,
LaTeX maths, mermaid diagrams, provisional labels inline next to numbers, closing plain-language glossary,
every number generated by `make_report.py` from run JSONs. Also write `experiments/<lane>/EXPORT.json`
(machine-readable key numbers + verdicts, with source paths) for the coordinator's final compilation, and
PNG plots for anything presentation-worthy. Retractions matter more than successes: if something contradicts
an earlier report, say so plainly.

## 11. Hand-back (final message to the coordinator) — keep it ≤ 60 lines
- what ran (job ids, GPUs, namespace, GPU-hours from the ledger);
- the key numbers, read from the generated report, with each **stop-rule outcome** (fired / not fired);
- what is provisional and why; anything wrong or retracted;
- every audit file with its verdict (one line each);
- paths of report, EXPORT.json, plots, DESIGN, ledger;
- the last lane commit and its mirror commit;
- deferred items and open questions.
