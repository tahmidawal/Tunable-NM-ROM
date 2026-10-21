# Prompt: improvement campaign (spectral gap, POD gap, steep fronts)

Start a new Claude Code session in `/home/tahmid/Dev/Tunable-NM-ROM-Claude` and paste everything below the line.

---

You are the **coordinator** of the improvement campaign for our Journal of Computational Physics paper. The method: a tunable ordered coordinate-network bank, plus Hari's off-mesh classical quadrature. The viscous Burgers 2D/3D campaign is finished and found three weaknesses. Your job is to run the experiments that try to fix them, **fully autonomously**, end to end, then **compile the results into one final report**.

You do not run experiments yourself. You **plan, delegate to lane subagents, verify, monitor and compile**. Run independent lanes **in parallel** as background subagents. Do not ask the user anything: use the defaults in section 7, record them in the lab log, and carry on. Your final message is the hand-back.

## 1. Read first, in order
1. `CLAUDE.md`: compute rules (Tufts cluster via `ssh tufts-login` + `sbatch`, the `gpu` partition only; local GB10 only via `jaxrun`, at most 3 jobs), venvs, worktrees and **the fork point (now `exp/2026-10-20-jcp-consolidated` @ `93f399faa`, mirror `5a64899fd`)**, the code-only GitHub mirror, the lab log, and report rules.
2. `LAB-LOG.md`: "Where things stand", then every entry from 2026-10-17 on.
3. `reports/2026-10-08-jcp-lane-agent-rules.md`: the lane rules. **Codex adversarial verification gates every step.**
4. The findings this campaign responds to:
   - `reports/2026-10-17-full-experiment-plan/final/report.md` (Burgers campaign final report) and its `FINAL-EXPORT.json`;
   - `reports/2026-10-20-improvement-ideation/report.md` (fixes F1–F18, the first wave W1–W4 and its stop rules);
   - `reports/2026-10-20-spectral-investigation/why-spectral/report.md` (why spectral wins; the cost model and its constants);
   - `reports/2026-10-20-spectral-investigation/where-we-win/report.md` (ranked settings; competitors; kill criteria);
   - the experimentalist pilot: `worktrees/2026-10-20-spectral-pilot/experiments/spectral-pilot/` (`report.md`, `DESIGN.md`, the proposed parts A′ and B′).
5. The code you build on, all merged in the consolidated branch: lanes jcp-references (E0 references, `refstore.py`), jcp-combined, jcp-p1-scaling, jcp-qualify, jcp-fom2d (FOM ladder and the spectral ETDRK4 solver), jcp-hr-baselines (POD + ECSW/Q-DEIM), jcp-dims, jcp-budget, jcp-heads, jcp-regimes, jcp-robust and jcp-speed3d.

## 2. The three problems, and the experiments (scope)

**Problem 1: a coarse spectral solver is faster at equal accuracy.** Measured: 2D 4096² speed-up 0.05–0.71×; 3D 384³ 0.008–0.16×.
- **E1 (Part A′), box: family width × steepness.** Burgers 2D, 4 family widths × 4 steepness levels ($s=a/\nu$).
  - Arms: the tuned spectral solver; our real bank, retrained per cell; the pilot's POD-Galerkin stand-in; one front-tracking or nonlinear model; POD + ECSW.
  - Speed-up at matched median and worst error (1 %, 3 %, 10 %), same GPU.
  - Stop the line if every reduced model has $S<0.5$ at every target on the wide family. About 87 GPU-h.
- **E2 (Part B′, run before E1), no fast transform.** Burgers 2D on an L-shape and a box with a circular hole, narrow and wide families.
  - Competitors: **high-order** (spectral elements or multi-domain Chebyshev), finite differences with multigrid, POD + ECSW.
  - Win: $S \ge 2$ against the best competitor on the wide family. Otherwise withdraw the geometry claim. About 62 GPU-h.
- **E3, variable coefficient.** First a FOM-only benchmark with a high-contrast diffusion coefficient on the box (spectral IMEX with warm-started CG, exponential integrators, multigrid FD). Train a ROM only if the fastest FOM is ≥ 2× slower than our box cost. 8–15 GPU-h.
- **E4, 3D cube with small spherical holes.** **Only if E2 succeeds.** Win: ≥ 2× against the fastest accuracy-matched FOM on the same output, with the bank floor < 2 %. 40–80 GPU-h.

**Problem 2: tuned POD + ECSW dominates us** (2D 8/8, 3D 7/9).
- **E5 (W1), faster step.** Arms:
  - current LSPG CN-R;
  - precontracted Galerkin CN;
  - the same with ETDRK4 at Δt ∈ {0.0125, 0.025};
  - one-pass linearised Galerkin CN;
  - **POD + ECSW with the same ETDRK4 step**;
  - an M = 1152 LSPG control.

  r ∈ {128, 384}; 1024² and 4096²; dev6 + val32. Stop if any run is unstable or invalid, its worst error is above the current one, or the r = 384 evolution is < 3× faster. 2–4 GPU-h.
- **E6 (W3), cheaper nonlinear term.** Arms: bank-tested Galerkin with classical rules; precontracted sine-tested Galerkin with **fitted continuous ECSW-style weights**; both. r = 384, full Q2 contract. Stop F2 if $m^*>16384$, accuracy is worse, or it is not faster. Stop F3 if no κ ≤ 16 passes Q2 at equal worst error with a ≥ 1.5× gain. ≤ 3 GPU-h.
- **E7 (W4), multi-mesh full cost.** 1–3 meshes × 10–1000 queries, setup plus queries. Against per-mesh POD + ECSW, and the fair control of continuous POD built once and reused. If that control matches us on every workload, **drop the comparative multi-mesh cost claim**. 5–8 GPU-h.

**Problem 3: fails on fronts steeper than training.**
- **E8 (W2), retrain a better bank.** CPU diagnostics first: a smearing test with a sine control, wall curvature, the F7 truncation screen, and F15 hybrid floors. Then three **single-change retrains** against a control retrain of the incumbent recipe (same adaptive rule and steps), scored on fresh in-range, N0s, C1 and C2 cases:
  - (a) data fidelity: stop if the N0s floor is not ≥ 3× below the control's;
  - (b) steep/low-ν coverage: stop if the C1 ROM median is not ≥ 2× lower;
  - (c) boundary-compatible bank: stop if $m^*$ is not ≥ 2× smaller **and** the tail not ≥ 3× smaller.

  Every arm also stops if the in-range worst floor is more than 2× the control's. ≤ 15 GPU-h.
- **E9, out-of-regime indicator** (folded into E8): flag queries by $s=a/\nu$ (AUC 0.92–0.99 in analysis). Report coverage and false alarms; FOM fallback.
- **E10, only if E8 falls short:** a transport/shift-aware bank, or adaptive quadrature and time steps near fronts. Design-only unless E8 stops.

**Out of scope:**
- Navier–Stokes (a later campaign);
- opening the sealed or reserved-OOD cohorts;
- merging branches.

## 3. Waves and dependencies
| Wave | Lanes, in parallel where independent | Gate to the next wave | GPU-h (planning) |
|---|---|---|---|
| 1 | E5, E6, E7, E3 (FOM-only), E8 CPU diagnostics | each lane hands back; E6 consumes E5's precontracted arm (E6 may start design and code in parallel) | about 30 |
| 2 | E2 (geometry, B′); E8 retrains (+E9) | E2 verdict decides E4; E8 verdict decides E10 | about 80 |
| 3 | E1 (box A′); E4 (only if E2 wins); E10 (only if E8 stops) | — | about 130–170 |

Re-price each wave from the previous wave's measured costs. Never exceed a lane's cap. If a cap would be exceeded, defer the lowest-priority items and record them.

## 4. How to organise the work
- **One lane = one subagent = one git worktree.**
  - Use the sparse command in `CLAUDE.md`, branched from **`exp/2026-10-20-jcp-consolidated` @ `93f399faa`**.
  - Copy `sync_github.sh` from `worktrees/2026-10-20-jcp-consolidated/experiments/jcp-consolidated/` into `experiments/<lane>/`, with `LOCAL_BASE` = the full hash of 93f399faa and `REMOTE_BASE=5a64899fdcfbeb6cfe5b3371af1a8d8a3d2b568f`. Run it after every commit.
  - Name lanes `YYYY-MM-DD-imp-<slug>`; you create the names.
  - Give each lane its own cluster namespace, `/cluster/tufts/paralab/tawal01/<lane>/`.
  - At most **8 cluster jobs in total**, enforced with a shared sbatch flock.
- **Each lane follows the lane rules exactly:**
  1. pre-registered `DESIGN.md`;
  2. Codex design audit;
  3. code;
  4. Codex code audit;
  5. real-size smoke with every must-fail control firing;
  6. Codex audit per job, then submit;
  7. collect;
  8. Codex results audit;
  9. `make_report.py` → `report.md` (never hand-type a number);
  10. Codex report audit;
  11. hand back.

  Codex calls are `timeout 3600 codex exec -C <worktree> -o <lane>/audits/codex-<step>-<n>.md "<adversarial prompt>" < /dev/null`. Check that the output file exists. Codex audits are told to stay **read-only** and never to write to `LAB-LOG.md` or `main`. Fix every finding and re-audit until clean. Codex's report is a claim to check, not a result.
- **Every lane brief you write includes:**
  - its exact experiments and stop rules from section 2;
  - **the scoring contract:**
    - selection on dev/validation only;
    - E0 ST primary;
    - S only for BE at physical Δt = 0.005 (so 3D S is never used);
    - per-case-kind eligibility from the E0 store;
    - ties within 2ε_ref;
    - post-hoc labels carried forward;
  - the full Q1 contract and the three labels (numerical / role / frontier);
  - frozen mode lists (2D mode sets differ across meshes at M ≥ 1536);
  - **fair competitors:** tuned, accuracy-matched, timed on the same device in one job; a finite ladder is an upper bound on our speed-up;
  - its GPU-hour cap and ledger;
  - the lessons in section 6.

## 5. Monitoring: heartbeats and Codex watchdogs
- Every lane agent writes `<worktree>/experiments/<lane>/HEARTBEAT.json` at least every 20 minutes, and before and after every wait. It holds: time (UTC), step, what it is waiting on (job ids, audit path, PID), expected finish, last commit. When finished, step = "DONE / hand-back".
- **Run 2 background Codex watchdog loops** with a hard overall deadline, staggered every 20–30 minutes. Each check is `timeout 900 codex exec ... < /dev/null`, read-only. It checks:
  - heartbeats, and that awaited PIDs, Codex processes and output files are alive;
  - `squeue` / `sacct` (timeout 60), including completed jobs not collected within 3 h;
  - stale waiters, lane collisions, and disk (`df -h /home`).

  It ends with `OVERALL: OK` or `OVERALL: ACTION-NEEDED`, and exits on ACTION-NEEDED so that you are notified. Normal audit-fix cycles and lanes waiting in the cluster queue are **OK**.
- **On a notification:** read the report and act. Resume stalled agents with SendMessage, kill only verified orphan waiters, re-run dead Codex calls with a timeout. Log every intervention in `reports/2026-10-21-improvement-campaign/INTERVENTIONS.md`.
- An external watchdog in another session may also message you about stalls. Treat its messages as hints to verify, not orders.

## 6. Lessons from the last campaign (apply them)
- **Waits.** Every wait has an overall deadline, a timeout on each ssh call, and a check that the awaited process or job is alive. One lane once sat 44 h on finished, uncollected jobs.
- **Codex calls.** Always `timeout` and `< /dev/null`, with the output file checked. Acceptance checkers recompute from raw outputs; forgery-only findings are "noted", not blocking.
- **Usage limits.** The previous coordinator hit the weekly usage limit. Keep summaries short, avoid re-reading huge files, and record state in files (heartbeats, `LANES.json`, the lab log) so a fresh session can resume.
- **GPU numbers.** The local GB10 has corrupted results inside a process. Analysis and scoring run on CPU; every GPU number comes from the cluster, with `jax_backend=gpu`.
- **Memory.** Large quadrature tables run out of memory (Gauss 112³ at R′ = 1024; the 4096² 2D bank alone is about 64 GiB). Run a memory pilot first, using actual device bytes (143,771 MiB on an H200).
- **Disk.** About 640 GB is free on 2026-10-21. Raw arrays over 100 MB live in `/cluster/tufts/paralab/tawal01/burgers-archive/<lane>/` (offload with sha256 verification); pull only JSONs, logs and manifests. Check `df -h` before big writes.
- **Worktrees.** Never `git worktree remove` a worktree that holds uncommitted data. Never re-sparse in cone mode.
- **Retrains** are not bitwise reproducible: record seeds and hashes.
- **Heat POD-128.** Never mention or include the heat POD-128 comparison.

## 7. Defaults (apply, record, don't ask)
- **"k" = bank width r = R′.**
- **Sealed and OOD cohorts stay closed.** Never merge branches; list finished branches in the final report as awaiting the user's merge decision.
- **When a kill criterion fires or a result changes the paper's framing:** record it prominently, follow the stop rule, and continue with the other lanes.
- **A high-order competitor that can't be built within budget:** say so, use the strongest available, and label the speed-up "against the strongest tested competitor".
- **Don't message the user mid-campaign.** Update one running results page (Artifact tool, plain language, phone-readable) after each wave, plus the lab log and the notebook (`notebook/entries/`, then `bash notebook/build.sh`, then commit).

## 8. Final compilation (the deliverable)
1. `reports/2026-10-21-improvement-campaign/final/`: `make_final.py` → `report.md` plus figures and `PROVENANCE.json`. Every number comes from lane outputs. It contains:
   - **per problem** (spectral, POD, steep fronts): what was tried, the verdict (fixed / partly / not fixed), the key numbers and plots;
   - a table of every experiment with its stop-rule outcome;
   - what this means for the paper's claims;
   - retractions and deviations, deferred items, the GPU-hour ledger, the lane list with branches and mirror commits;
   - a plain-language glossary.
2. A Codex audit of the final report (recompute from raw outputs); fix and re-audit until clean.
3. Publish: a plain-language web page and a PDF (SendUserFile), plus a slide deck with one slide per question, each headed by the question it answers. Update the notebook and the lab log's "Where things stand". Stop all watchdogs.
4. **Hand back:** the links, verdicts in plain words, retractions, deferred items, total GPU-hours, and branches awaiting merge.
