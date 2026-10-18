# Speed-and-accuracy campaign protocol (2026-09-20)

Shared rules for the parallel lanes launched 2026-09-20 toward the ICLR 2027 deadline (2026-09-25 AOE). This file contains no results; it is the contract every lane agent follows. Numbers quoted here are orientation from earlier audited panels, not new claims.

## Goal set by the user

The NM-ROM must be **very accurate and faster than the named full-order model (FOM)**, and better than other nonlinear-manifold ROMs. The user is away; work autonomously, keep iterating, and keep results prepared for their return.

**Lane success bar (pre-registered):** at the largest mesh reached, the *accurate* setting (corrections on) has worst same-grid relative $L^2$ error $\le 1\%$ (stretch $\le 0.5\%$) **and** speedup $S = T_{\mathrm{FOM}}/T_{\mathrm{ROM}} \ge 5$ against the named FOM in the same allocation. Report the fast setting beside it. Missing the bar is a result: report it, never soften the comparator, never tune on final/held-out cases.

## Standing mandate: find out why we are slow, fix it, measure again

Every lane runs a profile → hypothesis → fix → re-measure loop and keeps going until the bar is met, the job budget is spent, or the remaining ideas are exhausted. Investigate every plausible cause, at minimum:

- per-query breakdown: host↔device transfer, compile/retrace, kernel-launch overhead (earlier finding: Burgers query is launch-bound, ≈17 µs/step + 141 µs/iteration), residual evaluation, Jacobian, normal-equation solve, full-field decode, output transfer;
- iteration counts and exit reasons (stalls vs stationarity), damping retries, cold vs warm start, initial-fit cost;
- dense vs empirical-quadrature residual; number of weak tests $M$; correction elimination for linear PDEs; fused residual+Jacobian; folding the head output layer into the bank (both from `exp/2026-09-16-b-speed`);
- chunked / on-demand decode at large meshes; f64 necessity per stage (any precision change must pass a parity gate at 1e-12 on integers + stated field tolerance and be labelled);
- batching and `jit` boundaries; never close a jit over a large array.

Each optimisation is accepted only with a parity gate against the unoptimised path and a same-job before/after timing. Keep a running `SPEED-LOG.md` in the lane directory: hypothesis, change, measured effect, kept or reverted.

## Non-negotiable mechanics

- Read `/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/LAB-LOG.md` ("Where things stand") and the repo `CLAUDE.md` / `AGENTS.md` first.
- Write only inside your own worktree. Read other worktrees freely; copy code in with recorded source commit.
- 27 older worktrees are sparse (heat archive hidden); do not disable that. Never create a worktree.
- Cluster: `ssh tufts-login`, `gpu` partition only, never `preempt`; venv `/cluster/tufts/paralab/tawal01/ae-research/venv`; everything under your namespace `/cluster/tufts/paralab/tawal01/<ns>/`, one directory per job; check `squeue` before and after each submit; preflight must print `jax_backend=gpu` (exit 42 otherwise); `JAX_DEFAULT_MATMUL_PRECISION=highest`; f64; regenerate data from seed on the cluster. H200 + `--mem 240G` for ≥ 2048² / ≥ 128³. Helper scripts: `/home/tahmid/Dev/ae-research/scripts/tufts-*.sh` if present, else raw `sbatch` with the same requirements. No bare `scancel` (only your own job ids).
- Budget: ≤ 2 running and ≤ 8 total GPU jobs per lane; if the account already has 6 running, wait.
- Local GB10: sub-minute smoke only, via `jaxrun` with `/home/tahmid/Dev/.venv/bin/python`.
- Before the first GPU job: write `experiments/<lane>/DESIGN.md` (question, arms, pass bar, controls, stop rules) and get an independent audit with the Codex CLI (`codex exec`, read-only) if it is available; record the audit. If Codex is unavailable, record that and self-audit against the checklist below — do not block.
- Every timing job pairs ROM and FOM in one allocation with identical I/O scope, GPU burn-in, synchronised timing, ≥ 5 retained repetitions, medians reported, GPU model + UUID recorded.
- Required arms in every comparison job: NM-ROM fast (q=0) and accurate (q>0); named FOM; fastest tested FOM with error ≤ the ROM's; coarse-grid FOM at matched accuracy against a refined reference; any direct/transform solver as a labelled control.
- Controls must fail on real data: check every threshold/gate at one real mesh before trusting a verdict.
- After each job: independent NumPy re-computation of errors from saved fields, checksum-verified pull, delete the remote job dir. Commit code, configs, logs and **small** JSON summaries only (`summary.json` with source hashes) — no field arrays, nothing > 50 MB in git.
- Maintain `experiments/<lane>/HANDOFF.md` continuously (state, job ids, what is running, next step) so an interruption loses nothing. Commit often. Do not push (GitHub pushes fail on this repo).
- Lab log: append your dated entry (what ran, numbers, **what was wrong/retracted**, what is open) under a lock: `flock /home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/.lablog.lock -c '<append>'`. Append at each milestone, not only at the end. Do not edit other entries.
- Reports: numbers only from generated JSON; LaTeX math; mermaid diagrams; glossary at the end.
- Do not touch `best-results/`, root `paper/` (paper lane only), `main`, or other lanes' cluster namespaces. No merges.

## Final message from each agent

Return: the bar verdict per mesh, the headline table (error, speedup, comparator, status), the speed log's kept optimisations with measured gains, anything retracted, exact paths of `summary.json` files, and what is still running.
