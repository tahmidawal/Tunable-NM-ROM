# Handoff prompt — paste this into a fresh session

Everything below is self-contained. The new session's job is to launch two subagents (on
**Opus** — a Fable usage limit killed three agents on 16 Aug), monitor them, and report.

---

```
Continue the INR NM-ROM project. Read
/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-17-inr-rom-consolidated/HANDOFF.md
first — it is the consolidated tree (branch exp/2026-08-17-inr-rom-consolidated, pushed)
holding all four completed experiment cells, the reports pipeline, and the full state.

CONTEXT IN ONE PARAGRAPH. We replaced the paper's ViT-CP decoder with a FiLM coordinate
decoder (INR, u(x,t;z)). The latent solve used to stall ~8x above the decoder's own ceiling
because minimising the pointwise FD residual amplifies grid-scale decoder error by ~(N-1)^2.
The fix: minimise the residual projected onto M low smooth sine test modes (weak form,
Lambda^-1 weighting), with NNLS-EQ quadrature weights fitted on DECODER-OUTPUT snapshots,
m ~ 4M points, meshfree pool fine. That reaches the decoder ceiling on Poisson (7.65e-3 vs
7.11e-3), Burgers (1.65e-2 vs 1.15e-2) and Heat (1.87e-2 vs 1.16e-2), is init-insensitive,
and is flat in N. It FAILS on Wave for a structural reason (nonlinear manifold destroys the
time-reversal symmetry that makes the CN/Newmark recurrence conservative) — do not try to fix
Wave with knobs. Operating rules, including M > k and "hyper-reduce the cold start too", are
in HANDOFF.md section "Operating rules".

The user has approved two new experiments and two worktrees. Launch ONE SUBAGENT PER
EXPERIMENT, in parallel, both with model "opus", both branched off the consolidated branch:

  git worktree add -b exp/2026-08-17-cost-to-tolerance \
    worktrees/2026-08-17-cost-to-tolerance exp/2026-08-17-inr-rom-consolidated
  git worktree add -b exp/2026-08-17-rom-warmstart-fom \
    worktrees/2026-08-17-rom-warmstart-fom exp/2026-08-17-inr-rom-consolidated
  (push each with -u immediately)

Work goes in experiments/cost-to-tolerance/ and experiments/rom-warmstart-fom/ respectively.
Both cover POISSON-2D and BURGERS-2D only. Both reuse the existing harnesses in the tree
(poisson2d-rom-objective/, burgers2d-rom-latent-stepping/) — the reference ROM implementation
is blat_common.py / blat_rom.py; import, do not re-implement.

=========================================================================
AGENT 1 — experiments/cost-to-tolerance/
=========================================================================
QUESTION: how does the cost of the latent solve depend on k, and does that k-dependence
change with mesh resolution N? We currently have only two 1-D slices that cross at one point:
cost-vs-k at N=64, and cost-vs-N at k=8. We need the surface.

TWO PROTOCOL DEFECTS IN THE EXISTING k DATA THAT THIS EXPERIMENT MUST NOT REPEAT:
 (a) Iteration counts are to solver TERMINATION, not to a tolerance, so different k stop at
     different accuracy and "cost vs k" conflates work with target.
 (b) Cost and accuracy came from DIFFERENT runs (timing used mean-init on a single source;
     accuracy used nearest-init averaged over 16 sources). The cross-check column disagrees
     with the accuracy table by up to 7x at some k. Because of this you CANNOT build an
     iso-error Pareto curve from the committed tables. Cost and accuracy must come from the
     SAME run, same init, same sources.

TOLERANCE DESIGN (decided — do not restate the FD-residual version):
A tolerance in the true discrete residual ||Au-f||/||f|| is UNREACHABLE: at the weak-form
solution that quantity is ~2e-1 while the field error is 8e-3 (that amplification is the
whole point of the weak form). So stop on the RELATIVE REDUCTION OF THE OBJECTIVE ACTUALLY
BEING MINIMISED, measured from the initial guess:
    tau in {1e-1, 1e-2, 1e-3}     ("good enough" / "converged" / "at the manifold floor")
This is deployable — it needs no oracle. For every cell report, from the same run:
  - work: Jacobian evaluations, LM attempts, ms/iteration, total solve ms, cold-start ms
  - the FIELD ERROR actually achieved at each tau (the knob -> accuracy map)
  - the censoring rate: cells that never reach tau (report honestly, never drop them)
  - for reference, the achieved ||Au-f||/||f|| at each tau

GRID: k in {2,4,6,8,12,16,24,32} x N in {32,64,128,256,512} for Poisson and
{32,64,128,256} for Burgers. Fixed M=64 and m=256, EXCEPT M=256 whenever k >= 32 (the
weak form COLLAPSES when M <= k — heat M=16,K=16 -> 9.0e-2 against a 6.3e-3 ceiling;
burgers POD k=64,M=64 diverged). Include a POD arm at every (k, N) with the same
hyper-reduction and the same solver.

NO RETRAINING IS NEEDED. The coordinate decoder is meshfree, so the existing k-ladder
checkpoints (runs/ in the two experiment dirs) evaluate natively at every N — this is exactly
what the committed Burgers N-ladder did. EQ weights must be REFIT on each N.

TIMING PROTOCOL (non-negotiable): every cell of the whole grid runs SEQUENTIALLY IN ONE JOB
ON ONE GPU — cross-N and cross-k timings measured on different GPUs are invalid and this has
burned us before. Warm-up 2, median of 7, device sync (block_until_ready), the FOM compiled
identically to the ROM. One job for Poisson, one for Burgers.

DELIVERABLES: cost(k) curves overlaid across N (the test of whether the k-dependence is
genuinely N-independent — theory says it should be, we have never checked); iterations-to-tau
vs k; and the ISO-ERROR PARETO (error vs wall time, coordinate vs POD, per N) — the table the
project currently cannot produce. Expect ~1 h Poisson, ~4-6 h Burgers.

=========================================================================
AGENT 2 — experiments/rom-warmstart-fom/
=========================================================================
QUESTION: after the ROM reaches a tolerance, hand its solution to the FULL-ORDER solver as an
initial guess and finish to full accuracy. What is the total cost, and how does it scale with
N? This converts the ROM's accuracy ceiling from a limitation into a non-issue: the answer is
FOM-exact, so the only question left is cost — which sidesteps every accuracy objection the
reviewers raised, including "a direct solver beats you".

POISSON: decode the ROM solution, hand it to CG as the initial guess, run to
tau_FOM in {1e-6, 1e-8, 1e-10} (relative discrete residual). Measure CG iterations from the
ROM start vs from a zero start, and total = ROM solve + decode + CG. Then sweep the ROM's own
stopping tolerance (use Agent 1's tau ladder: 1e-1, 1e-2, 1e-3) — the headline figure is
TOTAL TIME vs ROM TOLERANCE, one line per N, against the flat pure-FOM baseline. There is an
optimum in there; finding it is the point. N in {32,64,128,256,512}.

BURGERS: use the ROM state at step n as the Newton initial guess instead of u_{n-1}. Count
Newton iterations AND inner BiCGStab iterations per step, from both starts, and the total
wall clock. N in {32,64,128,256}.

TWO RISKS — STATE THEM UP FRONT IN THE README, AND MEASURE THEM RATHER THAN ASSUMING:
 (a) On Poisson at small N the FOM is 7.8 ms while the ROM alone is 20 ms, so the hybrid
     CANNOT win below some N. Locating that crossover resolution is a real result either way.
 (b) On Burgers the FOM already warm-starts from the previous time step, which is a good
     guess, so the ROM must beat THAT. Expect the win concentrated in the first steps and at
     large dt, and it may be small or negative. Report it honestly.

Same one-GPU sequential timing protocol as Agent 1. Expect ~1 h Poisson, ~3 h Burgers.

=========================================================================
RULES FOR BOTH AGENTS
=========================================================================
COMPUTE (CLAUDE.md is authoritative): Tufts is the default; local GB10 only for sub-minute
smokes, and each agent gets ONE jaxrun slot (3 max on the box). Local:
  source /etc/profile.d/jax-mem.sh ; JAX_DEFAULT_MATMUL_PRECISION=highest ;
  /home/tahmid/Dev/.venv/bin/python ; f64.
CLUSTER: gpu partition only (never preempt), --gres=gpu:1, the exit-42 jax_backend=gpu
preflight, venv /cluster/tufts/paralab/tawal01/ae-research/venv, logs inside the cell dir.
Model run.sbatch on experiments/burgers2d-rom-latent-stepping/cluster/. Namespaces:
/cluster/tufts/paralab/tawal01/ctol/<cell>/ for Agent 1 and .../wsfom/<cell>/ for Agent 2 —
never touch the other's. ONE JOB PER DIRECTORY. scp code directly into the paralab path,
NEVER stage through login /tmp (login nodes are load-balanced with node-local /tmp; a stale
script silently corrupted a round once). Checksum staged files, check squeue BEFORE and AFTER
every submit, pull with checksums, delete the cluster cell dir when its results are local.
Data is regenerated on the cluster from seed, never synced. POD Gram/eigh on host CPU (the
all-slice f64 Gram OOMs an 80 GB A100 at N>=128).

VERIFICATION (the user asks for Codex explicitly): before the fan-out run `codex exec` in
read-only mode (the -s sandbox flag is broken — run without it and put read-only guardrails in
the prompt) as an adversarial review of the harness. Brief it on: the tolerance definition and
whether the stopping test is computable at deployment; cost/accuracy coming from the same run;
timing protocol validity (warm-up, sync, same-compile FOM, one-GPU sequencing); POD
comparability at each k; EQ refit when N or M changes; leakage of held-out fields into the ROM
path. Apply MUST items, archive as CODEX-REVIEW-*.md. AFTER results, a second codex pass
checking every README table and figure against the JSONs — on the last round that pass
re-derived 1311 cells and found no generated-table error but 19 hand-written prose errors, so
it earns its keep.

DELIVERABLES per agent: experiments/<cell>/README.md (design, tables, verdict, explicit
caveats, provenance: job ids, GPU type, jax_backend, commit), generated SUMMARY/tables from
the JSONs (never hand-typed), raw JSONs, figures via the reports pipeline style
(reports/make_report_figs.py + fu_style.py; write PNG+PDF and copy to
/home/tahmid/Dev/pod-ae-nmrom/Plots/), committed and pushed. Final message: a compact
summary with the headline numbers, the crossover(s), what Codex flagged, and what is missing.
Do not ask questions mid-task; make reasonable calls and document them.

ALSO, SMALL: check whether Slurm job 2481538 (wave N=128 K=8, namespace wlat/) has finished.
If it has, pull it with checksums, fold the N-flatness row into the wave README and
SUMMARY_TABLES on exp/2026-08-16-wave2d-rom-latent-stepping, re-audit only the changed
numbers, push, and delete /cluster/tufts/paralab/tawal01/wlat/wad_n128_k8.
```

---

## Notes for the session running this (not part of the prompt)

- Launch both agents in **one message** so they run concurrently, `model: "opus"`.
- Do not use `subagent_type: "fork"` — forks inherit the parent model and cannot spawn their
  own reviewer subagents.
- Expect the Burgers cells to take several hours; agents that stop with cluster jobs pending
  can be resumed with SendMessage and will pick up their own state.
