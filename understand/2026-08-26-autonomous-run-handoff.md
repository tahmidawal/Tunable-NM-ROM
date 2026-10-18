# Handoff: autonomous overnight run — exact linear terms → gradient-target EQ → learned nodes

Written 2026-08-26 by the session that ran the EQ fidelity ladder, for its own compacted
continuation. The user (Tahmid) is asleep and has authorized this run end to end. **This file
is the plan of record.** If conversation context disagrees with this file or with LAB-LOG.md,
the files win.

## User decisions already made (do not re-ask)

- Worktree `worktrees/2026-08-26-eq-learned`, branch `exp/2026-08-26-eq-learned`, cut from
  `exp/2026-08-25-sepdec-consolidated`. Cluster namespace `/cluster/tufts/paralab/tawal01/exlin/`.
- `exp/2026-08-25-eq-fidelity-ladder` stays a separate archive branch — do NOT merge it.
  Copy `sep_eq_ladder.py` (and `runs/gen_eq_ladder.py`) into the new worktree instead.
- Scope: stages 1–2 fully measured; then stage 3 (learned continuous nodes) implemented and
  its first training arms launched on Tufts overnight if 1–2 land cleanly.
- Notes for the user go under `understand/` (see "Notes discipline" below).

## Read first (order matters)

1. This file.
2. `LAB-LOG.md` — "Where things stand" + the two 2026-08-25 EQ-ladder entries.
3. `reports/2026-08-25-eq-fidelity-ladder.md` — findings F1–F7 + tables T-L1..T-L5.
4. `understand/2026-08-25-eq-fidelity-ladder-explained.md` — esp. §D (exact-linear derivation)
   and §E (the convexity qualification: g_s = J_sᵀR_s is QUADRATIC in w; the convex refit
   freezes one factor at the full-grid teacher).
5. `worktrees/2026-08-25-sepdec-consolidated/START-HERE.md` — architecture + non-negotiables.
6. On branch `exp/2026-08-25-eq-fidelity-ladder`: `experiments/separable-decoder/sep_eq_ladder.py`
   (the ladder driver — the `make_full` closure and the `mk()` sampled ops are the templates
   for stage 1) and `cluster/stage_eqladder.sh` / `push_eqladder.sh` / `runs/pull_eqladder.sh`
   (clone these as `*_exlin*` for the new namespace).

## Task 0 — bookkeeping (do first, ~15 min)

Pull `dn1024` (job 2837430) and `dn256b` (2837431) from namespace `burgacc/` into the
**consolidated** worktree (`worktrees/2026-08-25-sepdec-consolidated`, its
`runs/pull_burgacc.sh`, `--delete` after sha256+markers verify), commit on
`exp/2026-08-25-sepdec-consolidated`, push. This is the ONE sanctioned write to another
worktree this session (the lab log assigned it). Record their verdict (the cost-matched
recipe at full density, N=1024 confirmation) in the notes file and the final lab-log entry.
If a job failed or is still queued, record that and move on — do not block on it.

## Stage 1 — exact linear terms (no training; local smoke then 2 cluster jobs)

**Change.** In the online residual, replace the sampled linear terms with the exact ones:

```
A = Φᵀ G_int            # (M, R), precomputed once offline per (checkpoint, M)
r_w(z) = wt ⊙ [ A(h(z) − h(zⁿ)) + DT·( Φ_qᵀ N(u)|_nodes  +  ν λ ⊙ A h(z) ) ]
```

Advection stays sampled via `G_st`/`Phi_q` exactly as today. The IC fit (Gram-space) is
untouched. `prev` becomes the M-vector `A h(zⁿ)` cached per step (replaces `prev_of` at the
nodes for the linear part only; advection still needs nothing from prev).

**Driver.** New `sep_burgers_exlin.py` in the new worktree, built from `sep_speed_r5.py`'s
structure (checkpoint-loading, EQ fit, round-4 protocol) with the residual above. Also a
matching ladder re-run: adapt `sep_eq_ladder.py` so the SAMPLED side is the new residual
(the full-grid side is unchanged) — after the change, `b_lin` must be ≤1e-12 by construction
and (b) collapses to the advection part. That assertion IS the correctness gate.

**Gates (rule change — log it).** Gate 0 (bit-identity to the incumbent sampled residual)
no longer applies to the full residual BY DESIGN. Replace with, and assert all three:
- gate L: linear part vs full-grid linear part ≤1e-12 (exactness);
- gate A: sampled advection term identical to the incumbent's advection evaluation ≤1e-12
  (nothing changed there);
- gate F (N≤512): full-grid reference vs `make_weak_ops` on the whole interior ≤1e-12.
Record "gate 0 redefined, why, and the measured values" in the lab log — this is the
deliberate rule change the report's §3 item 1 announced.

**EQ refit.** Fit the m nodes on ADVECTION rows only: `ss.build_eq_system_burgers` restricted
to the `Nf` row blocks (drop the `uf` blocks), same NNLS, both set sizes (m=256, m=1024).
The freed budget now serves one term.

**Measure.** On the same four checkpoints as the ladder (r3a N=256, dense_mid N=256, r3d
N=1024, r4a6 N=1024): (i) ladder re-run — expect (b) ≈ old `b_adv` or better, (c1)/(c3)
improved; (ii) the round-4 matched-accuracy paired AB/BA protocol at N=256 and N=1024 —
rollout error and e2e ms vs the swept classical ladder, so the accuracy gain and the
unchanged cost are measurements. N=64 smoke locally FIRST (ckpt
`worktrees/2026-08-22-separable-decoder/.../runs/sepdec_r1/out/sep_burgers_N64_K16_R64.pkl`),
then cluster: N=256 on A100, N=1024 on H200 `--mem 240G`. The M=256 NNLS fit takes 12–24 min
(700 s H200 / ~1000 s A100) — size `--time` ≥ 3 h.

**Optional bonus (only if wall-clock allows after stage 2 is queued):** Poisson goes fully
quadrature-free (values-only weak form ⇒ the whole residual is `(Λ ⊙ ΦᵀG)h(z) − Φᵀf`).
A small N=256 confirmation cell; do not let it displace Burgers work.

## Stage 2 — same-target NNLS for advection (convex; no training)

With the linear part exact, refit the advection weights against what the solver consumes:
- Rows: (i) advection-projection rows at OFF-MANIFOLD states — collect LM iterates from real
  rollouts (the ladder driver already produces them) across several ν, plus oracle/training
  states; (ii) gradient-teacher rows with ONE FACTOR FROZEN at the full-grid teacher
  (`J_fᵀ R_s(w)` rows targeting `J_fᵀR_f`) — NOT the literal `J_sᵀR_s`, which is quadratic
  in w (see the explainer §E). Weights stay w ≥ 0, Lawson–Hanson capped at m — same solver,
  different rows.
- Certify ONLY with the ladder (rung c1/c3 on held-out iterates), never rel-fit or row tail.
- Compare three quadratures at fixed nodes-budget m: (a) stage-1 advection-only state-target,
  (b) stage-2 same-target, (c) the old incumbent set — ladder + rollout error each. This is
  the baseline table stage 3 must beat.

## Stage 3 — learned continuous nodes (training; overnight cluster arms)

Only if stages 1–2 land with gates green. Design constraints (from design doc §6.2 + the
audit, all still binding):
- FROZEN decoder, FROZEN full-grid teacher (anti-collusion). Nodes are the only new
  parameters; weights solved by inner NNLS at the current nodes (variable projection) so the
  convex family is always the fallback.
- Nodes reparameterized into the open box (sigmoid), min-separation penalty, fixed
  cardinality m (no recompile churn). `g` is meshfree so `g(x_j)`, `g(x_j ± Δx)` and the
  sine modes at `x_j` are all differentiable in `x_j`; the upwind `where(c>0,·)` switch is
  piecewise — use its subgradient as-is and say so in the notes.
- Loss: the ladder rungs vs the frozen teacher — `α_b‖R_s−R_f‖² + α_g‖J_fᵀR_s − J_fᵀR_f‖²
  (+ probe rows for c2)` — averaged over off-manifold iterates, several ν, and both early
  and late time (round 5: the sharp-blob first step is where quadrature was worst).
- Success bar: beat stage 2 (same m, same states) on held-out (c1)/(c3) AND on rollout error
  at unchanged e2e ms. If it can't beat convex, that is a REPORTABLE negative, not a failure
  of the session.
- Launch 2–4 arms on Tufts (e.g. m=256 nodes-learned vs stage-2 at N=256; one N=1024 arm),
  each its own job dir under `exlin/`, one verified-alive watcher for the wave. It is fine
  for these to still be running when the user wakes — the notes must say exactly what is
  in flight, job ids, and how to pull.

## Mechanics (non-negotiable, from CLAUDE.md — abbreviated, the file rules)

- Worktree creation: `git worktree add -b exp/2026-08-26-eq-learned
  worktrees/2026-08-26-eq-learned exp/2026-08-25-sepdec-consolidated` then push -u.
- Every job: own dir under `exlin/`, stage script with MANIFEST.sha256, gpu preflight
  (`jax_backend=gpu` or exit 42), `JAX_DEFAULT_MATMUL_PRECISION=highest`, squeue before AND
  after submit, pull + sha256 + markers + `--delete`, never park checkpoints.
- Local: `source /etc/profile.d/jax-mem.sh`, `jaxrun /home/tahmid/Dev/.venv/bin/python`,
  ≤3 concurrent. Banks/data as EXPLICIT jit arguments (captured-constant landmine).
- Codex runs: `codex exec` with stdin closed (`< /dev/null`) — it hangs on stdin otherwise
  (cost 40 min on 08-25). `pkill` patterns must not match your own shell's command line.
- Every generated table from a committed script + JSONs; no hand-typed numbers anywhere.
- Commit and push after every milestone, on the experiment branch; report/notes files on main.

## Notes discipline (the user's explicit ask)

- Running log: `understand/2026-08-26-overnight-notes.md` on main. Append a dated/timed
  entry after EVERY milestone (worktree created; smoke passed; jobs submitted with ids;
  results pulled with the numbers; gates measured; anything retracted or surprising;
  decisions taken and why). Plain language, advisor-readable, numbers copied from generated
  output only. This is what the user reads on waking — keep it current, not retrospective.
- End-state explainer: when stages 1–2 are measured, have Codex write
  `understand/2026-08-26-exact-linear-and-gradient-eq-explained.md` (same recipe as the
  ladder explainer: sources listed, every number from generated tables, verify its numbers
  programmatically before committing, credit any correction it makes).
- LAB-LOG.md: append the session entry and rewrite "Where things stand" before ending —
  wins, retractions, job ids, what is in flight, what the next session picks up. Use
  line-anchored edits, never `str.replace` with a possibly-empty anchor (that corrupted the
  file once on 08-25).
- Memory: update `eq-ladder-2026-08-25.md` or add a successor with the stage-1/2/3 outcome.

## Stop conditions

Stop and leave a clear note (rather than improvising) if: a gate fails and the cause is not
found within ~2 hours of focused debugging; the cluster share is full or all GPU types are
down; or a result contradicts a published number (record it as a candidate retraction in the
notes and lab log, do not silently "fix" it). Never widen scope beyond stage 3; never touch
`main`'s frozen packages or the read-only archive branches.
