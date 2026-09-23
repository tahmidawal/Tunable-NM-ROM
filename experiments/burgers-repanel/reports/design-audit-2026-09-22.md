# Independent audit of the burgers-repanel design (2026-09-22)

Commissioned before the second and third jobs, against the lane at commit `dbcf02aa`. `codex exec -m
gpt-6-astra -s read-only` **could not run** — its bubblewrap sandbox fails on this box
(`bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`) before reading any file, a recorded
landmine — so an independent subagent audited instead, with the same brief and read-only access. The brief
asked four questions: does the design measure what it claims; what could make the timings unfair between
arms; are any gates missing or at the wrong bar; and what is concretely wrong in the diff against
`burgers-eqcert/eqcert.py`.

The auditor read `DESIGN.md`, `repanel.py`, `make_configs.py` and the three configs, `audit_repanel.py`,
`cluster/{stage,submit,collect}.py`, `xfast.py`, `hfast.py`, `hops.py`, `topfix.py`, `arms.py`, `varpro.py`,
`engines.py` / `iterative_paths.py`, `b-panel/panel.py`, `hires-burgers/SPEED-LOG.md`, and
`paper/tables/headline-provenance.json` rows 17–19.

**The disposition of every finding is recorded in `DESIGN.md` §9.** Findings, as written:

## 1(a) Is `base` really the path the paper's rows were timed on?

- **F1 (major) — the fast row's baseline is the wrong path at $256^2$/$512^2$.** `headline-provenance` rows
  17/18 print `q0_M64_eqcert_g1em06_fastL4` and `q0_M64_eqxfer_g1em06_fastL4`. In `b-panel/panel.py:148,629`
  those are `family='fast'`, built by `F.make_query(..., FL.ARMS['L4'])` — an *already optimised* kernel, not
  the audited path. Only row 19 (`q0_M64_eqxfer_g1em06`, b-panel config-1024 has `fast_arm: null`) is
  `family='rom'` → `TF.make_query(...,'base')`. repanel set `roles.preoptimisation_fast` to the `base` arm at
  all three meshes, so at $256^2$/$512^2$ the label was false and the gain would be overstated.
- **F2 (major) — the $q{=}0$ rule is not the printed rule at $512^2$/$1024^2$.** The paper's fast rows use
  `eqxfer`, $m{=}922$ / $m{=}934$ — the qrg304 support transferred **and NNLS-refit** at that mesh
  (`panel.py:83,178`). repanel's `scaled` is the same support with weights $\times(L/256)^2$, no refit,
  $m{=}1024$: ~10 % more nodes, a different rule, a different error. DESIGN §3.1 disclosed exactly this for
  the *accurate* arm at $512^2$ but §3.2 was silent about the fast arm.
- **F3 (fine) — the accurate row's base arm is correct.** b-panel builds `family='rom'` arms with
  `TF.make_query(..., 'base', ...)` (`panel.py:623`) and dense-at-strict-gtol arms the same way
  (`panel.py:395`). repanel passes the same `ic_budget/step_budget/gtol/ic_gtol/linear/inner_damping/tau_y`.
  The one omitted argument, `Rb`, is read only when `adaptive_y` is set, and `ARMS['base']` has
  `adaptive_y=False` — a genuine no-op. $256^2$ is literally the printed rule
  (`same_rule_as_lane_scaled: true`); $1024^2$ dense has no rule; $512^2$ carries the disclosed deviation.

## 1(b) Parity chain

- **F4 (fine) — sound.** `algorithmic` covers clip|lamcarry|pred2|exact_steps|ic_gtol, and `parity_twin` is
  keyed by the same rule spec and the same gtol, so only `_fast` (fold) and `_fast_chol` at $10^{-6}$ claim
  the base as twin — exactly the two changes SPEED-LOG parity-gated at $2048^2$. `parity_bar=1e-9` is ~4
  orders looser than observed, i.e. correctly non-binding.
- **F5 (minor) — the reported accurate arm's rule is never in the chain.** `lat64` has no `base`, so the
  fold/chol parity is verified on `scaled` and transferred by assumption to `lat64`, the rule that carries
  the corrected row. SPEED-LOG did the same at $2048^2$. State it so the gate is not over-read.
- **F6 (minor) — silent parity hole.** `parity_and_prune` `continue`s with no `rep['parity']` row when either
  field is `None`, and the gate then passes on fewer pairs. Unreachable with this config, but silent.

## 1(c) Single-job row ratio

- **(fine) — yes.** `median_gpu_ms` over all 30 invocations and `worst_evolved_percent` over cases are the
  same statistics b-panel's audit used (`audit_panel.py:224,284`), then `speedup_gpu` from one job,
  `by_grid` over `fom_subsets`, and `optimisation_gain` pre/post ratios. Both grids are delivered as
  DESIGN §3.3 promises.

## 2. What could make the timings unfair between arms

- **F8 (major) — a 22 s arm in a 25 ms panel, separated by a 0.25 s burn.** The $1024^2$ dense
  pre-optimisation arm runs ~22 s of dense f64 per invocation inside the same randomised panel as the 25 ms
  optimised arms. `e.burn(0.25)` does not restore a clock/thermal steady state after that; the arm drawn
  next is systematically penalised, which would inflate everything the lane reports. Five reps × six cases
  randomises but does not remove it, and nothing recorded lets you test for it.
- **F9 / F10 (minor) — memory held for nothing.** `Gp = jnp.concatenate(G, 0)` is a second 4.3 GB copy of the
  bank at $1024^2$, freed only in phase 8, which is off. `Phi` for the dense base (9.1 GB) was built inside
  the per-tolerance loop, each tolerance keeping its own device copy — latent OOM the moment a second
  tolerance is added.
- **F12 (minor) — the base arm's `data` is not b-panel's.** `dict(data, G=G[0])` hands the audited EQ query
  `sx`/`sy`, which `arms.weak_eq` never reads. No compute difference, but the jit signature is not identical
  to the arm being reproduced.
- **F11 (fine) — everything else checks out.** One warm-up per timed arm after any `clear_caches`; fresh
  `device_put` per invocation; `nu/ntol/ltol` are runtime args so no per-case recompiles; `foms` cached and
  the FFT `pre` hoisted out of the timed region for both implementations; `tab`/`cold`/`data` built once and
  passed as jit *arguments*; `params`/`C` are the only closure constants and `topfix` and `hfast` close over
  them identically. The audited FOM's extra per-Newton diagnostic is a real cost, but it is the
  implementation the corrected rows were selected against, and `fom_notes` plus the `lean` subset disclose it.

## 3. Gates

- **F13 (major) — the DESIGN §5 "algorithmic arms" gate did not exist.** Nothing computed "within 1 %
  relative of the non-algorithmic twin" or "zero stalled exits". These are precisely the arms
  (clip/lamcarry/pred2/x1) that carry every corrected number.
- **F14 (major) — no tripwire that the base arm reproduces the paper's printed error.** At $256^2$,
  `q256_M1088_scaled_g1em06_base` must give 0.5129 % (row 17); at $1024^2$,
  `q256_M1088_exact_g1em06_base` must give 0.5861 % (row 19). Same mesh, same cohort, deterministic
  arithmetic. If it disagrees, the lane's whole premise is wrong and nothing currently notices.
- **F23 (major) — `missing_roles` is computed but never gated.** A typo in `roles`, or an arm dropped by the
  OOM path in `run_quick`, yields empty rows, no `optimisation_gain`, and still `accepted: true`.
- **F15 (minor) — vacuous certificate gate.** `control_rule_fails_certificate` was still emitted with
  `passed=None` in a job that deliberately runs no certificate.

## 4. Concrete bugs in the diff vs `eqcert.py`

F10, F12, F6 above; **F18 (nit)** dead locals; **F19 (nit)** inert config fields (`dense_deadline_seconds`,
rung-level `variants`/`gtols`, `keep_for_reference`, `q0_through_hfast`, `target_chunk`, `rho_bar`,
unused imports). **F20 (fine)** — the `skip_certificates` short-circuits are clean: everything downstream
that reads `populations`, `rules[...]['heldout_population']`, `arm_status`, `pops`, `targets` is inside the
same guard or behind `CERTIFIED_HERE`; `rep['population_source']` is still written; the cohort and
population disjointness assertions still run.

## 5. Config generator

- **F21 (fine) — no dangling arm names at the three real meshes.** Every name `register()` produces was
  enumerated and checked against `roles`, `audit_arms`, `profile_arms` and `fom_subsets`: zero misses. Every
  rule builds at every mesh (`lattice_rule` needs $L \bmod 64 = 0$; `transfer_nodes` needs
  $L \bmod 256 = 0$; `assert single` holds because $(L-1)^2\cdot512 < 2.146\times10^9$).
- **F22 (major) — the smoke config was broken and would hide exactly the failure it should catch.**
  `smoke()` renamed the rules but not `roles` or `fom_subsets`, so the smoke exercised neither the role
  lookup nor the by-grid selection, and its output (missing roles, empty `by_grid`) was indistinguishable
  from the same failure on a real job — which, per F23, still reported `accepted: true`.
