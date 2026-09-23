# DESIGN — burgers-repanel: re-time the Burgers 2D rows at $256^2$, $512^2$, $1024^2$

Written before the first job. Lane `exp/2026-09-22-burgers-repanel`, forked from
`exp/2026-09-20-hires-burgers` @ `0ab60014`. Cluster namespace
`/cluster/tufts/paralab/tawal01/brepanel_20260922/`. Budget: ≤ 1 running GPU job, ≤ 4 total.
Hard deadline: rows wanted by 2026-09-24.

## 1. The defect this lane repairs

The paper's Burgers 2D rows at $256^2$, $512^2$ and $1024^2$ (`paper/tables/headline-provenance.json`
rows 17–19; b-panel jobs 3789570, 3805065, 3789572) time the **accurate** NM-ROM setting on a
solver path that has since been superseded: LU on the damped normal matrix, no trust-radius
clipping, no damping carry-over, no quadratic predictor, the unfused residual/Jacobian, and at
$1024^2$ a **dense** residual instead of a quadrature rule.

| paper row | arm printed | ms | optimised arm measured elsewhere | ms |
|---|---|---|---|---|
| $256^2$ accurate | `q256_M1088_eqtop_g1em06` | 746.02 | `…lat64_g0p01_…pred2_x1` (bc256b 4143154) | 214.2 |
| $512^2$ accurate | `q256_M1088_eqtopxfer_g1em06` | 783.33 | same arm (bc512 4140818) | 194.97 |
| $1024^2$ accurate | `q256_M1088_dense_g1em06` | 22053.85 | `…lat64_g0p001_…pred2` (bc1024 4139290) | 108.58 |
| $1024^2$ fast | `q0_M64_eqxfer_g1em06` | 40.27 | `q0_M64_scaled_g0p001_fast_clip_lamcarry_pred2` | 25.00 |

The optimisation stack is parity-gated and error-unchanged at $2048^2$ and $4096^2$
(`experiments/hires-burgers/SPEED-LOG.md`): Cholesky 1.53–1.80×, clip 1.37–1.41×, damping
carry-over 1.04–1.06×, quadratic predictor 1.15×, fused residual+Jacobian / folded head 1.47–1.52×.
It is simply not applied at these three meshes.

The corrected numbers already exist (`experiments/burgers-eqcert`, the paper's "confirmed rule"
block), but they come from a **different job** than the arm they would replace, so "the optimised
path is $N\times$ faster" is currently an inference across jobs, and the full-order comparator grid
differs between the two lanes. **This lane measures both arms and the full-order grid in one
allocation, per mesh.**

## 2. What this lane is not

- **No retraining.** Same frozen checkpoint
  `experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl`, same $K=16$,
  $R=512$, same directions file, same head.
- **No re-certification and no re-selection of quadrature rules.** Every rule is frozen and is
  carried with the certificate status the `burgers-eqcert` lane gave it (`rule_status` in each
  config, copied verbatim into `result.json`). `skip_certificates` is set in all three configs;
  the held-out $\rho$ phases do not run. A rule that this lane finds fast does **not** thereby
  become certified.
- **No new science.** This is a timing correction.

## 3. Arms — identical in all three jobs unless stated

Every arm below runs in ONE allocation per mesh, in one randomised, burn-separated, synchronised
timed panel, ≥ 5 retained repetitions × 6 development cases, medians reported.

### 3.1 NM-ROM, accurate

| role | arm | notes |
|---|---|---|
| **optimised accurate** | $256^2$/$512^2$: `q256_M1088_lat64_g0p01_fast_chol_clip_lamcarry_pred2_x1`; $1024^2$: `q256_M1088_lat64_g0p001_fast_chol_clip_lamcarry_pred2` | the rule and tolerance the `burgers-eqcert` lane certified at that mesh; $\rho$ status quoted, not re-derived |
| **pre-optimisation control (EQ)** | `q256_M1088_scaled_g1em06_base` | audited `topfix` path, LU, gtol $10^{-6}$, b-eqtop $q{=}256$ rule $m{=}2560$ — the rule family the paper's $256^2$/$512^2$ rows print |
| **pre-optimisation control (dense)** | $1024^2$ only: `q256_M1088_exact_g1em06_base` | the dense residual on the audited path — the arm the paper's $1024^2$ row actually prints |
| parity chain | `…scaled_g1em06_fast` (fold only), `…scaled_g1em06_fast_chol` | same rule, same tolerance as the control: these are the arms the parity gate covers |
| optimisation on the paper's own rule | `…scaled_g1em06_fast_chol_clip_lamcarry_pred2`, `…scaled_g0p001_…` | isolates the solver-path gain from the rule change |
| dense on the optimised path | `q256_M1088_exact_g0p001_fast_chol_clip_lamcarry_pred2` | separates "dense → quadrature" from "old solver → new solver" |

**Parity covers the chain, not the headline rule.** The `lat64` rule that carries the corrected accurate row
has no audited twin (b-panel never ran it), so the fold and Cholesky parity is established on the `scaled`
rule and carried across. `SPEED-LOG.md` did the same at $2048^2$ ("no audited twin for this rule"); the
report states it rather than letting the gate be read as covering the headline arm.

**Known deviation, stated up front:** at $512^2$ the paper's printed arm uses `eqtopxfer` — the same
support with NNLS-refit weights. Reproducing that refit needs the fit population from b-panel's own
population spec, which this lane deliberately does not rebuild (it would be a *new* rule, not the
printed one). This lane times the **no-refit transfer of the same support** (`scaled`) instead and
labels it as such. The $256^2$ row's `eqtop` rule is, per its own provenance
(`same_rule_as_lane_scaled: true`), exactly the `scaled` rule at that mesh, so $256^2$ reproduces
the printed arm directly; $1024^2$ reproduces it directly (dense has no rule).

### 3.2 NM-ROM, fast

- **optimised fast**: `q0_M64_scaled_g0p001_fast_clip_lamcarry_pred2`.
- **pre-optimisation control**: whichever arm the paper actually prints at that mesh —
  `q0_M64_scaled_g1em06_fastL4` (the b-speed `L4` kernel) at $256^2$ and $512^2$, because rows 17 and 18
  print `…_fastL4` arms, which are *already* a fused kernel and not the audited path;
  `q0_M64_scaled_g1em06_base` (audited, LU) at $1024^2$, because row 19 prints the audited arm.
  Both are timed at every mesh, so the ladder audited → L4 → optimised is visible throughout.
- parity chain: `q0_M64_scaled_g1em06_fast` and the `L4` arm, both against `…_base`.

**Second known deviation, stated up front:** the paper's fast rows at $512^2$ and $1024^2$ use `eqxfer` —
the $q{=}0$ support *refit* at that mesh ($m{=}922$ and $m{=}934$) — while this lane times the no-refit
transfer (`scaled`, $m{=}1024$), for the same reason as §3.1: rebuilding the refit would produce a new rule,
not the printed one. At $256^2$ there is no deviation (`eqcert` is that file with unit weights). Where the
rule differs, the report says so beside the number and the tripwire in §5 does not fire.

### 3.3 Newton–BiCGStab candidate grid

Fifteen settings, the **union** of the two grids in play, so the paper's FOM rule (*the fastest
tested setting whose error is at least as small as the row's*) can be applied like-for-like:

- the b-panel grid the current rows were selected from (audited implementation,
  `iterative_paths.make_fom`): `fft_tight`, `nt1e-4_dt005`, `nt1e-3_dt005`, `nt1e-2_dt005`,
  `nt1e-4_dt01`, `nt1e-3_dt01`, `nt1e-2_dt01`;
- the grid the $2048^2$/$4096^2$ rows are selected from (`lean` = `engines.make_fom`, the same
  discretisation and solver without the per-Newton diagnostic): `lean_tight`,
  `lean_nt1e-4_dt005`, `lean_nt1e-3_l1e-3_dt005`, `lean_nt3e-3_l3e-3_dt005`,
  `lean_nt1e-2_l1e-2_dt005`, `lean_nt1e-3_l1e-3_dt01`, `lean_nt3e-3_l3e-3_dt01`,
  `lean_nt1e-2_l1e-2_dt01`.

The report gives the rule's pick on **both** the full grid and the b-panel-only subset, so the
change in the speedup can be attributed to the ROM arm and to the comparator separately.

### 3.4 POD-LSPG

Not carried by this harness (no POD basis is built anywhere in the `hires`/`eqcert` code path, and
building one at these meshes would be new work with its own training and selection). **Skipped, and
reported as skipped** — the brief admits it only "if the harness carries it cheaply".

## 4. Cohort, error convention, timing scopes

- **Cohort:** dev6 = `params_draw(7090702, 4)` + `params_draw(911702, 2)`, asserted against the
  b-panel cohort SHA256 `108f12dc…` — the same six development cases as every row being corrected.
  The final cohort is not opened.
- **Error convention:** *same-grid, evolved* — max over the five evolved output times of
  $\lVert u_{\text{arm}} - u_{\text{ref}}\rVert_2 / \lVert u_0 \rVert_2$, reference `fft_tight`
  (Newton–BiCGStab, $\Delta t = 0.005$, ntol $10^{-6}$, ltol $10^{-8}$) at the same mesh. This is the
  convention `headline-provenance.json` names for all three rows. Worst and median over the six
  cases are both reported.
- **Timing scopes:** `gpu_seconds` — dense initial field already resident on the GPU to six dense
  GPU output fields, `block_until_ready` both sides. `host_seconds` (complete query) — the same
  invocation plus the host upload of the input and the host copy of the six outputs. Identical for
  every subject, ROM and FOM.
- **Speedup:** always FOM ms ÷ ROM ms with **both taken from the same job**, never across jobs.

## 5. Gates — what must pass for a number to be reported

| gate | bar | on failure |
|---|---|---|
| backend | log prints `jax_backend=gpu`, x64 on, matmul precision `highest` | job aborts (preflight exit 42) |
| cohort | `physical_sha256` equals the b-panel cohort hash | assertion, job aborts |
| $\Phi$-free operators | $\lVert A - \Phi^{\mathsf T}G\rVert/\lVert\Phi^{\mathsf T}G\rVert \le 10^{-12}$, identical eigenvalues, rows to $10^{-14}$ | assertion, job aborts |
| **parity** | `…_fast` and `…_fast_chol` reproduce `…_base`'s field to $\le 10^{-9}$ relative on every case, with **identical** per-step iteration counts and stop reasons | **a finding, reported as such** — not worked around, not hidden. A parity failure invalidates the claim that the optimisation is error-free and the mesh's row falls back to the audited "confirmed rule" row |
| algorithmic arms | clip / lamcarry / pred2 / exact-first-step change the iterates, so they are *not* parity arms. Bar: worst evolved error within 1 % *relative* of the non-algorithmic twin's, and zero stalled exits (budget, tiny-step or rejected) | reported; the arm is labelled as failing if it moves the error |
| repetitions | ≥ 5 retained repetitions for every (arm, case); every timed repetition's full-field SHA256 identical to the untimed quick run of the same arm and case | gate recorded false; row not reported |
| **reproduces the printed row** | where this job rebuilds the paper's arm exactly (same rule, tolerance, kernel, cohort, mesh), its worst evolved error must return the published value to 2 % relative: $256^2$ accurate 0.5129 %, $256^2$ fast 1.8891 %, $1024^2$ accurate 0.5861 % | **a finding**: the lane is then not re-timing the setting it claims to |
| algorithmic arms, measured | every clip / carry-over / predictor / exact-step arm within 1 % *relative* of the non-algorithmic arm of the same rule | reported as failing |
| no role arm stalls | zero budget, tiny-step or rejected-trial exits on every reported arm | reported as failing |
| no order effect | each arm's median GPU time after a $\ge 1$ s neighbour within 5 % of its median after a short one — the $1024^2$ dense control runs ~22 s inside a panel of 25 ms arms | **a finding**: the panel is then not internally comparable |
| all roles present | every named role arm reached the timed panel | audit not accepted |
| independent audit | `audit_repanel.py` (NumPy only, no JAX) recomputes every error from the saved fields and reproduces the job's numbers | mismatch = finding |

**Success for a mesh** = a single-job table giving, for optimised accurate / optimised fast /
pre-optimisation control / FOM candidates: worst and median evolved error, GPU ms, complete ms, the
setting the FOM rule selects, and the speedup as one job's row ratio; plus the measured gain of the
optimised path at that mesh and the parity result. **Failure** = any gate above red; the fallback
the paper then uses is its existing audited confirmed-rule rows (0.091× at $512^2$, 0.32× at
$1024^2$).

## 6. Jobs and order

Largest defect first, one job each, one directory each, `gpu` partition, f64,
`JAX_DEFAULT_MATMUL_PRECISION=highest`, output under paralab only:

| order | attempt | mesh | GPU | notes |
|---|---|---|---|---|
| 1 | `br1024` | $1024^2$ | H200, `--mem 240G` | dense arm ≈ 22 s per query × 30 invocations; $\Phi$ at $M{=}1088$ is 9.1 GB |
| 2 | `br512` | $512^2$ | A100-80G or H200 | |
| 3 | `br256` | $256^2$ | A100-80G | |

After each: checksum-verified `collect.py`, independent NumPy audit, remote job directory deleted,
code/configs/logs and the small summary committed (nothing over 50 MB).

**Fourth job, only if 1–3 are on track (coordinator's addendum):** one $4096^2$ job re-timing the
whole tunability ladder — rungs $q{=}0/M{=}64$, $q{=}128/M{=}576$, $q{=}256/M{=}544$,
$q{=}256/M{=}1088$, $q{=}256/M{=}1088$ at the looser tolerance, $q{=}256/M{=}2176$ — on the
optimised path, in one allocation, against a candidate grid containing both
`lean_nt3e-3_l3e-3_dt005` and `lean_nt1e-3_l1e-3_dt01`, so Table 1 and Table 4 can print the same
ratio for the same arm without a cross-job division. This is the `hires-burgers` harness with the
$4096^2$ config, not the code above. If the budget will not stretch, it is skipped and the paper
keeps both ratios, each labelled with its job and comparator.

## 7. Code

`repanel.py` is `burgers-eqcert/eqcert.py` @ `176b2a9a` with two additions, both necessary and both
small:

1. `base` on a rule spec builds the audited pre-optimisation arm (`topfix.make_query`, arm `base`,
   LU, dense or EQ) at the named tolerances, inside the same job and the same timed panel, and the
   non-algorithmic optimised arms take it as their parity twin.
2. `skip_certificates` skips the held-out population and $\rho$ phases (this lane must not re-open a
   frozen rule), and `rule_status` carries each rule's certificate verbatim into the report.

Configs are generated by `make_configs.py`, so the three meshes differ only in the mesh, the
certified arm at that mesh, and whether the dense control is timed.

## 8. What would make this lane report a null

- Parity fails at any mesh → the optimisation is not error-free there; report it, do not print the
  optimised row, fall back to the confirmed-rule row.
- The optimised arm's error differs materially from the control's → the arms are not comparable and
  the "same setting, faster path" framing is wrong; report the error difference beside the times.
- Cannot finish a mesh before 2026-09-24 → say which mesh, early, and leave the paper on its
  existing audited rows.

## 9. Independent audit of this design, and its disposition

`codex exec -m gpt-6-astra -s read-only` could not run: its bubblewrap sandbox fails on this box
(`bwrap: loopback: Failed RTM_NEWADDR: Operation not permitted`) before reading any file — a recorded
landmine. An independent subagent audited instead, with the same brief; the full text is in
`reports/design-audit-2026-09-22.md`. Twenty-three findings; disposition:

| finding | severity | disposition |
|---|---|---|
| F1 the fast row's pre-optimisation arm is the `L4` kernel, not the audited path, at $256^2$/$512^2$ | major | **fixed**: the `L4` arm is built and timed, and `roles.preoptimisation_fast` points at the arm the paper prints per mesh. Does not affect $1024^2$ (row 19 prints the audited arm), so job `br1024`, already submitted, stands |
| F2 the $q{=}0$ rule is `eqxfer` (refit) at $512^2$/$1024^2$, not `scaled` | major | **fixed by disclosure**: §3.2 above, and beside the number in the report |
| F3 the accurate row's base arm is right | fine | — |
| F4 parity chain sound; F11 timing hygiene sound; F20 skip-certificate short-circuits clean; F21 no dangling arm names | fine | — |
| F5 parity does not cover the `lat64` headline rule | minor | **fixed by disclosure**: §3.1 above |
| F6 a released field silently drops a parity pair | minor | **fixed**: the pair is now recorded as `covered=False` with a reason |
| F8 the 22 s dense control sits in a panel of 25 ms arms; a 0.25 s burn may not isolate it | major | **fixed in the audit**, no job change: the invocation list is in execution order, so the audit splits every arm by whether its predecessor took $\ge 1$ s and gates the gap at 5 % |
| F9 second bank copy; F10 `Phi` rebuilt per tolerance | minor | F10 **fixed** (hoisted, built once); F9 left — 4.3 GB on a 141 GB H200 with phase 8 off |
| F12 unused `sx`/`sy` in the audited arm's data | minor | **fixed**: the audited arm now gets exactly the keys `arms.weak_eq` reads |
| F13 the "algorithmic arms do not move the error" gate did not exist | major | **fixed**: `algorithmic_arms_do_not_move_the_error` (1 % relative) and `no_role_arm_stalls` |
| F14 no tripwire that the base arm reproduces the paper's printed error | major | **fixed**: `reproduces_the_printed_error_of_the_paper_row`, 2 % relative, on the three arms this lane rebuilds exactly |
| F15 vacuous certificate-control gate | minor | **fixed**: not emitted under `skip_certificates` |
| F18 dead locals | nit | **fixed** |
| F19 inert config fields | nit | **left**: inherited from the parent harness, marked here rather than pruned mid-flight |
| F22 the smoke config's roles and FOM subsets did not follow the rule rename | major | **fixed**: both rebuilt inside `smoke()`, and the smoke re-run |
| F23 `missing_roles` computed but never gated | major | **fixed**: `all_roles_present` |
