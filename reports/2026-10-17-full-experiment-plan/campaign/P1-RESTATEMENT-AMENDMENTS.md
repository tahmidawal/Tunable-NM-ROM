# Priority-1 restatement — amendments after Codex audit 1 (2026-10-17)

Supersedes `P1-RESTATEMENT.md` wherever they differ. Answers `codex-p1-restatement-1.md`
(NEEDS-RESTATEMENT, six ranked blockers). Numbered by blocker.

## AM-1. Configuration identity and reuse (blocker 1)
- **No archived trajectory is used as a P1 result.** The 1024² P1d arms (R128, R256, R384 × BE, CN-R × 38
  cases = 228 rollouts) are **re-run fresh** with in-job Q1 certification, because fresh-Jacobian
  every-step certification needs a live replay anyway. The archived `c2grid` arrays become a
  **reproduction gate**: for R128 (`fib4181`) and R384 (`gauss128`), whose archived configuration matches
  in full (manifest equality below), the re-run must agree with `W_R128.npz` / `W_R384.npz` to ≤ 1e-10
  relative on every case. **R256 is not reproducible from `c2grid`**: the archive is `gauss96`, the P1
  frozen rule is `gauss128`; R256 has no reproduction gate (its `R256fb` fallback results are cited as
  prior evidence only).
- **Development manifest (the missing "item 0"),** written and hashed before any job:
  bank checkpoint + rotation hashes; per (r, mesh): mode identities and order (hash), with the
  mesh-dependent eigenvalues/projections recomputed per mesh from the frozen identities; rule name, points
  m, rule hash; scheme, Δt, solver and tolerances (`c2common.tol_of`: prod / tight / tighter, LSPG and
  GAL); initialisation contract; test weighting; output times and scoring metric; E0 reference version
  and per-case-kind eligibility; cohort case IDs; code commit. Reuse or reproduction is accepted only on
  full manifest equality, never on a mode hash alone.
- Rollout inventory therefore rises by 228 (1024² P1d) and is reflected in the certificate (AM-6).

## AM-2. Q2 defined (blocker 2)
For a configuration with rule $q$ and next finer rule $q^+$ (R128 `fib4181`→`fib6765`; R256/R384
`gauss128`→`gauss160`; R512 `gauss192`→`gauss256`):
- **Rollout bar (primary):** replay the arm with $q^+$ (same everything else); scoring-lattice relative
  distance, max over the five evolved output times, ≤ 2.5e-4 on **every** case.
- **ρ bar (secondary):** on the arm's **own reached states** (all six output times, every case of the
  population), $\rho(c) = \lVert N_q(c) - N_{q^+}(c)\rVert / \lVert N_{q^+}(c)\rVert$ of the tested
  advection vector (length M); $\rho_{\max}$ over states ≤ 0.116 (the B5 bar registered in
  `worktrees/2026-10-01-quadrature-study/experiments/quadrature-study/DESIGN.md:234`); report the full
  distribution.
- **Finer-rule convergence:** on dev6 only, $q^+$ against one further rule ($q^{++}$: fib10946,
  gauss192, gauss320) must meet the rollout bar; if it fails, the Q2 label is "unverified (finer rule not
  converged)", not pass.
- Q2 runs on: all P1d configurations at all meshes on dev6 + val32[0:10] (16 cases); P1c configurations
  inherit the P1d label of the same (r, mesh, scheme, rule, mode list).
- **P1e frozen:** 2D r=256 at 1024², BE, dev6: failing `gauss64`, deployed `gauss128`, finer `gauss160`
  (+`gauss192` for its convergence). 3D r=512 at 65³, BE, 6 cases of 923801: the lane freezes failing /
  deployed / finer from the speed3d W512 grid configuration (deployed = the speed3d W512 rule; failing =
  a rule with fewer points that combined or speed3d recorded as failing; finer = next Gauss order),
  stated in DESIGN.md before any data, with a memory projection.

## AM-3. Kernel fixtures (blocker 3)
- **P1a fixture source is `W_R512.npz`** (R512, `gauss192`, LSPG-CN-R Δt₀, 1024², dev6 cases 0–5, output
  indices 1 and 5), truncated to each r ∈ {32,…,512}. Label "kernel fixtures, not reached states of the
  smaller-r solver". Verify the archived R512 configuration from its manifest.
- **P1b nominal panel rule per r:** r ∈ {256, 512, 768, 1024}: the speed3d/combined W1024-prefix rule for
  that width (lane cites config file:line); r ∈ {64, 128}: the r=256 rule. Fixed panel unchanged
  ((M, m) = frozen 65³ list for r=512, Gauss 24³).
- Fixed-cost cross-check uses the **smallest rank** of each panel (2D r=32, 3D r=64).

## AM-4. Verdict procedures (blocker 4)
**H-cost-N.** A per triplet = **mean of the two flanking A timings**. Drift: if the two flanking A timings
differ by > 10 %, repeat the triplet (≤ 2 retries); if still drifting, the triplet is invalid. A
configuration with > 1/3 invalid triplets on any mesh → "panel invalid; verdict not reached". Missing
mesh → "verdict not reached" (and with 4096² dropped: "two-mesh evidence, registered three-mesh verdict
not reached").

**H-cost-r.** Verdict only on the **fixed-(M, m)** panel, per component. Fit $t=a+br^p$ with $a\ge0$,
$b>0$, $p\in[0,4]$; fit failure or a boundary solution → inconclusive. **Pass** = pre-declared p inside
the 95 % case-bootstrap interval **and** interval half-width ≤ 0.5; **fail** = pre-declared p outside the
interval; **inconclusive** = otherwise. Secondary label for a fail: "operation-model" if the FLOP/byte
model also departs from p over the range, else "hardware scaling". Fixed-m and nominal panels are
**descriptive**, plotted against the FLOP model (no verdict).

**H-acc-N.** Per (r, scheme), separately for each pair (256² vs 1024² on the 38 cases eligible at both;
4096² vs 1024² on the 16-case subset eligible at both): normalised differences
$d_i = (e_i(N) - e_i(1024^2))/\delta_i$, $\delta_i = \max(2\varepsilon_{{\rm ref},i}, 0.1\,e_i(1024^2))$.
**TOST** = two one-sided Wilcoxon signed-rank tests of $d_i$ against −1 (greater) and +1 (less), α = 0.05
each; equivalence iff both reject. "Within margin" fraction = share of cases with $|d_i|\le1$.
Trend: per-case OLS slope of $e$ vs $\log_2 N$ on the 16-case subset (three meshes); two-sided Wilcoxon of
slopes vs 0, α = 0.05 (zero slopes dropped per Wilcoxon convention; if fewer than 6 non-zero slopes, the
trend test is "not testable"). **Pass** = TOST equivalence for both pairs **and** ≥ 90 % within margin in
each pair's population (≥ 35/38 and ≥ 15/16). **Fail** = TOST fails for at least one pair **and** (trend
test rejects **or** < 50 % of a pair's cases within margin). **Inconclusive** = otherwise. Cases not
eligible at both meshes of a pair are dropped from that pair and counted. Applies with the two-mesh label
to accuracy too if 4096² is dropped.

**H-acc-r.** Per transition (128→256, 256→384 at every mesh; 384→512 at 1024²): resolved = eligible,
both rungs Q1-qualified, and |Δe| > 2ε_ref. Outcome **improves** = resolved coverage ≥ 50 % of eligible
cases **and** > 50 % of resolved cases improve **and** median paired change < 0; **does not improve** =
coverage ≥ 50 % and the improve criterion fails; **inconclusive** = coverage < 50 % or > 25 % of cases
lacking qualification. **Registered range for the monotone claim:** 128–384 at each mesh (primary) and
128–512 at 1024² (registered through 512 per V3-5; the known R384→R512 regression is expected to make it
"not monotone"). The label is "majority improvement on resolved cases", never "every case" or "worst
error". Worst-case change reported for every transition.

## AM-5. Controls with fixed fixtures and a claim-blocking map (blocker 5)
| Control | Kind | Fixture and predicate | Blocks if it misbehaves |
|---|---|---|---|
| Synthetic timing fault | negative | synthetic 6 cases × 3 meshes, all timings 1.00 × (1 + 0.01·N(0,1), seed 0), one mesh × 2.0 → H-cost-N must return **fail**; same fixture unmodified must return **pass** | H-cost-N |
| Production timing fault | negative | smoke data at 1024², every case × 3.0 on one mesh → must return fail (reported; procedure is not tuned on it) | H-cost-N |
| Doubled work | negative | B = two sequential evolutions, the second starting from the first's final state, both outputs returned; B/A ∈ [1.8, 2.2] | H-cost-N, H-cost-r |
| Component sum vs fused | positive | Σ components ≥ 0.9 × fused query and ≤ 1.6 × fused | H-cost-r |
| Split-query equivalence | positive | init+evolve+decode vs fused: fields ≤ 1e-12 rel., identical iteration counts | H-cost-N |
| Timed = scored outputs | positive | hash or ≤ 1e-12 | all timing claims |
| Reproduction gate | positive | AM-1 (R128, R384 at 1024²) | H-acc-N, H-acc-r |
| Wrong-mesh table | negative | every table carries (L, mesh hash, mode-list hash) metadata; a 256² table fed to a 1024² bank must raise | all |
| Under-resolved rule | negative | `gauss32` (1024 points < M=1536) at r=384 on the R384 own reached states at output index 5, dev6: must fail Q2 rollout bar on ≥ 1 case and have ρ_max > 0.116 | Q2, all fronts |
| Wrong-gradient solver | negative | Jacobian of the advection term with its sign flipped (linear part intact) — not a scalar multiple: must fail Q1 stationarity (> 1 % non-stationary pooled) | Q1 |
| Zero-update solver | negative | solver returns its input state: must fail Q1 distance | Q1 |
| Rank-deficient fixture | negative | A with a duplicated column: Q3 must report cond > 1e12 / rank deficit | Q3 |
| Case / time / reference-kind swap | negative | scorer asserts (cohort, case id, time index, reference kind, reference hash) per record; a swapped record must raise | all accuracy |
| Synthetic accuracy trend | negative | 16 cases, ε_ref = 0, e(1024²)=1 %, e(256²)=0.5 %, e(4096²)=1.5 % (+0.01 % noise, seed 0) → H-acc-N must return **fail**; with all three equal ± noise → **pass** | H-acc-N |
| CN-R temporal order | positive | GAL-CN-R (established second order in time2), dev6 case 0, R384 at 1024², Δt₀, Δt₀/2, Δt₀/4 vs Δt₀/16 anchor: observed order ∈ [1.8, 2.2]; LSPG-CN-R order measured and reported, no gate | scheme labels |
| Sensitivity probes | report | GPU type (A100 vs H200 dimensionless ratios), repeat seed; no lattice shifts or seeds in P1 (no randomised rules) | — |
Every control is shown to behave as required in the real-size smoke at 1024² (synthetic ones in CPU tests)
before any verdict is read; a misbehaving control blocks the listed claim until fixed and re-audited.

## AM-6. Resources and provenance (blocker 6)
- **Reservation rule:** submit only if spent + Σ certified projections of queued/running jobs + this
  job's projection ≤ 20 GPU-h. Smoke, memory pilots, compilation, failed attempts, certification, anchors,
  Q2 replays and controls all count.
- **P1g deduplicated:** shared with P1d = R384 frozen-list runs at 256² (38 cases) and at 4096² (16
  cases). Additional for P1g = R384 frozen-list at 4096² on the remaining 22 cases, plus the
  **nominal-policy** R384 arms at 256² and 4096² (38 cases each), both schemes. P1g's quantity: per case,
  frozen − nominal error at each mesh, and the change in the cross-mesh comparison. The lane lists, in
  DESIGN.md, the historical claims it re-checks (which reports/tables made cross-mesh 2D accurate-setting
  comparisons at M ≥ 1536, with file:line) and reports the moved numbers next to each.
- **P1g is never dropped, including at 4096².** If the 4096² memory pilot shows 4096² infeasible even
  with the prefix-retaining builder, the outcome is "P1 incomplete" (P1g not reached), not silent removal.
- **Q1 every configuration:** tight anchors for every rollout configuration (P1d all meshes, P1g extra
  and nominal arms, R512 diagnostic); tighter anchors on dev6 cases 0–5 per configuration; tolerances from
  `c2common.tol_of` (LSPG tight (1e-7, 0), tighter (1e-8, 0)).
- **Q3:** J at output indices 1, 3, 5 of each dev6 case's own trajectory; report median and max cond per
  configuration.
- **Speed3d imports** (coordinator, final report): g1–g3 → 3D Q1 for those arms and part of the 3D
  component picture (not the separated component profile; that is P1b); f1 → 3D FOM comparator set
  (upper bound on optimal FOM cost); m1 → 3D cost vs N under the **mesh-free / scoring-lattice contract**
  (labelled, never substituted for the full-field contract); p1 → batching/A7–A8. The P1 EXPORT schema
  carries a `contract` field on every cost number.

## AM-7. After Codex audit round 2 (`codex-p1-restatement-2.md`)
**Controls (blocker 1).**
- *Production timing fault:* demoted to a **reported sensitivity probe** (no required outcome). The
  synthetic fixture (frozen: `numpy.random.default_rng(0)`, 6 cases × 3 meshes × 3 repeats) remains the
  mandatory H-cost-N negative control.
- *Doubled work:* **workload-matched.** B = the same evolution executed twice from the **same** initial
  state with the second call data-dependent on a scalar of the first output (e.g. adds 0·sum(W₁) to the
  input) so it cannot be elided; both outputs consumed. Predicate: B/A ∈ [1.8, 2.2] **and** identical
  iteration counts in both executions (otherwise the control is invalid, not passed).
- *Component sum vs fused:* compared at matched scope only — the sum of per-iteration component times ×
  the recorded iteration count of the same query vs the evolution-only time of that query; bounds as AM-5.
- *Numerical negative controls (under-resolved rule, wrong-gradient, zero-update):* replaced by the rule
  "**each must be a discriminating fixture with an independently established expected result**":
  - under-resolved rule: the lane picks the fixture rule and shows **independently** (a CPU NumPy ρ
    computation against a converged rule on the frozen states) that it violates the ρ bar before using it
    as the Q2 negative control;
  - Q1 negative control: a solver mutation whose wrongness is established independently of the
    stationarity metric — e.g. a premature-exit solver (iteration cap 1 at production tolerances) on a
    frozen trajectory segment whose one-iteration residual is shown by an independent NumPy evaluation of
    γ to exceed 1e-3; the sign-flipped Jacobian is **reported only**;
  - zero-update: on a frozen trajectory whose independently computed motion (scoring-lattice distance
    between output 0 and output 5) exceeds 100 × 2.5e-4.
- *CN-R order gate:* interpreted only after showing the solver (tight vs tighter) and quadrature (Q2)
  contributions at the Δt₀/4 step are < 0.1 × the time-error difference being fitted; otherwise "order
  check not interpretable" (blocks only the scheme-order label).
- *Sensitivity probes:* correction — Fibonacci rules use a seeded random shift
  (`hari_quadrature.py:107–116`, seed 0). P1 uses **one frozen seeded realisation**; shift sensitivity is
  out of P1 scope (E1, priority 4).

**H-acc-r (blocker 2), with precedence:**
1. Population = eligible cases. If > 25 % of them lack Q1+Q2 qualification on either rung →
   **inconclusive** (stop).
2. Resolved = qualified, |Δe| > 2ε_ref. If resolved < 50 % of eligible → **inconclusive** (stop).
3. Else **improves** iff > 50 % of resolved improve and median paired change < 0; otherwise
   **does not improve**.
Whole-range monotone verdict: **monotone** iff every transition "improves"; **not monotone** iff any
transition "does not improve" (even if others are inconclusive); **inconclusive** otherwise.

**Scope notes adopted from round 2.** Percent thresholds use the actual eligible populations (35/38 and
15/16 were examples). Q2 is a validation-population check (16 cases); the other cases are not described
as individually Q2-tested; nominal-policy and R512 arms get the same Q2 disposition if they support any
claim. Reproduction compares the **numerical** configuration; provenance (code commit, scoring version)
is recorded old vs new, not required equal. P1g distinguishes historical `acc` (`gauss96`, R384) from
`R384` (`gauss128`); the frozen-vs-nominal delta is measured on both rules at R384 if the historical
claims being re-checked used `acc` (lane decides from its file:line mapping; +cost goes in the
certificate). The 3D P1e successor of the W512 lattice rule is an explicit Gauss comparator frozen in
the lane design (historical W512 `gl56` evidence: speed3d `DESIGN.md:415–418`).

## AM-8. After Codex audit round 3 (`codex-p1-restatement-3.md`)
- **Doubled work, replaced:** B = **two separately dispatched, separately synchronised executions** of the
  same compiled evolution from the same initial state (`block_until_ready` after each), timed as one B
  interval; A = one such execution. Predicate: B/A ∈ [1.8, 2.2] and identical iteration counts. The
  "0·sum(W₁)" anti-elision construction of AM-7 is withdrawn.
- Adopted as lane obligations (round 3 "safe lane work"): the under-resolved-rule fixture must fail the
  **actual Q2 predicate** (vs its successor); the Q1 negative fixture must fail the **pooled > 1 %**
  criterion, not just one step; zero-update motion is measured against the verified tight trajectory
  under the Q1 contract; CN-R order checks include anchor-error and asymptotic-regime checks.
