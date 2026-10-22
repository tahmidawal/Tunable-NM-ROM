# imp-fronts E10 DESIGN (design-only, no GPU): steep-front follow-up after E8

Status: v2.2 (2026-10-22). **Activated:** E8 arm (b) ended STOP (endpoint) (`checks/phaseb_verdicts.json`): both arm (b)
and K1 are STEP-UNCONVERGED on all 8 fresh FC1 test cases, and even the diagnostic FC1 medians (arm (b) vs K1, in
`checks/phaseb/score_b.json` / `score_K1.json`) miss the 2× bar. Activation does not authorise runs. No GPU use is authorised by this document. Numbers are read from
`checks/*.json`; every accuracy number is conditional on E0 / E0-style references (Richardson $\varepsilon_{\rm ref}$).

## 1. Evidence, stated at the strength it supports

- **A2 (in-range cohorts dev6 + val32):** decoded r = 384 ROM fields have a median 4096-sine tail ratio to the reference
  of ≈ 2 165 with wall share $\omega$ ≈ 0.95; the bank projections of the references show a similar wall-localised excess
  in range. On C1 and C2 the projection/reference tail ratios are only ≈ 1.7 and ≈ 0.7 (`a2_wall.json` groups), so on
  steep cohorts the *excess* over the solution's own tail is small, although the tail that remains is still
  wall-localised (ROM wall shares ≈ 0.96 / 0.95, projection ≈ 0.94 / 0.95): the boundary mechanism is strong in range
  and weak, not absent, on steep cohorts. The envelope $16x(1-x)y(1-y)$
  *permits* a non-zero normal second derivative of a bank field (it does not force it for every function); the decoded
  fields measured do carry one (gradient-difference estimate ≈ 29× the interior Laplacian scale).
- **Phase B K1 policy:** all FC1 cases terminate STEP-UNCONVERGED, i.e. the step check at factor 0.5 failed and factor
  0.25 has no finer check. This establishes **missing step certification** on steep states, not the size of the time
  error.
- **A4:** on C1, hybrid (bank 512 + 512 sines) 4.35 % < 1024 sines 5.85 % < bank 512 6.58 % (worst floors); the
  registered 2× hybrid threshold failed. This says the sine enrichment did not halve the floor, not that the learned
  span has no advantage (bank 512 beats 512 sines on C1).
- Phase B fresh test outcomes informed this design, so they are **development data** for E10; E10 confirmation uses
  new test-once seeds (§5).

## 2. Question

Which of three candidate fixes moves steep-front (C1-type, $s\in[250,500]$) ROM errors toward qualification: (i)
certified time refinement, (ii) state-triggered quadrature escalation, or (iii) a boundary-compatible bank — each as a
single change against the same control?

## 3. Arms (single changes; control = K1 bank with the Phase B policy, rescored on the new test seeds)

- **E10-s (step refinement, numerics only):** the Phase B state machine extended by one factor level (0.125) so that
  factor 0.25 gets a step check; same bank (K1), same integrator (LSPG-CN-R), same rules. Single change: the factor set.
- **E10-q (quadrature escalation, numerics only, F13):** state-triggered escalation along the registered ladder (the
  `next_candidate` map of DESIGN Amendment 3, not literal doubling) — any state whose successor-rule $\rho$ exceeds
  0.116 is re-solved with the next ladder rule — with Q2 then checked on the **full rollout** of the
  adaptive policy; same bank, integrator and factors. Stop rule (F13 verbatim): *stop the quadrature arm if C1 does not
  become Q2-qualified.*
- **E10-c (boundary-compatible bank; carried over from deferred E8 arm (c) / F16):** the B1(c) construction. Before any
  code, an amendment freezes: integer Fourier convention $B=\operatorname{round}(2B_0)$ from the seed-1 draw, frozen (no
  optimiser update), MLP initialisation and normalisation identical to the incumbent, seed 1, the incumbent data and
  training budget, the rotation step, and the four-reflection evaluation cost (the incumbent's per-evaluation cost does
  not transfer; it is measured).
- **E10-sq (time + quadrature, F13 combined arm): deferred** — registered only if both E10-s and E10-q CONTINUE, as a
  combination test, by amendment.
- **Considered and deferred, with reasons:** an adaptive embedded-pair time integrator (changes integrator and
  projection together, so it is not a single change; would need its own fixed-step comparator on the same bank and
  projection); F8 transport/shift-aware or registered banks, front-width dilation and local banks with local
  hyper-reduction (the ideation found the A−/C1 contrast to be amplitude/steepness at matched geometry, so transport is
  not the identified mechanism, which deprioritises shifts/registration; front-width dilation and local banks with
  local hyper-reduction are deferred for budget, not on evidence, and are the natural next step if E10-c and E8 (b)
  both fail).

## 4. Qualification, endpoints and stop rules

- Every arm keeps the full Phase B contract: per-run validity, fresh-Jacobian stationarity (pooled ≤ 1 %), every-case
  tight distance ≤ 2.5e-4, tight/tighter anchors, Q2 (ρ ≤ 0.116 and successor rollout distance ≤ 2.5e-4) on the arm's own
  reached states, step convergence; population minima, missing-data handling and guard precedence as DESIGN
  Amendments 3–5, 8.
- E10-s: report the number of FC1 cases that become step-converged (partial success, descriptive) separately from the
  registered endpoint: CONTINUE iff qualified on every eligible FC1 case and the FC1 median ≤ K1's diagnostic median / 2.
- E10-q: stop if C1 does not become Q2-qualified (F13); otherwise the same all-case endpoint as E10-s.
- E10-c: the brief's arm-(c) rule — stop if $m^*$ is not ≥ 2× smaller **and** the tail not ≥ 3× smaller (either route
  passes) — and stop if the in-range worst floor > 2× K1's.
- **Must-fail controls (frozen fixtures):** for E10-c, an incumbent-bank decoded field known (from A2) to violate wall
  compatibility must fail the oddness/curvature identity, and a passing compatible fixture (a sum of sines) must pass;
  for E10-s, a dev case known to be step-unconverged at 0.5 must be flagged; for E10-q, a deliberately undersized rule
  (fib4181) on a dev C1 case must trigger escalation.
- Timing (if any arm continues): paired same-job, burn-in, saved repetitions, coefficient recomputation, controller and
  certification overhead included, comparator tuned on dev before scoring.

## 5. Cohorts and budget (projection only; not requested)

New test-once fresh seeds (root distinct from "IMPF"): FI 6, FN0s 6, FC1 8 — references ≈ 20 × 433 s × 1.5 ≈ 3.6 GPU-h
on H200 (measured Phase B rate). E10-s ≈ 0.6 GPU-h, E10-q ≈ 0.6, K1 rescoring ≈ 0.3 (policy jobs). E10-c: training
4.2–4.6 (measured K1 range, plus a measured four-reflection overhead — uncertified until smoked), selection 0.35,
policy 0.5, smoke 0.5 → ≈ 5.6–6.0. Total ≈ 10.7–11.1 GPU-h, of which E10-s/E10-q/K1 + references ≈ 5.1.

## 6. To freeze by amendment before any release (open, acknowledged)

The E10-c architecture amendment (initialisation, normalisation, rotation, identity tolerances); the new test-once seed
root and protocol; NOT RUN semantics for the numerical arms and the extended escalation/round cap; control fixture case
and state identities with their measured violation thresholds; timing comparator search, equal-error tolerance,
medians and failure counts; smoke-derived projections for reflection overhead, controls, certification and comparator
searches; rejection/retry checks for every control.

## 7. Deliverables

This design after a clean Codex audit; the E10-c architecture amendment; a smoke plan. No runs until released.

## 8. Freeze (2026-10-22, coordinator release of E10-s and E10-q with an E10 cap of 10 GPU-h; before any E10 code)

- **Released arms:** E10-s and E10-q only (E10-c is E8 arm (c), Amendment 19 of DESIGN.md, under E8's cap). Defer order if
  the cap binds: E10-q first, then E10-s.
- **Cohorts (coordinator decision):** the Phase B fresh test cohorts and references (refT: FI 6, FN0s 8, FC1 8, FC2 2) and
  the 16 fresh dev/val cases, not new seeds. These test cases were already scored for K1 and arm (b), so for E10 they are
  **re-used test cases**, reported as such (not test-once); the control is K1's Phase B policy result, re-used unchanged.
- **E10-s (single change: factor set):** factors {1, 0.5, 0.25, 0.125} (driver factors, $dt=DT0\cdot$factor); the step check
  at 0.25 compares with 0.125; STEP-UNCONVERGED only at 0.125. Everything else = Phase B policy (maps, Q2, Q1 on prod,
  anchors FI 4, 5 and FC1 4, 5, pooled fraction). Round cap 1 + 7 + 3 = 11; BUDGET-CAP at the job ceiling.
- **E10-q (single change: state-triggered escalation):** inside each prod run, after each accepted step, the successor-rule
  ρ is evaluated at the accepted state; if ρ > 0.116 the step is re-solved with `next_candidate` and that rule is kept for
  the rest of the run (no de-escalation), up to gauss384. Q2 for the adaptive run = rollout distance ≤ 2.5e-4 between the
  run and the same policy with every rule used replaced by its `q2_successor` (gauss384 → gauss512), plus ρ ≤ 0.116 at the
  run's own accepted states. Factors {1, 0.5, 0.25} and the step check as in Phase B. Stop rule (F13 verbatim): *stop the
  quadrature arm if C1 does not become Q2-qualified* — operationalised: fewer than all eligible FC1 test cases Q2-qualified.
- **Endpoint for both (unchanged):** CONTINUE iff numerically qualified on every eligible FC1 test case and the FC1 median
  ≤ K1's diagnostic median / 2; otherwise STOP; partial counts (cases newly step-converged / Q2-qualified) reported as
  descriptive. NOT RUN if the job is not admitted or fails before any test solve.
- **Must-fail controls (frozen fixtures, smoke on fresh dev cases):** E10-s — FC1 dev j = 0, whose Phase-B SM-r dry run
  failed the step check at 0.5, must again fail it at 0.5 (a check that cannot fail is a bug); E10-q — a forced fib4181
  start on FC1 dev j = 0 must trigger at least one escalation (SM-r's forced round escalated). If a control does not fire
  the job is blocked; one retry after a code fix, re-audited.
- **Timing:** no timing claim is made by E10-s / E10-q (endpoints are accuracy and qualification); if either CONTINUES, a
  timing amendment comes first.
- **Budget (E10 cap 10 GPU-h; ceilings):** smoke SM-e 0.4 (H200: both controls + one FC1 dev case per arm), E10-s policy
  job 0.8 (H200; factor 0.125 doubles the finest-step work), E10-q policy job 0.6. Total 1.8.

## 9. Amendment E10-2 (2026-10-22, after `audits/codex-e10-4.md`; supersedes §§1, 3, 5–7 where they conflict, and §8)

- **Supersession.** E10 is released (coordinator); runs are allowed under the E10 cap; confirmation uses the re-used Phase B
  fresh test cases (§8), not new seeds; E10-c is DESIGN Amendment 19–20 under E8's cap; §5's budget is replaced by §8/§9.
- **E10-q: resolution withdrawn (after `audits/codex-e10-5.md`).** The recorded ρ (max ≈ 0.032) covers only the six output
  states of each run, not every accepted step, so it cannot show that the trigger never fires. Pending: an every-step
  computation of the successor ρ on K1's saved per-step states (`rom/rho_every_step.py` → `checks/phaseb/e10q_every_step.json`,
  validated against the recorded output-state values). **If** every accepted state of every prod run has ρ ≤ 0.116, the
  E10-q adaptive run executes exactly K1's solves (no escalation: same rules, factors, successor, tight and half runs), so
  its outcome is K1's and it STOPs without a job; **otherwise** E10-q stays deferred (its adaptive-policy semantics in
  audit e10-4 are not implemented in this lane) and is reported as such.
- **E10-s, precedence (complete):** NOT RUN (job not admitted or failed before any test solve) → INCONCLUSIVE if fewer than 6
  eligible FC1 test cases or K1's FC1 diagnostic median is non-finite → INCONCLUSIVE also if a must-fail control or a
  prerequisite is unavailable → STOP if not numerically qualified on every eligible FC1 case, or the E10-s FC1 median is
  non-finite or > K1 diagnostic median / 2 → CONTINUE. Missing certification (BUDGET-CAP, unavailable half step)
  counts as not qualified; reported separately from measured step failure.
- **E10-s semantics.** Exactly the Phase B executor with the factor set {1, 0.5, 0.25, 0.125}: at factor $f>0.125$ the half
  run is at $f/2$; there is no half solve at 0.125 (STEP-UNCONVERGED there); promotion of a half run to prod gets its tight solve (Amendment 5); tighter anchors at the terminal state; Q2
  at every visited factor as Phase B (successor replayed from prod's initial state at the same factor). Round bound
  1 + 7 + 3 = 11. Must-fail control: FC1 dev j = 0 fails the step check at 0.5 (SM-r record) — rerun in the smoke, must fail
  again; plus a synthetic-record unit test for every new transition.
- **Budget (E10 cap 10):** SM-e 0.3 (H200, E10-s control + one FC1 dev case at all four factors, timing for the
  projection), POL-e10s 0.8 (H200; projection from SM-e's whole-step wall clock incl. escalation, certification and anchors, scaled to
  40 cases × 1.5 using K1's Phase B path distribution extended by the 0.125 level, certified against 90 %); admission as Amendment 5
  against the E10 bucket; retries charged to the same bucket.

## 10. Amendment E10-3 (2026-10-22): E10-q every-step evidence

`checks/phaseb/e10q_every_step.json` (CPU recompute with the r2acc code path on K1's saved per-step states): 171 prod runs,
17 771 states (171 initial + 17 600 accepted-step states); maximum successor ρ 0.0386 overall and 0.0148 on FC1; no state exceeds 0.116. The recompute
reproduces the recorded output-state ρ to 2.5e-9 relative (137 runs; registered tolerance changed from "≈ 1e-12" to
1e-6 relative, since the recorded values come from the GPU job and the recompute is on CPU; the margin to the 0.116
trigger is a factor ≈ 3). Therefore the E10-q trigger never fires on any visited prod run, the E10-q adaptive run executes
exactly K1's solves, and **E10-q's outcome is K1's: STOP** (FC1 already Q2-qualified, not step-converged on any case).
No E10-q job is run; 0 GPU-h.

**E10-3 addendum (after `audits/codex-e10-6.md`).** The E10-s must-fail control is pinned: FC1 dev j = 0, rule gauss256,
factor 0.5 (the state whose step check failed in the SM-r dry run). The audit's independent CPU ρ spot-checks timed out;
the every-step maxima rest on `rom/rho_every_step.py`'s validation against the recorded output-state values (2.5e-9).
