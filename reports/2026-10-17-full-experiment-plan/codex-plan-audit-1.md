**Verdict: NEEDS-RESTATEMENT. Do not use this draft as a launch specification.** Most experiments are sensible questions, but the plan mixes existing capability with new development, assumes conclusions contradicted by current results, understates dependencies, and miscounts the sealed cohorts.

I made no filesystem changes, ran no experiments, and did not access sealed solutions. Running-lane status below comes from local records, not a live scheduler query.

For readable citations, these abbreviations refer to exact files; line numbers below refer to those files:

| Key | File |
|---|---|
| PLAN | [New experiment plan](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/PLAN.md) |
| OLD | [Governing JCP plan](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-06-jcp-offmesh-paper-plan.md) |
| INV | [Evidence inventory](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-06-jcp-offmesh-paper-plan/01-evidence-inventory.md) |
| COMB | [Combined-lane report](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/report.md) |
| WIDE | [Wide-bank report](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/report.md) |
| TIME | [Time-stepping report](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2/report.md) |
| REF | [E0 report](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/report.md) |
| SPEED | [Speed3d design](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/DESIGN.md) |
| Q2 | [Original 2D quadrature report](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-study/reports/2026-10-01-burgers2d-offmesh-quadrature.md) |

**Existing assets and knob ranges**

| Knob/capability | Verdict | Repository evidence and required correction |
|---|---|---|
| Meaning of \(k\) | **NEEDS-RESTATEMENT** | Use \(r=R'\) for bank coefficients and \(K\) for head latent dimension. The deployed span is linear in \(r\) coefficients. The original 2D configuration has \(R=512\), head \(K=16\), span rungs 128/384; these are different dimensions (`quadrature-study/qcore.py:3–6,53–57`). PLAN:76 cannot remain unresolved while interpreting scaling. |
| 2D \(r=32,64,128,256,384,512\) | **CORRECT as available prefixes** | A frozen R512 checkpoint and ordered rotation exist. `qcore.py:141–145` explicitly uses `T[:, :Rp]`; `trust_linear`, lines 152–156, also accepts the prefix. No new training is intrinsically required for 32/64. **They are not qualified deployment rungs**: WIDE tested 128–512, not every proposed prefix. |
| 3D \(r=64,\ldots,1024\) | **CORRECT as available prefixes** | Both M2 and W1024 banks are loaded by `jcp-combined/c3common.py:36–39,99`. W1024’s `bank.pkl` exists locally, 113,318,738 bytes. Its 256/512/768/1024 prefixes were evaluated; smaller prefixes need qualification. WIDE:9,111–113. |
| Banks “for each mesh” | **NEEDS-RESTATEMENT** | These are continuous banks reused across meshes, with mesh-dependent operators/projection. Original 2D evidence covers 256²/1024²/4096²; wide 2D principally 1024². Original 3D covers 64³/128³/256³; W1024 evidence initially covers 64³/128³. Availability of a bank does not establish high-mesh table feasibility or accuracy. INV:9,15; WIDE:38,192–198. |
| \(M/r=1,2,4,8\) | **NEEDS-RESTATEMENT** | No bank retraining is required. Wide-bank drivers accept configurable \(M\)/ratios; 2,3,4 already ran. However, combined 2D hard-codes `4*Rp` (`c2common.py:109–140`), and 3D uses `complete_M(n,4*r,...)` (`c3common.py:124`). Changing these drivers requires implementation and requalification. |
| “Currently \(M=4r\)” | **WRONG literally in 3D** | Complete eigenvalue shells make \(M\) approximate: R512 uses M2052; R1024 uses M4097. Mechanism report:17; WIDE:192. Report actual M, not only nominal ratio. |
| \(M/r=1\) as normal operating point | **NEEDS-RESTATEMENT** | Treat it as a deliberately weak-test-space control, not an equally credible production setting. A small tested residual at square \(M=r\) does not establish small physical error. Shell completion also means nominal ratio 1 may not mean exact equality. |
| \(m\), failing through converged | **CORRECT** | Necessary, but “converged” must be established for the bank, solver, time step, and reached states. W1024 required escalation to Gauss 80³, checked against 96³; 112³ failed with the existing builder. COMB:137–191,446. |
| Mesh range | **NEEDS-RESTATEMENT** | 8192² reference capability is not 8192² ROM-table capability. E0’s standard scoring range ends at 4096²/384³; outside it `restrict` is diagnostic-only. REF:277. Large full-grid bank storage needs chunking; speed3d already owns that work. |
| \(\Delta t_0/8\)–\(5\Delta t_0\), schemes | **NEEDS-RESTATEMENT** | Define dimension-specific \(\Delta t_0\), exact output-time divisibility, startup and solve form. 3D offmesh code asserts exact alignment (`offmesh.py:119–120`). LSPG-CN is not established second order; some LSPG-CNR/BDF2 settings also fail order establishment. TIME:17–23. |
| Rule families | **NEEDS-RESTATEMENT** | Gauss/lattice/Sobol/Smolyak infrastructure exists. **2D Halton already has a generic full-solve path**, although historical configs place it in the \(\rho\) ladder. MC is absent from the inspected production dispatch; 3D `rules.py` has neither Halton nor MC generators. |
| LSPG/Galerkin/low mixed frequencies | **NEEDS-RESTATEMENT** | LSPG versus GAL is a change of solve equation, not simply a different sine-test list. GAL uses \(Q^\top\mathcal R=0\) with \(A=QR\). TIME:27–31. Low mixed-frequency tests need new selection/build logic and independent stability checks. |
| Adaptive/fixed/QN solvers | **CORRECT, with qualification** | Implementations exist; QN work is in speed3d. Fixed-sweep is not synonymous with converged LM. COMB:159–169 documents persistent failures. |
| Viscosity/IC range | **NEEDS-RESTATEMENT** | Specify numerical ranges, development OOD versus reserved OOD, reference resolution, and acceptance criteria before evaluation. “Smaller viscosity” is not a complete experiment. |
| Bank seed/resolution/smoothness | **CORRECT as experiments, not existing matched assets** | Smooth-bank has treatment and seed diagnostics, not 3–5 matched repetitions of every proposed recipe. Its report:14–26 shows substantial seed/rotation effects. |
| Batch 1–256 | **NEEDS-RESTATEMENT** | Speed3d currently specifies 1,4,16,64 or largest fitting, with trajectory-equivalence controls. SPEED:160–164. Batch 256 needs a memory pilot and latency definition. |

More specifically, the code availability questions resolve as follows:

- **Smolyak full solves exist.** 2D includes `ctrl_smolyak8` in rollout arms (`make_configs.py:23`); 3D generates `smol8` and feeds non-tensor rules through the offmesh solver (`rules.py:184`; `qpanel.py:214,537`).
- **Halton is not a greenfield 2D solver task.** `qcore.py:83–100` generates it, and `qstudy.py:270–292` builds arbitrary point-rule arms. Adding rollout configuration and qualification is necessary; claiming an existing accepted Halton rollout result would be wrong.
- **Allen–Cahn code exists in Hari’s imported prototype**, including cubic point reaction and time-dependent setup: `external/quadrature-study-2026-09-30/quadrature/nmrom/pdes.py:106–141` in the quadrature-study worktree. That is not a production, full-scale JCP lane.
- **The imported exponential example is steady Bratu**, explicitly `steady=True`, `T=0` at `pdes.py:167–174`. It is not the proposed time-dependent Frank–Kamenetskii experiment.
- **NS3D exists**, including coordinate-bank training, mesh ROM, and spectral comparison. It is not already a Burgers-style offmesh deployment. INV:25,54 records the distinction and the weaker bank floor.
- **Non-box geometry has historical L-shaped Poisson evidence**, but that does not supply a nonlinear offmesh geometry implementation. The existing Burgers problem already has nonperiodic zero-Dirichlet walls, and E0 uses a sine spectral solver successfully. REF:47–55.
- **Central+CN-R and spectral Burgers FOMs exist in both dimensions.** E0 `refcore.py:102–109` is dimension-parametric; `pscore.py:111–167` provides sine pseudo-spectral ETDRK4. Speed3d ports these into comparison infrastructure. Do not mistake the earlier upwind-in-space CN comparator for central+CN.
- **Production POD+DEIM/ECSW/GNAT comparisons are not established.** OLD:32 and INV:24 identify this gap; the dedicated hr-baselines lane currently has README, sync script, and vendors, not a completed implementation/report.

**Per-item audit**

“CORRECT” below means a sound experiment to retain, not a guaranteed positive result.

| Item | Verdict | Evidence and correction |
|---|---|---|
| **A1** | **NEEDS-RESTATEMENT** | Fix bank, \(r,M,m\), scheme, tolerances, output contract and workload across meshes. Distinguish fixed-work kernel timing from adaptive rollout timing. Q2:727 times *median query minus median decode*, not a directly isolated solve. Its registered bar was endpoint ratio [0.8,1.25]. |
| **A2** | **NEEDS-RESTATEMENT** | The stated powers are conditional leading-order models. Residual-only nonlinear work is \(O(mr+mM)\); Jacobian is \(O(mMr)\); normal-equation formation is \(O(Mr^2)\); factorization is \(O(r^3)\). With \(M\propto r\), these reduce to the advertised powers, but iteration counts and implementation fusion matter. See derivation below. |
| **A3** | **CORRECT** | Essential factorial experiment. Add a fixed-M panel: fixing m alone does not isolate r when M grows with r. Wide-bank’s independently configurable \(r,M\) supports this (`w2d.py:310–380`). |
| **A4** | **NEEDS-RESTATEMENT** | Define \(m^*\) over a discrete candidate list, rule family, shifts, reached-state population, and both \(\rho\) and rollout-distance bars. COMB:175–189 demonstrates that the original selection can return **none**, while a post-hoc alternative selects rules. |
| **A5** | **NEEDS-RESTATEMENT** | \(m(2r+M)\) and \(Mr^2\) count advection-table scalars, not peak memory. WIDE:236 reports 1.61 GB advection tables versus 75.76 GB sampled peak. Include f64’s 8 bytes, temporary matrices, mesh bank, compilation, and concurrently live rules. |
| **A6** | **NEEDS-RESTATEMENT** | Prefix deployment does not retrain a bank per r. Separate shared bank-training cost, prefix-specific setup, per-mesh setup, selection, compilation and query cost. Break-even exists only for an accuracy-qualified competitor with a positive per-query saving. WIDE:111–113. |
| **A7** | **NEEDS-RESTATEMENT** | Define throughput, latency, memory ceiling, heterogeneous viscosity batches and equivalence tolerances. Reuse speed3d’s route 3 rather than duplicating it. SPEED:160–164. |
| **A8** | **CORRECT** | Necessary to delimit the claim. Full-field output has mesh-dependent cost. Mesh-free projection plus scoring-lattice decode is a different output contract, already under test in speed3d. SPEED:150–159; Q2:727. |
| **B1** | **NEEDS-RESTATEMENT** | E0 ST is appropriate within its contract; S is BE-matched at dt0 only. Flagged case-kinds remain visible but unranked. COMB:44–48. A bank prefix is not a promise of monotonic accuracy. |
| **B2** | **NEEDS-RESTATEMENT** | Retain separate ablations, but do not add norms as an exact error decomposition. Include interactions, IC projection, spatial operators, scoring error and reference uncertainty. OLD:86 explicitly requires interactions; COMB:119–125 rejects the proposed cancellation explanation. |
| **B3** | **NEEDS-RESTATEMENT** | Existing generic solver paths make this feasible in stages. It is not a ready full Cartesian sweep. MC and 3D Halton need generators/configuration and gates; random families need repeated draws. Evaluate own reached states, not just tensor-reached states. |
| **B4** | **CORRECT** | Necessary and substantially supported by time2/combined infrastructure. Measure observed order against a same-form converged anchor and verify solver/quadrature errors are smaller. Never award order two from the method name. TIME:17–23. |
| **B5** | **NEEDS-RESTATEMENT** | Plot only qualified configurations on the headline frontier; keep failed ones visible separately. COMB:100–105 shows R384→R512 median 0.198→0.199%, worst 1.78→2.17%; the dial need not improve monotonically. |
| **B6** | **NEEDS-RESTATEMENT** | M/r changes need no retraining but require driver changes in combined. Separate sine-mode selection, residual weighting and GAL equation changes. Include exact M=r as a diagnostic control, not a presumed useful setting. TIME:31; WIDE design:35–40. |
| **C1** | **NEEDS-RESTATEMENT** | Add **FOM mesh** as a selection dimension; “at every N” cannot force an unnecessarily fine comparator. Define eligible accuracy using both median/worst or predeclared casewise criteria and uncertainty. Confirm frozen validation-selected matches in paired timings. OLD:88; SPEED:184–187. |
| **C2** | **NEEDS-RESTATEMENT** | Depends on qualified B5 and C1. Distinguish same-mesh crossover from best-accuracy-matched crossover allowing a different FOM mesh. Speed3d already covers substantial 3D work. SPEED:166–187. |
| **C3** | **NEEDS-RESTATEMENT** | Scientifically essential, but new baseline development rather than an existing run matrix. Equal rank alone is insufficient: include smaller POD ranks, native solvers, common-solver ablations, offline costs and equal tuning budgets. OLD:89 contains these requirements; the new draft drops several. |
| **C4** | **NEEDS-RESTATEMENT** | Define equal data, training budget, retained rank, basis normalization, derivative accuracy and representation floor. Otherwise a decoder comparison is confounded by fitting quality and ordering. Imported small CPU studies are exploratory, not full-scale evidence. INV:20. |
| **D1** | **NEEDS-RESTATEMENT** | Freeze distributions and distinguish parameter extrapolation, IC-family change and stiffness. Reserved OOD solutions cannot become development data. E0 generation/QC did not authorize ROM scoring. REF:237–258; SPEED:208. |
| **D2** | **NEEDS-RESTATEMENT** | Allen–Cahn prototype exists; production assets/references need development. \(Mr^3\) is the storage of a direct cubic tensor, not a lower bound for every competing implementation. Register the predicted **absence of an advection-stencil gap**. OLD:90. |
| **D3** | **INFEASIBLE as an existing-assets run** | The located exponential PDE is steady Bratu, not time-dependent Frank–Kamenetskii. Requires a new problem, training data/bank, reference and stability regime. It is feasible as a separately scoped development project. |
| **D4** | **NEEDS-RESTATEMENT** | NS3D code/spectral FOM exist, but offmesh transfer is unimplemented in the inspected campaign path. Its bank floor is already substantially worse than POD-64. Include pressure/divergence, translation treatment and representation controls. INV:25,54. |
| **D5** | **WRONG as motivation; needs new scope** | Nonperiodic walls do not imply spectral methods struggle: the current zero-Dirichlet box admits the sine spectral FOM. A non-box nonlinear test would be new work; historical linear L-shaped Poisson does not establish it. REF:47–55. |
| **E1** | **NEEDS-RESTATEMENT** | “64 random lattice shifts per rule” does not apply unchanged to deterministic Gauss/Smolyak. Use shifts/scrambles where appropriate and validate coverage for the **deployed single draw**, not the shift mean. Include an independent-family bad-vector control. OLD:92. |
| **E2** | **CORRECT** | Necessary. Repeat the complete training/ordering/selection recipe, not only checkpoint initialization. Smooth-bank report:24–26 shows meaningful seed variation, including +128.7% in one gradient-error comparison. |
| **E3** | **CORRECT; priority too low** | Must precede expensive accuracy/rule sweeps. Fixed CN K=2 fails persistently on 7–15/64 cases, with discrepancies up to 0.436. COMB:161–169. This cannot wait until priority 4. |
| **E4** | **NEEDS-RESTATEMENT** | Measure singular values/conditioning of A, weighted J and normal equations separately. Orthogonalization can preserve the span but change truncation/regularization semantics; regularization changes the problem. Freeze fixes on validation. WIDE:160; solver uses \(J^\top J\). |
| **E5** | **NEEDS-RESTATEMENT** | Make paired method comparisons within each physical GPU/job, then compare the dimensionless conclusions across hardware. Same GPU model across different jobs is insufficient. Record memory model, clocks, precision and software. Lane rules:42. |
| **F1** | **NEEDS-RESTATEMENT** | Freeze **globally**, including all baselines, thresholds, OOD policy, fallback rules, scoring, failure handling and indicator calibration. Merely hashing the principal method choices is insufficient. OLD:97. |
| **F2** | **WRONG** | Counts are doubled. REF:245–252 lists four batches of **32 physical 2D cases**, each with S and ST: **128 cases, 256 case-kinds**. REF:255–258 lists two batches of 32 physical 3D cases: **64 cases, 128 case-kinds**. Opening remains contingent on global freeze and unresolved custody/opening procedures. |

**The scaling argument needs this correction**

For the actual stored offmesh matrices,

\[
B,D\in\mathbb R^{m\times r},\qquad P\in\mathbb R^{m\times M},
\]

the code computes

\[
N(c)=P^\top[(Bc)\odot(Dc)],\qquad
J_N(c)=P^\top[\operatorname{diag}(Dc)B+\operatorname{diag}(Bc)D].
\]

This is explicit in [offmesh.py:84](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/offmesh.py:84).

Consequently,

\[
C_N=O(mr+mM),\quad
C_{J_N}=O(mMr),\quad
C_{J^\top J}=O(Mr^2),\quad
C_{\mathrm{Cholesky}}=O(r^3).
\]

The normal equations and Cholesky are explicit at lines 129–134. With \(M\approx4r\), A2’s powers are reasonable **per-operation** approximations. They are not a rollout complexity theorem:

- The 3D quadratic implementation often obtains the nonlinear value from \(\tfrac12J_Nc\), so separately timing an ideal residual-only kernel does not measure the deployed path.
- The original fixed-sweep solver first takes three adaptive steps; subsequent predictor work and rejected updates still matter. `offmesh.py:115,147–183`.
- Adaptive iterations can depend on r, mesh-conditioned operators and tolerances.
- Increasing r while selecting \(m^*(r)\) and \(M(r)\) measures a **deployment policy**, not isolated r scaling.
- Small dense factorizations, matrix-vector products and large matrix multiplications need not exhibit their asymptotic slopes over this range.

**The 5% flatness bar is unsupported as currently phrased.** A range of 0.97–1.04 already has max/min \(1.04/0.97\approx1.072\), exceeding 5%, despite each endpoint being near one. Moreover, Q2:746 actually reports **1.060** for fast Gauss48². That report’s bar was [0.8,1.25], and speed3d uses max/min≤1.25, not 1.05.

A prospectively chosen 5% equivalence margin is permissible, but specify its statistic, repeat count, uncertainty interval and drift sensitivity. Do not infer it from favorable historical rows.

**The accuracy claim must become a hypothesis, not a requirement to “support.”** PLAN:7 is too strong because the deployed hybrid still uses mesh-dependent A, Laplacian eigenvalues and projection. Poorer R512 accuracy disproves an unconditional monotone-r dial; it does **not by itself** disprove approximate mesh invariance at fixed r. Likewise, 4–7% worst errors disprove a blanket high-accuracy narrative, not necessarily mesh invariance. These are separate claims requiring separate tests.

**Integrity and controls**

The top “Where things stand” block is stale: it says E0 has not started, while the later log records E0 and combined completion. The lane rules’ first-order-reference statement is likewise historical. Use REF and the later chronology, not those stale status sentences. [LAB-LOG.md:19013](/home/tahmid/Dev/Tunable-NM-ROM-Claude/LAB-LOG.md:19013) and [later completion](/home/tahmid/Dev/Tunable-NM-ROM-Claude/LAB-LOG.md:19038).

Required distinctions:

- **Validation-only selection; test scoring after freeze.** Historical `test64`/923901 are replication-only. “Scoring only on validation” is not the rule; *selection* only on development/validation is.
- **E0 is conditional evidence.** Its uncertainty is a Richardson estimate, not a rigorous bound. Thirteen historical 2D **case-kinds**, not necessarily thirteen distinct physical cases, are non-rankable. REF:40–43.
- **S does not remove time error for arbitrary schemes.** COMB:44 explicitly labels non-BE-dt0 S comparisons unmatched.
- **Preserve post-hoc labels.** E0 A10 changed the cross-check rule after inspecting data; combined’s 3D rule selection is post hoc; its A15 selection was retracted. These cannot be erased by freezing the new plan.
- **Do not convert reference QC counts into sample sizes.** The F2 error appears to propagate the lab-log summary’s case-kind counts.
- **Do not open reserved OOD cohorts during D1.** Develop new regimes using explicitly designated development cases, then freeze before final evaluation.

Each group needs explicit controls; PLAN currently provides none:

| Group | Minimum controls |
|---|---|
| A | Inject a timing fault exceeding the drift threshold; doubled-work timing; synchronization/dispatch checks; wrong-mesh table rejection; component-sum versus fused-query consistency. |
| B | Underresolved Gauss/lattice/Smolyak; case/time/reference-kind swaps; inadequate fixed-sweep solver; mismatched output-grid/normalization; manufactured temporal-order check. |
| C | Reproduce a trusted FOM implementation; reject nonconverged/overly loose comparators; reject a falsely matched accuracy pair; validate each ROM baseline’s residual/Jacobian independently. |
| D | PDE-specific manufactured/reference checks; Allen–Cahn no-advection-gap control; exponential-reaction stability checks; NS divergence/pressure checks; geometry boundary-condition checks. |
| E | Bad generating vector tested by an independent rule; zero-update/wrong-gradient solver; ill-conditioned/rank-deficient fixture; permuted-viscosity batching control. |
| F | Reject changed hashes, wrong reference version, restricted cohorts and ineligible rankings; verify global freeze before granting access. |

These should be demonstrated on a real-size smoke, as required by [lane rules:21–24](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-08-jcp-lane-agent-rules.md:21). Speed3d already specifies many reusable controls at SPEED:189–201.

**Priority and budget audit**

| Priority | Verdict | Assessment |
|---|---|---|
| 1: 10–20 GPU-h | **NEEDS-RESTATEMENT** | Plausible for a sharply bounded existing-bank pilot, not all widths, all meshes, accuracy qualification, component profiling and mesh-free implementation. WIDE’s 1024² grid took 2.28 h; its 3D dial 2.64 h. Those are much narrower tasks. |
| 2: 30–50 | **NEEDS-RESTATEMENT** | Solver qualification and reference eligibility must come first. “Every family at each k” plus M/test-space/time sweeps is unspecified combinatorially. Large converged-rule builds have already OOMed. |
| 3: 100–150 | **NEEDS-RESTATEMENT** | C3 needs substantial implementation; GPU-hours do not account for that. Move a minimal standard-ROM baseline earlier because it can invalidate the paper framing. Reuse speed3d for 3D C1/C2. |
| 4: 150–250 | **NEEDS-RESTATEMENT** | E3 and conditioning diagnostics belong before selection, not here. New PDEs require their own training/reference budgets. E1 cost depends critically on whether 64 draws mean kernel evaluations or complete rollouts. |
| 5: 100–150 | **NEEDS-RESTATEMENT** | A6 should start immediately from existing logs; A7 overlaps speed3d. NS/non-box development is not interchangeable with routine hardware repetition. |
| 6: 30–50 | **NEEDS-RESTATEMENT** | Depends on the actual frozen arm count and corrected cohort sizes. Include failures, qualified fallbacks and timing repetitions. |
| Total / wall time | **WRONG arithmetic; unsupported schedule** | Listed ranges sum to **420–670**, not 450–650 GPU-h. At 3–4 continuously busy GPUs that is roughly **4.4–9.3 days** of occupancy. Two–three weeks may be reasonable with queues and development, but the plan provides no critical-path model or contingency. |

Useful duration anchors are TIME:9–13, WIDE:16–23, and REF:36: E0 consumed **39.7 GPU-h**, including **8.06 h** for one 2D calibration. Conversely, W1024 training itself took only **1.85 H200-h**. Neither “everything is cheap” nor “everything requires hundreds of hours” follows; the expense is often qualification and reference work.

The large-table failure is concrete: COMB:485 records Gauss112³ failing on a **42.9 GiB allocation**; line 487 records another W1024 budget attempt failing after **3.6 h** while two large rules remained live.

Speed3d overlap extends beyond the parenthetical C1/C2 note:

- A7/A8: batching, projection and mesh-free query.
- E3: fixed sweeps, predictors and QN.
- C1/C2: upwind/central/spectral FOM comparison.
- A1 and portions of B1/B5: cross-mesh cost and accuracy.

Its namespace is `jcp-speed3d`, cap three, and its design estimates about **34 GPU-h**. Local ledger jobs are g2 **5301264**, g3 **5301270**, g1 **5301580**; FOM ladder f1 **5301274** is marked collected. The ledger’s `hours:24` entries are requested limits, not measured durations. Do not create competing copies in that namespace or assume unfinished outputs are accepted results.

**Missing reviewer questions**

The draft also needs:

- An explicit **linear-span versus nonlinear-manifold** claim. Existing headline span results do not establish a benefit from a nonlinear head.
- Accuracy-matched **coarse FOMs**, not merely same-mesh opponents.
- A fixed-M, fixed-m experiment distinguishing representation size from test-space and quadrature growth.
- Longer-time stability and error accumulation; current Burgers outputs end at \(T=0.25\).
- Physical diagnostics: boundary error, energy/dissipation behavior, overshoots/positivity where applicable.
- Initial-condition projection and output sampling error, including whether the scoring lattice misses fine features.
- Reference independence: central-CN comparisons partly share E0’s discretization, so retain spectral cross-checks.
- Statistical treatment of failed cases, repeated selection, training seeds and lattice shifts; do not bootstrap correlated time snapshots as independent cases.
- Shared versus per-prefix/per-mesh offline cost, storage, compilation and reproducible artifact provenance.
- Representation-floor comparisons using identical norms, times and normalization. WIDE:113 explicitly warns that its training-validation floors are not rollout errors.

**Ranked top 10 problems**

1. **F2 doubles physical sample sizes:** 128/64 sealed cases, not 256/128.
2. **The headline claims assume the desired outcome:** mesh independence and increasing-r accuracy are not unconditional properties of the deployed hybrid.
3. **Solver and quadrature qualification come too late:** selected-arm failures can invalidate the proposed accuracy/cost frontier.
4. **A1 confounds r, M, m and solver work; its 5% criterion is undefined and stronger than historical evidence.**
5. **C1 omits freely varying comparator mesh**, risking another over-solved FOM comparison.
6. **Existing assets are confused with validated configurations:** small prefixes exist, but new M/test spaces/PDEs are not turnkey.
7. **E0 is treated too uniformly:** BE-only S matching, non-rankable case-kinds and conditional uncertainty must govern every comparison.
8. **Global freeze and OOD custody are underspecified**, with real risk of cross-lane selection leakage.
9. **Memory accounting and budget exclude expensive setup/qualification**, despite documented H200 OOMs.
10. **The plan duplicates speed3d while delaying standard-ROM baselines**, which are among the earliest possible falsifiers of the paper’s central claim.

**Corrected priority-1 experiments using existing assets**

1. **Freeze a development-only manifest and reuse current evidence.** Use 2D dev6/val32 and 3D 923801; record checkpoint/rotation hashes, actual M, E0 eligibility and post-hoc status. Reuse saved combined results before rerunning anything.

2. **Run an isolated kernel panel on existing bank prefixes.**  
   Use 2D r=32/64/128/256/384/512 and 3D W1024 prefixes r=64/128/256/512/768/1024. Time residual, Jacobian, normal-equation formation and factorization separately, with fixed m and fixed M panels plus nominal M≈4r panels. Label small prefixes as kernel experiments until rollout-qualified. Existing wide-bank builders avoid retraining.

3. **Run a bounded 2D mesh-cost panel first.**  
   Start with r=128/256/384 at 256²/1024²/4096², using one frozen rule per r across meshes. Measure directly isolated stepping, complete query and decode separately; retain paired repetitions and iteration counts. Include R512 as a diagnostic, not a presumed better rung.

4. **Qualify accuracy before constructing B5.**  
   For those rungs, use existing BE and LSPG-CNR implementations, E0 ST, solver-distance checks and converged/finer-rule checks. Failed quadrature rungs remain reported but excluded from a qualified Pareto claim. S appears only with its BE-dt0 qualification.

5. **Run a small fixed-r quadrature-cost slice.**  
   At one middle width per dimension, use a failing rule, deployed rule and finer rule already available in the repository. This supplies A3 evidence without launching every rule family, shift and time step.

6. **Reuse speed3d for 3D mesh-free timing and FOM comparisons.**  
   Accept its results only after its audits; do not duplicate A7/A8/E3/C1/C2. Keep the new work focused on kernel scaling and any explicitly uncovered fixed-M question.

7. **Report storage and total peak memory separately, and inventory offline costs now.**  
   Reuse existing setup/training logs; add measured peaks to the timing panel. No retraining, new PDE, sealed opening, or Gauss112³ construction belongs in this first tranche.

This is a feasible existing-assets first tranche with limited driver/instrumentation work. Treat **10–20 GPU-h as a pilot cap**, then price expansion from measured setup, compilation and qualification costs.