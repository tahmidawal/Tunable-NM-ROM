# Red team: Reviewer 2 on "Off-mesh quadrature hyper-reduction for NM-ROMs with coordinate-network decoders"

Evidence read: the 3D report (`reports/2026-10-01-burgers3d-offmesh-quadrature.md`, "3D" below), the 2D report (`reports/2026-10-01-burgers2d-offmesh-quadrature.md`, "2D"), Hari's `SUMMARY.md`, `LITERATURE.md` and `DECODERS.md` ("Hari"), LAB-LOG, memory notes, `paper_latex/main.tex` and the 2026-09-20 claim audit. Every number below is quoted from those files and none is new. Severity: **F** = fatal, **M** = major, **m** = minor.

## 1. Ranked objections

**1. "Your 'beats the full-order model' result is a first-order-upwind artefact, and so is your reference." (F)**
*Referee:* "The FOM is first-order sign-upwind with backward Euler. The 'refined' reference is the same scheme at 513³. It differs from the 257-node solution by 1.70%, it is tolerance-limited at about 1e-5, and the Richardson extrapolation was added post hoc. An off-mesh ROM that drops the upwind numerical diffusion will of course beat that FOM at 64³. You are measuring the weakness of the baseline discretisation, not the strength of your method."
*Current answer:* only partly. 3D §2 states the bias openly. At 256³, R′=256, the selected rule and the tensor differ by 0.58 pp, inside the reference uncertainty (3D §8). Off-mesh 2.83% vs tensor 10.46% at 64³ is far outside the 2.28% Richardson displacement, so the ranking holds there. But "more accurate than the same-mesh FOM" (2.83% vs FOM best 9.85% at 64³) is exactly the kind of result a second-order FOM would erase. 2D is in better shape: 8192² with Δt/16.
*Neutralise:* (a) build a reference with demonstrated convergence: a second-order or higher FV/WENO solver, or a pseudo-spectral one, on three nested levels, reporting the observed order and the extrapolated error bar. Then state every accuracy difference against that error bar. (b) Add a second-order FOM to the speed panel (central/MUSCL advection, IMEX or SSP-RK) and rerun the speedup rule against it.

**2. "This is super-resolution from training data, not hyper-reduction." (F)**
*Referee:* "The 3D bank was trained on fields up to 129³ (32/64/128³ data). The off-mesh error is 2.83% at 64³, 128³ and 256³, identical to three digits. The online solve never sees the mesh. The 'mesh invariance' is a tautology, and the win over the 64³ FOM is information imported from 128³ training data."
*Current answer:* conceded in 3D §8 ("a property of the trained bank, not a free lunch"). The ROM's 2.83% also beats the FOM *at its own training resolution*: FOM best at 128³ is 5.22%. That fact suggests the gain comes from dropping upwind diffusion rather than from training resolution, but the two mechanisms are not separated. In 2D the bank was trained at 256² only (`main.tex` l.3245), which is a weaker confound, but nobody has made this argument explicitly.
*Neutralise:* two controls. (i) Train a bank only on the coarsest-mesh data (64³ in 3D, which 2D already effectively is) and repeat the 64³ comparison. (ii) Add an arm that uses the **analytic derivative at the mesh nodes** (an equal-weight uniform sub-lattice) to separate "continuum derivative" from "off-mesh points" (see 3). Stop calling mesh invariance a result; call it a design property.

**3. "The mechanism is the analytic gradient, not off-mesh points." (M, borderline F for the title)**
*Referee:* "Every resolved rule lands on the same rollout (dist conv ≤ 0.09% for lat8192/gl24/gl32/kor). The upwind sub-lattice `lat64` loses because it evaluates the *stencil*, not because its points are on the mesh. A trapezoidal rule on mesh nodes with the analytic gradient is presumably spectrally accurate for an integrand that vanishes on the boundary. Hari's own 'lattice ≈ periodic trapezoid' argument says so. Where is that arm?"
*Current answer:* no. Every on-mesh arm uses the stencil, and every off-mesh arm uses the analytic gradient.
*Neutralise:* add an "on-mesh nodes + analytic gradient" arm at matched m. If it matches Gauss/lattice, retitle to "continuum residual evaluation" and present off-mesh placement as a convenience. If it does not match, you have a real result about point placement.

**4. "No comparison with the standard hyper-reduced linear ROMs." (F)**
*Referee:* "Where are POD-Galerkin/LSPG with DEIM, Q-DEIM, ECSW or GNAT? Your POD-LSPG rows use a dense residual. That is a straw man that is 1–3 orders slower by construction (LAB-LOG l.18032). Burgers is quadratic: POD-Galerkin with a precomputed tensor is the textbook baseline, and it is exactly your 'incumbent tensor' with a POD basis."
*Current answer:* no. The project's own history is unfavourable. On NS 3D the per-mesh POD-64 bank reaches 0.15% against 1.09–1.28% for the coordinate network (LAB-LOG top block). On Poisson, POD-128 dominates every neural checkpoint (tunability campaign). Burgers 2D is the one favourable case: the neural head reaches 2.56% vs POD-LSPG k=128 at 10.1% for 7× the cost, but that was a dense residual.
*Neutralise:* in the same job, on the same cohorts, run POD-LSPG and POD-Galerkin at matched online dimension (R′ = 256, 512) with ECSW (NNLS on mesh elements) and with Q-DEIM, plus POD + exact quadratic tensor. Report accuracy against the converged reference (objection 1) and the timing. If POD+ECSW wins, the paper has to be about the quadrature question for a *given* smooth basis, not about the ROM being competitive.

**5. "The nonlinear-manifold setting failed. The title is false." (F)**
*Referee:* "In 2D the head setting failed the confirmation (B3 worst 0.585/1.105/1.258% against a 0.50% bar) and failed mesh invariance for every arm, continuum included (ratio 1.90–1.92). In 3D the head was 'not run'. All successful settings, `acc`/`fast` in 2D and the 3D 'span model', are linear rungs on a fixed basis. This is a linear ROM with a neural-field basis."
*Current answer:* the evidence confirms the objection (2D "Answers in brief" test table; 3D answer 4).
*Neutralise:* rewording. Drop "nonlinear-manifold" from the title and the claims, and report the head failure in the main text as a limitation. Alternatively, fix and re-run the head in 2D and 3D under frozen pre-registration, which is costly and of uncertain outcome.

**6. "Burgers is quadratic. The exact tensor already exists, so you have not shown a need." (M→F)**
*Referee:* "For a quadratic nonlinearity, Galerkin ROMs have used an exact precomputed tensor for decades. Your method only beats that tensor because of the upwind stencil (objection 1) and because of memory: 4.30 GB vs 0.10–0.34 GB at R′=512, a fair point. The case for fixed quadrature is non-polynomial nonlinearity, and there your only evidence is a scaled-down CPU study whose ROM errors are 13–22% (Hari §3)."
*Current answer:* weak. Hari tests Allen–Cahn, HJ and Bratu, but at R=128, on CPU, with a 3D bank floor of 40% worst.
*Neutralise:* one full-scale GPU case with a genuinely non-polynomial nonlinearity, where no tensor exists and the competitor is ECSW/EQ. Candidates: Bratu/Arrhenius e^u, or porous-medium diffusion u^m with non-integer m. Use the same protocol as 3D: validation selection, frozen held-out, must-fail controls. The memory argument should be put forward explicitly, since R′² scaling is the tensor's real weakness in 3D.

**7. "Speedup over an efficient FOM is small, negative, or a ladder-granularity artefact." (F if claimed, m if not)**
*Referee:* "At 64³ the reduced model is slower than Newton–BiCGStab in every arm (0.06–0.32×, 3D §5). At 128³ it ranges 0.20–1.13×. In 2D at 256², the FOM `lean_nt3e-3` takes 18.1 ms at 0.048% same-grid error, against 77–81 ms for `acc`. At 4096² the 525 ms vs ~53–60 ms comparison sets a 0.05% FOM against a ~1% ROM. Your headline 6.45× at 256³ exists because the next FOM rung (3.01%) misses the ROM's 2.83% by 0.18 pp, well inside the reference's own ~2% uncertainty. One rung lower gives 2.59×."
*Current answer:* the reports are honest about all of this. The 2.21×→2.59× (same-grid rule) and 6.01× (R′=256) at 256³ are the only defensible numbers.
*Neutralise:* report a continuous FOM cost-vs-error curve (dense Δt and tolerance ladder, interpolated) instead of a "fastest rung at least as accurate" rule. Show the crossover mesh. Use the second-order FOM from objection 1. Never print a single speedup without the accuracy pair beside it. Expect the referee to ask about a pseudo-spectral or direct-method FOM on this box geometry as well (see 12).

**8. "Mesh-independent cost is true only for the advection term." (M)**
*Referee:* "At 256³ the query is dominated by projecting u₀ and decoding six output fields on the mesh (3D §8). In 2D the full query is not flat (decode 0.2→18 ms). Offline, A = ΦᵀĜ, the discrete eigenvalues and the projection are all O(N)."
*Current answer:* the reports say so (3D §5, 2D D5). The claim is honestly scoped there but will not stay scoped in an abstract.
*Neutralise:* (a) project u₀ by the same quadrature at the rule points when u₀ is analytic, and decode outputs only at requested probes, then show a truly mesh-free online path. (b) Give a cost model (online and offline flops and bytes as functions of m, R′, M and N) and a measured breakdown table. (c) Report bank training cost: 4608 trajectories in 2D, and A100-hours.

**9. "The fully converged continuum target needs more points than the mesh has." (M)**
*Referee:* "The 2D continuum target is Gauss 640² = 409,600 points, against 65,536 mesh nodes at 256². Gauss 128² still gives continuum ρ of 2.1e-3 to 7e-3, and the authors write 'our bank converges much more slowly in the number of points than Hari's, cause not isolated'. Your deployed rules use m = 4096–13,824 in 3D for R′ = 256–512, i.e. m ≈ 8–27 R′, whereas ECSW typically needs m = O(R′). This is not hyper-reduction in the usual sense."
*Current answer:* partly. The deployed rules (Gauss 96², Fibonacci 1597) are far smaller than 640², and the rollout converges well before ρ does (spread across resolved rules 0.009%, 3D §8).
*Neutralise:* separate "rollout-converged m" from "ρ-converged m". Plot rollout error vs m with the knee marked. Explain why a 409k-point target is needed (bank bandwidth: measure the bank's Fourier spectrum). Compare m against ECSW's m at equal ρ on the same basis.

**10. "'A-priori error control' is claimed, but the rule was selected empirically on reached states." (F for that claim)**
*Referee:* "Continuum ρ of a fixed rule *grows* with mesh because the reached states are sharper. lat4096 at R′=512 crosses the 0.116 bar at 256³ (3D §8). The selection used validation reached states, just as EQ certification does. Where is the theorem? An RFF/SIREN network is analytic but not band-limited, and no bound is computed."
*Current answer:* no bound exists, and the selection is empirical (3D §7).
*Neutralise:* either (a) derive a computable bound: lattice error in terms of the Fourier tail of the integrand on the dual lattice, with the bank's spectrum estimated numerically, plus a Gauss bound via the Bernstein ellipse of the decoded integrand; or (b) adopt a randomized-shift lattice with an **online a-posteriori estimator** from ≥ 8 shifts, and show that it tracks the true ρ. Option (b) is cheap and would be a genuine contribution. Otherwise, drop "a priori".

**11. "Single lattice shift, single Sobol scramble, selection from validation, reused held-out cohorts." (M)**
*Referee:* "One random shift (seed 0) per lattice and one scramble for Sobol, so there is no distribution. In 2D the pre-registered recommendation was *none* in all three settings, and the deployed rules are post-hoc (amendments A3–A5 after seeing data). The two-sided B1 bar was reinterpreted one-sided. Both held-out cohorts had been evaluated by earlier lanes (3D §8 last bullet, 2D limitations). In 3D, the selection rule re-applied to held-out data picks a different rule at R′=512."
*Current answer:* all of this is disclosed, which is good, but it is still the record.
*Neutralise:* ≥ 16 shifts/scrambles with median and max, plus a fresh, never-touched held-out cohort for the final table. State in the paper that 2D rules were chosen post hoc. The referee will read the selection-flip as instability, so show that the flip changes the error by less than 0.01 pp.

**12. "This is a spectral Petrov–Galerkin method with a learned basis. Gauss quadrature of the nonlinear term is textbook." (M, novelty)**
*Referee:* "Your tests are M ≈ 4R′ Dirichlet sines on a box, with diagonal discrete Laplacian eigenvalues. Evaluating the nonlinear term by quadrature at off-grid points is the pseudo-spectral/dealiasing literature (Canuto–Hussaini–Quarteroni–Zang; Boyd). Smolyak failing on tensor sines is hyperbolic-cross exactness theory (Novak–Ritter). CROM already evaluates the INR off-mesh with analytic gradients and claims mesh-independent cost. Weder–Schwerdtner–Peherstorfer decouple collocation points from the grid. VPINN analysis (Berrone–Canuto–Pintore) covers quadrature with fixed test functions. Hari's own DECODERS table shows a *fixed, untrained sine bank* gives the best end-to-end error (8.95%), and that POD with cubic interpolation also works end to end (14.50 vs 14.52%). So why a neural network?"
*Current answer:* `LITERATURE.md` §2 already concedes most of this. The novel residue is narrow: fixed classical rules replacing *fitted* EQ in an LSPG ROM, plus the characterisation of rule families.
*Neutralise:* rewording plus one experiment. Position the work against spectral methods and CROM explicitly. Make the decoder-smoothness result (analytic vs C¹ vs C⁰ bases, Hari §8) a central finding at full scale. Include a sine-bank and a POD+interpolation arm in the main experiment so the reader can see what the neural basis buys, if anything.

**13. "Smolyak and Sobol are straw men." (m→M if presented as findings)**
*Referee:* "Nobody would integrate tensor sines up to frequency ~25 with a Clenshaw–Curtis sparse grid, and plain equal-weight Sobol is a 1/m method. These are controls, not competitors. The relevant competitors are a uniform trapezoid grid with the analytic gradient, Gauss with a 3/2-rule count, and a continuous empirical cubature (CECM, Hernández 2024) fitted on continuum points."
*Current answer:* the 3D report labels Smolyak a must-fail control, which is correct. Hari's summary presents "Smolyak: no" as a verdict.
*Neutralise:* label both families as controls throughout. Add trapezoid and CECM arms. Cut the "Smolyak fails" claim from the novelty list, or keep it as a one-paragraph remark with the hyperbolic-cross explanation.

**14. "Box domain, homogeneous Dirichlet, sine tests. Does it generalise?" (M)**
*Referee:* "Lattice rules and Gauss rules on the unit cube with integrands that vanish on the boundary are the ideal case for both. What happens on an L-shape, a cylinder in a channel, a curved boundary, or a Neumann condition? Do you then need a cut-cell or mapped quadrature, and do you lose the super-algebraic rate?"
*Current answer:* no. The project has an L-shape Poisson case, but no quadrature study was run on it.
*Neutralise:* one non-box geometry, e.g. an L-shape or a disk via a mapped/composite Gauss rule, with a distance-function Dirichlet factor. Report ρ ladders and one rollout. Otherwise, scope the title and abstract to tensor-product domains.

**15. "The discretisation is inconsistent: discrete diffusion plus continuum advection." (M)**
*Referee:* "You keep the mesh's discrete Laplacian eigenvalues Λ and the mesh-projected linear terms, but integrate continuum advection. Which equation is the ROM consistent with? The answer changes with N on the diffusion side only. Has stability been analysed? LSPG with an inexact, sign-indefinite quadrature (Smolyak's negative weights cause 54–61 non-stationary exits) has no energy argument."
*Current answer:* no.
*Neutralise:* add an all-continuum arm, with continuum eigenvalues and a quadrature projection, so that the ROM is mesh-free and consistent. Add a short stability discussion: positive weights plus exact linear terms give a quadrature-perturbed residual, with a perturbation bound. Run longer horizons, since the current ones are 25 steps in 3D and 50 steps to t = 0.25 in 2D.

**16. "Low viscosity and shocks are untested." (M)**
*Referee:* "The required m grows with state sharpness (3D §8). Burgers' interesting regime is small ν, where coordinate networks have spectral bias and quadrature of a near-shock needs many points. Your bump widths and ν are benign."
*Neutralise:* a ν ladder (or narrowest-width ladder), reporting the rollout-converged m and bank floor vs ν. State where the method stops working.

**17. "The FOM is not state-of-the-art, and the box admits fast direct or spectral solvers." (M)**
*Referee:* "Is the backward-Euler Newton–BiCGStab well preconditioned (multigrid)? For viscous Burgers on a box, an explicit or IMEX pseudo-spectral solver with a 3/2 rule may beat every ROM here at 64³–256³."
*Current answer:* no. An earlier memory note found a direct Poisson solve 494× faster than the iterative FOM. The user previously restricted paper comparisons to named iterative FOMs (memory: paper FOM scope), but a JCP referee will ask regardless.
*Neutralise:* add at least one modern FOM (multigrid-preconditioned, or IMEX pseudo-spectral) to the speed panel, or justify the iterative FOM as representative of general geometries. That justification only holds if objection 14 is addressed.

**18. "Results come from one trained bank, one seed and one PDE family." (M)**
*Neutralise:* a second bank training seed in 3D (and in 2D, if cheap), with the rollout error spread. Two seeds is the minimum a referee will accept.

**19. "Hari's study is not evidence at journal standard." (M)**
*Referee:* "CPU only, R=128, six evaluation cases, timings under heavy load, an EQ re-implementation fitted on 16 states that reaches only ρ = 0.6 in 3D, and absolute ROM errors of 13–51%."
*Neutralise:* do not cite Hari's numbers as support for generality or for EQ's weakness. Use his study as motivation, and re-run anything quoted at full scale under the 3D protocol.

**20. "The EQ comparator is weak." (M)**
*Referee:* "Hari's 3D EQ (ρ = 0.6) and the 2D head EQ (m=1024, ρ 1.0e-1 dev / 2.8e-2 test) are under-trained. The tunability campaign showed that EQ fitted on reachable states certifies q ≤ 64 at m=1024. Compare against the best EQ, not a convenient one."
*Neutralise:* an EQ refit on reached states with a generous snapshot budget, at matched m, certified by held-out ρ (the project's own rule).

**21. "The flop and timing comparison is hardware-specific." (m)**
The off-mesh Jacobian is 29 GFLOP vs the tensor's 1.1 GFLOP and is faster only because of H200 memory bandwidth. *Neutralise:* add a roofline argument and one other GPU, and state the bytes vs flops trade.

**22. "The acceptance bar is arbitrary." (m)**
ρ ≤ 0.116 is inherited from a discarded manuscript. *Neutralise:* justify it by rollout sensitivity (ρ vs rollout distance curve), or replace it with a rollout-based criterion.

**23. "Reproducibility." (M)**
*Referee:* "The checkpoints are multi-GB, the code backup is a filtered mirror, the cluster jobs used specific GPUs, and the 2D bank needs 4608 training trajectories."
*Current answer:* good internal hygiene: generated tables, NumPy audits, sha256-pinned selection, must-fail controls.
*Neutralise:* a public archive (Zenodo) with the bank weights, rule point sets (CBC vectors are given), seeds, one-command reproduction of the 3D tables, and an explicit statement of the offline training cost.

## 2. Fatal-unless-fixed subset and minimal experiment set

**Fatal:** 1 (biased first-order reference and FOM), 2 and 3 (super-resolution and mechanism confound), 4 (no hyper-reduced POD baseline), 5 (title claims a nonlinear manifold that failed), 6 (quadratic-only, where the exact tensor exists), 10 (a-priori claim vs empirical selection), and 7 *if any speedup is headlined*.

**Minimal experiment set (in priority order, all under the 3D lane protocol: validation selection, frozen fresh held-out, same-job timing, NumPy audit):**

1. **Converged reference plus a second-order FOM** (3D at 64³–256³; 2D can reuse 8192²/Δt/16 with an observed-order check). This covers objections 1 and 7.
2. **Mechanism controls:** an on-mesh analytic-gradient arm, and a bank trained on coarsest-mesh data only. This covers 2 and 3.
3. **POD-LSPG/Galerkin with ECSW and with the exact quadratic tensor, plus best-practice EQ on the same bank,** at matched R′. This covers 4 and 20.
4. **One full-scale non-polynomial nonlinearity** (e^u or u^m), where tensors are impossible. This covers 6.
5. **Randomized shifts (≥ 16) with an online shift-variance estimator.** This covers 10 and 11 and converts a weakness into a contribution.
6. **A cost model and breakdown, mesh-free u₀ projection, and the crossover mesh.** This covers 8 and 9.

Second tier, to do if time allows: a ν ladder (16), a non-box geometry (14), a second seed (18), an all-continuum consistent arm (15).

## 3. Framing

**Dishonest or rejectable framings:**
- "Mesh-independent online cost" without the qualifier "for the nonlinear term; projection and output decode are O(N)".
- "The ROM is more accurate than the full-order model", when the FOM is first-order upwind and the bank saw finer data.
- Any single headline speedup, especially 6.45× (256³, which depends on ladder granularity inside the reference error) or ~10× at 4096² (which compares 0.05% against ~1% accuracy). Silence about the FOM being faster at 64³ and at 256² in 2D is also unacceptable.
- "Nonlinear-manifold ROM" in the title, when every successful setting is a linear span and the head failed confirmation.
- "A-priori error control, no certification needed", when rules were selected on reached validation states and ρ grows with the mesh.
- "Lattice rules are the right 3D rule" as a headline, when the frozen R′=512 selection is Gauss 24³.
- "Smolyak/QMC fail" listed as a contribution.
- "Mesh invariance" presented as an achievement, when it holds by construction.
- Using Hari's CPU Allen–Cahn/HJ/Bratu results as evidence of generality.

**Honest and still publishable framing (JCP, or SISC/CMAME as a fallback):**

> *Continuum residual evaluation for reduced models with smooth neural-field bases: replacing mesh-fitted hyper-reduction by fixed quadrature.*
> When the trial basis is a smooth function of x, hyper-reducing the nonlinear term becomes a numerical-integration problem with a well-defined continuum target. This has three consequences, each measured. (i) The ROM no longer inherits the mesh stencil's O(h) consistency error: rollouts agree across 64³–256³ to 0.07%, while the stencil-based tensor's error halves per refinement. (ii) No per-mesh or per-state fitting is required, and memory falls from M R′² (4.3 GB) to m(2R′+M) (0.1–0.34 GB). (iii) Which rule works is governed by integrand smoothness: tensor Gauss and rank-1 lattices converge fast, quasi-random and sparse-grid rules do not, and randomized shifts supply an online error estimate.
> Limitations stated up front:
> - the cost advantage over a good FOM appears only at large meshes;
> - projection and output remain O(N);
> - the study covers linear-span ROMs on box domains;
> - the nonlinear-head variant did not confirm.

This framing survives if experiments 1–3 and 5 come out as the current data suggest. It becomes a stronger JCP paper if experiment 4 shows the method beating ECSW on a non-polynomial nonlinearity, because that is the case where the classical tensor alternative does not exist.
