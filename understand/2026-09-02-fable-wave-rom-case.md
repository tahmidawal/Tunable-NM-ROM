# The case for (and against) wave ROMs on a nonlinear manifold — independent analysis, 2026-09-02

**Provenance.** Written by a Claude Fable 5.1 subagent asked to argue, adversarially against the
2026-08-30 retirement of waves, whether the separable-decoder ROM can work on the wave equation.
It read the 08-14 and 08-16 wave cells' JSONs and Codex reviews, and ran a **1D CPU check**
(`understand/2026-09-02-wave-manifold-check/wave_manifold_check2.py`, numpy/scipy only, no GPU,
nothing in the repository modified). **Its numbers below are from that 1D check and from the
08-16 cell's JSONs; none of them is a result of this project's 2D pipeline.** The 2D cell that
tests these claims is designed in `understand/2026-09-02-handoff-wave2d-cell.md`.

Verbatim report follows.

---

## Bottom line first

Your conclusion ("don't attempt waves") is defensible, but your *reason* is wrong, and the wrong reason is now written into the lab log and `WAVES-RETIRED.md` as "architecture-independent". The 08-16 record plus a small CPU check say:

- The energy/reversibility argument is **not decisive and not intrinsic**. A Lagrangian manifold ROM conserves the pulled-back energy for *any* smooth decoder, needs no change to `u = G h(z)`, and with the exact-residual machinery costs `R x R` matrices. "Structure-preserving latent stepping is a genuinely open problem" is false for this PDE class.
- **But fixing the structure does not fix the accuracy.** In the check below, the variational formulation takes energy drift from 0.62 to 1.0000 and the trajectory error does not improve (0.87 vs 0.69). The 08-16 failure is a *manifold-quality* failure (a wrinkled, poorly generalising autodecoder: train 0.07, held-out 0.19), and conservative dynamics integrate that tangent-space error instead of damping it. That is the part of your intuition that survives, restated.
- Quadrature error was **not** a contributor: the 08-16 full-grid, no-EQ variants failed identically to the EQ ones.
- Even a working wave ROM **cannot pay** on the *linear* wave for exactly the reason heat did not pay. The only reason to run it is as a certified stepping stone toward conservative nonlinear systems (compressible Euler/acoustics on your roadmap), and to correct the record.

Recommendation: yes, one cheap 1D cell, in a specific form (section 5), with the explicit expectation that it settles *mechanism*, not headline speed.

## 1. What the 08-16 record actually says (numbers you did not quote)

From `wlat_rom_N64_K8*.json`, `wlat_stepdiag_N64_K8_k8*.json`, and the two Codex reviews:

| fact | number | consequence |
|---|---|---|
| `lspg:full:weak64` (full grid, exact projection, no EQ) | 0.850 | quadrature ruled out |
| `lspg:full:fd` (full-grid strong residual, N=128) | 0.864 | same |
| `lspg:eq256:weak64` (the headline arm) | 0.843 | EQ neither helps nor hurts |
| kinematic-recursion energy ratio (your 0.27) | 0.22–0.30 | |
| dynamic-velocity energy ratio, same runs | **8 to 50** (growth) | the two energy proxies disagree by ~100x; the second Codex review says the kinematic one is dominated by decoder noise, `O(e/dt)` alternating. "Energy ratio 0.27" is not a clean measurement of anything. |
| autodecoder train / held-out reconstruction | 0.070 / 0.189 | a wrinkled manifold |
| FiLM decoder conditioned on true `(params, t)`, dim 6 (08-14 cell) | 0.028–0.035 held-out | the manifold IS low-dimensional and smooth; the autodecoder did not learn it smoothly |
| ROM error at RS=8 / 20 / 40 | 0.843 / 0.878 / 0.906 | error grows weakly with *more* steps: horizon-proportional systematic dynamics error, not per-step noise |
| stepdiag from oracle starts, excess over floor at H=1/2/5/10 | 0.009 / 0.053 / 0.20 / 0.46 | one snapshot interval is nearly free; accumulation is superlinear |
| stepdiag `hold` (freeze latent) at H=1/5/10 | 0.29 / 0.91 / 1.35 | the ROM does track the wave, roughly 2x better than doing nothing |

Note also that the POD-8 ROM (0.3424) sits *on* its projection floor (0.3417). Linear symplectic-lift ROMs track their floor even at 34% error. That asymmetry is the whole story: a linear Galerkin wave ROM is forgiving of representation error; a curved manifold is forgiving only if its *tangent space* is right.

## 2. Q1: intrinsic property, or artefact of the stepping? Answer: artefact, in the sense you meant; intrinsic, in a sense you did not mean

**The structural claim is wrong.** Take the FOM as a Lagrangian system, `L(u,u_t) = 1/2|u_t|^2 - 1/2 c^2 u^T(-L)u`, and *any* smooth decoder `u = g(z)`. Restricting to the manifold gives `L_r(z, z_t) = L(g(z), J_g z_t)`, and Lagrange–d'Alembert says the Euler–Lagrange equations of `L_r` are exactly the tangent-space Galerkin projection of the second-order FOM, `J_g^T (u_tt - c^2 L u) = 0` (the `J_g^T H_g(z_t,z_t)` curvature terms cancel identically; I checked the algebra). This system conserves `E_r = 1/2 |J_g z_t|^2 + V(g(z))` exactly in continuous time, and a variational integrator (Marsden–West) makes it symplectic on `T*R^K` with bounded energy error for any `g`. Nothing about `g` has to be symplectic, because the u-only second-order form puts velocity in the tangent bundle automatically. That is the cotangent-lift trick of Peng–Mohseni / Lall–Krysl–Marsden, applied to a curved manifold — 2003 material, not an open problem.

For `u = G h(z)` this is almost embarrassingly cheap: `M_r(z) = J_h^T (G^T G) J_h`, `K_r = G^T(-L)G`, `V = 1/2 c^2 h^T K_r h`, force `= J_h^T K_r h`. Two precomputed `R x R` matrices, zero quadrature, a `K x K` Newton solve per step. The 08-16 scheme (LSPG on the FOM's Newmark residual with `J_g` at the new time level, or weighted sine-mode Petrov–Galerkin) is not variational; it is close to it only when the manifold is nearly flat.

**The CPU check.** 1D wave, N=128, Dirichlet, Gaussian-bump ICs with `v0 = 0`, CN FOM at 80 substeps, ROM at dt=1e-3 (RS=20), 4 held-out trajectories. Manifold `u = G h(z)` with a POD bank and a quadratic head, optionally with a small high-frequency "wrinkle" in directions orthogonal to the data (an overfit autodecoder in caricature). Arms: A = the 08-16 LSPG-Newmark; B = Galerkin with `J_g` at the new time; C/D = variational Verlet / midpoint on the pulled-back Lagrangian; P = POD-K Galerkin.

| manifold | oracle floor | A error / E_final | C error / E_final | POD-K error |
|---|---|---|---|---|
| K=3, smooth | 0.447 | 0.537 / 0.999 | 0.537 / 1.0000 | 0.517 |
| K=4, smooth | 0.292 | 0.298 / 1.000 | 0.298 / 1.0000 | 0.290 |
| K=6, smooth | 0.119 | 0.125 / 1.000 | 0.126 / 1.0000 | 0.115 |
| K=10, smooth | 0.022 | 0.024 / 1.000 | 0.024 / 1.0000 | 0.020 |
| K=6, wrinkle 5% | 0.124 | 0.213 / **0.757** | 0.204 / 1.0000 | 0.115 |
| K=6, wrinkle 15% | 0.152 | 0.692 / **0.616** | 0.871 / 1.0000 | 0.115 |
| K=6, wrinkle 15%, dt 4x coarser | 0.152 | 0.889 / 0.611 | 0.861 / 0.999 | 0.115 |

Three readings:

1. **A smooth manifold with a 45% floor tracks its floor under the 08-16 formulation, with energy 0.999.** Floor size does not cause the failure. Your formulation is fine on a well-behaved manifold; it drifts only when curvature is high.
2. **A wrinkle that moves the floor from 0.115 to 0.124 costs 1.7x the floor and 25% of the energy; at 0.152 it reproduces the 08-16 phenomenology exactly** (4.5x floor, energy 0.62, insensitive to dt). Reconstruction floor is a bad predictor of ROM viability; tangent-space quality is the predictor (tangent-space velocity residual at oracle points: 0.415 for POD-6, 0.488 and 0.672 for the two wrinkle levels).
3. **The variational arms remove the energy drift completely and do not recover the accuracy.** Energy conservation was a symptom.

So the honest decomposition is: (a) intrinsic to nonlinear manifolds — nothing; (b) consequence of the stepping choice — the energy drift, which is fixable and turns out not to matter; (c) intrinsic to *the manifolds this project trains* (per-snapshot autodecoders that generalise 2.7x worse than they fit) — the accuracy failure, which dissipative PDEs hide and conservative ones expose. "Not fixable by tuning" was right. "Because the symmetry is gone" was wrong.

## 3. Q2: the literature, and what each would require of `u = G h(z)`

- **Lagrangian/mechanical route (the natural one here).** [Lall, Krysl, Marsden 2003](https://www.cds.caltech.edu/~marsden/bib/2003/08-LaKrMa2003/LaKrMa2003.pdf) (Galerkin on a Lagrangian system yields a Lagrangian system) and [Carlberg, Tuminaro, Boggs 2015](https://arxiv.org/abs/1401.8044) (approximate the Lagrangian ingredients — metric, potential, forces — then apply Euler–Lagrange; energy conservation and symplectic maps retained). Requirement of our decoder: **none**; `J_h` full rank so `M_r` is nonsingular (K <= R). Requirement of the pipeline: a different online residual (tangent-space Galerkin, variational integrator, square Newton instead of LM), initial `z_t` from the velocity IC by least squares, and ideally a velocity-consistency term in training so `u_t` lies in the tangent space. That last item is the one that addresses the actual failure mode.
- **Symplectic manifold Galerkin on first-order canonical systems.** [Buchfink, Glas, Haasdonk, SISC 2023](https://doi.org/10.1137/21m1466657): project a canonical Hamiltonian system onto a symplectic trial manifold; reduced system is Hamiltonian with `H_r = H o g`, energy and Lyapunov stability preserved; the decoder must satisfy `J_g^T J_2n J_g = J_2k` (they enforce it weakly by penalty). For a `(u,v)` decoder `diag(G,G)[h_u; h_v]` this is a constraint on `J_hu^T (G^T G) J_hv` — imposable by penalty, i.e. an architecture-level change. [Brantner and Kraus 2023](https://arxiv.org/abs/2312.10004) make it exact by construction with SympNet-style layers — a full replacement of `h`. The cotangent-lift `h_v = J_hu(z) p` satisfies the constraint exactly and *is* the Lagrangian route above; so for a second-order PDE the u-only form gets you exact symplecticity for free.
- **Quadratic manifolds with structure.** [Sharma and Kramer](https://arxiv.org/abs/2203.06361) (Lagrangian structure, data-driven) and the Sharma–Najera-Flores–Todd–Kramer quadratic-manifold symplectic ROM: `u = V z + W(z⊗z)` with reduced operators derived analytically. Our head is a general MLP; a quadratic head is a special case, and the exact tensor machinery you already have makes it cheap. Not needed for the linear wave.
- **Linear symplectic subspaces.** Peng–Mohseni PSD (2016), Afkham–Hesthaven (2017), Gong–Wang–Wang (2017), Hesthaven–Pagliantini–Ripamonti rank-adaptive (2022), Gruber–Tezaur Hamiltonian OpInf (2023). These are the POD control you already ran; POD with the same basis for `u` and `v` is a symplectic lift, which is why 1.000003 came for free.
- **What genuinely is open:** structure-preserving *hyper-reduction* of non-polynomial forces on nonlinear manifolds (Chaturantabut–Beattie–Gugercin 2016 handle port-Hamiltonian with DEIM, linear subspaces). Your exact linear/quadratic collapse sidesteps that entirely for degree <= 2 — which is the one place the new machinery is a real asset for conservative problems.

## 4. Q3: is n-width separable from stepping? Yes, and it binds differently for the separable decoder

- The *linear* n-width is slow (measured, 4x worse than heat). The *manifold* is not: the conditioned FiLM at latent dim 6 reached 3% held-out on the same 2D wave. The 08-16 ceiling of 17% at K=8 was a training/latent-organisation failure, not an n-width one.
- For `u = G h(z)` the ceiling is the POD-R floor by construction (0.082 at R=64, N=64; unmeasured above). The head buys parameter efficiency, not escape from the span. So on waves the separable decoder is bounded by an R-width that decays slowly; at R=256 at N=64 you would probably land at a few percent. Adequate, not spectacular.
- The 08-30 maxim "on a linear PDE the nonlinear head can never buy accuracy" **does not transfer** to time-dependent linear PDEs: `cos(ω c t)` is nonlinear in `(c, t)`, the manifold over (IC, c, t) is curved, and K=8 beat POD-8 by 2x in reconstruction in 08-16. What does transfer is the *cost* verdict: a POD-R linear wave ROM is a precomputed `R x R` recurrence with no iteration, exactly like heat's "13–38x faster than FOM, nothing iterative competes".

## 5. Q4: does the exact residual change anything? Not by removing quadrature error

Quadrature error was already zero-effect in 08-16 (section 1). What the exact machinery *does* change is that the tangent-space Galerkin/Lagrangian form becomes exact at `R x R` cost, so the structure-preserving arm is now as cheap as the incumbent arm. That is the honest "changes something".

## 6. Q5: the cheapest experiment that settles it

1D reflective wave, N=256, `c ~ U(0.8,1.2)`, Gaussian-bump ICs with `v0=0`, T=1 and a 4T long-horizon run; the existing `b1d` separable pipeline, K=8, R=64, f64. Runs locally in well under an hour per arm; no cluster needed.

Arms and gates, in order — each can fail:

- **G0 manifold gates (before any rollout):** train/held-out oracle gap; tangent-space velocity residual `||(I - P_T) v||/||v||` at oracle-projected held-out states, alongside the POD-K value for the same states; stepdiag from oracle starts at H=1,2,5,10 with the `hold` control. Pass: gap < 1.5x and stepdiag excess at H=10 < 0.5x floor. Expect FAIL with autodecoder training as-is — that would already confirm the manifold, not the structure, was the problem.
- **G1 structure gate:** arm A (incumbent LSPG, exact `A = Φ^T L G`, `B = Φ^T G`) versus arm C (variational midpoint on `M_r`, `K_r`); report the *pulled-back* energy `E_r(z, z_t)` for both (drop the kinematic recursion), over 4T. Pass for C: bounded oscillation, no secular trend, `|ΔE/E| < 1e-2`. Expect PASS regardless of manifold quality.
- **G2 accuracy gate:** ROM error <= 1.5x oracle floor at T and 4T, and <= 0.5x POD-K. Expect FAIL under G0-FAIL; expect PASS if training adds a velocity-consistency loss (fit `v ≈ G J_h(z) z_t` with `z_t` free per snapshot — a modest change).
- **G3 cost, reported not gated:** arm C versus POD-K and POD-R Galerkin CN. Expect LOSS to POD-R; write it as such.

Likely outcome: G1 pass, G0/G2 fail on the current training, G2 pass with the velocity-consistency loss, G3 lost. That result corrects the log ("fails for manifold reasons, not structural ones; structure is free"), and gives you a certified conservative-stepping arm and a tangent-space gate that you will need before compressible Euler in any case.

## 7. Verdict, by claim

- "Cannot work on the wave equation": **overstated**. It can be made to work at roughly POD-2K-to-3K accuracy; it will not pay.
- "Boundary of applicability is dissipative, fast-decaying n-width": **half right**. Restate as: conservative dynamics do not forgive tangent-space error, and the project's autodecoders have unmeasured tangent-space error.
- The energy mechanism and "structure-preserving latent stepping is open": **wrong**; the 08-16 energy number itself is not a robust measurement.
- "Not fixable by tuning": **right**, for the wrong reason.
- Quadrature as the alternative explanation: **dead**, by the 08-16 data.

Should you attempt waves again? **Yes, in the 1D form above, as a mechanism study and a stepping stone — not as a result cell.** Then either retire it with a correct reason or carry the Lagrangian arm and the tangent gate forward to Euler.

Sources: [Buchfink, Glas, Haasdonk 2023](https://doi.org/10.1137/21m1466657) · [Brantner and Kraus 2023](https://arxiv.org/abs/2312.10004) · [Sharma and Kramer](https://arxiv.org/abs/2203.06361) · [Lall, Krysl, Marsden 2003](https://www.cds.caltech.edu/~marsden/bib/2003/08-LaKrMa2003/LaKrMa2003.pdf) · [Carlberg, Tuminaro, Boggs 2015](https://arxiv.org/abs/1401.8044) · [Chaturantabut, Beattie, Gugercin 2016](https://doi.org/10.1137/15m1055085)

---

## How this changed the plan (session note, not part of the report)

The user's request is the **2D** cell with both boundary conditions and a resolution cost
ladder, so the 2D design keeps that scope but absorbs every finding above: quadrature is
dropped as a candidate cause; manifold-quality gates (G0) run before any rollout; the
variational arm C runs beside the incumbent arm A; three head-training arms
(`auto`, `sup` on $(\mu,t)$, `auto+vc`) are the experimental variable; the cost ladder is
predeclared as likely lost to POD-$R$. The one claim in the report this session has **not**
independently verified is the algebra of the Lagrange–d'Alembert cancellation in §2; it is
item 4 of the Codex design audit.
