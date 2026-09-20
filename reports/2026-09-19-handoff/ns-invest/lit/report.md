# Nonlinear-manifold / neural ROMs on 2D incompressible Navier–Stokes: what the literature does, why ours fails, what to change

Literature review for the closed `ns2d` lane (branch `exp/2026-09-17-ns2d`, commit `9830d202`), written 2026-09-18.
Read-only; nothing in any worktree was touched. Paper texts were pulled from arXiv and are cached under
`lit/txt/` (pdftotext output) so every number below can be re-checked by grep. Numbers for *our* setup come
from `experiments/ns2d/DESIGN.md` (§A4–§A13) and `experiments/ns2d/reports/2026-09-17-ns2d.md`.

**State of the numbers:** literature numbers are quoted from the papers (final); the per-DOF-step cost
comparison in §4 is my arithmetic on those quoted numbers (provisional, marked); the recommendations in §6 are
judgement, not results.

---

## 0. Our setup, in the literature's vocabulary

| element | ours (ns2d) |
|---|---|
| problem | decaying 2D incompressible NS, vorticity–streamfunction on the periodic unit torus, $N=256^2$ ($n=65{,}536$), Re $=1/\nu\in[100,1000]$, $T=1$, 500 implicit-midpoint steps ($\Delta t=2\times10^{-3}$) |
| FOM | certified 2nd-order Arakawa + implicit midpoint, Newton–Krylov with an **exact FFT Helmholtz preconditioner**; **0.42 s per 500-step query at ntol $10^{-3}$ on an A100 (0.84 ms/step, $\approx13$ ns per DOF-step)** |
| parameter family | 12 i.i.d. Gaussian amplitudes on 6 low modes + $\nu$: **intrinsic dimension 13 (+t)**; 512 random training trajectories (2048 in ns302), 64 dev |
| decoder | $u = G\,(h(z)+C_q y)$: coordinate-network bank $G\in\mathbb R^{n\times R}$, $R\in\{256,512\}$, MLP head $h:\mathbb R^K\to\mathbb R^R$, $K\in\{16,32\}$ |
| latent | **auto-decoder**: one free code per snapshot, `Z = 0.1*normal`, joint Adam over bank, head and codes; no encoder, no $(t,\nu)$ conditioning, no code regulariser (`ns2d_decoder.train_autodecoder`) |
| online | weak LSPG on a **fixed** test space of the $M$ lowest Fourier modes ($M=4(K{+}q)$; 2176 in ns304), exact precomputed $M\times R\times R$ advection tensor (4.56 GB at $R=512$), Levenberg–Marquardt per step with `jacfwd`, ROM $\Delta t$ = FOM $\Delta t$ |
| incompressibility | automatic (vorticity form), enstrophy/energy-conserving Arakawa |
| outcome | bank floor 0.06–0.2 % at $t=0$ but **1.6–3.7 % median / 7–14 % worst on evolved states**; head oracle 12–20 % held-out vs POD-$K$ 14–24 % (ratio 1.15–1.46, bar 2.0); training reconstruction 1.4–5 % (held-out/train gap 3–4×); solved error 41 % median at $q=0$; **13–53 s per query vs 0.42 s FOM (30–120× slower)**; POD-LSPG better and cheaper at every rung |

---

## 1. Comparison table

Legend: *latent* = how the reduced coordinates are obtained; *online* = how the reduced dynamics are computed;
*div-free* = whether incompressibility is enforced in the ROM. "n/a" = compressible or non-fluid problem.
Errors are the papers' own metrics (usually max or mean relative $L_2$ over time).

| paper | problem | FOM (size, solver, hardware) | reduced dim | error (held-out) | speedup, vs what | latent | online | div-free |
|---|---|---|---|---|---|---|---|---|
| Lee & Carlberg 2020, JCP 404 | 1D inviscid Burgers, $n_\mu=2$, 80 train pts on a 10×8 grid; 2D chemically reacting flow, $n_\mu=2$, 64 train pts | $N=256$ / $N=8192$; backward Euler, Newton; CPU | $p=5$ / $p=3$ ($p^\star=3$) | ≈0.1 % / <0.1 % (POD-LSPG ≈10 % / >40 % at same $p$) | **none reported** (App. D estimates decoder FLOPs only) | CAE **encoder** (init code = encoder of centred zero) | manifold LSPG / Galerkin on the **full $N$-dim residual**, Gauss–Newton, no hyper-reduction | n/a |
| Kim, Choi, Widemann, Zohdi 2022, JCP 451 | 1D inviscid Burgers ($n_\mu=1$, 2–4 train pts); 2D viscous Burgers Re $10^4$, 60×60, $n_\mu=1$, 4 train pts | $n_x=1001$ / 3364 nodes per component; backward Euler, Newton, **Python on a Xeon CPU** (2D FOM 140.7 s) | $n_s=5$ | 1D ≈1 %; 2D 0.93–0.98 % (LS-LSPG-HR 34–38 %) | 1D 2.6× (avg 2.7×); **2D 11.7×** with GNAT-SNS HR (55 residual basis, 58 samples); without HR: NM-LSPG 1.8×, NM-Galerkin 1.0× | shallow **masked-AE encoder**, $\hat x_0 = h(x_0 - x_{\rm ref})$ | NM-LSPG-HR: sampled residual, Gauss–Newton per BE step | n/a |
| Fresca & Manzoni 2022, CMAME 388 (POD-DL-ROM) | NS flow past a cylinder, velocity only, $n_\mu=1$ (Re 66–133), 11 train / 10 test params, 400 sampled times | $N_h=64{,}892$ and 257,528; P2–P1 FEM, BDF2 semi-implicit, 4000 steps; CPU | rPOD $N=256$ per component, latent $n=2$ | $\epsilon_{\rm rel}=1.4\times10^{-2}$; flow-rate error 1.65 % | **$2.15\times10^5$× / $6.59\times10^5$×** (0.1 s on a V100 vs the FEM FOM) | CAE encoder + DFNN regression $(t,\mu)\mapsto$ latent | **no residual, no time stepping** (pure regression) | no (pressure discarded) |
| Romor, Stabile, Rozza 2023, JSC (NM-LSPG + ROC/GNAT) | 2D nonlinear conservation law (Burgers-type), $\nu=10^{-4}$, $n_\mu=1$, 12 train × 501 snapshots | 60×60 FV cells (7200 DOF), 1.21 ms/step; CPU | latent 4 (POD needs ≈50) | mean rel $L_2$ O(1 %) in range, worse in extrapolation | **none**: NM-LSPG 55× *slower* than FOM (133 s vs 2.4 s); hyper-reduced NM-LSPG-GNAT/ROC did not reach a speedup ("differently from the NM-LSPG-GNAT … methods"); LSTM surrogate 3× | CAE encoder | LSPG, Levenberg–Marquardt, ≤7 residual evals per step, 50–150 magic points | n/a |
| Romor, Stabile, Rozza 2025, JCP 524 (explicable hyper-reduced NM-ROM) | compressible NS Ma 2–5 (4500 / 32,160 cells); **incompressible RANS Ahmed body**, 198,633 cells, $d=993{,}165$, Re $2.8\times10^6$, params = slant angle + time, 20 train angles | OpenFOAM sonicFoam / PISO; 1 core (FOM-1 13.2 min) and 8 cores (3.6 min) | AE on 150 rSVD modes, latent 4 | few-% mean $L_2$ (Fig. 19), poor at early times; INS-2 (large deformation) fails | CNS coarse: "no evident speedup" at same $\Delta t$, ≈2× with 4× $\Delta t$; **INS-1: ≈26× vs 1-core PISO with 5× $\Delta t$** (12–66 s vs 13.2 min; ≈3–18× vs 8-core FOM) | CAE encoder on rSVD coefficients | NM-LSPG, LM, reduced over-collocation $r_h=500$–2500 nodes, adaptive submesh | FOM PISO; ROM residual on momentum + pressure, no div-free basis |
| Barnett & Farhat 2022, JCP 464 (quadratic manifold) | Ahmed body DES wake flow (compressible), reproductive in time | $N=17.3\times10^6$, 2nd-order BDF, **240 cores, 15.1 h** | HPROM $n=627$ vs HQPROM $n=39$ | QoI errors 0.10 / 0.71 / 0.54 / 2.66 % (drag, lift, $v_x$, $v_z$) | **131× wall-clock** (8 cores vs 240-core HDM; 3940× CPU-time); linear HPROM 4× | POD projection + quadratic correction fitted by least squares | LSPG + ECSW hyper-reduction, Newton | n/a |
| Barnett, Farhat, Maday 2023, JCP 492 (PROM-ANN) | 2D inviscid Burgers, $n_\mu=2$ | 125,000 DOF; 1 CPU core, HDM 718 s | HPROM-ANN $n=10$ (vs HPROM $n=95$) | 1.44 % (HPROM 1.38 %) | **51.9×** (HPROM 8.9×) | POD projection; ANN maps dominant to subdominant POD coefficients | LSPG + ECSW, Newton | n/a |
| Geelen, Wright, Willcox 2023, CMAME 403 (quadratic-manifold OpInf) | 1D transport ($n=4096$, 1 param), 2D wave ($n=321{,}201$) | – | $r=15$–40 | quadratic $r=30$ ≈ linear $r=60$ | none reported | POD projection (+ quadratic term) | learned quadratic ODE (non-intrusive) | n/a |
| Chen et al. 2023, ICLR (CROM) | 1D advection/Burgers, thermodynamics, **Karman vortex street NS** (100×200, 40k DOF, single trajectory), elastodynamics | in-house solvers; CPU/GPU | $r=3$–4 (NS $r=4$) | NS: qualitative; elastodynamics 1.46 %; thermo 0.43 % with 22/501 samples | **109× CPU / 89× GPU** (solid mechanics with 40 of 66,608 samples); **NS: no hyper-reduction, no speedup reported** | **encoder** (explicitly preferred over auto-decoder to obtain a "smoothly varying latent space") | sampled PDE time-step on $|\mathcal M|$ points then Gauss–Newton projection back to the manifold | no |
| Fries, He, Choi 2022, CMAME 399 (LaSDI) | 2D Burgers Re $10^4$, 60×60, $n_\mu=2$, 25 train pts (5×5 grid) | backward Euler, 1500 steps | AE latent 3 (POD 5) | ≈1–5 % max rel | **≈800×** (NM) | AE encoder | SINDy cubic latent ODE, local models; no residual | n/a |
| Bonneville, Choi et al. 2024 (GPLaSDI) | 1D/2D Burgers, radial advection | as above | 3–5 | ≤7 % | 200–100,000× | AE encoder | GP-interpolated latent ODE | n/a |
| Klein & Sanderse 2024, JCP (structure-preserving hyper-reduction) | **incompressible 2D shear-layer roll-up**, 256² staggered FV, 800 RK4 steps; also 2D HIT 1024² | energy-conserving FV, RK4; single i7 CPU core (FOM ≈190 s) | POD $r=30$, DEIM $m=40$ | $\epsilon_u\sim10^{-2}$–$10^{-3}$ | **636–808×** (linear POD-DEIM, reproductive) | POD projection | Galerkin + (energy-conserving) DEIM, explicit RK4 | **yes** (div-free POD, energy-conserving) |
| Benner, Goyal, Heiland, Pontes Duff 2020/22 (OpInf for incompressible flows) | lid-driven cavity Re 500 ($n_v=3042$); cylinder wake Re 60 ($n_v=5812$), reproductive | Taylor–Hood FEM, IMEX Euler, 512 steps | $r_v=30$ quadratic | $\sim10^{-2}$–$10^{-3}$ rel $L_2$ | none reported | POD of Leray-projected snapshots | learned quadratic ODE | **yes** (Leray projection) |
| Pérez De Jesús & Graham 2023, PRF 8 (Kolmogorov flow) | 2D Kolmogorov flow Re 13.5–14.4, 32² | pseudo-spectral | AE latent 5–9 | good short-time tracking and statistics | none (data-driven) | AE encoder (+ symmetry factoring) | learned discrete-time map | FOM only |
| Pan, Brunton, Kutz 2023, JMLR (NIF) | cylinder Re≈123; 3D HIT 128³ (compression only) | – | $r=1$–10 | beats POD/CAE at equal $r$ | none (no ROM) | ParameterNet($t,\mu$, sensors) → latent (**conditioned, not free codes**) | none | – |
| Puri et al. 2024/25 (SNF-ROM) | 1D/2D advection, 1D/2D Burgers (**512²**, intrinsic dim 2), KS | Fourier spectral, SSP-RK, **GPU (2080 Ti), 13.44 s** for 500 steps of 524k DOF | $N_{\rm ROM}=2$ | 0.17–1.6 % | **1.92× at same $\Delta t$ and 16,384 collocation pts; 199× at 10 $\Delta t$ and 64 pts** | smooth **hypernetwork latent $\tilde u(t;\mu)$** + Lipschitz/weight regularisation; init by encoder then Gauss–Newton | Galerkin (= LSPG for explicit) with collocation hyper-reduction, explicit Euler | n/a |
| Kim, Wang, Choi 2024 (CNF-ROM) | 1D Burgers, $\mu\in$ 8 train / 5 test | – | small | parameter inter/extrapolation OK after PINN fine-tune | none reported | $\mu$-conditioned PNODE latent; exact IC/BC so no encoder | latent ODE + PDE residual loss | n/a |
| Diaz, Choi, Heinkenschloss 2024, CMAME 425 (DD-NM-ROM) | 2D steady Burgers, 25,056 DOF | Newton; CPU | 48 DoF (shallow **wide** sparse AE, width 26,290 for 5238 inputs) | 1.28e-3 (no HR) / 1.64e-3 (HR); LS-ROM 1.98e-2 / 1.44e-2 | 21.7× / **43.9×** (LS-ROM 30× / 347.6×) | masked-AE encoder | Lagrange–Gauss–Newton SQP, collocation HR (100 nodes/subdomain) | n/a |
| Magargal, Khodabakhshi, Rodriguez 2026 (HRF Newton) | 1D Burgers $N=1024$, $n_\mu=2$ | CPU, FOM 1.1–1.6 s | POD $r$ from $\epsilon_{\rm POD}$ | = Galerkin/LSPG error | HRF-G ≈100×, HRF-LSPG ≈10× (≈1× at $\epsilon_{\rm POD}=10^{-4}$) | POD projection | precomputed polynomial tensors, reduced Newton; **cost grows as $r^2$–$r^3$** | n/a |
| Berman & Peherstorfer 2024, ICML (CoLoRA) | 2D Burgers, Vlasov, RDE | – | $q=2$–few | $\sim10^{-3}$ | CoLoRA-EQ ≈$10^2$×, CoLoRA-D ≈$10^4$× vs "traditional numerical models" | latent $\phi(t;\mu)$ smooth by construction (LoRA-style adaptation) | Neural-Galerkin residual minimisation or hypernet | n/a |
| Vlachas et al. 2022 NMI / Graph-LED 2025 (LED) | cylinder Re 696; backward-facing step Re 5000 | – | small | good statistics | 900× / 100× | AE encoder | autoregressive latent model (non-intrusive) | no |
| Hesthaven, Peherstorfer, Unger 2026, Acta Numerica (survey) | – | – | – | – | "Achieving runtime speedups in the online phase compared to the full model is **challenging** with autoencoder-based nonlinear parametrization as used in Lee and Carlberg (2020)" | – | – | – |

Not found: a Wan & Sapsis paper on this question; the "Kolmogorov barrier + data-driven manifold" line is
represented above by Linot/Graham and Pérez De Jesús/Graham. Nothing 2023–2026 reports an **intrusive** NM-ROM
speedup on 2D incompressible NS beyond Romor 2025's ≈26× (single-core FOM, 5× larger ROM step).

---

## 2. What the successful papers have that we lack — ranked

Ranked by how plausibly each explains our two failures. "Explains" is my judgement on the evidence in the table and
in DESIGN §A6–§A13.

1. **A low-dimensional, densely sampled parameter family, with the latent dimension pinned to it.** Every paper
   with a sub-1 % held-out NM-ROM has $n_\mu\le2$ (intrinsic manifold dimension 2–3 with time), 4–80 training
   instances on a *grid*, test points inside the hull, and $p=p^\star$ or $p^\star+2$ (Lee & Carlberg: "the
   intrinsic solution-manifold dimensionality is at most the number of parameters plus one"). Ours is
   13-dimensional with 512 random draws (≈1.6 points per axis); the 8-dimensional ns303 family has ≈2.2 per axis
   and still fails. The signature we measured, a 3–4× held-out/train gap that closes only when the head is driven
   into a capacity limit (ns302), with a data slope decelerating from −0.4 to −0.06 (ns301), is interpolation
   failure in parameter space, which no paper in the table ever attempted. This is a design choice of the
   family, not of the network, but it is the single largest departure from the literature. *Most plausible
   explanation of the 10–20 % oracle*, jointly with item 2; they are not separable on the runs done so far.

2. **An encoder-trained or explicitly $(t,\mu)$-conditioned, smooth latent instead of free per-snapshot codes.**
   Lee & Carlberg, Kim–Choi, Romor (×2), Diaz, LaSDI, LED: jointly trained encoders. CROM tried both and chose
   the encoder "as a tool for training the smoothly varying latent space". SNF-ROM (the auto-decoder-like
   neural field closest to ours) had to add a hypernetwork $\tilde u(t;\mu)$ **and** Lipschitz/weight
   regularisation to make a neural field usable online, and then beat the CAE. NIF, CNF-ROM, CoLoRA,
   Fresca–Manzoni all condition the latent on $(t,\mu)$. Ours: `Z = 0.1*normal(S,K)` optimised jointly with no
   tie between consecutive snapshots of one trajectory, no $\nu$ input, no encoder; the head can therefore
   memorise training codes (1.4–5 %) without learning a map that a held-out state lands on (12–20 %). Our
   AdamW/early-stop arms (§A10) regularised the *weights*, not the *latent*, so they do not test this. *Second
   most plausible cause of the oracle*; cheapest to test.

3. **A decoder whose image is not confined to a fixed $R$-dimensional linear span.** CAE transposed
   convolutions (Lee & Carlberg), masked shallow AEs with width ≫ latent (Kim–Choi; Diaz: width 26,290 for
   5238 outputs), neural fields (CROM, SNF-ROM). Our manifold is a nonlinear subset of $\operatorname{span}(G)$,
   $R=256/512$, whose Kolmogorov floor on *evolved* states is 1.6–3.7 % median and 7–14 % worst (the 0.06–0.2 %
   floor holds only at $t=0$). It caps every rung, forces $R$ large, and $R$ then sets the online cost (item 5).
   It does not explain the oracle (the oracle sits 10–30× above the floor) but it explains why the top rung
   $q=R$ lands at 2–6 % and why POD-512 (4–6 %) is where the linear controls also stall.

4. **Tests adapted to the manifold (LSPG/Galerkin with the decoder Jacobian, or GNAT-sampled residual) rather
   than a fixed low-pass Fourier test space.** All intrusive papers minimise $\|r\|$ over the full or sampled
   residual (Lee & Carlberg §3; Kim–Choi; Romor; SNF-ROM Galerkin). Ours enforces the residual only on the $M$
   lowest modes, $M=4(K{+}q)$ (128 at $K=32,q=0$; 2176 in ns304, i.e. $|k|\lesssim26$ on a grid resolving
   $|k|\le128$ at Re 1000). Layer 3 sits 4–10× above layer 2 at every rung (0.41 vs 0.106 at $q=0$; 0.024 vs
   0.0023 at $q=512$): the projection/time-stepping loses more than the head restriction does. *Explains the
   solved 40 % vs oracle 10–20 %, not the oracle.*

5. **A small online dimension, or hyper-reduction, so that the nonlinear term does not cost $O(R^2)$.** The
   quadratic/OpInf line (Geelen, Kramer–Peherstorfer–Willcox, HRF-Newton) precomputes tensors only for
   $r\lesssim40$ because cost scales as $r^2$–$r^3$; HRF-LSPG is already as slow as the FOM at
   $\epsilon_{\rm POD}=10^{-4}$. Ours contracts an $M\times R\times R$ tensor ($R=512$): one residual is
   $2MR^2\approx1.1$ GFLOP, and `jacfwd` over $K{+}q$ tangents makes each LM iteration $\approx(K{+}q{+}1)MR^2
   \approx2\times10^{10}$ flop, versus $\sim10^7$ flop for the whole FFT-preconditioned FOM step. That is the
   30–120× slowdown: **the bank width, not the head, sets the online cost, and the exact-tensor route scales
   as $R^2$ so no rung of the ladder can be cheap.** The literature's remedy is node sampling with a decoder
   that is cheap per node (Kim–Choi masked AE: $O(zbf)+O(fz^2)$; Romor ROC $r_h=500$–1500; CROM/SNF-ROM
   collocation), $O(mR)$ per residual. Our bank is a coordinate network and $\Psi=-\Delta_h^{-1}G$ is
   precomputable, so sampling is available (the Burgers lane's `engines.build_rom` already does it). *Most
   plausible explanation of the slowdown.*

6. **A ROM time step larger than the FOM's.** Romor 2025 uses 5× (INS) and 4× (CNS) the FOM step and gets its
   only speedups that way; SNF-ROM's 199× is at 10 $\Delta t$ (1.9× at 1 $\Delta t$). Our implicit ROM is not
   CFL-bound but ran at the FOM's $\Delta t=2\times10^{-3}$ for all 500 steps; the FOM's own temporal error is
   $1.6\times10^{-7}$ (F-TG-semi), so a 5–10× ROM step costs nothing visible against 10–50 % manifold errors.
   *Secondary cost factor (5–10×).*

7. **POD-anchored manifold.** Barnett–Farhat, PROM-ANN, Geelen, POD-DL-ROM, Romor 2025 all build the nonlinear
   part on top of a POD (or rSVD) projection, so the $q=R$ rung is exactly the POD-LSPG control and the linear
   part is optimal by construction. Our bank is a random-Fourier coordinate net; at $t=0$ and $K=32$ POD-32 is
   *better* than the oracle (0.82), and the paper's own linear cells found the bank 1.x× worse than POD at
   equal rank. A POD bank would make the ladder nested inside the classical control and remove the bank-floor
   penalty in item 3 (POD-256: 3.1 % median vs bank 3.7 %; POD-512: 1.1 % vs 1.6 %).

8. **Parameter conditioning at query time.** Fresca ($\mu$), LaSDI (local models in $\mu$), CNF-ROM (PNODE on
   $\mu$), NIF (ParameterNet). $\nu$ spans a decade and is known at query time; the head never sees it. Minor,
   but free. (Note for the record: the 12 amplitudes are the Fourier coefficients of $\omega_0$ and are also
   exactly recoverable at query time, so a fully $(a,\nu,t)$-conditioned latent is admissible for this family.)

9. **Incompressibility / vorticity formulation.** Not missing. Ours is the strongest in the table (exactly
   div-free, conserving Arakawa). Klein–Sanderse and Benner et al. enforce div-free POD; Fresca and CROM do
   not enforce anything. Not a differentiator for either failure.

10. **Warm start from an encoder.** Lee & Carlberg / Kim–Choi initialise $\hat x_0$ by the encoder; CROM and
    SNF-ROM by Gauss–Newton fit. Ours (nearest training code + LM fit at $t=0$) is equivalent; single-start is
    within 1.05–1.10× of the multistart oracle. Not a cause.

---

## 3. Which item explains what

- **10–20 % held-out oracle:** items 1 and 2 together. The measured signature (large held-out/train gap,
  decelerating data slope, regularisation of weights inert, gap unchanged at 8-dim, gap closing only when
  capacity binds at 4× data) fits "no neighbours in a 13-dim family for free codes to interpolate between"
  better than any capacity story. The literature never runs above $n_\mu=2$ and always trains an encoder; the
  two are confounded in every paper, which is why the pre-registration in §6 separates them.
- **Solved 40 % vs oracle 10–20 %:** item 4 (fixed low-pass test space), with item 6 as a small contributor.
- **30–120× slowdown:** item 5 (exact $M\times R\times R$ tensor with $R=512$ and `jacfwd` per LM iteration,
  500 implicit steps) against a FOM that is already among the cheapest per DOF-step in the literature (§4);
  item 6 multiplies it.

---

## 4. The honest speedup picture on 2D NS

**What is actually reported on incompressible NS or advection-dominated 2D flow, and against what:**

| class | best NS/2D-flow numbers | FOM they beat |
|---|---|---|
| non-intrusive regression / latent ODE (POD-DL-ROM, LaSDI, LED, CoLoRA-D) | $2\times10^5$–$6.6\times10^5$× (Fresca cylinder), 800× (LaSDI 2D Burgers), 900× (Graph-LED cylinder), $10^4$× (CoLoRA-D) | CPU FEM/FD with implicit steps; e.g. Fresca's FOM is $\approx2\times10^4$ s (0.1 s × $2.15\times10^5$) for 4000 BDF2 steps of 65k velocity DOF |
| intrusive NM-ROM with hyper-reduction | 11.7× (Kim–Choi 2D Burgers, 6.7k DOF Python FOM); ≈26× (Romor 2025 Ahmed RANS, 1M DOF, vs **1-core** OpenFOAM, with 5× ROM step; ≈3–18× vs 8 cores); 43.9× (Diaz steady Burgers); **none** (Romor 2023 hyper-reduced NM-LSPG; Lee & Carlberg report no timings; Hesthaven–Peherstorfer–Unger call it "challenging") | CPU, mostly single-core, implicit Newton FOMs |
| POD + hyper-reduction (linear or quadratic manifold) | 636–808× (Klein–Sanderse shear layer, $r=30$, CPU RK4 FOM); 131× wall-clock (Barnett–Farhat Ahmed DES, 17M DOF, 240-core industrial code; 3940× CPU-time); 51.9× (PROM-ANN, 1 core) | the only *industrial-strength* FOM in the set is Barnett–Farhat's (AERO-F on 240 cores) |
| same-hardware GPU spectral FOM | **SNF-ROM: 1.92× at the FOM's $\Delta t$ with 16,384 collocation points; 18× at 256 points; 199× only at 10 $\Delta t$ and 64 points** (2D Burgers 512², explicit spectral, 2080 Ti) | the closest analogue to our comparator, and the honest number at matched step is ≈2× |

**Per-DOF-step cost of the FOMs being beaten (my arithmetic from the quoted timings; provisional):**

| FOM | DOF | steps | time | per step | per DOF-step |
|---|---|---|---|---|---|
| **ours** (A100, f64, FFT-preconditioned Newton–Krylov, ntol $10^{-3}$) | 65,536 | 500 | 0.42 s | 0.84 ms | **≈13 ns** |
| SNF-ROM (2080 Ti, Julia Fourier spectral, explicit SSP-RK; "640 GiB allocated") | 524k | 500 | 13.4 s | 27 ms | ≈51 ns |
| Romor 2025 INS-1 (OpenFOAM PISO, 1 core) | 993k | 1000 | 791 s | 0.79 s | ≈0.8 µs |
| Klein–Sanderse (staggered FV, RK4, one i7 core) | ≈131k | 800 | ≈190 s | 0.24 s | ≈1.8 µs |
| Fresca cylinder (P2–P1 FEM, BDF2, CPU) | 64,892 (+p) | 4000 | ≈$2.2\times10^4$ s | ≈5 s | ≈80 µs |
| Kim–Choi 2D Burgers (Python BE-Newton, Xeon) | 6,728 | – | 141 s | – | (≫ µs) |
| Barnett–Farhat Ahmed DES (240 cores, 2nd-order BDF) | 17.3M | ≈2500 | $5.45\times10^4$ s wall | 22 s wall | ≈300 µs CPU-time |

Our FOM is 4× cheaper per DOF-step than the one GPU-spectral comparator in the literature and 60–6000× cheaper
than every CPU FOM that the headline speedups are measured against. **No paper compares a neural NM-ROM against
a tuned same-hardware implicit spectral/FFT-preconditioned solver.** The one that comes closest (SNF-ROM, GPU
spectral, explicit) gets ≈2× at matched step and hyper-reduction on 1/16 of the grid. The design's own
disclaimer ("500 implicit steps with a per-step LM will not be cheaper than an FFT-preconditioned Newton–Krylov
FOM at 256²") is consistent with the literature: at $n=65$k and a 0.84 ms FOM step, a per-step nonlinear solve
on a GPU is kernel-launch-bound at ≳0.3–1 ms regardless of $R$, so **a wall-clock speedup at 256² is not on
offer to any method in this table; the honest ceiling is parity at 256² and a few × at 1024², and only with
hyper-reduction plus a larger ROM step.** The Burgers hero's corrected 1.8× (LAB-LOG 2026-08-17) is the same
finding.

---

## 5. Notes on individual papers that matter for the diagnosis

- **Lee & Carlberg 2020.** The result that launched the field is a *reproductive-in-parameter* 1D/2D test with
  $n_\mu=2$, 64–80 grid-sampled training instances, $p=p^\star$ or $p^\star+2$, an encoder, and no timings. The
  manifold wins by 100× over POD at equal $p$ because the linear Kolmogorov width of a 3-dim manifold of shocks
  is terrible; that is the regime where the H-ORACLE bar (2×) is easy. Our family's POD-$K$ error is 14–24 %
  for the same reason but our manifold cannot exploit it because it is 13-dimensional and sparsely sampled.
- **Kim–Choi 2022.** Same regime ($n_\mu=1$, 4 training points, $n_s=5$). The paper is explicit that without
  hyper-reduction NM-ROMs "do not achieve any speed-up"; NM-Galerkin is 1.0× and NM-LSPG 1.8× on a 6.7k-DOF
  Python FOM. Their masked decoder makes the per-node cost $O(z\,b\,f)$; ours is $O(R)$ per node with a
  precomputed bank, which is the same property, unused in the NS lane.
- **Romor 2023/2025.** The only group that reports intrusive NM-ROM on (RANS) incompressible NS, with OpenFOAM
  as the FOM; they say plainly that the finite-volume FOM "is highly optimized and it is therefore more
  challenging to achieve a speedup for small test cases", get none on the coarse compressible mesh at matched
  $\Delta t$, and 26× on a 1M-DOF case versus a *single core* with a 5× ROM step.
- **SNF-ROM.** Closest architecture to ours (neural field, grid-free bank, Galerkin online, collocation HR). Two
  ingredients we lack: the latent is a smooth function $\tilde u(t;\mu)$ produced by a hypernetwork (not free
  codes), and the decoder is regularised for smoothness in $(x,\tilde u)$. Their CAE baseline (encoder, no HR)
  fails to capture the 2D Burgers shock and their POD baseline needs $N_{\rm ROM}=16$ vs 2.
- **CROM.** Explicit encoder-vs-auto-decoder discussion (their §3): auto-decoding (DeepSDF style) is rejected
  for ROM because the encoder "is a tool for training the smoothly varying latent space and for determining
  the initial latent space vector". Their NS example is qualitative, single-trajectory, without hyper-reduction
  and without a reported speedup; the 89–109× headline numbers are solid mechanics.
- **Barnett–Farhat 2022.** The only large-scale, industrial-code, wall-clock-honest speedup on a turbulent flow
  (131× on 8 vs 240 cores), obtained with a *quadratic* manifold on POD ($n=39$ vs 627) and ECSW, reproductive
  in time. Their offline dimension heuristic $n_2\sim\sqrt{n_1}$ is a useful check: for our POD-512 control the
  analogue would be $n_2\approx23$, close to our $K=16$–32, so a POD-anchored quadratic head is a natural
  control arm that no run has tried.
- **Klein–Sanderse.** Shows what a *linear* POD-DEIM ROM does on 2D incompressible shear flow at 256²: $r=30$,
  $m=40$, $\sim10^{-2}$–$10^{-3}$ error, 636–808× against an explicit FV FOM on one CPU core, i.e. the same
  order of speedup that our exploratory ladder's POD-LSPG control would show if our FOM were 2000× slower per
  DOF-step.
- **HRF-Newton 2026.** Confirms the tensor-scaling point independently: the precomputed-polynomial route is a
  win at small $r$ and "roughly as expensive as the FOM" once $r$ grows to meet $\epsilon_{\rm POD}=10^{-4}$.
- **Hesthaven–Peherstorfer–Unger 2026** (Acta Numerica survey) frames the state of the art as three elements
  (nonlinear parametrisation, reduced dynamics, online solver) and states that runtime speedups with
  autoencoder-induced parametrisations are "challenging" because the residual lives in $\mathbb R^N$ and the
  per-step nonlinear solve lifts to $N$ every iteration; hyper-reduction is the only route they name.

---

## 6. Two recommended changes

Each with the paper it comes from, the cost given our code, and a pre-registrable pass criterion. Both are new
Phase-2/Phase-3 cells under the lane rules (new pre-registration, own attempt directory, bar not lowered).

### Change A — encoder-trained, smooth, $\nu$-conditioned latent; run on the 12-parameter family *and* on a 2-parameter grid family as the discriminating control

**From:** Lee & Carlberg 2020 (jointly trained encoder, code = encoder(state)); SNF-ROM 2024/25 (latent
$\tilde u(t;\mu)$ as a smooth function, Lipschitz/weight regularisation of the field); CROM 2023 (encoder chosen
over auto-decoder for exactly our failure mode).

**What to implement (all in `experiments/ns2d/`, read from the current tree):**

1. `ns2d_decoder.train_autodecoder`: replace the free variable `Z (S,K)` by `z = E(c_pod, nu)` where
   `c_pod = Phi_256^T u` are the POD-256 coefficients already computed in `ns2d_phase2.py` (`report['pod']`) and
   `E` is a 3-layer MLP 257→512→512→K trained jointly with bank and head on the same loss (≈40 lines; the
   loss/`step` closure keeps its signature, `pz` gains the encoder params). Add one term
   $\lambda_t\|z_{i+1}-z_i\|^2$ over consecutive snapshots of a trajectory (indices are known from the cohort
   layout) — the smooth-in-time latent of SNF-ROM/CoLoRA. Query path unchanged: the encoder gives $z_0$ in place
   of the nearest-training-code search in `ns2d_rom.make_query` (the whitened LM fit stays as the polish).
2. `ns2d_fom.params_draw`: add a 2-parameter sub-family (one amplitude ratio $\theta$ and $\nu$, all other
   amplitudes fixed) sampled on an $8\times8$ grid (64 training trajectories) with dev at the 49 cell midpoints —
   the literature's regime (`params_draw` is already mode-count aware; ≈30 lines).
3. Two Phase-2 jobs (`ns2d_phase2.py`, A100, ≈4–5 h each as ns203): arm A1 = encoder head on the
   12-amplitude family (512 trajectories, $K=16$, $R=256$, ns203 recipe); arm A2 = encoder head on the
   2-parameter grid family ($K=4$ = intrinsic 3 + 1, $R=256$).

**Cost:** ≈1 day of code + smoke at 32², two A100 jobs (≈10 h wall), no new machinery beyond what
`ns2d_phase2.py` and `audit_phase2.py` already gate (B-DATA/B-ORTH/B-FLOOR/H-TRAIN/H-SOLVED/H-ORACLE).

**Pre-registered reading (stated before the jobs run):**

- Pass for the head class: A1 reaches **H-ORACLE ratio ≥ 2.0** (POD-16 median / oracle median on the 384 dev
  states) **and** held-out/train ratio ≤ 1.5. Then the auto-decoder was the cause and Phase 3 opens on A1.
- If A1 fails but A2 passes (ratio ≥ 2.0 at $K=4$ against POD-4, gap ≤ 1.5): the family dimension/sampling
  is the cause, not the head class; the paper's NS claim must be restated for a low-dimensional family and the
  12-amplitude family is reported as out of reach for any published NM-ROM design at this data budget.
- If both fail: the separable bank×head class is the cause (item 3); the next arm is a non-separable decoder,
  not more data.
- Negative control (must fail): A2 re-run with free codes (current recipe) at $K=4$ — if that also passes, the
  encoder is not what moved it and the verdict is "family dimension" alone.

### Change B — hyper-reduced, tangent-tested online solve with a larger ROM step

**From:** Kim, Choi, Widemann, Zohdi 2022 (NM-LSPG-HR: "hyper-reduction is essential to achieve a speed-up";
sampled residual with GNAT-SNS, 55–58 nodes for 6.7k DOF); Romor, Stabile, Rozza 2025 (reduced over-collocation
$r_h=500$–1500 nodes on 1M DOF, ROM step 5× the FOM's); Lee & Carlberg 2020 (LSPG with the decoder Jacobian as
test space).

**What to implement (`ns2d_rom.py`):**

1. Replace `make_weak`'s tensor contraction $(Q c_m)c_m$ by a node-sampled residual: precompute the vorticity
   bank $G$ and streamfunction bank $\Psi=-\Delta_h^{-1}G$ restricted to a node set $\mathcal S$ plus its
   Arakawa stencil (9 neighbours), evaluate $J_A(\Psi c,\,Gc)$ only on $\mathcal S$ ($O(|\mathcal S|R)$ per
   residual instead of $O(MR^2)$), keep the linear terms exact (they are diagonal in the Fourier tests, as now).
   Node selection: the Burgers lane's `engines.nnls_capped`/`build_rom` (NNLS-weighted nodes fitted on stored
   codes) or plain GNAT-style greedy on residual snapshots; certify by the held-out $\rho$ of the projected
   advection term, never by the NNLS fit (campaign rule).
2. Test space: use the decoder-tangent LSPG form $\min_w\|W\,r(u(w))\|$ with $W$ the sampled rows (GNAT) rather
   than the fixed $M$ lowest modes; `make_lm` is unchanged (it takes any `fun`). Keep the fixed-mode arm as the
   control so the two effects are separable.
3. ROM step $\Delta t_{\rm ROM}\in\{1,5,10\}\times\Delta t_{\rm FOM}$ (`nsteps`, `dt` are already arguments of
   `make_query`); the implicit-midpoint ROM is A-stable, and the FOM's temporal error at $\Delta t$ is
   $1.6\times10^{-7}$, so $10\Delta t$ costs $\lesssim2\times10^{-5}$.
4. Run on the ns204 manifold ($K=32$, $R=512$, exploratory label stays) so the cost comparison is against ns304's
   own numbers (13.4–52.6 s per query; POD-LSPG 0.21–54 s; FOM 0.42 s), same job, same GPU, three timed reps.

**Cost:** 2–3 days (sampled Arakawa on a node set with periodic stencil, node selection, $\rho$ certification
gate, a second `fun_step`), one A100 job (≈3 h as ns304). The FFT tensor build stays as the exact oracle for the
R-TQ-style gate on the sampled term.

**Pre-registered pass criterion (stated before the job runs):**

- Cost: per-query median time at $q=0$ **≤ 0.5 s at 256²** (≥ 25× below ns304's 13.4 s) at
  $\Delta t_{\rm ROM}=5\Delta t$, and ≤ 0.25 s at $10\Delta t$; certification $\rho_{\max}\le$ the campaign's
  EQ bar on held-out states.
- Accuracy: worst-evolved error at $q=0$ **≤ ns304's $q=0$ value (0.688)** and median ≤ 0.414 — i.e. the sampled,
  tangent-tested solve must not be worse than the exact-tensor, fixed-mode solve; the *hypothesis* (item 4) is
  that it lands nearer the layer-2 oracle (0.106 median), and that is reported as a finding, not a gate.
- Verdict: **PASS if at least one neural rung enters the same-job non-dominated set on (median ms,
  worst-evolved) against the POD-LSPG ladder and the FOM tolerance ladder**; on ns304's numbers that means a
  rung under ≈0.21 s with error < 0.546, or under ≈0.26 s with error < 0.256. **FAIL otherwise**, and the paper's
  NS paragraph keeps the sentence "nothing reduced is on the full-order frontier at 256²". No speedup over the
  FOM is promised at 256² (see §4); a 1024² timing arm is the only place a >1× number is plausible and is not
  part of this pass criterion.

**Why these two and not others.** Item 1 (family) is folded into A's control arm because it is the
discriminator. Item 3 (non-separable decoder) is the fallback if A fails on both arms; it is a larger change
and the literature (SNF-ROM vs CAE) says a bank×head field can work when the latent is smooth. Item 7 (POD
bank) is a cheap additional control (`bank_on_grid` → POD-256 matrix) worth adding to A1 if a third job is
available, because it makes $q=R$ identical to POD-LSPG-R and removes the bank-floor confound.

---

## 7. Sources

- Lee, Carlberg. *Model reduction of dynamical systems on nonlinear manifolds using deep convolutional autoencoders.* JCP 404 (2020) 108973. https://arxiv.org/abs/1812.08373
- Kim, Choi, Widemann, Zohdi. *A fast and accurate physics-informed neural network reduced order model with shallow masked autoencoder.* JCP 451 (2022) 110841. https://arxiv.org/abs/2009.11990
- Fresca, Manzoni. *POD-DL-ROM.* CMAME 388 (2022) 114181. https://arxiv.org/abs/2101.11845
- Romor, Stabile, Rozza. *Non-linear manifold ROM with convolutional autoencoders and reduced over-collocation method.* J. Sci. Comput. (2023). https://arxiv.org/abs/2203.00360
- Romor, Stabile, Rozza. *Explicable hyper-reduced order models on nonlinearly approximated solution manifolds of compressible and incompressible Navier–Stokes equations.* JCP (2025) 113729. https://arxiv.org/abs/2308.03396
- Barnett, Farhat. *Quadratic approximation manifold for mitigating the Kolmogorov barrier.* JCP 464 (2022) 111348. https://arxiv.org/abs/2204.02462
- Barnett, Farhat, Maday. *Neural-network-augmented projection-based model order reduction.* JCP 492 (2023) 112420. https://arxiv.org/abs/2212.08939
- Geelen, Wright, Willcox. *Operator inference for non-intrusive model reduction with quadratic manifolds.* CMAME 403 (2023) 115717. https://arxiv.org/abs/2205.02304
- Chen et al. *CROM: Continuous reduced-order modeling of PDEs using implicit neural representations.* ICLR 2023. https://arxiv.org/abs/2206.02607
- Fries, He, Choi. *LaSDI: Parametric latent space dynamics identification.* CMAME 399 (2022). https://arxiv.org/abs/2203.02076
- Bonneville, Choi, Ghosh, Belof. *GPLaSDI.* CMAME (2024). https://arxiv.org/abs/2308.05882
- Klein, Sanderse. *Structure-preserving hyper-reduction and temporal localization for reduced order models of incompressible flows.* JCP (2024). https://arxiv.org/abs/2304.09229
- Benner, Goyal, Heiland, Pontes Duff. *Operator inference and physics-informed learning of low-dimensional models for incompressible flows.* ETNA (2022). https://arxiv.org/abs/2010.06701
- Pérez De Jesús, Graham. *Data-driven low-dimensional dynamic model of Kolmogorov flow.* PRF 8 (2023) 044402. https://arxiv.org/abs/2210.16708
- Pan, Brunton, Kutz. *Neural implicit flow.* JMLR (2023). https://arxiv.org/abs/2204.03216
- Puri et al. *SNF-ROM: Projection-based nonlinear reduced order modeling with smooth neural fields.* JCP (2025). https://arxiv.org/abs/2405.14890
- Kim, Wang, Choi. *Physics-informed reduced order model with conditional neural fields.* NeurIPS ML4PS 2024. https://arxiv.org/abs/2412.05233
- Diaz, Choi, Heinkenschloss. *A fast and accurate domain decomposition nonlinear manifold reduced order model.* CMAME 425 (2024). https://arxiv.org/abs/2305.15163
- Magargal, Khodabakhshi, Rodriguez. *Hyper-reduction-free reduced-order Newton solvers.* Eng. Comput. (2026). https://arxiv.org/abs/2603.03420
- Berman, Peherstorfer. *CoLoRA.* ICML 2024. https://arxiv.org/abs/2402.14646
- Hesthaven, Peherstorfer, Unger. *Nonlinear model reduction for transport-dominated problems.* Acta Numerica 35 (2026). https://arxiv.org/abs/2602.01397
- Kramer, Peherstorfer, Willcox. *Learning nonlinear reduced models from data with operator inference.* Annu. Rev. Fluid Mech. 56 (2024).
- Vlachas et al. *Multiscale simulations of complex systems by learning their effective dynamics.* Nat. Mach. Intell. 4 (2022); Graph-LED https://arxiv.org/abs/2502.07990
- Bruna, Peherstorfer, Vanden-Eijnden. *Neural Galerkin schemes with active learning.* JCP (2024). https://arxiv.org/abs/2203.01360

---

## Glossary

- **FOM / ROM / NM-ROM / LS-ROM:** full-order model; reduced-order model; nonlinear-manifold ROM (decoder is a neural net); linear-subspace ROM (POD).
- **LSPG / Galerkin:** least-squares Petrov–Galerkin (minimise the norm of the projected discrete residual at each time step) / project the residual onto the tangent of the trial space.
- **Hyper-reduction (HR), GNAT, ECSW, DEIM, ROC, collocation:** ways of evaluating a nonlinear residual on a small set of mesh nodes so the online cost does not scale with the full dimension.
- **Auto-decoder:** decoder trained with one free latent code per training snapshot (no encoder); **encoder:** a network mapping a state to its code, trained jointly with the decoder.
- **Kolmogorov $n$-width / barrier:** the best error any $n$-dimensional linear subspace can achieve; slow decay ("barrier") for advection-dominated flows.
- **Bank / head / latent code / correction rank $q$:** see DESIGN.md glossary; the bank floor is the projection error onto the bank's span.
- **Oracle (best-found):** the best latent fit of a known field by multi-start LM; an upper bound on what the ROM can reach on that state.
- **H-ORACLE ratio:** POD-$K$ held-out median error divided by the head-oracle median; the pre-registered bar is 2.0.
- **Held-out/train ratio:** oracle on dev trajectories divided by reconstruction on training snapshots; ≈1 means capacity-limited, large means generalisation-limited.
- **Layer 1 / 2 / 3:** bank floor / manifold best-fit / solved ROM error on the same states.
- **Non-dominated set:** subjects no other subject beats on both time and error at once.
- **Per-DOF-step cost:** FOM wall time divided by (number of unknowns × number of time steps); a hardware-and-solver-quality normalisation used in §4 to compare the FOMs different papers beat.
- **Reproductive vs predictive:** ROM evaluated on a trajectory it was trained on vs on unseen parameters.
- **$\rho$ certification:** held-out relative error of the hyper-reduced advection term against the exact projected term over states the solver actually reaches (campaign rule: never certify by the NNLS fit residual).
