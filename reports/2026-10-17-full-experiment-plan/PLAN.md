# Full experiment plan: every angle, every dimension (draft for audit, 2026-10-17)

Status: DRAFT, not pre-registered. To be audited by Codex before any lane is designed from it. Written by the coordinator after the user's advisor asked to show that cost depends on the latent dimension k, not the mesh size N.

## Claims the paper must support
1. Cost depends on the reduced size k (= R', bank columns used), not the mesh size N.
2. Accuracy depends on k, the quadrature and the time step, not on N.
3. The method matches or beats full-order models (first-order upwind+BE, central+CN, spectral) and standard ROMs (POD+DEIM/ECSW/GNAT).
4. It generalises: other PDEs, parameters, 2D/3D, seeds.
5. The mechanism is understood: every error source is measured.

## Knobs
- k = R': 2D 32, 64, 128, 256, 384, 512; 3D 64, 128, 256, 512, 768, 1024.
- M/k = 1, 2, 4, 8 (currently M = 4R').
- m: quadrature points, from failing through just-passing to converged.
- N: 2D 256^2–4096^2 (+8192^2 if feasible); 3D 64^3–384^3.
- dt: dt0/8 to 5 dt0. Schemes: BE, CN, CN-R, BDF2, GAL-CN.
- Rule families: Gauss, lattice (Fibonacci/CBC/Korobov), Sobol, Halton, Monte Carlo, Smolyak.
- Test space: LSPG with sine tests, Galerkin, low mixed-frequency sine tests.
- Solver: adaptive LM, fixed-sweep, quasi-Newton (Jacobian reuse).
- nu and IC families: in-distribution, out-of-distribution, smaller nu.
- PDE: Burgers 2D/3D, Allen–Cahn, e^u reaction–diffusion (Frank-Kamenetskii), NS 3D.
- Bank: width, seed, training resolution, smoothness. Batch size 1–256.

## Group A: cost scaling
- A1. Cost against k × N with the rule and dt fixed; same GPU, same job, paired A–B–A, warm JIT. Measure solve time, per-iteration time, full-query time. Pass if solve time varies ≤ 5 % across N at each k. Plots: time vs k (one line per N), time vs N (one line per k), k × N heat map.
- A2. Per-component scaling: nonlinear term ~ m k, Jacobian ~ m k^2 (M = 4k), LM solve ~ k^3, linear terms; log–log slopes vs theory; FLOP and bytes roofline.
- A3. Decouple k and m: vary k at fixed m, and m at fixed k.
- A4. m*(k): smallest m passing the rho bar against k and M/k; also with a low mixed-frequency test space.
- A5. Memory against k: measured vs m(2k+M) and the tensor's M k^2.
- A6. Offline cost against k (training, table precompute, rule selection) and the break-even query count.
- A7. Batch throughput against k (batch 1–256).
- A8. Input/output cost against N (IC projection, decode) and a mesh-free online path.

## Group B: accuracy scaling
- B1. Error against k × N, scored against the E0 order-verified references (S beside ST in 2D).
- B2. Error budget against k: floor, quadrature, time, test space, solver, each by its own ablation.
- B3. rho and full-solve error against m for every rule family at each k, with Halton and MC as full solves.
- B4. Error against dt × scheme × k: observed order and where time error stops dominating.
- B5. Accuracy–cost dial: error against cost parameterised by k at every mesh; Pareto front.
- B6. Error against M/k and test-space type (LSPG sine, Galerkin, low mixed-frequency).

## Group C: competition
- C1. FOM ladder: upwind+BE, central+CN, spectral; dt × tolerance sweep at every N; comparator chosen on validation; speed-up = fastest FOM at least as accurate / ROM time.
- C2. Crossover map over k × N for each opponent.
- C3. Standard ROMs: POD-LSPG/Galerkin at the same k with DEIM/ECSW/GNAT; POD + exact tensor; fitted EQ on our bank; CROM-style sampler; equal tuning on validation.
- C4. Decoder ablation: our bank vs RFF, SIREN, RBF, fixed sine bank, POD + polynomial.

## Group D: generality
- D1. nu sweep including smaller nu; in- and out-of-distribution ICs.
- D2. Allen–Cahn (cubic; tensor would need M k^3).
- D3. e^u reaction–diffusion (non-polynomial; no exact tensor).
- D4. NS 3D: cost and mesh-independence vs spectral.
- D5. Non-box geometry or non-periodic boundaries where spectral methods struggle.

## Group E: robustness
- E1. 64 random lattice shifts per rule: error bars and an error indicator with measured coverage.
- E2. 3–5 bank training seeds.
- E3. Solver sensitivity: stationarity, failing cases, fixed-sweep vs QN vs adaptive.
- E4. Conditioning against k; orthogonalisation or regularisation fixes.
- E5. Hardware: A100 and H200; timing never compared across GPU types.

## Group F: final confirmation
- F1. Freeze all choices (rules, k values, schemes, comparators, thresholds), hashed.
- F2. Open the sealed test cohorts (2D 256 cases, 3D 128) once.

## Priority and budget
1. A1, A2, A3, A5, A8, B1, B5 with existing banks: 10–20 GPU-h.
2. A4, B2, B3 (+Halton, MC), B4, B6: 30–50.
3. C1, C2 (partly in the running 3D speed lane), C3: 100–150.
4. D1, D2, D3, E1, E2, E3, E4: 150–250.
5. C4, D4, D5, A6, A7, E5: 100–150.
6. F1, F2: 30–50.
Total about 450–650 GPU-h; 3–4 busy GPUs, 2–3 weeks.

## Open question
Which "k" the advisor means: bank width R' (assumed) or the nonlinear head's latent size (the head setting failed in 2D; add a head-size sweep 4/8/16/32 if so).
