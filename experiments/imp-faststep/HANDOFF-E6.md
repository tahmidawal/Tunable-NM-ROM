# Hand-off to E6 (lane imp-hrcost): the precontracted Galerkin stepper of imp-faststep

**Status (2026-10-21, before production).** The precontraction and the three steppers in `fscore.py` passed the Codex
code audit as written (`audits/codex-code-3-main.md`, item 1: "precontraction and three stepper equations — CORRECT");
the surrounding driver was then hardened (table/map gates, A5.1). Production numbers do not exist yet; this file gives
E6 the module so it can build on it now. It will be updated with the frozen production configuration hashes and the
measured per-step costs once `f1024` is collected.

- Branch `exp/2026-10-21-imp-faststep`, mirrored at `origin/codeonly/exp/2026-10-21-imp-faststep`.
- Module: `experiments/imp-faststep/fscore.py`, sha256 `3626350c4165d4db686db68f1be59730de31d8c4e6c06ee2876a8c19b86cbec5`
  (commit `df34a847e` onwards; check `sha256sum` before use). It imports `experiments/jcp-p1-scaling/p1core.py` (and through
  it the vendored `quad2d` code and the `dn256b` checkpoint); nothing else.
- Design: `experiments/imp-faststep/DESIGN.md` §3 (equations) and amendments A1.1–A1.3, A5.1 (gates).

## What it computes

The t2core GAL semi-discretisation of the deployed 2D bank ROM, $Q^\top[A\dot c + N(c) + \nu\Lambda Ac] = 0$ with
$A = QR_m$, $N(c) = \Psi^\top[(G_qc)\odot(G_sc)]$, rewritten offline in **diffusion coordinates**
$$Q^\top\Lambda Q = VDV^\top,\quad z = V^\top R_m c,\quad \dot z = -P_z^\top[(B_zz)\odot(D_zz)] - \nu Dz,$$
$$B_z = G_qR_m^{-1}V,\quad D_z = G_sR_m^{-1}V,\quad P_z = \Psi QV\quad (m\times r).$$
$\|z\| = \|Ac\|$, so the t2core GAL tolerances transfer unchanged. One advection evaluation is three $(m, r)$ products;
no $M$-sized operation is online.

## Functions (signatures)

```python
import fscore as F
tab, checks = F.precontract(dg, mutation=None, basis=None)
#   dg: the t2core GAL data dict of one (mesh, r, rule): p1common.Setting(..., forms=('LSPG','GAL')).data[('main','GAL')]
#       (keys A, lam, Qm, Rm, QtLA, Gq, Gs, Psi, G, hmask); basis=(D, V) reuses the main rule's eigenpairs for a successor rule
#   tab: dict(Bz, Dz, Pz, D, V, RinvV, VtRm); checks: D_min, D_max, qr_orth, eig_resid, v_orth, A_rank, A_cond
#   gates used in production (A5.1): eig_resid <= 1e-10, v_orth <= 1e-10, qr_orth <= 1e-10, D_min > 0
ev = jax.jit(F.make_evolve(kind, F.adv_pre, F.jac_pre))      # kind in {'cn', 'etd', 'ros'}
W, stats = ev(c0, nu, tab, F.sched(dt, level='prod'))
#   c0: initial coefficients (p1core split-query init), nu: viscosity (traced), dt traced
#   W (6, r): output coefficients c at t = 0, 0.05, ..., 0.25 (decode with the p1core split-query decode)
#   stats: Z (801, r) states (row k = z after k steps, NaN beyond), it / tolratio / reason per step (cn),
#          stage_max (etd), growth = max_k ||z_k|| / ||z_0||, steps, z_out (6, r)
F.TOL_CN   # {'prod': 1e-10, 'tight': 1e-12, 'tighter': 1e-13}: CN Newton tolerance on ||rho|| / ||z_n|| (A1.3)
cert = F.make_cert_cn(nl)     # fresh GAL-CN residual of every saved step through the ORIGINAL tables
cert = F.make_cert_etd(nl)    # per-step ETDRK4 reconstruction (independent phi coefficients F.etd_coefs_indep)
cert = F.make_cert_ros(nl)    # fresh autodiff Jacobian + saved-increment residual (one-pass linearised CN)
```

- `kind='cn'`: Crank–Nicolson, Newton with the exact Jacobian $I + \tfrac{\Delta t}{2}(J_N + \nu D)$ and LU,
  predictor $2z_n - z_{n-1}$, budget 30 (arm 2, "G-CN").
- `kind='etd'`: ETDRK4 (Cox–Matthews, Kassam–Trefethen 32-point contour coefficients on $-\nu D$; arms 3a/3b at
  $\Delta t$ = 0.0125 / 0.025).
- `kind='ros'`: one-pass linearised CN $(I + \tfrac{\Delta t}{2}J(z_n))\delta = -\Delta t\,g(z_n)$ (arm 4).

For an E6 hyper-reduction variant with **fitted weights on the same tests**, replace `adv_pre` / `jac_pre` by a provider
`adv(z, tab)` (and its Jacobian, or `None` for autodiff); `make_evolve` is generic in the provider. The certificate
pattern is `make_cert_etd_adv(twin)` with an un-precontracted twin of the provider.

## Configuration of the incumbent tables (frozen policy, as p1-scaling)

| r | rule (m) | successor | M | mode list |
|---|---|---|---|---|
| 128 | fib4181 (4181) | fib6765 | 512 | `modes_lean(1024, 2048)[:M]`, eigenvalues at the mesh |
| 384 | gauss128 (16384) | gauss160 | 1536 | same |

Bank: checkpoint `dn256b` (K = 16, R = 512, rotation R512), edges `[0, 128, 256, 384]`. Configs:
`configs/smk1.json` (and the production `f1024.json` / `f4096.json` when committed).

## How to time it (as this lane does, `fstiming.py`)

Paired A–B–A in one job on one GPU, warm JIT, `qstudy.burn(0.05)` before each timed call, f64 with
`JAX_DEFAULT_MATMUL_PRECISION=highest`; the evolve call includes the coefficient construction and the coordinate maps.
A timed evolve must be bitwise equal to the scored run (same compiled program and inputs). Drift band [1/1.1, 1.1],
≤ 2 retries; six dev6 cases × 3 repetitions; per-case median, then the median over cases.

## Production update (after f1024 / f4096)

Production ran with `fscore.py` sha256 3626350c4165d4db686db68f1be59730de31d8c4e6c06ee2876a8c19b86cbec5 and configs `configs/f1024.json` (sha256 cf42a84a37dba45a…) and `configs/f4096.json` (sha256 4096af7033c1855b…). Measured per-query init / evolve / decode times and per-step evolution cost of every arm are in `EXPORT.json` (`jobs.<job>.arms.<arm|r>.seconds`) and `report.md`. All Galerkin arms are Q1+Q2 qualified; the E5 stop rule fires on worst error for every arm (see report), so E6 should treat the precontracted Galerkin form as a numerically sound, faster stepper whose worst-case accuracy is slightly below the incumbent's on the hardest case.
