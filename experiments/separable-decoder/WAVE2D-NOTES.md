# Wave 2D mechanism cell — notes (what ran, what was found, what was retracted)

Branch `exp/2026-09-03-wave2d-mechanism`; design `WAVE2D-DESIGN.md` (r3) and its two Codex
audits. **Every number in this file is spliced in by `wav2d_tables.py` from `runs/wav2d/*.json`;
nothing is hand-typed.** Retractions and design amendments are numbered and never deleted.

## Retractions and amendments (newest last)

1. **Retraction 1 — gate V1 threshold 1e-11 → 1e-9 (2026-09-03, seen after the first N=64 run).**
   The u-only damped Newmark recurrence vs the independently assembled block CN (scipy `splu`)
   read 1.8e-10 at N=64 reflective with CG tolerance 1e-13 on the recurrence side. The design's
   1e-11 ignored that 4000 iterative solves at a 1e-13 residual tolerance accumulate. The gate now
   also requires the same comparison at CG 1e-11 to read ≥ 10× worse (it read ~4000× worse),
   which is the evidence that the discrepancy is solver-tolerance-limited, not algebraic; the
   negative control (a → 1.01a, reflective; damping sign flipped, absorbing) fires at 1e-1 / ≫1.
2. **Amendment 1 — F2 spatial study on nested grids and a wall-compatible bump (2026-09-03).**
   The design's $N\in\{64,128,256\}$ vs 512 are not nested ($N-1$ = 63, 127, 255, 511), so the
   coincident-node comparison is impossible without interpolation; the study uses
   $N\in\{33,65,129\}$ vs 257. And the family's blobs are chopped to zero at the walls by the
   hard mask (reflective) or leave $\partial_n u_0 \ne 0$ with $v_0=0$ (absorbing), so the widest
   family blob converges at order 0.35 (ref) / 1.19 (abs) — **a property of the inherited data,
   not of the scheme.** The gate now uses a centred $w=0.1$ bump (wall values ~4e-6 of the peak)
   and reads order 2.19 / 2.22; the family blob's order is reported beside it, not gated. This is
   an amendment to a specification, not to a threshold; the thresholds (2 ± 0.3) are unchanged.
3. **Amendment 2 — F2 temporal control (2026-09-03).** "Backward Euler reads order 1 ± 0.3" was
   wrong as a control: at these step sizes BE's error is dominated by its O(1) damping and its
   fitted order reads 0.6–0.9. The control is now a separation ratio: BE error ≥ 10× CN error at
   the finest step (read 64× / 191×).

4. **Retraction 2 — F0a/F0b eigenvector-residual normalisation (2026-09-03, seen at N=128).**
   $\|L\Phi+\Phi\Lambda\|_F/\|\Phi\Lambda\|_F$ read 1.09e-13 (ref) / 1.37e-13 (abs) at N=128 against
   1e-13, after 2.3e-14 / 3.2e-14 at N=64: the stencil cancels O(1) values to produce O($\lambda$),
   amplifying roundoff by $\sim 1/(k\pi\Delta x)^2$, so the raw relative residual **scales with N** —
   exactly the absolute-threshold-on-a-mesh-scaling-quantity failure the design forbids. The gate now
   uses the backward-error form $\|L\Phi+\Phi\Lambda\|_F/(\|L\|_2\|\Phi\|_F)$ with $\|L\|_2 \le 8/\Delta x^2$
   (reads ~1e-16 at both N); the control threshold moved from 1e-4 to 1e-9 (the 1%-perturbed
   eigenvalue reads 2.5e-5 / 6.1e-6 and still fires).
5. **Retraction 3 — V1 threshold 1e-9 → 1e-8; the binding criterion is solver-limitedness
   (2026-09-03, seen at N=128).** The absorbing V1 read 1.36e-9 at N=128: CG-tolerance accumulation
   grows with the condition number, so retraction 1's 1e-9 was again an absolute bound on a
   mesh-scaling quantity. The pass rule is now: value at CG 1e-13 ≤ 1e-8 (an algebraic error would
   read ≥ 1e-4, cf. the controls at 1e-1 / ≫1) **and** the CG 1e-11 value ≥ 10× the CG 1e-13 value
   (read ~10³–10⁴×, i.e. solver-limited).
6. **Amendment 4 — F2 temporal study on the smooth bump (2026-09-03).** At N=128 the family blob's
   temporal order read 1.67 (ref) / 1.73 (abs) at SUBSTEPS {10,20,40} vs 320: its wall
   discontinuity excites the high modes whose CN phase error dominates at coarse $\Delta t$, more
   so at larger N. The gated study now uses the same centred $w=0.1$ bump as the spatial one (reads
   2.010 / 2.011 at both N); the family blob's order is reported beside it.

## Phase 1 — FOM gates (both BCs) — generated table

<!-- phase1-table -->
| N | BC | gate | value | pass rule | negative control | verdict |
|---|---|---|---|---|---|---|
| 128 | ref | F0-stencil | 1.82e-16 | ≤ 1e-13 | 2.54e-08 | PASS |
| 128 | ref | F0a | 1.41e-16 | ≤ 1e-13 | 6.11e-06 | PASS |
| 128 | ref | F0d-sym | 0 | ≤ 1e-15 | — | PASS |
| 128 | ref | F1a | 6.94e-15 | ≤ 1e-10 | 0.598 | PASS |
| 128 | ref | F1a-form | 1.54e-16 | ≤ 1e-14 | — | PASS |
| 128 | ref | F4 | 0 | ≤ 1e-10 | — | PASS |
| 128 | ref | V1 | 2.99e-11 | ≤ 1e-08 | 0.171 | PASS |
| 128 | ref | F2-spatial | order 2.1892 (errors 0.052, 0.013, 2.52e-03; N [33, 65, 129] vs 257) | see design | — | PASS |
| 128 | ref | F2-temporal | order 2.0104 (errors 8.14e-04, 2.03e-04, 5.02e-05) | see design | BE separation 575.00x | PASS |
| 128 | ref | V0 | 3.36e-15 | ≤ 1e-13 | 9.46e-06 | PASS |
| 128 | abs | F0-stencil | 2.02e-16 | ≤ 1e-13 | 2.81e-08 | PASS |
| 128 | abs | F0b-zero-mode | 3.27e-18 | ≤ 1e-13 | — | PASS |
| 128 | abs | F0b | 8.97e-17 | ≤ 1e-13 | 3.45e-06 | PASS |
| 128 | abs | F0d-sym | 0 | ≤ 1e-15 | 0.056 | PASS |
| 128 | abs | F0d-spd | min eig/max(M) = 0.271 | see design | — | PASS |
| 128 | abs | F0c | 2.18e-14 | ≤ 1e-12 | 0.058 | PASS |
| 128 | abs | F1b | 7.98e-14 | ≤ 1e-10 | 3.13e-05 | PASS |
| 128 | abs | F4 | 0 | ≤ 1e-10 | 2.24e+45 | PASS |
| 128 | abs | V1 | 1.36e-09 | ≤ 1e-08 | 1.15e+158 | PASS |
| 128 | abs | F2-spatial | order 2.2151 (errors 0.026, 6.02e-03, 1.20e-03; N [33, 65, 129] vs 257) | see design | — | PASS |
| 128 | abs | F2-temporal | order 2.0107 (errors 3.97e-04, 9.91e-05, 2.45e-05) | see design | BE separation 576.25x | PASS |
| 128 | abs | F5 | -4.51e-15 | ≤ 1e-12 | 2.828 | PASS |
| 128 | abs | F3 | slope 4.0096; fraction/prediction 1.0215, 1.0052, 1.0013, 1.0003 at N [64, 128, 256, 512] | see design | reflective retains 1.0000 | PASS |
| 64 | ref | F0-stencil | 2.43e-16 | ≤ 1e-13 | 3.33e-08 | PASS |
| 64 | ref | F0a | 1.20e-16 | ≤ 1e-13 | 2.48e-05 | PASS |
| 64 | ref | F0d-sym | 0 | ≤ 1e-15 | — | PASS |
| 64 | ref | F1a | 2.49e-15 | ≤ 1e-10 | 0.387 | PASS |
| 64 | ref | F1a-form | 1.19e-16 | ≤ 1e-14 | — | PASS |
| 64 | ref | F4 | 4.44e-16 | ≤ 1e-10 | — | PASS |
| 64 | ref | V1 | 1.81e-10 | ≤ 1e-08 | 0.101 | PASS |
| 64 | ref | F2-spatial | order 2.1892 (errors 0.052, 0.013, 2.52e-03; N [33, 65, 129] vs 257) | see design | — | PASS |
| 64 | ref | F2-temporal | order 2.0104 (errors 8.09e-04, 2.02e-04, 4.98e-05) | see design | BE separation 576.87x | PASS |
| 64 | ref | V0 | 2.27e-15 | ≤ 1e-13 | 8.09e-06 | PASS |
| 64 | abs | F0-stencil | 2.25e-16 | ≤ 1e-13 | 3.08e-08 | PASS |
| 64 | abs | F0b-zero-mode | 3.24e-18 | ≤ 1e-13 | — | PASS |
| 64 | abs | F0b | 8.42e-17 | ≤ 1e-13 | 1.40e-05 | PASS |
| 64 | abs | F0d-sym | 0 | ≤ 1e-15 | 0.079 | PASS |
| 64 | abs | F0d-spd | min eig/max(M) = 0.260 | see design | — | PASS |
| 64 | abs | F0c | 8.41e-15 | ≤ 1e-12 | 0.058 | PASS |
| 64 | abs | F1b | 7.00e-14 | ≤ 1e-10 | 1.51e-05 | PASS |
| 64 | abs | F4 | 0 | ≤ 1e-10 | 1.48e+20 | PASS |
| 64 | abs | V1 | 2.16e-11 | ≤ 1e-08 | 5.23e+139 | PASS |
| 64 | abs | F2-spatial | order 2.2151 (errors 0.026, 6.02e-03, 1.20e-03; N [33, 65, 129] vs 257) | see design | — | PASS |
| 64 | abs | F2-temporal | order 2.0107 (errors 3.99e-04, 9.95e-05, 2.46e-05) | see design | BE separation 576.34x | PASS |
| 64 | abs | F5 | -4.51e-15 | ≤ 1e-12 | 2.828 | PASS |
| 64 | abs | F3 | slope 4.0096; fraction/prediction 1.0215, 1.0052, 1.0013, 1.0003 at N [64, 128, 256, 512] | see design | reflective retains 1.0000 | PASS |

| N | BC | reported quantity | value |
|---|---|---|---|
| 128 | abs | family-blob energy ratio E(T)/E0, E(4T)/E0 (F1/F4 trajectory) | 7.29e-04, 1.58e-05 |
| 128 | abs | family widest blob (w=0.190) spatial order, NOT gated | 1.1920 (errors 8.77e-03, 3.85e-03, 1.68e-03) |
| 128 | ref | family widest blob (w=0.190) spatial order, NOT gated | 0.3479 (errors 0.044, 0.036, 0.027) |
| 128 | ref | V0 energy agreement with the frozen 08-14 FOM | 1.56e-15 |
| 128 | both | provenance | commit 39130cc0, backend gpu, jax 0.10.1, matmul highest, wall 59.4 s |
| 64 | abs | family-blob energy ratio E(T)/E0, E(4T)/E0 (F1/F4 trajectory) | 1.00e-03, 3.12e-05 |
| 64 | abs | family widest blob (w=0.190) spatial order, NOT gated | 1.1920 (errors 8.77e-03, 3.85e-03, 1.68e-03) |
| 64 | ref | family widest blob (w=0.190) spatial order, NOT gated | 0.3479 (errors 0.044, 0.036, 0.027) |
| 64 | ref | V0 energy agreement with the frozen 08-14 FOM | 4.79e-15 |
| 64 | both | provenance | commit 39130cc0, backend gpu, jax 0.10.1, matmul highest, wall 45.7 s |
<!-- /phase1-table -->

**Reading.** F3 is the strongest single result of phase 1: the reflected-energy fraction of an
isolated right-going pulse through the ghost-closed Engquist–Majda face matches the discrete
prediction $|R|^2 = \tan^4(\theta/4) \approx \tfrac{15}{1024}(\Delta x/w)^4$ derived in the r1
audit, within a few percent at every resolution, slope 4.01 — the absorber is exactly what the
design says it is. V0 shows the new reflective FOM reproduces the frozen 08-14 rollout to
roundoff, so the reflective data are bit-comparable with the 08-16 negative.
