Verdict: the MAC FOM is correct and phase 2 may proceed, but only after two gate additions: a generic manufactured solution and a repaired deterministic S3 control. The closed-form claim is true; the S3 replacement claim is not.

## Central claim

### (a) Closed-form identities — CONFIRMED

Let \(U_{i,j}\) live at \((ih,(j+\tfrac12)h)\), \(V_{i,j}\) at \(((i+\tfrac12)h,jh)\), and \(t=\pi h\).

For the cell \((i,j)\),

\[
\frac{U_{i+1,j}-U_{i,j}}h
 =\frac{\pi\sin t}{h}
 \sin(2\pi x_{i+1/2})\sin(2\pi y_{j+1/2}),
\]

while

\[
\frac{V_{i,j+1}-V_{i,j}}h
 =-\frac{\pi\sin t}{h}
 \sin(2\pi x_{i+1/2})\sin(2\pi y_{j+1/2}).
\]

They cancel exactly. This includes boundary cells because the eliminated normal faces equal the analytic endpoint values \(0\). Thus \(D u_{\rm ex}=0\) algebraically for this particular field.

For \(U\),

\[
\delta_{xx}\sin^2(\pi x_i)
 =\frac{2\sin^2t}{h^2}\cos(2\pi x_i)
 =\frac{\sin^2t}{t^2}\,\partial_{xx}\sin^2(\pi x_i),
\]

including \(i=1,N-1\), since the omitted neighbors are the exact zero endpoint values.

In the tangential direction,

\[
\delta_{yy}^{\rm odd}\sin(2\pi y_{j+1/2})
 =-\frac{4\sin^2t}{h^2}\sin(2\pi y_{j+1/2}).
\]

At the lower wall the analytic continuation is

\[
\sin(-t)=-\sin(t),
\]

exactly the odd ghost; the upper wall is identical. Hence the wall-adjacent \(-5/h^2\) rows obey the same identity. Swapping \(x,y\) proves it for \(V\). Therefore

\[
L_hu_{\rm ex}=\frac{\sin^2t}{t^2}\Delta u
\]

on every active face, not just in the interior.

Finally,

\[
\frac{\sin(2\pi(x_i+h/2))-\sin(2\pi(x_i-h/2))}{h}
 =\frac{\sin t}{t}\,2\pi\cos(2\pi x_i),
\]

with the analogous cosine difference in \(y\). Thus

\[
\operatorname{Grad}_h p_{\rm ex}=\frac{\sin t}{t}\nabla p.
\]

The cell-centred sine and cosine sums are exactly zero for these frequencies, so the pressure already satisfies the mean-zero gauge.

Consequently,

\[
u_h=\left(\frac{t}{\sin t}\right)^2u_{\rm ex},\qquad
p_h=\frac{t}{\sin t}p_{\rm ex}
\]

is the unique discrete solution for every \(\nu\).

“Exactly” here means exact real-arithmetic identities. Direct floating-point stencil evaluation gave:

| \(N\) | \(\max|Du_{\rm ex}|\) | relative \(L\)-identity defect | relative Grad identity defect |
|---:|---:|---:|---:|
| 32 | \(5.68\times10^{-14}\) | \(1.18\times10^{-14}\) | \(1.46\times10^{-15}\) |
| 256 | \(6.25\times10^{-13}\) | \(6.69\times10^{-13}\) | \(8.92\times10^{-15}\) |

Those are trigonometric-evaluation and cancellation roundoff, not approximation error. The wall rows were no worse than the interior rows.

### (b) Prior anchors — CONFIRMED

The anchors from r1/r2 are analytic constants:

\[
\epsilon_u=(t/\sin t)^2-1,\qquad
\epsilon_p=t/\sin t-1.
\]

An independent Kronecker-product assembly and sparse solve reproduced, digit for digit:

- \(N=32:\ \epsilon_u=3.218964440079\times10^{-3}\), \(\epsilon_p=1.608189083975\times10^{-3}\).
- \(N=64:\ \epsilon_u=8.035776793722\times10^{-4}\), \(\epsilon_p=4.017081549652\times10^{-4}\).
- Orders \(2.002087\) and \(2.001217\).

So the earlier calculation was numerically independent code, but it was unknowingly evaluating predetermined analytic constants. The archived `mac_check.py` contains the operator checks, not the manufactured-solve script itself.

### (c) Weakening of S-FOM — OVERSTATED in wording, correct in substance

The pure-amplitude conclusion is correct. Therefore the convergence table adds almost no information beyond the three identities: global, boundary-only, and other relative norms must all report the same scalar.

But “a 2-D invariant subspace” is not literally correct. For example, the sampled \(\sin^2(\pi x)\) factor has 8 nonzero Dirichlet sine components at \(N=16\) and 16 at \(N=32\). The field is not an eigenvector of \(L\), nor does it occupy a two-dimensional \(L\)-invariant Krylov space. What is special is that the sampled velocity and pressure happen to receive uniform scalar consistency factors.

S-FOM still strongly detects odd-versus-even ghosts and several sign/scaling errors. It is not sufficient as the only manufactured convergence test.

A second generic MMS is warranted before expensive phase-2 data generation. I tested:

\[
\psi_g=\sin^2(\pi x)\sin^2(2\pi y)
 +0.3\sin^2(3\pi x)\sin^2(\pi y),
\qquad
u_g=(\partial_y\psi_g,-\partial_x\psi_g),
\]

\[
p_g=\sin(4\pi x)+0.37\cos(6\pi y)
 +0.21\sin(2\pi x)\cos(4\pi y),
\qquad
f_g=-\nu\Delta u_g+\nabla p_g.
\]

CPU results were:

| \(N\) | velocity error | pressure error |
|---:|---:|---:|
| 32 | \(1.541713\times10^{-2}\) | \(1.833939\times10^{-1}\) |
| 64 | \(3.820960\times10^{-3}\) | \(4.577326\times10^{-2}\) |
| 128 | \(9.531800\times10^{-4}\) | \(1.144216\times10^{-2}\) |

Orders were \(2.0125,2.0031\) for velocity and \(2.0024,2.0001\) for pressure. The error/solution cosine was about \(0.912\), not \(1\), so this genuinely tests spatial error structure.

## 1. Code — CONFIRMED, with gate-harness gaps

The actual numerical implementation is correct:

- `D` has the right boundary-normal elimination and signs.
- `Grad=-D^\top` under the uniform \(h^2I\) masses, despite separate assembly.
- `L` correctly uses odd tangential ghosts and total wall diagonal \(-5/h^2\); the even control has \(-3/h^2\).
- `C` is the compatible vertex incidence curl, with \(DC=0\).
- The bordered system and unweighted \(\mathbf1^\top p=0\) gauge are correct because the pressure mass is constant.
- The multiplier argument is correct algebraically: \(\mathbf1^\top D=0\) forces \(\lambda=0\).
- The analytic forcing signs and component Laplacians are correct.

The independent Kronecker implementation reproduced every odd- and even-ghost result through \(N=128\). The generated block in `STOKES-NOTES.md` also matches `stk2d_tables.py` output exactly. The three Python files are unchanged from the commit recorded in the JSON.

There are nevertheless gate-enforcement defects in [stk2d_fom_gates.py](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-30-stokes-vector/experiments/separable-decoder/stk2d_fom_gates.py:360):

- S0 records JAX provenance but asserts none of GPU/x64/precision.
- S-FOM records anchors but only asserts order.
- S-EXACT’s rule mentions prediction agreement, but only field agreement is asserted. Its recorded `worst_pred_dev_p=2.331e-8` exceeds the stored \(10^{-8}\) tolerance.
- Rank results, the S-ADJ negative control, S-PRESS, and S-FREESLIP are recorded but not asserted.
- The JSON can therefore say `complete=true` even though its own frozen S3 control floor fails.

These do not invalidate the recorded FOM, but they should be fixed before calling the phase-2 suite executable gates rather than diagnostics.

## 2. Free-slip control — CONFIRMED

An \(O(1)\) plateau is exactly the expected behavior. Even ghosts impose a different continuum boundary condition, \(\partial_nu_t=0\), while this manufactured velocity has nonzero tangential normal derivative at the walls.

Richardson extrapolation from \(N=64,128\) gives limiting errors approximately

\[
E_{u,\infty}=1.2729591,\qquad E_{p,\infty}=2.5543888.
\]

The deviations from those limits decrease by approximately \(4\times\), meaning the free-slip discretization is itself converging second order—to the wrong boundary-value problem. The wall-relative error grows as \(h^{-1}\), because the exact wall-adjacent velocity is \(O(h)\) while the free-slip limit is \(O(1)\).

There is no evidence of an additional broken free-slip arm.

## 3. Threshold retractions — CONFIRMED

S-NU is roundoff, not a physics or discretization defect. At \(N=32\):

| SuperLU ordering | pressure invariance error |
|---|---:|
| default COLAMD | \(1.0774\times10^{-11}\) |
| MMD\_ATA | \(9.06\times10^{-14}\) |
| NATURAL | \(5.12\times10^{-14}\) |

A real defect would not disappear by changing only the factorization permutation. Small-mesh dense condition estimates grew approximately as \(N^{2.35}\); the \(\nu=7\) saddle condition was about \(48\times\), not merely \(7\times\), worse than \(\nu=1\). Thus “\(h^{-2}\) plus factor seven” is a rough explanation, but it understates the block-conditioning effect rather than concealing a defect. The \(10^{-9}\) tolerance is conservative but defensible.

The \(1.579\times10^{-10}\) N=256 S-EXACT pressure discrepancy is likewise consistent with roundoff: its backward residual is \(2.868\times10^{-12}\), pressure forward error grows with refinement, and changing factorization ordering at \(N=32\) reduced the exact-pressure error from \(1.379\times10^{-12}\) to \(8.28\times10^{-14}\). No real defect is indicated.

## 4. S3 disagreement — WRONG

The observed numbers are real, but the diagnosis and proposed gate are not.

For mass-normalized columns, the Frobenius metric is

\[
\frac{\|\Psi^\top M_u g\|}
{\|\Psi\|_F\|M_ug\|}
=
\sqrt{\frac1M\sum_{j=1}^M
\cos_M(\psi_j,g)^2}.
\]

It is the RMS of per-column physical cosines. There is a \(1/\sqrt M\) aggregation, but no unavoidable residual \(h\)-factor.

The reported decay occurs because `p` is grid-white random noise while the control contains only 64 smooth low-frequency modes. As \(N\) increases, almost all gradient energy moves outside that fixed control space. Indeed, the proposed “resolution-independent” maximum cosine itself decays:

\[
3.753\times10^{-2},\
6.687\times10^{-3},\
1.881\times10^{-3},\
5.195\times10^{-4}
\]

at \(N=32,64,128,256\). It is plainly not resolution-independent for this pressure choice, and a maximum also depends on how many columns are included.

A direct counterexample disproves “the \(10^{-2}\) floor is unachievable.” Taking \(p=\chi_{11}\), one of the matched control pressures, produced for \(M=64\):

- control Frobenius metric \(=0.125=1/\sqrt{64}\) at every \(N=32,64,128,256\);
- matched-control cosine \(=1\) to \(3\times10^{-15}\);
- solenoidal cosine between \(4.2\times10^{-16}\) and \(2.0\times10^{-15}\).

So the original floor can pass cleanly once the negative-control pressure is chosen coherently. The design is under-specified, not intrinsically impossible.

Recommended S3 repair:

- Use a deterministic \(p=\chi_{k\ell}\) aligned with each selected control column.
- Require matched-control cosine \(\ge0.99\).
- Require solenoidal projection/cosine \(\le10^{-13}\).
- Do not gate on a ratio whose denominator is roundoff.

## Phase-2 verdict

**PROCEED WITH SPECIFIC ADDITIONS.**

The FOM operators and solver may remain unchanged. Before phase-2 training or bulk snapshot generation:

1. Add the generic MMS above, at least on \(N=32,64,128\).
2. Replace the random-pressure S3 control with deterministic aligned pressures.
3. Turn the currently descriptive rank, negative-control, S0, and prediction checks into actual assertions.

No repository files were modified; all verification was CPU-only NumPy/SciPy.