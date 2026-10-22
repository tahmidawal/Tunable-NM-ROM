**Verdict: NEEDS-RESTATEMENT before implementation.** Read-only audit completed; no files changed or jobs launched.

- **CORRECT — Symmetry:** Integer cosine inputs and SiLU make $g$ smooth, separately even and 2-periodic. Multiplication by $\sin(\pi x)\sin(\pi y)$ gives separate oddness and 2-periodicity, hence every even normal derivative vanishes at both walls, including corners.

- **WRONG — “The compatibility the PDE imposes”:** The incumbent solves $u_t+u(u_x+u_y)=\nu\Delta u$. Smooth homogeneous Dirichlet solutions satisfy $u_{nn}=0$, but generally **not every higher even normal derivative**. At $x=0$, writing $a=u_x(0,y,t)$ gives $\nu u_{xxxx}=2a\,a_y$, generally nonzero. Describe the architecture as imposing stronger symmetry than the PDE requires.

- **NEEDS-RESTATEMENT — Expressivity:** It cannot exactly reproduce incumbent fields with nonzero wall curvature, nor all smooth PDE solutions in a derivative-sensitive norm. Nevertheless, with $\cos(\pi x),\cos(\pi y)$ available, sine polynomials have the form envelope × polynomial in those coordinates, giving dense approximation in $L^2$. This does **not** guarantee equivalent approximation with the fixed 1024×2 network and 512-column bank; retain the in-range floor guard.

- **CORRECT — Frequency mapping:** $|\operatorname{round}(2B_0)|$ has the correct angular-frequency conversion; absolute value is harmless for cosine, and zeros are valid constants. A CPU check following the actual seed-1 key splits found frequency 1 on both axes, seven zeros per axis, and 19 distinct frequencies per axis. Thus coordinate identification survives, but 256 inputs contain substantial duplication.

- **NEEDS-RESTATEMENT — “Single intervention”:** Defensible as one **bundled bank-architecture replacement**, not an isolated compatibility experiment. It changes envelope, mixed versus axiswise features, sine/cosine channels, frequency quantization, trainability of $B$, and effective input diversity. Matching component frequency scale does not match the feature spectrum or optimization problem.

- **WRONG — Normalization rationale:** Equal maxima do not preserve RMS. On the continuous unit square, envelope RMS changes from $8/15$ to $1/2$, a ratio $15/16$; feature statistics also change. Keeping `out_scale = RMS(training data)` is a defensible frozen recipe choice, but does not preserve initial output or Gram-penalty scale.

- **NEEDS-RESTATEMENT — Identity tolerances:** The stated f64 thresholds are plausible, not established. Specify fixtures, max/RMS aggregation, nonzero scale handling, all four walls, **both** periodic shifts, and gradient-specific normalization. A gradient difference is not an analytic second derivative; its centered wall stencil benefits from exact symmetry, but roundoff is amplified by $1/\delta$. Add an autodifferentiated Hessian check and retain the stencil as an independent check.

- **NEEDS-RESTATEMENT — Must-fail control:** A2’s approximately 29× statistic on decoded fields does not establish the same statistic on arbitrary K1 bank columns. Pin the checkpoint, coefficients, wall nodes, reference-Laplacian denominator and threshold; rerun that exact fixture through the same test. Include a known nonzero-curvature polynomial control.

- **CORRECT — Endpoint thresholds:** The brief’s AND between failed improvement routes, factors 2 and 3, and worst-floor guard are copied correctly. Explicitly inherit the existing minimum populations, paired valid-tail population, UNAVAILABLE handling and guard precedence; missing evidence must not silently become STOP.

- **CORRECT — Budget arithmetic:** Ledger charges sum to 6.327 GPU-h; remaining budget is 8.673. Proposed ceilings total 5.95, leaving 2.723. The ledger’s deferred-arm-(c) entry is stale relative to Amendment 19.

- **WRONG — Fourfold runtime argument:** TR-cov’s 9,987 s is arm **(b)** Stage-A wall time, not K1’s. K1 records 9,766 s Stage A, 3,501 s Stage B and 19 s rotation. Four network evaluations do not quadruple data generation, head work, reconstruction products or all optimizer work. Label 11.1 h a crude extrapolation, not an established exclusion.

- **NEEDS-RESTATEMENT — Admission projection:** One Stage-A ratio cannot scale all Stage-A/B costs. Measure paired incumbent/new timings after burn-in for subsampled and full-grid training and bank extraction; retain unchanged generation/head costs, compilation and rotation separately. Persist repetitions and compare the complete projection against 16,440 s (4.5667 h). The 0.45 h smoke budget remains provisional until this inventory fits.