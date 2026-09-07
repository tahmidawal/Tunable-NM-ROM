# Absorbing-wave global moment diagnostic

This generated note analyzes the previously archived dynamics pilot without changing a model or rerunning a solver. It is a provisional mechanism diagnostic; no causal correction experiment is included.

For the homogeneous outgoing boundary, the actual discrete operator satisfies

$$I(t)=\langle 1,v(t)\rangle_M+c\langle 1,u(t)\rangle_B,\qquad dI/dt=0.$$

On the unit square, $h=1/N$ and $M$ is the tensor product of trapezoidal weights. The actual equations are $\dot u=v$, $\dot v=-c^2L_Nu-cD_0v$. Each outgoing edge contributes $2/h$ to the diagonal of $D_0$; corners receive both contributions, and $B=MD_0$. The positive Laplacian has one-dimensional endpoint rows $2(u_0-u_1)/h^2$ and $2(u_N-u_{N-1})/h^2$, with the usual centered interior rows. Therefore $1^TML_N=0$, and the speed factor in the boundary term is exactly $c$. The one-dimensional summation identity is checked independently, and its tensor sum gives the two-dimensional identity.

The boundary weights are the mass weights times the unit-speed damping ratio, including both corner contributions. The reference moment, bank-projected initial moment, head-fitted initial moment and subsequent autonomous drift are distinct measurements.

| Intervals | Constant test projection error | Contains constant to stated tolerance | Stiffness moment row norm | Damping moment row norm |
|---|---:|---|---:|---:|
| 256 | 0.00409844609136 | False | 1.50336051123 | 0.112979423731 |
| 512 | 0.00409384396709 | False | 1.50842933719 | 0.11323482171 |

The exact-constant diagnostic tolerance is $10^{-10}$. The retained bank obeys $G^TMG=I$. Let $\ell=G^TM1$, $d=G^TB1$, $K=G^TML_NG$, and $D=G^TBG$. The full-bank equations are $\dot a=b$, $\dot b=-c^2Ka-cDb$. Thus $\dot I=-c^2\ell^TKa+c(d^T-\ell^TD)b$. The table gives Euclidean coefficient-row norms before the recorded case speed factors are applied. Nonzero rows mean that this reduced generator does not preserve the original discrete moment for general states. This is an algebraic property, not proof that it explains every observed prediction error.

| Intervals | Case | Input moment | Reference maximum drift | Unrestricted bank-projected initial moment |
|---|---:|---:|---:|---:|
| 256 | 0 | 0 | 4.68375338514e-17 | 0.00074164886025 |
| 256 | 1 | 5.93719517994e-13 | 6.59194920871e-17 | -0.000974759572889 |
| 512 | 0 | 0 | 2.77555756156e-17 | 0.000744056739409 |
| 512 | 1 | -1.73472347598e-18 | 4.33680868994e-17 | -0.000976556237386 |

| Intervals | Case | Method | Fitted initial error | Maximum drift from its own initial moment | Final moment error | Maximum error / fixed scale |
|---|---:|---|---:|---:|---:|---:|
| 256 | 0 | rom | 0.0061401481157 | 0.0174032003748 | -0.00151372039889 | 0.00764798647499 |
| 256 | 0 | affine16 | 0.00146541110173 | 0.0229013860725 | 0.00697624047123 | 0.0133180652109 |
| 256 | 0 | affine32 | -0.000100444833144 | 0.00155661739427 | -0.00078033370653 | 0.00102952456481 |
| 256 | 0 | full64 | 0.000741648860252 | 0.000897420297635 | -3.21976133949e-05 | 0.000460782768121 |
| 256 | 1 | rom | 0.0164960793548 | 0.0182329921 | -0.00149229097651 | 0.0155220006215 |
| 256 | 1 | affine16 | 0.000605491957221 | 0.0214639198677 | 0.0116529022479 | 0.0128334719058 |
| 256 | 1 | affine32 | 0.000604067503806 | 0.00130946333915 | 0.000271868332341 | 0.000434005749343 |
| 256 | 1 | full64 | -0.000974759573482 | 0.000739274227133 | -0.00159525959798 | 0.00105458593142 |
| 512 | 0 | rom | 0.0061420490497 | 0.0173840772813 | -0.00151221529952 | 0.0076464810254 |
| 512 | 0 | affine16 | 0.00146752858683 | 0.0229040689982 | 0.00697850144422 | 0.0133177899282 |
| 512 | 0 | affine32 | -9.88198781179e-05 | 0.00155776528863 | -0.000779687689051 | 0.00102917974755 |
| 512 | 0 | full64 | 0.000744056739415 | 0.000898699520583 | -3.08990302449e-05 | 0.000462257022823 |
| 512 | 1 | rom | 0.0164977297022 | 0.0182351774973 | -0.00149248527355 | 0.0155266679504 |
| 512 | 1 | affine16 | 0.000606667376302 | 0.0214628818357 | 0.0116532850291 | 0.0128314489843 |
| 512 | 1 | affine32 | 0.000605823868663 | 0.0013106591429 | 0.000272537780268 | 0.000433638514856 |
| 512 | 1 | full64 | -0.000976556237383 | 0.000740449173717 | -0.00159802068559 | 0.00105635983851 |

Every coefficient-based initial/final moment is checked against reconstructed full-grid fields. The fixed denominator is $\sqrt{2E(0)}\sqrt{\langle1,1\rangle_M}$, not the almost-zero moment itself. Raw traces, moment operators and displacement means are preserved beside this note. The original physical error metrics and runtime claims are unchanged.

A future control may separate adding the exact constant test direction from correcting the fitted initial moment. Neither control has been run here, and the current larger-head pilot is unchanged. A corrected moment is not by itself a guarantee of local field or energy-state accuracy.

## Plain-language glossary

- **Moment / invariant:** a global weighted combination of the fields / a quantity the exact discrete equations keep constant.
- **Mass / boundary weights:** area integration weights / outgoing-boundary integration weights, including corners.
- **Bank / constant test / projection:** learned spatial span / testing against the spatial constant function / closest field in that span under the mass norm.
- **Generator row / drift / fixed scale:** algebraic derivative of the moment under reduced dynamics / change from its own starting value / a nonvanishing initial normalization.
- **Unrestricted / head fitted / reference:** any coefficient in the learned span / coefficient constrained by the decoder / retained full solver trajectory.
- **Causal control / artifact / hash:** an intervention that isolates a mechanism / stored numerical output / content fingerprint.
