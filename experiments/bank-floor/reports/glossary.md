## Glossary

- **bank**: the $n\times R$ matrix $G$ of spatial basis functions the decoder $u = G\,h(z)$ is built on; frozen online. $n$ = interior grid points, $R$ = rank (number of columns).
- **floor**: relative $L^2$ error $\lVert u - P_G u\rVert/\lVert u\rVert$ of the best approximation of a true solution inside span($G$). No ROM on that bank can be more accurate. "worst" = maximum over the cohort's snapshots; "median" likewise.
- **inc512**: the incumbent learned bank (coordinate network, $R=512$) of the frozen checkpoint (Burgers `sep_hfit_dense_mid_N256_dense.pkl`; Poisson `primary_K32.pkl` = `new_K32`).
- **ft512**: the incumbent network fine-tuned by variable projection at the same rank. **cat1024 / cat2048**: incumbent network concatenated with a fresh random-Fourier-feature coordinate network to total rank 1024 / 2048, trained the same way (warm start asserted in-job).
- **variable projection**: training the bank directly against its own floor — the linear coefficients are eliminated in closed form, so no head and no latent codes are involved.
- **pod R**: optimal linear basis of rank $R$ (proper orthogonal decomposition) of the *same* unit-normalised training snapshots; the exact minimiser of the learned arms' training objective, computed by a deflated snapshot-Gram eigen-decomposition. **podraw**: classical un-normalised POD. **_sub**: built from only the incumbent bank's own training subset. POD banks are **grid-bound**: they exist on one mesh and cannot be evaluated on another, unlike coordinate networks.
- **random512**: an untrained 512-column network; shows the floor metric is not trivially small.
- **cond**: condition number of the raw bank columns (orthonormal bases have 1).
- **head-transplant defect**: floor of the incumbent head's own decoded fields on the new bank; if it is far below the head's error (2.5 % Burgers, 3.1 % Poisson) the existing head can be reused on the new bank by a linear re-fit without retraining.
- **q**: number of linear bank directions solved for online beside the $K$ latent coordinates; **q = R / full bank**: all of them (identity head). **M**: number of sine test functions in the weak residual, here $4R$.
- **solved error**: error of the actual online reduced solve against the same-grid full-order solution. **all-times / evolved**: worst over all six output times / over the five times after $t=0$.
- **dense vs EQ**: dense evaluates the nonlinear term on every grid point; EQ (empirical quadrature) on a few weighted points. An EQ rule must be re-fitted per bank and certified by the held-out ratio $\rho$ of quadrature error to residual norm on reachable states — never by its NNLS training fit. No EQ rule was built here.
- **dev6 / hold64 / dev12 / common256 / fresh256**: held-out validation cohorts (parameters never trained on; they drove promotion). **confirm / confirm256**: confirmation cohort reserved in DESIGN amendment A1 and opened once in Phase 2.
- **DST direct**: fast-sine-transform exact solver (labelled control). **CG**: unpreconditioned conjugate gradients, the named full-order solver. **coarse DST / coarse FOM**: full-order solve on a coarser mesh, interpolated — the matched-accuracy control.
- **total ms**: median wall time from host input array to host output field(s), including transfers. **decode FLOPs**: $nR$ per field. **full-bank operator entries**: $M\times R$ numbers pre-assembled offline.
- **P1**: the pre-registered promotion gate.
