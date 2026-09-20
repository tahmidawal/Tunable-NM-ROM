# Burgers 3D: current corrected NM-ROM overnight development pilot

Prospective protocol committed before submission on 2026-09-20. This is a bounded development experiment; it does not promote the historical failed head or provide final-cohort evidence.

## Scope and provenance

The user approved `exp/2026-09-20-paper-b3d` from `25434a27bfc8857d81859784250b0bfd9a78adfd`, with namespace `/cluster/tufts/paralab/tawal01/paper_b3d_20260920`. The checkout is sparse to avoid copying inherited archives. `VENDOR.json` pins every imported primitive and the retained trained checkpoint; the block-damped Levenberg–Marquardt function is copied exactly from the current correction implementation. Three-dimensional FOM primitives and the retained checkpoint originate at repair commit `303bb6f6e582f876d9e4034da141f394b17f1dda`.

The equation is scalar $u_t+u(u_x+u_y+u_z)=\nu\Delta u$ on the unit cube with zero Dirichlet walls. Keep the original positive multi-blob family, viscosity distribution, backward Euler, sign-dependent upwinding, and mesh convention: $N$ counts nodes including the walls. The first pilot uses $N=33$, $\Delta t=0.005$, and $50$ steps. Family descriptors do not enter the neural head.

## First experiment

The retained network has $R=128$ bank functions and $K=32$ latent coordinates. Its historical representation failures remain failures. In exact QR coordinates, $G=QR_B$, the new corrected model is

$$u=Q\bigl(R_B h_\theta(z)+C_q y\bigr).$$

The columns of $C_q$ are nested orthonormal residual directions fitted on training-only snapshots. The initial ladder is $q\in\{0,32,64\}$. The unrestricted $R$-coefficient bank is a separate endpoint, with no redundant latent coordinates. POD controls have ranks $32,64,96,128$. All weak residuals use the same lowest $384$ sine modes, extended only to complete a degenerate shell; actual $M$ is recorded. This exceeds every online dimension comfortably. Dense quadrature is the initial correctness path. There is no tensor shortcut or empirical quadrature claim in this job.

Regenerate the seed-zero family on the cluster. Use the first $64$ original training trajectories for the new direction/POD diagnostic, at steps $0,1,2,5,10,20,35,50$. The retained network had more training data: this explicitly unmatched-data pilot cannot establish neural superiority over POD. Evaluate original validation rows $512$ through $519$ at all steps; no sealed final cases are opened. Retain a deterministic subset of training snapshots for multistart residual-direction fitting. Retain the full selected training fields, coefficients, fitted codes and direction matrix for later diagnosis. Later fair comparisons must match all learned-component training membership.

Initial fitting uses the supplied dense field, with the nearest stored training code as its online start and extra deterministic starts for the separate best-found representation diagnostic. Cost includes this dense projection and complete dense trajectory output. Hyper-reducing the cold start is subsequent work; this dense pilot makes no grid-independent cost claim. The full previous decoded augmented state enters the next weak step. No reference solution enters an online solve.

The online weak residual is

$$r(w)=\frac{A c(w)-A c_{\rm prev}+\Delta t\left[\Phi^\top\mathcal N_{\rm upwind}(Qc(w))+\nu\lambda\odot A c(w)\right]}{1+\Delta t\nu\lambda},\qquad A=\Phi^\top Q.$$

Use the current block-LM algorithm with separate damping on latent and correction coordinates, normalized-gradient tolerance $10^{-6}$, initial-fit budget $400$, and evolution budget $200$ initially. Record every iteration count, stopping reason, weak residual and normalized gradient. A residual exit is distinct from stationarity. A bounded extension is allowed only as an explicitly labelled development amendment if iteration budgets bind.

## Verification and controls

Before numerical comparison: independent NumPy sign-upwind and Laplacian checks, directional-derivative check away from the switching surface, independent bank/head evaluation, QR identity/rank, sine/Laplacian identity, zero-boundary check, and an axis-omission negative control. Independent saved-field checks recompute backward-Euler reference defects and physical errors. Nonfinite data or invalid reference residuals stop acceptance.

The same-grid reference is tolerance-terminated Newton–BiCGStab with the 3D discrete-sine Helmholtz preconditioner, nonlinear tolerance $10^{-10}$ and linear tolerance $10^{-11}$. Same-job performance controls use nonlinear tolerances $10^{-2},10^{-4},10^{-6}$ with recorded linear tolerances. These are comparison points, not a claim to have exhausted classical tuning. The original fixed-eight-Newton data generator is not used as the performance baseline. A two-case finer-space/time diagnostic is planned after the primary same-grid pilot; no physical-reference headline is allowed before that diagnostic.

No arbitrary accuracy win is required to retain a row. Promotion to a final panel requires verified operators, finite trajectories, declared stopping behavior, and adequate reference accuracy. A failed model is shown as failed; incomplete or budget-limited training is identified beside its result.

## Measurement and resource contract

Every actual run asserts GPU backend, float64 and highest matmul precision. One single-GPU job at a time in this lane, initially at most two hours, in a unique directory. Burn in before timed blocks, warm every program, then retain three repetitions per case and method. Cost and accuracy come from the same returned invocation; method order is deterministically shuffled. Save whole trajectories and all solver arrays. Medians, tails and failure counts are generated from records. No timing ratios cross allocations.

All scripts/configuration/checkpoint hashes, source commit, seed, grid, GPU, Slurm job and precision are saved. Results are checksum-collected before removal of the exact remote attempt. The coordinator owns the canonical lab-log append. No merge is authorized.

## Follow-up order

Diagnose bank floor, initial compression, best-found fit and solved rollout separately. If bank capacity is limiting, train a larger learned bank; if the head is limiting, test early-state coverage and per-state relative loss before another architecture search. New FNO/U-Net ports and fair shared-training panels follow the first verified pilot. Remaining operator families, repeated seeds, quadrature, a second resolution and sealed final evaluation remain explicit backlog until actually completed.

## Exploratory amendment after the first development pilot

The first pilot's bank floor, remaining head error and dense query cost motivate a fresh learned bank with larger rank and a better-conditioned relative-loss training objective. `config-train.json` is the frozen second-attempt configuration. It uses the full original training membership for every new learned component and the original eight training output times, with extra early-time coverage already included. The coordinate-bank last hidden layer is wider than its output rank. Spatial Fourier features, nonlinear hidden layers and boundary masking remain the B3D learned architecture. The two-stage optimizer recipe is adapted from the independently implemented heat trainer at `e6460d73`; no heat data or PDE is imported.

Head training uses relative reconstruction error plus a relative-error tail penalty. PCA initializes latent training codes and the head's linear skip only; it does not replace the learned spatial bank. Every checkpoint and learning curve is retained; a finite training budget is not labelled convergence. The pilot keeps one head latent dimension and evaluates nested correction ranks at a fixed weak-space size, with matched-dimension POD controls and the unrestricted bank endpoint. New loose-inner-tolerance FOM controls address the first pilot's incomplete classical tuning. Two development cases receive time, space, and combined reference refinements before any physical-accuracy interpretation. Model choice remains exploratory on the same opened validation cohort. Sealed final cases remain untouched.

Repeated identical full-field outputs may share an artifact only after every returned array is compared bit-for-bit; each invocation still records its own timing and achieved accuracy. This avoids retaining multiple identical copies while keeping cost and accuracy coupled.

The next independent comparison job must train FNO3D and U-Net3D on these exact original training rows, the same observed training times and physical input fields plus viscosity. It must evaluate all requested output times and include the current learned ROM, POD and tuned FOM in one allocation. DeepONet, Transolver, independent training seeds and sealed evaluation remain unfinished until measured.

## Glossary

- **Bank/head:** learned spatial features / neural map from latent state to coefficients.
- **Correction rank $q$:** additional linear coefficient directions solved online.
- **$N,K,R,M$:** nodes per axis, latent coordinates, bank functions, weak test functions.
- **QR/POD:** an exact coordinate factorization / a linear basis fitted to training snapshots.
- **FOM:** a numerical solve over every interior grid node.
- **Weak residual:** the PDE residual projected onto smooth test functions.
- **Best-found:** a truth-assisted fitting diagnostic, unavailable as an online prediction.
- **Stationarity:** small normalized objective gradient, not proof of the globally best fit.
- **Development/final cohort:** data available for selection / untouched cases used only after freezing choices.
- **Same-grid:** measured against a converged discretized solve on the query mesh.
- **Dense quadrature:** evaluating the weak nonlinear functional on the complete interior grid.
- **Invocation:** one complete solver call, supplying both its measured cost and its returned field errors.
