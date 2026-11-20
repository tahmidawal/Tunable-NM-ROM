# B8 exact root-Jacobian profiling

Prospective diagnostic only. Maximum allocation: one GPU, 15 minutes. No full-trajectory replay is authorized by this profile. Preserve original forward differentiation as the default and retain profile failures.

The 32-variable nonlinear root has one scalar output at each point. Replace 32-direction forward differentiation with a vectorized scalar-output reverse gradient, independently per point. This is not full-field jacrev. Child Jacobian blocks, all five fine stencil positions, sign-upwind branches, diffusion eigenvalues, test projection, LM arithmetic and stopping tolerances remain unchanged.

Use verified b8childprimary weights and b8treepde retained codes at indices0,250,499. Four retained children yield96 total variables. Match L16384, Q128, M512, dt0.0005, nu0.005, theta0.5, LM60, station1e-8/update1e-10. The driver verifies source/checkpoint hashes and records both paths. Defaults can select alternative compatible jobs explicitly, but may not change these numerical conditions.

Profile root Jacobians, complete residual Jacobians, and complete one-step solves. Burn the GPU, compile separately, warm both paths, synchronize every timed call, and alternate forward/reverse order for at least five repetitions by default. Record all samples and compilation/assembly costs; compare three paired median step times. Numerical acceptance: Jacobian max absolute difference<=1e-9 and relative<=1e-10; step relative physical field and residual-vector differences<=1e-10; identical iteration count, accepted count and convergence flag, with both converged. Retain raw stationarity/update values.

Only parity plus all three faster step medians permits proposing a separately registered500-step replay with all five requested4096 outputs and complete-query accounting. This microbenchmark cannot establish complete runtime advantage, time/integration convergence or fair CROM/SNF superiority. Independent chronology/data gates remain unchanged.
