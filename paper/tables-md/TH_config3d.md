| Problem | Mesh | Cases | Reduced model | Named FOM | Residual |
| --- | --- | --- | --- | --- | --- |
| Poisson 3D | 32^3, 64^3 | 64 final | q\in\{0, 32, 96\} (k=16) | CG, rtol 10^{-2}, no preconditioner | dense |
| Heat 3D (new bank) | 32^3, 64^3, 128^3 | 64 sealed final (all times) | q\in\{0, 288\} (k=32, R=320) | CN–CG, setting per row by the rule; named \Delta t=0.025, rtol 10^{-6} | exact (linear) |
| Navier–Stokes 3D | 32^3, 64^3, 96^3 periodic | 16 dev. + 32 held-out (96^3) | POD bank R=64 in a moving frame; head k=8; span R'\in\{64, 48, 32, 16, 8\} | CNAB2 (spectral), 50–70 steps (per row by the rule) | exact (precomputed quadratic tensor) |
