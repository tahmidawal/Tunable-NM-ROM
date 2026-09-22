| Problem | Mesh | Cases | Correction ranks | Named FOM | Residual |
| --- | --- | --- | --- | --- | --- |
| Burgers 3D | 32^3 | 32 final | 0, 192 | Newton–BiCGStab, \Delta t=0.01 | dense |
| Poisson 3D | 32^3, 64^3 | 64 final | 0, 32, 96 (k=16) | CG, rtol 10^{-2}, no preconditioner | dense |
| Heat 3D (new bank) | 32^3, 64^3, 128^3 | 64 sealed final (all times) | 0, 288 (k=32, R=320) | CN–CG, setting per row by the rule; named \Delta t=0.025, rtol 10^{-6} | exact (linear) |
| Navier–Stokes 3D | 32^3 periodic | 32 final | 0, 32, 64, 128, 256 (k=64, R=1536, M=2048) | CNAB2, \Delta t=0.01 | dense |
