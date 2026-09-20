| Problem | Mesh | Cases | Correction ranks | Named FOM | Residual | Allocation |
| --- | --- | --- | --- | --- | --- | --- |
| Burgers 3D | 33^3 nodes | 32 final | 0, 192 | Newton–BiCGStab, \Delta t=0.01 | dense | job \texttt{4021709} |
| Poisson 3D | 32^3, 64^3 | 64 final | 0, 32, 96 (k=16) | CG, rtol 10^{-2}, no preconditioner | dense | job \texttt{4028642} |
| Heat 3D | 32^3, 64^3 | 64 final | 0, 32, 64, 96 (k=32) | CN–CG, \Delta t=0.05, rtol 10^{-4}, warm start | dense | job \texttt{4033346} |
| Navier–Stokes 3D | 32^3 periodic | 32 final | 0, 32, 64, 128, 256 (k=64, R=1536, M=2048) | CNAB2, \Delta t=0.01 | dense | job \texttt{4027788} |
