| Problem | Mesh | Accurate err. (%) | Accurate speedup | Fast err. (%) | Fast speedup | FOM err. (%) | FOM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Poisson 2D (development) | 256^2 | 0.97 | **3.13×** | 3.16 | **3.33×** | 0.15 | CG, rtol 10^{-2} |
| Poisson 2D (development) | 1024^2 | 0.96 | **13.9×** | 3.15 | **15.2×** | 0.072 | CG, rtol 10^{-2} |
| Poisson 2D (development) | 2048^2 | 0.96 | **72.8×** | 3.15 | **74.3×** | 0.049 | CG, rtol 10^{-2} |
| Poisson 2D (development) | 4096^2 | 0.96 | **116×** | 3.15 | **117×** | 0.45 | CG, rtol 10^{-1} |
| Poisson, L-shape 2D (development) | 256^2 | 2.13 | **3.84×** | 3.86 | **4.04×** | 0.64 | CG, rtol 10^{-2} |
| Poisson, L-shape 2D (development) | 512^2 | 2.12 | **6.07×** | 3.85 | **6.18×** | 0.38 | CG, rtol 10^{-2} |
| Poisson, L-shape 2D (development) | 1024^2 | 2.20 | **8.38×** | 3.85 | **8.59×** | 1.05 | CG, rtol 3{\times}10^{-2} |
| Poisson, L-shape 2D (development) | 2048^2 | 2.20 | **17.0×** | 3.85 | **17.1×** | 0.76 | CG, rtol 3{\times}10^{-2} |
| Heat 2D (development; earlier checkpoint, one setting) | 64^2 | — | — | 4.56 | 0.22× | 1.59 | CN–CG, rtol 10^{-2} |
| Heat 2D (development; earlier checkpoint, one setting) | 256^2 | — | — | 4.56 | 0.61× | 0.93 | CN–CG, rtol 10^{-2} |
| Heat 2D (development; earlier checkpoint, one setting) | 1024^2 | — | — | 4.56 | **4.85×** | 0.77 | CN–CG, rtol 10^{-2} |
| Heat (wide bank) 2D (sealed held-out; opened once) | 1024^2 | 0.49 | **1.59×** | 1.36 | **1.57×** | 0.18 | CN–CG, \Delta t=0.05, rtol 10^{-3} |
| Heat (wide bank) 2D (sealed held-out; opened once) | 2048^2 | 0.49 | **9.74×** | 1.36 | **9.59×** | 0.18 | CN–CG, \Delta t=0.05, rtol 10^{-3} |
| Heat (wide bank) 2D (sealed held-out; opened once) | 4096^2 | 0.49 | **36.0×** | 1.36 | **35.4×** | 0.47 | CN–CG, \Delta t=0.05, rtol 10^{-2} |
| Heat (wide bank, batched fit) 2D (sealed held-out; opened once) | 1024^2 | 0.49 | **13.9×** | 1.36 | **13.0×** | 0.18 | CN–CG, \Delta t=0.05, rtol 10^{-3} |
| Heat (wide bank, batched fit) 2D (sealed held-out; opened once) | 2048^2 | 0.49 | **60.0×** | 1.36 | **57.0×** | 0.18 | CN–CG, \Delta t=0.05, rtol 10^{-3} |
| Heat (wide bank, batched fit) 2D (sealed held-out; opened once) | 4096^2 | 0.49 | **103×** | 1.36 | **101×** | 0.47 | CN–CG, \Delta t=0.05, rtol 10^{-2} |
| Burgers 2D (development) | 256^2 | 0.51^{s} | 0.043× | 1.89 | 0.79× | 0.049 | Newton–BiCGStab, tol 10^{-3} |
| Burgers 2D (development) | 512^2 | 0.55^{s} | 0.068× | 2.14 | **1.31×** | 0.052 | Newton–BiCGStab, tol 10^{-3} |
| Burgers 2D (development) | 1024^2 | 0.59^{d} | 0.0037× | 2.29 | **2.02×** | 0.034 | Newton–BiCGStab, tol 10^{-4} |
| Burgers 2D (development; 6 cases, arm chosen here) | 2048^2 | 0.87^{\ell} | 0.99× | 2.37 | **4.16×** | 0.049 | Newton–BiCGStab, tol 3{\times}10^{-3} |
| Burgers 2D (development; 6 cases, arm chosen here) | 4096^2 | 0.60^{\ell} | **4.87×** | 2.41 | **12.9×** | 0.050 | Newton–BiCGStab, tol 3{\times}10^{-3} |
| Burgers (held-out cases) 2D (held-out; 64 cases, one timing repetition) | 2048^2 | 1.31^{\ell} | **1.43×** | 9.03 | **5.63×** | 0.13 | Newton–BiCGStab, tol 10^{-3} |
| Burgers (held-out cases) 2D (held-out; 64 cases, one timing repetition) | 4096^2 | 1.33^{\ell} | **5.38×** | 9.03 | **13.2×** | 0.14 | Newton–BiCGStab, tol 3{\times}10^{-3} |
| Burgers, confirmed rule 2D (development; re-drawn quadrature rule) | 512^2 | 0.56^{v} | 0.091× | 2.14 | 0.75× | 0.050 | Newton–BiCGStab, tol 3{\times}10^{-3} |
| Burgers, confirmed rule 2D (development; re-drawn quadrature rule) | 1024^2 | 0.59^{v} | 0.32× | 2.29 | **1.39×** | 0.048 | Newton–BiCGStab, tol 3{\times}10^{-3} |
| Burgers (earlier model) 2D (development; earlier model, stalled exits permitted) | 1024^2 | — | — | 3.91 | **1.63×** | 2.39 | Newton–BiCGStab, relaxed |
| Burgers (wider learned bank), held-out 64 2D | 2048^2, 4096^2 | results incoming |  |  |  |  | lane burgers-heldout |
| Poisson 3D (accepted final) | 32^3 | 0.26 | 0.94× | 1.40 | 0.99× | 0.16 | CG, rtol 10^{-2} |
| Poisson 3D (accepted final) | 64^3 | 0.26 | **1.33×** | 1.39 | **1.38×** | 0.11 | CG, rtol 10^{-2} |
| Poisson (dev. sources) 3D (development) | 128^3 | 0.16 | **6.75×** | 0.55 | **6.98×** | 0.075 | CG, rtol 10^{-2} |
| Poisson (dev. sources) 3D (development) | 256^3 | 0.16 | **23.2×** | 0.55 | **23.7×** | 0.049 | CG, rtol 10^{-2} |
| Heat 3D (wider bank) 3D | 64^3, 128^3 | results incoming |  |  |  |  | lane heat3d-bank |
