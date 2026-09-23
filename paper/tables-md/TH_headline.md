| Problem | Mesh | Accurate err. (%) | Accurate speedup | Fast err. (%) | Fast speedup | FOM err. (%) | FOM |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Poisson 2D (development; ordered bank; provisional timing (A-B-A drift gate failed)) | 256^2 | 0.75 | **13.4×** | 2.31 | **17.5×** | 0.55 | CG, rtol 3{\times}10^{-2} |
| Poisson 2D (development; ordered bank; provisional timing (A-B-A drift gate failed)) | 1024^2 | 0.74 | **27.9×** | 2.31 | **68.5×** | 0.22 | CG, rtol 3{\times}10^{-2} |
| Poisson 2D (development; ordered bank) | 2048^2 | 0.74 | **54.1×** | 2.31 | **174×** | 0.61 | CG, rtol 10^{-1} |
| Poisson 2D (development; ordered bank) | 4096^2 | 0.74 | **106×** | 2.31 | **358×** | 0.45 | CG, rtol 10^{-1} |
| Poisson, L-shape 2D (development) | 256^2 | 2.13 | **3.84×** | 3.86 | **4.04×** | 0.64 | CG, rtol 10^{-2} |
| Poisson, L-shape 2D (development) | 512^2 | 2.12 | **6.07×** | 3.85 | **6.18×** | 0.38 | CG, rtol 10^{-2} |
| Poisson, L-shape 2D (development) | 1024^2 | 2.20 | **8.38×** | 3.85 | **8.59×** | 1.05 | CG, rtol 3{\times}10^{-2} |
| Poisson, L-shape 2D (development) | 2048^2 | 2.20 | **17.0×** | 3.85 | **17.1×** | 0.76 | CG, rtol 3{\times}10^{-2} |
| Heat (wide bank, batched fit) 2D (sealed held-out; opened once) | 1024^2 | 0.49 | **13.9×** | 1.36 | **13.0×** | 0.18 | CN–CG, \Delta t=0.05, rtol 10^{-3} |
| Heat (wide bank, batched fit) 2D (sealed held-out; opened once) | 2048^2 | 0.49 | **60.0×** | 1.36 | **57.0×** | 0.18 | CN–CG, \Delta t=0.05, rtol 10^{-3} |
| Heat (wide bank, batched fit) 2D (sealed held-out; opened once) | 4096^2 | 0.49 | **103×** | 1.36 | **101×** | 0.47 | CN–CG, \Delta t=0.05, rtol 10^{-2} |
| Burgers 2D (development) | 256^2 | 0.51^{s} | 0.043× | 1.89 | 0.79× | 0.049 | Newton–BiCGStab, tol 10^{-3} |
| Burgers 2D (development) | 512^2 | 0.55^{x} | 0.068× | 2.14 | **1.31×** | 0.052 | Newton–BiCGStab, tol 10^{-3} |
| Burgers 2D (development) | 1024^2 | 0.59^{d} | 0.0037× | 2.29 | **2.02×** | 0.034 | Newton–BiCGStab, tol 10^{-4} |
| Burgers 2D (development; 6 cases, arm chosen here) | 2048^2 | 0.87^{\ell} | 0.99× | 2.37 | **4.16×** | 0.049 | Newton–BiCGStab, tol 3{\times}10^{-3} |
| Burgers 2D (development; 6 cases, arm chosen here) | 4096^2 | 0.60^{w} | **4.87×** | 2.41 | **10.10×** | 0.050 | Newton–BiCGStab, tol 3{\times}10^{-3} |
| Poisson 3D (accepted final) | 32^3 | 0.26 | 0.94× | 1.40 | 0.99× | 0.16 | CG, rtol 10^{-2} |
| Poisson 3D (accepted final) | 64^3 | 0.26 | **1.33×** | 1.39 | **1.38×** | 0.11 | CG, rtol 10^{-2} |
| Poisson (dev. sources) 3D (development) | 128^3 | 0.16 | **6.75×** | 0.55 | **6.98×** | 0.075 | CG, rtol 10^{-2} |
| Poisson (dev. sources) 3D (development) | 256^3 | 0.16 | **23.2×** | 0.55 | **23.7×** | 0.049 | CG, rtol 10^{-2} |
| Heat (new bank, batched fit) 3D (accepted final) | 32^3 | 0.11 | 0.62× | 2.00 | 0.57× | 0.078 | CN–CG, \Delta t=0.025, rtol 10^{-4} |
| Heat (new bank, batched fit) 3D (accepted final) | 64^3 | 0.11 | 0.94× | 2.00 | 0.85× | 0.081 | CN–CG, \Delta t=0.025, rtol 10^{-4} |
| Heat (new bank, batched fit) 3D (accepted final) | 128^3 | 0.11 | **2.22×** | 2.00 | **2.10×** | 0.082 | CN–CG, \Delta t=0.025, rtol 10^{-4} |
| Navier--Stokes 3D (development; co-moving frame) | 32^3 | 0.15 | **1.09×** | 2.96 | **1.87×** | 0.094 | CNAB2 (spectral), 50 steps |
| Navier--Stokes 3D (development; co-moving frame) | 64^3 | 0.15 | **2.89×** | 2.96 | **5.09×** | 0.093 | CNAB2 (spectral), 50 steps |
| Navier--Stokes 3D (development; co-moving frame) | 96^3 | 0.15 | **6.67×** | 2.96 | **11.6×** | 0.063 | CNAB2 (spectral), 60 steps |
