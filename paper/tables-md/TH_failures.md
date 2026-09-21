| Problem | Setting | NM-ROM err. (%) | FOM err. (%) | Speedup | FOM |
| --- | --- | --- | --- | --- | --- |
| Burgers 3D, $32^3$ (final) | $q=0$ | 16.26 | 2.31 | 0.038× | Newton–BiCGStab |
| Burgers 3D, $32^3$ (final) | $q=192$ | 4.40 | 2.31 | 0.023× | Newton–BiCGStab |
| Navier--Stokes 3D, $32^3$ (final) | $q=0$ | 20.34 | 2.47 | 0.0025× | CNAB2 |
| Navier--Stokes 3D, $32^3$ (final) | $q=256$ | 18.92 | 2.47 | 0.0007× | CNAB2 |
| Wave 2D, $1024^2$ (dev.) | $q=0$ | 11.34 | 4.29 | **3.03×** | midpoint–CG |
| Wave 2D, $1024^2$ (dev.) | $q=32$ | 5.12 | 4.29 | 0.23× | midpoint–CG |
| Heat 3D, $32^3$ (final, evolved) | $q=0$ | 1.56 | 0.32 | 0.13× | CN–CG |
| Heat 3D, $32^3$ (final, evolved) | $q=96$ | 0.76 | 0.32 | 0.066× | CN–CG |
| Heat 3D, $64^3$ (final, evolved) | $q=0$ | 1.55 | 0.34 | 0.27× | CN–CG |
| Heat 3D, $64^3$ (final, evolved) | $q=96$ | 0.75 | 0.34 | 0.13× | CN–CG |
| Heat 3D, $128^3$ (final, all times) | $q=0$ | 3.18 | 1.58 | 0.43× | CN–CG |
| Heat 3D, $128^3$ (final, all times) | $q=96$ | 1.93 | 1.58 | 0.20× | CN–CG |
