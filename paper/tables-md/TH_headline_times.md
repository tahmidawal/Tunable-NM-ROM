| Problem | Mesh | Accurate | Fast | Accurate ms | Fast ms | FOM setting | FOM ms | Timing | Other scope or FOM: acc. / fast | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Poisson 2D | 256^2 | q=256 | q=0 | 7.12 | 6.70 | CG, rtol 10^{-2} | 22.29 | GPU query | — | development |
| Poisson 2D | 1024^2 | q=256 | q=0 | 4.32 | 3.94 | CG, rtol 10^{-2} | 60.02 | GPU query | — | development |
| Poisson 2D | 2048^2 | q=256 | q=0 | 5.47 | 5.35 | CG, rtol 10^{-2} | 397.80 | GPU query | 28.7× / 28.8× (complete query) | development |
| Poisson 2D | 4096^2 | q=256 | q=0 | 16.92 | 16.74 | CG, rtol 10^{-1} | 1954.80 | GPU query | 32.7× / 33.2× (complete query) | development |
| Poisson 2D^{\ast} | 4096^2 | q=256 | q=0 | 17.17 | 17.01 | CG, rtol 2{\times}10^{-1} | 1779.89 | GPU query | 27.4× / 27.6× (complete query) | development |
| Poisson, L-shape 2D | 256^2 | q=64 | q=0 | 3.03 | 2.88 | CG, rtol 10^{-2} | 11.64 | complete query | — | development |
| Poisson, L-shape 2D | 512^2 | q=64 | q=0 | 4.80 | 4.72 | CG, rtol 10^{-2} | 29.13 | complete query | — | development |
| Poisson, L-shape 2D | 1024^2 | q=128 | q=0 | 5.81 | 5.67 | CG, rtol 3{\times}10^{-2} | 48.70 | complete query | 13.7× / 14.1× (GPU query) | development |
| Poisson, L-shape 2D | 2048^2 | q=128 | q=0 | 18.75 | 18.66 | CG, rtol 3{\times}10^{-2} | 318.22 | complete query | 29.8× / 30.2× (GPU query) | development |
| Heat 2D | 64^2 | — | single | — | 11.23 | CN–CG, rtol 10^{-2} | 2.50 | GPU query | — | development |
| Heat 2D | 256^2 | — | single | — | 11.34 | CN–CG, rtol 10^{-2} | 6.94 | GPU query | — | development |
| Heat 2D | 1024^2 | — | single | — | 12.21 | CN–CG, rtol 10^{-2} | 59.18 | GPU query | — | development |
| Heat (wide bank) 2D | 1024^2 | q=32 | q=0 | 27.33 | 27.77 | CN–CG, \Delta t=0.05, rtol 10^{-3} | 43.51 | GPU query | 4.70× / 4.62× (vs. CN–CG rtol 10^{-6}) | sealed held-out |
| Heat (wide bank) 2D | 2048^2 | q=32 | q=0 | 27.97 | 28.42 | CN–CG, \Delta t=0.05, rtol 10^{-3} | 272.55 | GPU query | 28.8× / 28.3× (vs. CN–CG rtol 10^{-6}) | sealed held-out |
| Heat (wide bank) 2D | 4096^2 | q=32 | q=0 | 31.68 | 32.19 | CN–CG, \Delta t=0.05, rtol 10^{-2} | 1139.83 | GPU query | 149× / 146× (vs. CN–CG rtol 10^{-6}) | sealed held-out |
| Heat (wide bank, batched fit) 2D | 1024^2 | q=32 | q=0 | 3.13 | 3.36 | CN–CG, \Delta t=0.05, rtol 10^{-3} | 43.51 | GPU query | 41.0× / 38.3× (vs. CN–CG rtol 10^{-6}) | sealed held-out |
| Heat (wide bank, batched fit) 2D | 2048^2 | q=32 | q=0 | 4.54 | 4.78 | CN–CG, \Delta t=0.05, rtol 10^{-3} | 272.55 | GPU query | 177× / 169× (vs. CN–CG rtol 10^{-6}) | sealed held-out |
| Heat (wide bank, batched fit) 2D | 4096^2 | q=32 | q=0 | 11.03 | 11.33 | CN–CG, \Delta t=0.05, rtol 10^{-2} | 1139.83 | GPU query | 428× / 416× (vs. CN–CG rtol 10^{-6}) | sealed held-out |
| Burgers 2D | 256^2 | q=256, M=1088, EQ m=2560 | q=0, M=64, EQ m=1024 | 746.02 | 40.36 | Newton–BiCGStab, tol 10^{-3} | 31.79 | GPU query | — | development |
| Burgers 2D | 512^2 | q=256, M=1088, EQ m=2438 | q=0, M=64, EQ m=922 | 783.33 | 40.49 | Newton–BiCGStab, tol 10^{-3} | 52.92 | GPU query | — | development |
| Burgers 2D | 1024^2 | q=256, M=1088, dense | q=0, M=64, EQ m=934 | 22053.85 | 40.27 | Newton–BiCGStab, tol 10^{-4} | 81.31 | GPU query | — | development |
| Burgers 2D | 2048^2 | q=256, M=544, EQ lattice m=3969 | q=0, M=64, EQ m=1024 | 124.15 | 29.48 | Newton–BiCGStab, tol 3{\times}10^{-3} | 122.76 | GPU query | 0.99× / 2.01× (complete query); 6.07× / 25.5× (vs. tight Newton) | development |
| Burgers 2D | 4096^2 | q=256, M=1088, EQ lattice m=3969 | q=0, M=64, EQ m=1024 | 107.51 | 40.73 | Newton–BiCGStab, tol 3{\times}10^{-3} | 523.85 | GPU query | 2.18× / 2.68× (complete query); 30.4× / 80.3× (vs. tight Newton) | development |
| Burgers (held-out cases) 2D | 2048^2 | q=256, M=1088, EQ lattice m=3969 | q=0, M=64, EQ m=1024 | 114.91 | 29.19 | Newton–BiCGStab, tol 10^{-3} | 164.43 | GPU query | 1.28× / 2.49× (complete query); 6.48× / 25.5× (vs. tight Newton) | held-out |
| Burgers (held-out cases) 2D | 4096^2 | q=256, M=1088, EQ lattice m=3969 | q=0, M=64, EQ m=1024 | 99.96 | 40.60 | Newton–BiCGStab, tol 3{\times}10^{-3} | 537.39 | GPU query | 2.28× / 2.74× (complete query); 32.3× / 79.6× (vs. tight Newton) | held-out |
| Burgers, exact first step 2D^{\ast} | 4096^2 | q=256, M=1088, EQ lattice m=3969, first step exact | — | 5308.00 | — | Newton–BiCGStab, tol 3{\times}10^{-3} | 523.76 | GPU query | — | confirmed rule |
| Burgers, exact first step 2D^{\ast} | 4096^2 | q=256, M=1088, EQ lattice m=3969, first step exact | — | 6557.77 | — | Newton–BiCGStab, tol 3{\times}10^{-3} | 536.84 | GPU query | — | confirmed rule |
| Burgers, confirmed rule 2D | 512^2 | q=256, M=1088, EQ lattice m=3969, first step exact | q=0, M=64, EQ m=1024 | 194.97 | 23.62 | Newton–BiCGStab, tol 3{\times}10^{-3} | 17.74 | GPU query | — | development |
| Burgers, confirmed rule 2D | 1024^2 | q=256, M=1088, EQ lattice m=3969 | q=0, M=64, EQ m=1024 | 108.58 | 25.00 | Newton–BiCGStab, tol 3{\times}10^{-3} | 34.69 | GPU query | — | development |
| Burgers (earlier model) 2D | 1024^2 | — | single | — | 41.68 | Newton–BiCGStab, relaxed | 68.04 | GPU query | — | development |
| Burgers (earlier model) 2D^{\ast} | 1024^2 | — | single | — | 41.68 | Newton–BiCGStab, tight | 605.75 | GPU query | — | development |
| Poisson 3D | 32^3 | q=96 | q=0 | 2.63 | 2.49 | CG, rtol 10^{-2} | 2.47 | GPU query | — | accepted final |
| Poisson 3D | 64^3 | q=96 | q=0 | 3.36 | 3.22 | CG, rtol 10^{-2} | 4.46 | GPU query | — | accepted final |
| Poisson (dev. sources) 3D^{\ast} | 64^3 | q=96 | q=0 | 1.43 | 1.38 | CG, rtol 10^{-2} | 2.88 | GPU query | 1.57× / 1.56× (complete query) | development |
| Poisson (dev. sources) 3D | 128^3 | q=96 | q=0 | 1.86 | 1.80 | CG, rtol 10^{-2} | 12.55 | GPU query | 2.70× / 2.66× (complete query) | development |
| Poisson (dev. sources) 3D^{\ast} | 128^3 | q=96 | q=0 | 1.90 | 1.76 | CG, rtol 10^{-2} | 12.40 | GPU query | 2.65× / 2.70× (complete query) | development |
| Poisson (dev. sources) 3D | 256^3 | q=96 | q=0 | 6.08 | 5.95 | CG, rtol 10^{-2} | 140.87 | GPU query | 4.18× / 4.21× (complete query) | development |
| Heat (new bank) 3D | 32^3 | q=288 | q=0 | 58.16 | 25.41 | CN–CG, \Delta t=0.025, rtol 10^{-4} | 3.33 | GPU query | — | accepted final |
| Heat (new bank) 3D | 64^3 | q=288 | q=0 | 58.69 | 25.45 | CN–CG, \Delta t=0.025, rtol 10^{-4} | 5.23 | GPU query | — | accepted final |
| Heat (new bank) 3D | 128^3 | q=288 | q=0 | 62.03 | 28.25 | CN–CG, \Delta t=0.025, rtol 10^{-4} | 17.44 | GPU query | — | accepted final |
| Heat (new bank, batched fit) 3D | 32^3 | q=288 | q=0 | 5.36 | 5.88 | CN–CG, \Delta t=0.025, rtol 10^{-4} | 3.33 | GPU query | — | accepted final |
| Heat (new bank, batched fit) 3D | 64^3 | q=288 | q=0 | 5.54 | 6.14 | CN–CG, \Delta t=0.025, rtol 10^{-4} | 5.23 | GPU query | — | accepted final |
| Heat (new bank, batched fit) 3D | 128^3 | q=288 | q=0 | 7.84 | 8.30 | CN–CG, \Delta t=0.025, rtol 10^{-4} | 17.44 | GPU query | — | accepted final |
