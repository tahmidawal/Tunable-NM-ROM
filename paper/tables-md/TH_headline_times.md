| Problem | Mesh | Accurate | Fast | Accurate ms | Fast ms | FOM ms | Timing | Other scope or FOM: acc. / fast | Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Poisson 2D | 256^2 | q=256 | q=0 | 7.12 | 6.70 | 22.29 | GPU query | — | development |
| Poisson 2D | 1024^2 | q=256 | q=0 | 4.32 | 3.94 | 60.02 | GPU query | — | development |
| Poisson 2D | 2048^2 | q=256 | q=0 | 5.47 | 5.35 | 397.80 | GPU query | 28.7× / 28.8× (complete query) | development |
| Poisson 2D | 4096^2 | q=256 | q=0 | 16.92 | 16.74 | 2472.81 | GPU query | 41.1× / 41.7× (complete query) | development |
| Poisson 2D^{\ast} | 4096^2 | q=256 | q=0 | 17.17 | 17.01 | 2465.60 | GPU query | 37.6× / 37.9× (complete query) | development |
| Poisson, L-shape 2D | 256^2 | q=64 | q=0 | 3.03 | 2.88 | 11.64 | complete query | — | development |
| Poisson, L-shape 2D | 512^2 | q=64 | q=0 | 4.80 | 4.72 | 29.13 | complete query | — | development |
| Poisson, L-shape 2D | 1024^2 | q=128 | q=0 | 5.81 | 5.67 | 56.72 | complete query | 16.0× / 16.5× (GPU query) | development |
| Poisson, L-shape 2D | 2048^2 | q=128 | q=0 | 18.75 | 18.66 | 362.30 | complete query | 33.9× / 34.3× (GPU query) | development |
| Heat 2D | 64^2 | — | single | — | 11.23 | 2.50 | GPU query | — | development |
| Heat 2D | 256^2 | — | single | — | 11.34 | 6.94 | GPU query | — | development |
| Heat 2D | 1024^2 | — | single | — | 12.21 | 59.18 | GPU query | — | development |
| Heat (wide bank) 2D | 1024^2 | q=32 | q=0 | 31.20 | 36.89 | 66.84 | GPU query | 6.33× / 5.35× (vs. CN–CG rtol 10^{-6}) | development |
| Heat (wide bank) 2D | 2048^2 | q=32 | q=0 | 27.97 | 28.42 | 272.55 | GPU query | 28.8× / 28.3× (vs. CN–CG rtol 10^{-6}) | sealed held-out |
| Heat (wide bank) 2D | 4096^2 | q=32 | q=0 | 31.68 | 32.19 | 1139.83 | GPU query | 149× / 146× (vs. CN–CG rtol 10^{-6}) | sealed held-out |
| Heat (wide bank, batched fit) 2D | 2048^2 | q=32 | q=0 | 4.54 | 4.78 | 272.55 | GPU query | 177× / 169× (vs. CN–CG rtol 10^{-6}) | sealed held-out |
| Heat (wide bank, batched fit) 2D | 4096^2 | q=32 | q=0 | 11.03 | 11.33 | 1139.83 | GPU query | 428× / 416× (vs. CN–CG rtol 10^{-6}) | sealed held-out |
| Burgers 2D | 256^2 | q=256, M=1088, EQ m=2560 | q=0, M=64, EQ m=1024 | 746.02 | 40.36 | 31.79 | GPU query | — | development |
| Burgers 2D | 512^2 | q=256, M=1088, EQ m=2438 | q=0, M=64, EQ m=922 | 783.33 | 40.49 | 52.92 | GPU query | — | development |
| Burgers 2D | 1024^2 | q=256, M=1088, dense | q=0, M=64, EQ m=934 | 22053.85 | 40.27 | 81.31 | GPU query | — | development |
| Burgers 2D | 2048^2 | q=256, M=544, EQ lattice m=3969 | q=0, M=64, EQ m=1024 | 124.15 | 29.48 | 122.76 | GPU query | 0.99× / 2.01× (complete query); 6.07× / 25.5× (vs. tight Newton) | development |
| Burgers 2D | 4096^2 | q=256, M=1088, EQ lattice m=3969 | q=0, M=64, EQ m=1024 | 107.51 | 40.73 | 523.85 | GPU query | 2.18× / 2.68× (complete query); 30.4× / 80.3× (vs. tight Newton) | development |
| Burgers (held-out cases) 2D | 2048^2 | q=256, M=1088, EQ lattice m=3969 | q=0, M=64, EQ m=1024 | 114.91 | 29.19 | 164.43 | GPU query | 1.28× / 2.49× (complete query); 6.48× / 25.5× (vs. tight Newton) | held-out |
| Burgers (held-out cases) 2D | 4096^2 | q=256, M=1088, EQ lattice m=3969 | q=0, M=64, EQ m=1024 | 99.96 | 40.60 | 537.39 | GPU query | 2.28× / 2.74× (complete query); 32.3× / 79.6× (vs. tight Newton) | held-out |
| Burgers (earlier model) 2D | 1024^2 | — | single | — | 41.68 | 68.04 | GPU query | — | development |
| Burgers (earlier model) 2D | 1024^2 | — | single | — | 41.68 | 605.75 | GPU query | — | development |
| Poisson 3D | 32^3 | q=96 | q=0 | 2.63 | 2.49 | 2.47 | GPU query | — | accepted final |
| Poisson 3D | 64^3 | q=96 | q=0 | 3.36 | 3.22 | 4.46 | GPU query | — | accepted final |
| Poisson (dev. sources) 3D^{\ast} | 64^3 | q=96 | q=0 | 1.43 | 1.38 | 2.88 | GPU query | 1.57× / 1.56× (complete query) | development |
| Poisson (dev. sources) 3D | 128^3 | q=96 | q=0 | 1.86 | 1.80 | 12.55 | GPU query | 2.70× / 2.66× (complete query) | development |
| Poisson (dev. sources) 3D^{\ast} | 128^3 | q=96 | q=0 | 1.90 | 1.76 | 12.40 | GPU query | 2.65× / 2.70× (complete query) | development |
| Poisson (dev. sources) 3D | 256^3 | q=96 | q=0 | 6.08 | 5.95 | 140.87 | GPU query | 4.18× / 4.21× (complete query) | development |
| Heat 3D | 32^3 | q=96 | q=0 | 77.78 | 38.35 | 5.16 | GPU query | — | accepted final |
| Heat 3D | 64^3 | q=96 | q=0 | 85.44 | 40.38 | 10.84 | GPU query | — | accepted final |
