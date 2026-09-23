| Setting | Dev. err (%) | Dev. ms | Dev. speedup | Held-out err (%) | Held-out ms | Held-out speedup |
| --- | --- | --- | --- | --- | --- | --- |
| Poisson 2D, ordered bank, 40962 (12 development sources) |  |  |  |  |  |  |
| Head, k=32, R'=512 | 3.15 | 43.4 | 103× | –- | –- | –- |
| Bank span, R'=512 | 0.74 | 42.4 | 106× | –- | –- | –- |
| Bank span, R'=384 | 0.77 | 32.5 | 138× | –- | –- | –- |
| Bank span, R'=256 | 0.94 | 22.6 | 199× | –- | –- | –- |
| Bank span, R'=128 | 2.31 | 12.5 | 358× | –- | –- | –- |
| Bank span, R'=64 | 5.33 | 7.1 | 631× | –- | –- | –- |
| Bank span, R'=32 | 9.17 | 4.4 | 1022× | –- | –- | –- |
| FOM: CG, rtol 10-1 | 0.45 | 4490 | 1× | –- | –- | –- |
| Navier–Stokes 3D, co-moving frame, ordered bank, 963 (16 development / 32 held-out cases) |  |  |  |  |  |  |
| Head, k=8 | 0.15 | 16.0 | 6.67× | 0.21 | 15.8 | 7.79× |
| Bank span, R'=64 | 0.62 | 15.4 | 6.92× | 1.12 | 15.3 | 8.05× |
| Bank span, R'=48 | 1.00 | 12.9 | 8.27× | 1.26 | 12.6 | 9.76× |
| Bank span, R'=32 | 1.26 | 11.6 | 9.14× | 1.46 | 11.5 | 10.7× |
| Bank span, R'=16 | 2.96 | 9.2 | 11.6× | 3.24 | 9.2 | 13.4× |
| Bank span, R'=8 | 5.55 | 8.2 | 13.0× | 6.91 | 8.1 | 15.2× |
| FOM: CNAB2 (spectral), 60 / 70 steps | 0.063 | 106 | 1× | 0.056 | 123 | 1× |
| Burgers 2D, correction rank (6 development / 64 held-out cases) |  |  |  |  |  |  |
| q=0, M=64 | 2.41 | 40.7 | 12.9× | 9.03 | 40.6 | 13.2× |
| q=128, M=576 | 1.09 | 77.4 | 6.77× | 2.43 | 74.6 | 7.21× |
| q=256, M=544 | 0.88 | 112 | 4.67× | 2.93 | 105 | 5.11× |
| q=256, M=1088 | 0.60 | 127 | 4.12× | 1.33 | 113 | 4.77× |
| q=256, M=1088, looser tol.\star | 0.60 | 108 | 4.87× | 1.33 | 100 | 5.38× |
| q=256, M=2176\dagger | 0.46 | 132 | 3.96× | 1.13 | 126 | 4.25× |
| FOM: Newton–BiCGStab, tol. 3×10-3 | 0.050 | 524 | 1× | 0.14 | 537 | 1× |
| Heat 2D, wide bank, correction rank (16 sealed held-out cases) |  |  |  |  |  |  |
| q=0 | –- | –- | –- | 0.84 | 32.2 | 146× |
| q=32 | –- | –- | –- | 0.22 | 31.7 | 149× |
| q=96 | –- | –- | –- | 0.082 | 24.4 | 193× |
| FOM: CN–CG, \Delta t=0.025, rtol 10-6 | –- | –- | –- | 0.045 | 4713 | 1× |
