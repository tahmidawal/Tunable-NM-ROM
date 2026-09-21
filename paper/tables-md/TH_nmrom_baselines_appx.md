| Mesh | Method | Residual path | k | Worst (%) | Median (%) | GPU ms (this job) | MB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 256^2 | Kim et al. NM-LSPG | dense | 8 | 163.93 | 43.34 | 1301 | 2480 |
| 256^2 | Kim et al. NM-LSPG | dense | 16 | 145.40 | 43.87 | 2384 | 2567 |
| 256^2 | Kim et al. NM-LSPG | dense | 32 | 127.51 | 43.61 | 3846 | 2742 |
| 256^2 | Kim et al. NM-LSPG, data-matched | dense | 16 | 120.33 | 38.71 | 3886 | 2567 |
| 256^2 | POD-LSPG | dense | 8 | 55.52 | 22.72 | 138 | 18 |
| 256^2 | POD-LSPG | dense | 16 | 41.53 | 10.45 | 259 | 30 |
| 256^2 | POD-LSPG | dense | 32 | 24.78 | 5.65 | 519 | 55 |
| 256^2 | This work, fast (q=0, k=16) | dense (reference path) | 16 | 6.79 | 0.66 | 262 | 625 |
| 256^2 | This work, accurate (q=256, k=16) | dense (reference path) | 272 | 0.88 | 0.089 | 3746 | 1769 |
| 256^2 | Kim et al. NM-LSPG-HR (exploratory) | hyper-reduced, not reproduced | 8 | 115.60 | 51.78 | 65 | 2452 |
| 256^2 | Kim et al. NM-LSPG-HR (exploratory) | hyper-reduced, not reproduced | 16 | 202.87 | 52.51 | 126 | 2506 |
| 256^2 | Kim et al. NM-LSPG-HR (exploratory) | hyper-reduced, not reproduced | 32 | 122.68 | 41.98 | 475 | 2794 |
| 512^2 | Kim et al. NM-LSPG | dense | 8 | 175.71 | 55.48 | 746 | 9959 |
| 512^2 | Kim et al. NM-LSPG | dense | 16 | 174.99 | 58.56 | 1497 | 10511 |
| 512^2 | Kim et al. NM-LSPG | dense | 32 | 252.33 | 56.72 | 2059 | 10879 |
| 512^2 | POD-LSPG | dense | 8 | 55.82 | 24.08 | 314 | 71 |
| 512^2 | POD-LSPG | dense | 16 | 42.01 | 13.83 | 617 | 122 |
| 512^2 | POD-LSPG | dense | 32 | 25.63 | 7.74 | 383 | 222 |
| 512^2 | This work, fast (q=0, k=16) | dense (reference path) | 16 | 7.72 | 0.69 | 404 | 2368 |
| 512^2 | This work, accurate (q=256, k=16) | dense (reference path) | 272 | 1.05 | 0.080 | 6269 | 6891 |
| 512^2 | Kim et al. NM-LSPG-HR (exploratory) | hyper-reduced, not reproduced | 8 | 273.27 | 53.08 | 30 | 9813 |
| 512^2 | Kim et al. NM-LSPG-HR (exploratory) | hyper-reduced, not reproduced | 16 | 185.17 | 58.52 | 62 | 10099 |
| 512^2 | Kim et al. NM-LSPG-HR (exploratory) | hyper-reduced, not reproduced | 32 | 194.47 | 56.00 | 138 | 10569 |
