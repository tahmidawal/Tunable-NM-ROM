# Empirical quadrature — error and cost against the number of points

Generated fragment. Poisson and Burgers m-ladders at N=64, k=8, M=64 test modes.

---

| points m | share of grid | error | cost | quadrature fit |
|---|---|---|---|---|
| **Poisson** | | | | |
| 64 | 2% | 5.48e-02 | 11.1 | 1.4e-01 |
| 128 | 3% | 1.80e-02 | 13.9 | 1.6e-02 |
| 256 | 7% | 8.66e-03 | 15.1 | 2.4e-03 |
| 512 | 13% | 7.68e-03 | 15.2 | 2.4e-04 |
| 1024 | 27% | 7.65e-03 | 18.9 | 1.6e-05 |
| every point | 100% | 7.65e-03 | 35.6 | exact |
| **Burgers** | | | | |
| 64 | 2% | 6.54e-02 | 3.4 | 2.1e-01 |
| 128 | 3% | 1.95e-02 | 4.5 | 4.9e-02 |
| 256 | 7% | 1.74e-02 | 6.2 | 6.2e-03 |
| 512 | 13% | 1.68e-02 | 10.8 | 1.0e-03 |
| 1024 | 27% | 1.67e-02 | 20.7 | 1.5e-04 |
| every point | 100% | 1.65e-02 | 70.9 | exact |
