<!-- Generated from hash-pinned campaign evidence. -->
| Method | Median (\%) | Worst (\%) | GPU ms | p95 ms | NF/NS | Outliers/calls |
| --- | --- | --- | --- | --- | --- | --- |
| NM-ROM K32, q0 | 1.1494 | 5.9683 | 37.143 | 48.919 | 0/0 | 0/48 |
| NM-ROM K32, q8 | 1.0223 | 5.1921 | 40.126 | 80.283 | 0/0 | 5/48 |
| NM-ROM K32, q32 | 0.8224 | 4.4151 | 40.445 | 56.685 | 0/0 | 1/48 |
| NM-ROM K32, q64 | 0.5591 | 3.1815 | 52.322 | 92.258 | 0/0 | 5/48 |
| NM-ROM K32, q96 | 0.3903 | 1.3555 | 113.590 | 151.651 | 0/0 | 1/48 |
| Full bank, exact Galerkin | 0.2563 | 1.3166 | 0.564 | 0.766 | 0/0 | 1/48 |
| Full bank, exact weak | 0.2542 | 1.3113 | 1.495 | 1.694 | 0/0 | 0/48 |
| POD32 | 3.0811 | 9.3730 | 0.311 | 0.647 | 0/0 | 9/48 |
| POD128 | 0.1414 | 1.2338 | 0.565 | 0.616 | 0/0 | 0/48 |
| FNO, direct transfer | 15.1234 | 17.6641 | 16.246 | 17.031 | 0/0 | 0/48 |
| FNO, native + interpolation | 0.7669 | 0.9891 | 10.681 | 11.592 | 0/0 | 0/48 |
| U-Net, direct transfer | 75.7115 | 92.6756 | 10.960 | 11.808 | 0/0 | 0/48 |
| U-Net, native + interpolation | 0.7512 | 0.9468 | 4.240 | 4.764 | 0/0 | 0/48 |
| DeepONet, direct transfer | 33.8829 | 52.5118 | 11.270 | 12.017 | 0/0 | 0/48 |
| DeepONet, native + interpolation | 7.0168 | 21.7903 | 4.029 | 4.545 | 0/0 | 0/48 |
| Transolver, direct transfer | 19.8616 | 25.2954 | 20.828 | 22.118 | 0/0 | 0/48 |
| Transolver, native + interpolation | 0.9475 | 1.2766 | 6.854 | 7.952 | 0/0 | 0/48 |
| Direct DST | 0.0000 | 0.0000 | 1.591 | 1.926 | 0/0 | 0/48 |
