# Three-dimensional NM-ROM comparisons: generated results

These tables contain retained, audited development experiments from the overnight campaign. They are provisional paper material: tuning, operator comparisons and independent final evaluation must be assessed per attempt before a row supports a manuscript claim.

The explicit input manifest controls which attempts appear; no best run is selected automatically. Timed comparisons derive errors and costs from the same invocation records; untimed snapshot fits are separated explicitly. Every table retains unsuccessful methods and reports failure counts.

## Burgers 3D — b3d001

**Provisional:** Development diagnostic: retained network and smaller POD/direction training subset are unmatched; physical refinement and operator comparisons are absent. Dense initialization and dense nonlinear evaluation are charged. Final cohort remains unopened.

Source `1d2010aacff49a77ff7f8fe2c3221b4c6c739dc6`; job `3989410`; GPU `NVIDIA A100 80GB PCIe`. [Invocation data](../worktrees/2026-09-20-paper-b3d/experiments/paper-b3d/runs/b3d001/collected/out/result.json) and [independent audit](../worktrees/2026-09-20-paper-b3d/experiments/paper-b3d/runs/b3d001/collected/out/audit-local.json).

Errors use the initial-field norm. The evolved error excludes the initial compression; all-times and initial errors are also shown. GPU timing begins with the dense input already on device and ends with every requested dense output on device. Host transfers are unmeasured in this attempt. Mesh size counts nodes per axis.

| Mesh | Method | Cases | Error median (%) | Error worst (%) | All-times worst (%) | Initial worst (%) | Physical worst (%) | GPU median (ms) | Total median (ms) | Nonfinite / nonstationary cases | Timing outliers / calls | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 33 | `fom_nt1e-02` | 8 | 0.0382 | 0.6543 | 0.6543 | 0.0000 | — | 17.930 | — | 0 / 0 | 0 / 24 | development |
| 33 | `fom_nt1e-04` | 8 | 0.0188 | 0.0207 | 0.0207 | 0.0000 | — | 38.131 | — | 0 / 0 | 0 / 24 | development |
| 33 | `fom_nt1e-06` | 8 | 0.0000 | 0.0003 | 0.0003 | 0.0000 | — | 91.753 | — | 0 / 0 | 0 / 24 | development |
| 33 | `free_R128` | 8 | 2.0228 | 5.9695 | 7.0334 | 7.0334 | — | 143.348 | — | 0 / 0 | 0 / 24 | development |
| 33 | `pod_128` | 8 | 1.8758 | 5.3034 | 6.0555 | 6.0555 | — | 142.684 | — | 0 / 0 | 0 / 24 | development |
| 33 | `pod_32` | 8 | 10.4499 | 19.1398 | 20.8596 | 20.8596 | — | 91.625 | — | 0 / 0 | 0 / 24 | development |
| 33 | `pod_64` | 8 | 4.9116 | 10.0613 | 10.8955 | 10.8955 | — | 109.704 | — | 0 / 0 | 0 / 24 | development |
| 33 | `pod_96` | 8 | 2.8234 | 6.5582 | 7.5205 | 7.5205 | — | 138.375 | — | 0 / 0 | 0 / 24 | development |
| 33 | `rom_q0` | 8 | 4.5930 | 12.9762 | 14.5016 | 14.5016 | — | 162.856 | — | 0 / 1 | 0 / 24 | stopping failures |
| 33 | `rom_q32` | 8 | 3.9653 | 10.6624 | 11.9876 | 11.9876 | — | 193.751 | — | 0 / 0 | 0 / 24 | development |
| 33 | `rom_q64` | 8 | 3.0091 | 8.2111 | 9.3453 | 9.3453 | — | 237.352 | — | 0 / 0 | 0 / 24 | development |

A missing physical-error or total-time cell means unmeasured, not zero. Physical errors require the attempt's separate reference-refinement qualification. A passing numerical audit verifies the recorded experiment; it does not establish good predictive accuracy, convergence of training or a competitive method.

## Heat 3D — pilot01

**Provisional:** Development pilot with matched training membership and fixed diffusivity. The learned bank limits accuracy; longer training and operator comparisons are pending. Sampled quadrature arms failed their certificates. Final cohort remains unopened. Finer-grid POD is rebuilt from the same training members and is a mesh-adapted classical control.

Source `e6460d73c7d4d3292c9e9ef313ddb79c88bf59dd`; job `3989545`; GPU `NVIDIA A100 80GB PCIe`. [Invocation data](../worktrees/2026-09-20-paper-h3d/experiments/paper-h3d/runs/pilot01/archive/out/result.json) and [independent audit](../worktrees/2026-09-20-paper-h3d/experiments/paper-h3d/runs/pilot01/audit-local.json).

Errors use each reference field's current norm. Evolved errors exclude the initial state. Total timing includes host transfers; GPU timing includes initialization, evolution and dense output. Mesh size counts intervals per axis.

| Mesh | Method | Cases | Error median (%) | Error worst (%) | All-times worst (%) | Initial worst (%) | Physical worst (%) | GPU median (ms) | Total median (ms) | Nonfinite / nonstationary cases | Timing outliers / calls | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 32 | `dst_exact` | 16 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5119 | 1.264 | 2.640 | 0 / 0 | 0 / 48 | development |
| 32 | `linear_bank_cn` | 16 | 4.0772 | 9.7539 | 16.5976 | 16.5976 | 9.7085 | 0.453 | 2.423 | 0 / 0 | 10 / 48 | development |
| 32 | `linear_bank_exact` | 16 | 4.0768 | 9.7512 | 16.5976 | 16.5976 | 9.7055 | 0.408 | 2.440 | 0 / 0 | 12 / 48 | development |
| 32 | `linear_bank_galerkin_exact` | 16 | 4.1364 | 9.6202 | 15.3813 | 15.3813 | 9.5175 | 0.133 | 2.113 | 0 / 0 | 4 / 48 | development |
| 32 | `nmrom_K16_q0_dense` | 16 | 4.8700 | 11.9300 | 19.5033 | 19.5033 | 11.8542 | 17.151 | 18.188 | 0 / 0 | 3 / 48 | development |
| 32 | `nmrom_K16_q0_eq` | 16 | 64.8220 | 72.7290 | 73.1431 | 73.1431 | 72.6938 | 18.003 | 18.964 | 0 / 0 | 9 / 48 | failed EQ certificate |
| 32 | `nmrom_K16_q16_dense` | 16 | 4.2717 | 9.8300 | 16.6081 | 16.6081 | 9.7703 | 19.234 | 20.509 | 0 / 0 | 3 / 48 | development |
| 32 | `nmrom_K16_q16_eq` | 16 | 64.8965 | 72.8083 | 73.3330 | 73.3330 | 72.7727 | 21.818 | 22.857 | 0 / 0 | 1 / 48 | failed EQ certificate |
| 32 | `nmrom_K16_q32_dense` | 16 | 4.0740 | 9.7263 | 16.1032 | 16.1032 | 9.6749 | 22.290 | 23.296 | 0 / 0 | 5 / 48 | development |
| 32 | `nmrom_K16_q32_dense_dt_half` | 16 | 4.0693 | 9.7222 | 16.1032 | 16.1032 | 9.6711 | 34.540 | 35.671 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K16_q32_eq` | 16 | 64.8423 | 72.7958 | 73.3537 | 73.3537 | 72.7603 | 24.104 | 25.165 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 32 | `nmrom_K16_q8_dense` | 16 | 4.4447 | 10.8586 | 17.8493 | 17.8493 | 10.7918 | 17.995 | 19.129 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K16_q8_eq` | 16 | 64.9967 | 72.7896 | 73.2394 | 73.2394 | 72.7545 | 19.092 | 20.231 | 0 / 0 | 3 / 48 | failed EQ certificate |
| 32 | `nmrom_K8_q0_dense` | 16 | 5.7671 | 15.8873 | 26.0909 | 26.0909 | 15.7541 | 12.454 | 13.523 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K8_q0_eq` | 16 | 68.6077 | 74.2657 | 74.5765 | 74.5765 | 74.2326 | 12.924 | 14.047 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 32 | `nmrom_K8_q16_dense` | 16 | 4.5969 | 10.5655 | 17.5570 | 17.5570 | 10.5096 | 13.681 | 14.703 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K8_q16_eq` | 16 | 68.6507 | 74.3537 | 74.6838 | 74.6838 | 74.3212 | 14.319 | 15.411 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 32 | `nmrom_K8_q32_dense` | 16 | 4.1820 | 9.9107 | 16.6309 | 16.6309 | 9.8587 | 14.929 | 16.053 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K8_q32_dense_dt_half` | 16 | 4.1838 | 9.9035 | 16.6309 | 16.6309 | 9.8513 | 22.981 | 23.974 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K8_q32_eq` | 16 | 68.5743 | 74.4527 | 74.7857 | 74.7857 | 74.4206 | 14.645 | 15.691 | 0 / 0 | 1 / 48 | failed EQ certificate |
| 32 | `nmrom_K8_q8_dense` | 16 | 5.1616 | 11.5299 | 19.7313 | 19.7313 | 11.4473 | 13.069 | 14.128 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K8_q8_eq` | 16 | 68.7541 | 74.3825 | 74.6208 | 74.6208 | 74.3428 | 13.907 | 14.963 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 32 | `pod16_exact` | 16 | 8.6384 | 25.2986 | 36.4964 | 36.4964 | 25.1430 | 0.127 | 2.083 | 0 / 0 | 2 / 48 | development |
| 32 | `pod24_exact` | 16 | 5.5732 | 11.6555 | 19.6057 | 19.6057 | 11.6993 | 0.134 | 2.206 | 0 / 0 | 7 / 48 | development |
| 32 | `pod32_exact` | 16 | 3.0942 | 9.4366 | 16.6338 | 16.6338 | 9.3574 | 0.134 | 2.091 | 0 / 0 | 12 / 48 | development |
| 32 | `pod40_exact` | 16 | 2.2006 | 7.5846 | 13.3221 | 13.3221 | 7.5508 | 0.129 | 2.106 | 0 / 0 | 7 / 48 | development |
| 32 | `pod48_exact` | 16 | 1.3691 | 4.8625 | 9.8946 | 9.8946 | 4.8286 | 0.133 | 2.178 | 0 / 0 | 7 / 48 | development |
| 32 | `pod64_exact` | 16 | 0.8393 | 3.5288 | 6.8345 | 6.8345 | 3.5194 | 0.136 | 2.199 | 0 / 0 | 6 / 48 | development |
| 32 | `pod8_exact` | 16 | 19.9122 | 39.3251 | 50.6560 | 50.6560 | 39.1660 | 0.131 | 2.166 | 0 / 0 | 11 / 48 | development |
| 64 | `dst_exact` | 16 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1273 | 1.186 | 4.521 | 0 / 0 | 1 / 48 | development |
| 64 | `linear_bank_cn` | 16 | 4.0861 | 9.6921 | 16.5981 | 16.5981 | 9.6796 | 0.793 | 4.578 | 0 / 0 | 0 / 48 | development |
| 64 | `linear_bank_exact` | 16 | 4.0858 | 9.6892 | 16.5981 | 16.5981 | 9.6766 | 0.802 | 4.595 | 0 / 0 | 1 / 48 | development |
| 64 | `linear_bank_galerkin_exact` | 16 | 4.1619 | 9.5951 | 15.3816 | 15.3816 | 9.5681 | 0.298 | 4.094 | 0 / 0 | 8 / 48 | development |
| 64 | `nmrom_K16_q0_dense` | 16 | 4.8717 | 11.8659 | 19.5036 | 19.5036 | 11.8463 | 17.588 | 20.750 | 0 / 0 | 3 / 48 | development |
| 64 | `nmrom_K16_q0_eq` | 16 | 86.8397 | 91.4002 | 91.9654 | 91.9654 | 91.3961 | 21.171 | 24.513 | 0 / 2 | 0 / 48 | failed EQ certificate; stopping failures |
| 64 | `nmrom_K16_q16_dense` | 16 | 4.2825 | 9.7771 | 16.6085 | 16.6085 | 9.7612 | 19.599 | 22.857 | 0 / 0 | 3 / 48 | development |
| 64 | `nmrom_K16_q16_eq` | 16 | 87.2661 | 91.5172 | 92.1770 | 92.1770 | 91.5130 | 22.213 | 25.439 | 0 / 1 | 0 / 48 | failed EQ certificate; stopping failures |
| 64 | `nmrom_K16_q32_dense` | 16 | 4.0851 | 9.6681 | 16.1038 | 16.1038 | 9.6543 | 22.852 | 26.041 | 0 / 0 | 2 / 48 | development |
| 64 | `nmrom_K16_q32_dense_dt_half` | 16 | 4.0799 | 9.6643 | 16.1038 | 16.1038 | 9.6505 | 34.799 | 38.114 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K16_q32_eq` | 16 | 87.0579 | 91.5605 | 92.2312 | 92.2312 | 91.5565 | 21.810 | 24.991 | 0 / 1 | 3 / 48 | failed EQ certificate; stopping failures |
| 64 | `nmrom_K16_q8_dense` | 16 | 4.4524 | 10.8042 | 17.8496 | 17.8496 | 10.7867 | 17.966 | 21.226 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K16_q8_eq` | 16 | 87.0575 | 91.4660 | 91.9430 | 91.9430 | 91.4620 | 21.857 | 25.062 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 64 | `nmrom_K8_q0_dense` | 16 | 5.7792 | 15.7870 | 26.0910 | 26.0910 | 15.7536 | 12.516 | 15.710 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K8_q0_eq` | 16 | 87.1870 | 90.7488 | 91.5126 | 91.5126 | 90.7445 | 14.171 | 17.430 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 64 | `nmrom_K8_q16_dense` | 16 | 4.6023 | 10.5048 | 17.5573 | 17.5573 | 10.4902 | 13.547 | 16.697 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K8_q16_eq` | 16 | 87.5606 | 91.0330 | 91.8275 | 91.8275 | 91.0283 | 14.445 | 17.578 | 0 / 0 | 3 / 48 | failed EQ certificate |
| 64 | `nmrom_K8_q32_dense` | 16 | 4.1915 | 9.8501 | 16.6312 | 16.6312 | 9.8362 | 15.260 | 18.452 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K8_q32_dense_dt_half` | 16 | 4.1934 | 9.8425 | 16.6312 | 16.6312 | 9.8285 | 22.857 | 26.156 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K8_q32_eq` | 16 | 87.5203 | 91.2574 | 91.8283 | 91.8283 | 91.2531 | 16.218 | 19.545 | 0 / 0 | 3 / 48 | failed EQ certificate |
| 64 | `nmrom_K8_q8_dense` | 16 | 5.1675 | 11.4515 | 19.7315 | 19.7315 | 11.4304 | 13.209 | 16.457 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K8_q8_eq` | 16 | 87.5568 | 90.9647 | 91.6837 | 91.6837 | 90.9601 | 14.817 | 17.991 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 64 | `pod16_exact` | 16 | 8.6633 | 25.1877 | 36.5082 | 36.5082 | 25.1487 | 0.177 | 3.935 | 0 / 0 | 10 / 48 | development |
| 64 | `pod24_exact` | 16 | 5.5947 | 11.7352 | 19.6233 | 19.6233 | 11.7442 | 0.203 | 3.944 | 0 / 0 | 6 / 48 | development |
| 64 | `pod32_exact` | 16 | 3.0811 | 9.3730 | 16.6440 | 16.6440 | 9.3524 | 0.224 | 4.055 | 0 / 0 | 5 / 48 | development |
| 64 | `pod40_exact` | 16 | 2.1906 | 7.5604 | 13.4307 | 13.4307 | 7.5504 | 0.242 | 4.072 | 0 / 0 | 7 / 48 | development |
| 64 | `pod48_exact` | 16 | 1.3733 | 4.8685 | 9.9616 | 9.9616 | 4.8571 | 0.252 | 4.024 | 0 / 0 | 4 / 48 | development |
| 64 | `pod64_exact` | 16 | 0.8208 | 3.5334 | 6.8850 | 6.8850 | 3.5268 | 0.294 | 4.087 | 0 / 0 | 7 / 48 | development |
| 64 | `pod8_exact` | 16 | 20.0080 | 39.1939 | 50.6608 | 50.6608 | 39.1541 | 0.153 | 3.931 | 0 / 0 | 9 / 48 | development |

A missing physical-error or total-time cell means unmeasured, not zero. Physical errors require the attempt's separate reference-refinement qualification. A passing numerical audit verifies the recorded experiment; it does not establish good predictive accuracy, convergence of training or a competitive method.

## Poisson 3D — pilot01

**Provisional:** Development pilot with matched training membership. Bank/head training is incomplete; operator comparisons, independent training seeds and final evaluation are pending. Sampled quadrature arms failed their certificates and remain failure diagnostics. Selected latent states were not saved; stationarity is supported by the recorded stopping values and internal consistency only, not an independent weak-gradient reconstruction. Finer-grid POD is rebuilt from the same training members and is a mesh-adapted classical control.

Source `7158a27ab493bc1a5d0a8adefe13d4c20933a591`; job `3989715`; GPU `NVIDIA A100 80GB PCIe`. [Invocation data](../worktrees/2026-09-20-paper-p3d/experiments/paper-p3d/runs/pilot01/archive/out/result.json) and [independent audit](../worktrees/2026-09-20-paper-p3d/experiments/paper-p3d/runs/pilot01/audit-local.json).

Errors use the reference solution norm. There is one stationary output field; evolved, initial and all-times terminology does not apply. Total timing includes host transfers. Mesh size counts intervals per axis.

| Mesh | Method | Cases | Error median (%) | Error worst (%) | All-times worst (%) | Initial worst (%) | Physical worst (%) | GPU median (ms) | Total median (ms) | Nonfinite / nonstationary cases | Timing outliers / calls | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 32 | `dst_exact` | 16 | 0.0000 | 0.0000 | — | — | 0.2759 | 0.187 | 2.123 | 0 / 0 | 8 / 48 | development |
| 32 | `linear_bank_galerkin` | 16 | 2.2785 | 4.3629 | — | — | 4.2527 | 0.191 | 2.051 | 0 / 0 | 7 / 48 | development |
| 32 | `linear_bank_weak_qR` | 16 | 2.0891 | 4.1688 | — | — | 4.1525 | 0.211 | 1.991 | 0 / 0 | 5 / 48 | development |
| 32 | `nmrom_K16_q0_dense` | 16 | 3.6360 | 6.4366 | — | — | 6.4050 | 5.149 | 6.166 | 0 / 0 | 3 / 48 | development |
| 32 | `nmrom_K16_q0_eq` | 16 | 64.1649 | 69.1550 | — | — | 69.1072 | 5.687 | 6.482 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 32 | `nmrom_K16_q16_dense` | 16 | 2.2697 | 4.9518 | — | — | 4.9220 | 4.903 | 5.802 | 0 / 0 | 9 / 48 | development |
| 32 | `nmrom_K16_q16_eq` | 16 | 64.1598 | 69.1257 | — | — | 69.0788 | 4.501 | 5.476 | 0 / 0 | 14 / 48 | failed EQ certificate |
| 32 | `nmrom_K16_q32_dense` | 16 | 2.1142 | 4.4918 | — | — | 4.4721 | 4.946 | 5.792 | 0 / 0 | 13 / 48 | development |
| 32 | `nmrom_K16_q32_eq` | 16 | 64.1553 | 69.1243 | — | — | 69.0774 | 5.974 | 6.845 | 0 / 0 | 6 / 48 | failed EQ certificate |
| 32 | `nmrom_K16_q8_dense` | 16 | 2.6221 | 5.3206 | — | — | 5.2903 | 5.007 | 6.008 | 0 / 0 | 1 / 48 | development |
| 32 | `nmrom_K16_q8_eq` | 16 | 64.1628 | 69.1239 | — | — | 69.0766 | 4.812 | 5.747 | 0 / 0 | 8 / 48 | failed EQ certificate |
| 32 | `nmrom_K8_q0_dense` | 16 | 3.5283 | 8.1168 | — | — | 8.0825 | 4.999 | 6.006 | 0 / 0 | 9 / 48 | development |
| 32 | `nmrom_K8_q0_eq` | 16 | 74.9776 | 78.5647 | — | — | 78.5331 | 5.329 | 6.171 | 0 / 0 | 6 / 48 | failed EQ certificate |
| 32 | `nmrom_K8_q16_dense` | 16 | 2.2499 | 4.8099 | — | — | 4.7760 | 4.011 | 4.859 | 0 / 0 | 6 / 48 | development |
| 32 | `nmrom_K8_q16_eq` | 16 | 74.9474 | 78.3614 | — | — | 78.3285 | 5.176 | 6.105 | 0 / 0 | 3 / 48 | failed EQ certificate |
| 32 | `nmrom_K8_q32_dense` | 16 | 2.1064 | 4.4237 | — | — | 4.3964 | 3.935 | 4.982 | 0 / 0 | 3 / 48 | development |
| 32 | `nmrom_K8_q32_eq` | 16 | 74.9665 | 78.3801 | — | — | 78.3472 | 3.867 | 4.797 | 0 / 0 | 6 / 48 | failed EQ certificate |
| 32 | `nmrom_K8_q8_dense` | 16 | 2.7125 | 5.4715 | — | — | 5.4149 | 4.183 | 5.078 | 0 / 0 | 6 / 48 | development |
| 32 | `nmrom_K8_q8_eq` | 16 | 74.9387 | 78.3954 | — | — | 78.3624 | 5.131 | 6.097 | 0 / 0 | 5 / 48 | failed EQ certificate |
| 32 | `pod16_galerkin` | 16 | 1.8426 | 5.3174 | — | — | 5.2258 | 0.180 | 2.004 | 0 / 0 | 4 / 48 | development |
| 32 | `pod24_galerkin` | 16 | 0.6949 | 2.9930 | — | — | 2.9595 | 0.178 | 2.065 | 0 / 0 | 2 / 48 | development |
| 32 | `pod32_galerkin` | 16 | 0.4246 | 1.7517 | — | — | 1.7524 | 0.182 | 1.981 | 0 / 0 | 4 / 48 | development |
| 32 | `pod40_galerkin` | 16 | 0.2412 | 0.8478 | — | — | 0.8874 | 0.196 | 1.986 | 0 / 0 | 5 / 48 | development |
| 32 | `pod48_galerkin` | 16 | 0.1506 | 0.5672 | — | — | 0.6383 | 0.197 | 2.119 | 0 / 0 | 5 / 48 | development |
| 32 | `pod64_galerkin` | 16 | 0.0654 | 0.2617 | — | — | 0.3826 | 0.194 | 2.010 | 0 / 0 | 3 / 48 | development |
| 32 | `pod8_galerkin` | 16 | 4.8920 | 8.8900 | — | — | 8.8274 | 0.176 | 2.009 | 0 / 0 | 0 / 48 | development |

A missing physical-error or total-time cell means unmeasured, not zero. Physical errors require the attempt's separate reference-refinement qualification. A passing numerical audit verifies the recorded experiment; it does not establish good predictive accuracy, convergence of training or a competitive method.

## Navier–Stokes 3D — pilot01

**Provisional:** Representation-only development pilot: the small learned bank and POD controls have large held-out errors. These are fits to known reference snapshots, not predicted trajectories or timed solution queries. Larger banks, more data, operator baselines and final evaluation are pending. The pretraining amplitude amendment and failed original resolution setting remain documented in the lane design.

Source `72ddfef3c53708f4dce68ab44e6d542eb08c33cb`; job `3989800`; GPU `NVIDIA A100 80GB PCIe, 81920 MiB`. [Records](../worktrees/2026-09-20-paper-ns3d/experiments/ns3d/runs/pilot01/collected/output/result.json) and [independent audit](../worktrees/2026-09-20-paper-ns3d/experiments/ns3d/runs/pilot01/collected/audit.json).

The table contains 16 validation trajectories on a 32³ periodic grid, with 6 snapshots per trajectory. Snapshots from the same trajectory are correlated. Errors here use each snapshot's own velocity norm; they must not be confused with the campaign's initial-normalized predicted-trajectory metric.

| Representation | Snapshots | Median error (%) | Worst error (%) | Snapshots above declared target | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Learned bank R64 projection | 96 | 64.4573 | 90.3634 | 96 | failed representation target |
| Neural head K8 best-found fit | 96 | 68.2681 | 99.4488 | 96 | failed representation target |
| POD-8 projection | 96 | 93.9016 | 99.2008 | 96 | failed representation target |
| POD-16 projection | 96 | 87.2185 | 98.1796 | 96 | failed representation target |
| POD-24 projection | 96 | 80.6187 | 97.7022 | 96 | failed representation target |
| POD-40 projection | 96 | 69.5360 | 96.5670 | 96 | failed representation target |
| POD-64 projection | 96 | 59.6095 | 91.9224 | 96 | failed representation target |

The empirical reference refinement gate passes: worst discrepancy 0.3871% against a declared 0.5000% budget. Passing the numerical reference checks does not remedy the representation failures above. No rollout error, runtime or operator comparison is inferred from these snapshot fits.

## Poisson 3D — tuned02

**Provisional:** Audited development comparison with matched training membership. Dense correction ladders improve same-grid accuracy, but POD and DST remain stronger controls. Native-grid operator prediction with interpolation and direct resolution transfer are separate methods; direct transfer fails. Sampled quadrature certificates fail. These implementation timings include avoidable static head-parameter transfers; the next paired panel places all frozen weights on the GPU during setup. Independent retraining and final evaluation remain pending. Selected latent states were not saved; stationarity is supported by the recorded stopping values and internal consistency only, not an independent weak-gradient reconstruction. Finer-grid POD is rebuilt from the same training members and is a mesh-adapted classical control.

Source `68fd0cca50c44b8c866986d1fcbf44d9ee8b4c1b`; job `3990494`; GPU `NVIDIA A100 80GB PCIe`. [Invocation data](../worktrees/2026-09-20-paper-p3d/experiments/paper-p3d/runs/tuned02/archive/out/result.json) and [independent audit](../worktrees/2026-09-20-paper-p3d/experiments/paper-p3d/runs/tuned02/audit-local.json).

Errors use the reference solution norm. There is one stationary output field; evolved, initial and all-times terminology does not apply. Total timing includes host transfers. Mesh size counts intervals per axis.

| Mesh | Method | Cases | Error median (%) | Error worst (%) | All-times worst (%) | Initial worst (%) | Physical worst (%) | GPU median (ms) | Total median (ms) | Nonfinite / nonstationary cases | Timing outliers / calls | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 32 | `dst_exact` | 16 | 0.0000 | 0.0000 | — | — | 0.2759 | 0.183 | 2.056 | 0 / 0 | 9 / 48 | development |
| 32 | `fno3d_w24_m8` | 16 | 0.1502 | 0.2394 | — | — | 0.3379 | 11.787 | 12.578 | 0 / 0 | 0 / 48 | development |
| 32 | `linear_bank_galerkin` | 16 | 0.0505 | 0.1471 | — | — | 0.3114 | 0.218 | 2.079 | 0 / 0 | 2 / 48 | development |
| 32 | `linear_bank_weak_qR` | 16 | 0.0497 | 0.1450 | — | — | 0.3110 | 0.267 | 2.151 | 0 / 0 | 4 / 48 | development |
| 32 | `nmrom_K16_q0_dense` | 16 | 0.2070 | 0.5892 | — | — | 0.6145 | 2.902 | 3.687 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K16_q0_eq` | 16 | 0.2221 | 0.5939 | — | — | 0.6269 | 2.913 | 3.702 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 32 | `nmrom_K16_q16_dense` | 16 | 0.1427 | 0.4603 | — | — | 0.5314 | 3.052 | 3.848 | 0 / 0 | 1 / 48 | development |
| 32 | `nmrom_K16_q16_eq` | 16 | 0.2145 | 0.5074 | — | — | 0.5764 | 2.704 | 3.579 | 0 / 0 | 1 / 48 | failed EQ certificate |
| 32 | `nmrom_K16_q32_dense` | 16 | 0.1112 | 0.3284 | — | — | 0.3993 | 2.915 | 3.710 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K16_q32_eq` | 16 | 0.1932 | 0.3947 | — | — | 0.4854 | 2.908 | 3.750 | 0 / 0 | 1 / 48 | failed EQ certificate |
| 32 | `nmrom_K16_q64_dense` | 16 | 0.0669 | 0.2018 | — | — | 0.3393 | 3.011 | 3.839 | 0 / 0 | 1 / 48 | development |
| 32 | `nmrom_K16_q64_eq` | 16 | 0.2181 | 0.4824 | — | — | 0.5642 | 3.021 | 3.883 | 0 / 0 | 4 / 48 | failed EQ certificate |
| 32 | `nmrom_K16_q96_dense` | 16 | 0.0513 | 0.1605 | — | — | 0.3182 | 3.076 | 3.862 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K16_q96_eq` | 16 | 0.2684 | 0.5880 | — | — | 0.6552 | 4.432 | 5.288 | 0 / 0 | 9 / 48 | failed EQ certificate |
| 32 | `nmrom_K8_q0_dense` | 16 | 0.5633 | 1.7154 | — | — | 1.6730 | 2.664 | 3.513 | 0 / 0 | 3 / 48 | development |
| 32 | `nmrom_K8_q0_eq` | 16 | 0.5636 | 1.7161 | — | — | 1.6748 | 2.666 | 3.453 | 0 / 0 | 3 / 48 | failed EQ certificate |
| 32 | `nmrom_K8_q16_dense` | 16 | 0.3087 | 0.9815 | — | — | 0.9964 | 2.598 | 3.422 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K8_q16_eq` | 16 | 0.3266 | 0.9936 | — | — | 1.0109 | 2.575 | 3.492 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 32 | `nmrom_K8_q32_dense` | 16 | 0.1936 | 0.6150 | — | — | 0.6603 | 2.754 | 3.617 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K8_q32_eq` | 16 | 0.2356 | 0.6634 | — | — | 0.7060 | 2.663 | 3.447 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 32 | `nmrom_K8_q64_dense` | 16 | 0.0838 | 0.3434 | — | — | 0.4326 | 2.753 | 3.492 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K8_q64_eq` | 16 | 0.1783 | 0.5207 | — | — | 0.5883 | 2.753 | 3.561 | 0 / 0 | 1 / 48 | failed EQ certificate |
| 32 | `nmrom_K8_q96_dense` | 16 | 0.0560 | 0.2094 | — | — | 0.3433 | 2.835 | 3.702 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K8_q96_eq` | 16 | 0.2361 | 0.6410 | — | — | 0.7018 | 2.829 | 3.677 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 32 | `pod104_galerkin` | 16 | 0.0154 | 0.0776 | — | — | 0.2864 | 0.214 | 2.060 | 0 / 0 | 3 / 48 | development |
| 32 | `pod112_galerkin` | 16 | 0.0117 | 0.0685 | — | — | 0.2839 | 0.213 | 2.070 | 0 / 0 | 5 / 48 | development |
| 32 | `pod128_galerkin` | 16 | 0.0067 | 0.0370 | — | — | 0.2779 | 0.219 | 2.071 | 0 / 0 | 1 / 48 | development |
| 32 | `pod16_galerkin` | 16 | 1.8426 | 5.3174 | — | — | 5.2258 | 0.181 | 1.974 | 0 / 0 | 3 / 48 | development |
| 32 | `pod24_galerkin` | 16 | 0.6949 | 2.9930 | — | — | 2.9595 | 0.182 | 2.015 | 0 / 0 | 5 / 48 | development |
| 32 | `pod32_galerkin` | 16 | 0.4246 | 1.7517 | — | — | 1.7524 | 0.184 | 2.056 | 0 / 0 | 1 / 48 | development |
| 32 | `pod40_galerkin` | 16 | 0.2412 | 0.8478 | — | — | 0.8874 | 0.190 | 2.082 | 0 / 0 | 6 / 48 | development |
| 32 | `pod48_galerkin` | 16 | 0.1506 | 0.5672 | — | — | 0.6383 | 0.191 | 2.143 | 0 / 0 | 3 / 48 | development |
| 32 | `pod72_galerkin` | 16 | 0.0445 | 0.2366 | — | — | 0.3666 | 0.198 | 2.106 | 0 / 0 | 1 / 48 | development |
| 32 | `pod80_galerkin` | 16 | 0.0363 | 0.1980 | — | — | 0.3425 | 0.201 | 2.054 | 0 / 0 | 3 / 48 | development |
| 32 | `pod8_galerkin` | 16 | 4.8920 | 8.8900 | — | — | 8.8274 | 0.181 | 2.084 | 0 / 0 | 3 / 48 | development |
| 32 | `unet3d_w16` | 16 | 0.1326 | 0.1932 | — | — | 0.3357 | 4.586 | 5.384 | 0 / 0 | 0 / 48 | development |
| 64 | `dst_exact` | 16 | 0.0000 | 0.0000 | — | — | 0.0687 | 0.229 | 2.296 | 0 / 0 | 10 / 48 | development |
| 64 | `fno3d_w24_m8` | 16 | 16.0680 | 17.1866 | — | — | 17.2392 | 17.493 | 18.897 | 0 / 0 | 0 / 48 | development |
| 64 | `fno3d_w24_m8_native32_interpolate` | 16 | 0.2852 | 0.3656 | — | — | 0.3525 | 11.579 | 12.871 | 0 / 0 | 0 / 48 | development |
| 64 | `linear_bank_galerkin` | 16 | 0.0530 | 0.1474 | — | — | 0.1624 | 0.473 | 2.520 | 0 / 0 | 3 / 48 | development |
| 64 | `linear_bank_weak_qR` | 16 | 0.0520 | 0.1444 | — | — | 0.1599 | 0.894 | 2.938 | 0 / 0 | 1 / 48 | development |
| 64 | `nmrom_K16_q0_dense` | 16 | 0.2149 | 0.5726 | — | — | 0.5707 | 3.191 | 4.435 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K16_q0_eq` | 16 | 0.2610 | 0.6521 | — | — | 0.6523 | 2.643 | 3.935 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 64 | `nmrom_K16_q16_dense` | 16 | 0.1492 | 0.4580 | — | — | 0.4629 | 3.252 | 4.558 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K16_q16_eq` | 16 | 0.2748 | 0.6267 | — | — | 0.6302 | 2.838 | 4.149 | 0 / 0 | 1 / 48 | failed EQ certificate |
| 64 | `nmrom_K16_q32_dense` | 16 | 0.1159 | 0.3215 | — | — | 0.3252 | 3.230 | 4.585 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K16_q32_eq` | 16 | 0.2804 | 0.7197 | — | — | 0.7245 | 2.801 | 4.128 | 0 / 0 | 2 / 48 | failed EQ certificate |
| 64 | `nmrom_K16_q64_dense` | 16 | 0.0687 | 0.2001 | — | — | 0.2113 | 3.287 | 4.573 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K16_q64_eq` | 16 | 0.3607 | 0.8852 | — | — | 0.8906 | 3.066 | 4.341 | 0 / 0 | 7 / 48 | failed EQ certificate |
| 64 | `nmrom_K16_q96_dense` | 16 | 0.0539 | 0.1598 | — | — | 0.1739 | 3.282 | 4.598 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K16_q96_eq` | 16 | 0.4959 | 1.2060 | — | — | 1.2104 | 6.019 | 7.230 | 0 / 0 | 9 / 48 | failed EQ certificate |
| 64 | `nmrom_K8_q0_dense` | 16 | 0.5622 | 1.6683 | — | — | 1.6544 | 3.015 | 4.310 | 0 / 0 | 3 / 48 | development |
| 64 | `nmrom_K8_q0_eq` | 16 | 0.5653 | 1.6689 | — | — | 1.6553 | 2.453 | 3.733 | 0 / 0 | 3 / 48 | failed EQ certificate |
| 64 | `nmrom_K8_q16_dense` | 16 | 0.3133 | 0.9645 | — | — | 0.9617 | 3.214 | 4.511 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K8_q16_eq` | 16 | 0.3441 | 0.9913 | — | — | 0.9893 | 2.488 | 3.711 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 64 | `nmrom_K8_q32_dense` | 16 | 0.1969 | 0.6047 | — | — | 0.6055 | 3.092 | 4.326 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K8_q32_eq` | 16 | 0.2998 | 0.6795 | — | — | 0.6865 | 2.448 | 3.766 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 64 | `nmrom_K8_q64_dense` | 16 | 0.0860 | 0.3366 | — | — | 0.3415 | 3.022 | 4.318 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K8_q64_eq` | 16 | 0.3466 | 0.8552 | — | — | 0.8617 | 2.624 | 3.840 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 64 | `nmrom_K8_q96_dense` | 16 | 0.0566 | 0.2067 | — | — | 0.2170 | 3.134 | 4.367 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K8_q96_eq` | 16 | 0.4525 | 1.1390 | — | — | 1.1440 | 2.811 | 4.094 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 64 | `pod104_galerkin` | 16 | 0.0150 | 0.0761 | — | — | 0.1024 | 0.427 | 2.569 | 0 / 0 | 7 / 48 | development |
| 64 | `pod112_galerkin` | 16 | 0.0115 | 0.0672 | — | — | 0.0958 | 0.446 | 2.556 | 0 / 0 | 4 / 48 | development |
| 64 | `pod128_galerkin` | 16 | 0.0066 | 0.0360 | — | — | 0.0771 | 0.465 | 2.601 | 0 / 0 | 5 / 48 | development |
| 64 | `pod16_galerkin` | 16 | 1.8304 | 5.2702 | — | — | 5.2463 | 0.206 | 2.410 | 0 / 0 | 8 / 48 | development |
| 64 | `pod24_galerkin` | 16 | 0.6890 | 2.9688 | — | — | 2.9582 | 0.226 | 2.320 | 0 / 0 | 5 / 48 | development |
| 64 | `pod32_galerkin` | 16 | 0.4204 | 1.7372 | — | — | 1.7334 | 0.246 | 2.289 | 0 / 0 | 4 / 48 | development |
| 64 | `pod40_galerkin` | 16 | 0.2383 | 0.8384 | — | — | 0.8402 | 0.267 | 2.235 | 0 / 0 | 1 / 48 | development |
| 64 | `pod48_galerkin` | 16 | 0.1484 | 0.5576 | — | — | 0.5639 | 0.283 | 2.398 | 0 / 0 | 7 / 48 | development |
| 64 | `pod72_galerkin` | 16 | 0.0438 | 0.2329 | — | — | 0.2440 | 0.345 | 2.486 | 0 / 0 | 2 / 48 | development |
| 64 | `pod80_galerkin` | 16 | 0.0357 | 0.1951 | — | — | 0.2080 | 0.357 | 2.480 | 0 / 0 | 3 / 48 | development |
| 64 | `pod8_galerkin` | 16 | 4.8633 | 8.8440 | — | — | 8.8280 | 0.198 | 2.336 | 0 / 0 | 11 / 48 | development |
| 64 | `unet3d_w16` | 16 | 168.0982 | 219.8928 | — | — | 220.0031 | 11.985 | 13.358 | 0 / 0 | 0 / 48 | development |
| 64 | `unet3d_w16_native32_interpolate` | 16 | 0.2828 | 0.3769 | — | — | 0.3562 | 4.328 | 5.648 | 0 / 0 | 0 / 48 | development |

A missing physical-error or total-time cell means unmeasured, not zero. Physical errors require the attempt's separate reference-refinement qualification. A passing numerical audit verifies the recorded experiment; it does not establish good predictive accuracy, convergence of training or a competitive method.

## Burgers 3D — b3d003

**Provisional:** Audited matched-training development comparison. All learned models, POD and directions use the same trajectories and eight saved training times. Operator predictions pass through the exact initial field and interpolate the predicted time knots to the requested dense trajectory; their interpolation control is retained. The fixed-head correction ladder improves accuracy but the learned head still has a substantial held-out representation gap. The physical refinement check covers only two development cases and does not establish a full-cohort physical accuracy claim. Independent retraining and final evaluation remain pending.

Source `dc91edb2c81712fde9d1b05fec7028c47bd213b4`; job `3992083`; GPU `NVIDIA A100 80GB PCIe`. [Invocation data](../worktrees/2026-09-20-paper-b3d/experiments/paper-b3d/runs/b3d003/collected/out/result.json) and [independent audit](../worktrees/2026-09-20-paper-b3d/experiments/paper-b3d/runs/b3d003/collected/out/audit-local.json).

Errors use the initial-field norm. The evolved error excludes the initial compression; all-times and initial errors are also shown. GPU timing begins with the dense input already on device and ends with every requested dense output on device. Host transfers are unmeasured in this attempt. Mesh size counts nodes per axis.

| Mesh | Method | Cases | Error median (%) | Error worst (%) | All-times worst (%) | Initial worst (%) | Physical worst (%) | GPU median (ms) | Total median (ms) | Nonfinite / nonstationary cases | Timing outliers / calls | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 33 | `fno3d` | 8 | 1.6413 | 1.9979 | 1.9979 | 0.0000 | — | 5.959 | — | 0 / 0 | 0 / 24 | development |
| 33 | `fom_nt1e-02_lt5e-01` | 8 | 1.2739 | 1.6185 | 1.6185 | 0.0000 | — | 16.141 | — | 0 / 0 | 0 / 24 | development |
| 33 | `fom_nt1e-02_lt5e-03` | 8 | 0.0382 | 0.6543 | 0.6543 | 0.0000 | — | 18.239 | — | 0 / 0 | 0 / 24 | development |
| 33 | `fom_nt1e-03_lt5e-01` | 8 | 0.4692 | 0.5248 | 0.5248 | 0.0000 | — | 19.954 | — | 0 / 0 | 0 / 24 | development |
| 33 | `fom_nt1e-04_lt1e-06` | 8 | 0.0188 | 0.0207 | 0.0207 | 0.0000 | — | 37.326 | — | 0 / 0 | 0 / 24 | development |
| 33 | `fom_nt1e-04_lt5e-01` | 8 | 0.0267 | 0.0303 | 0.0303 | 0.0000 | — | 32.302 | — | 0 / 0 | 0 / 24 | development |
| 33 | `fom_nt1e-06_lt1e-01` | 8 | 0.0002 | 0.0003 | 0.0003 | 0.0000 | — | 48.915 | — | 0 / 0 | 0 / 24 | development |
| 33 | `free_R256` | 8 | 0.8437 | 2.4345 | 2.8901 | 2.8901 | — | 316.894 | — | 0 / 0 | 0 / 24 | development |
| 33 | `pod_160` | 8 | 0.7675 | 2.5057 | 3.0153 | 3.0153 | — | 224.232 | — | 0 / 0 | 0 / 24 | development |
| 33 | `pod_224` | 8 | 0.3988 | 1.2100 | 1.5306 | 1.5306 | — | 284.363 | — | 0 / 0 | 0 / 24 | development |
| 33 | `pod_256` | 8 | 0.3292 | 1.0363 | 1.2631 | 1.2631 | — | 315.911 | — | 0 / 0 | 0 / 24 | development |
| 33 | `pod_32` | 8 | 9.5224 | 22.6947 | 24.9424 | 24.9424 | — | 95.765 | — | 0 / 0 | 0 / 24 | development |
| 33 | `pod_96` | 8 | 1.9855 | 5.5785 | 6.6983 | 6.6983 | — | 150.552 | — | 0 / 0 | 0 / 24 | development |
| 33 | `rom_q0` | 8 | 5.7071 | 10.8780 | 11.7177 | 11.7177 | — | 185.964 | — | 0 / 0 | 0 / 24 | development |
| 33 | `rom_q128` | 8 | 3.3258 | 6.6491 | 7.1187 | 7.1187 | — | 352.971 | — | 0 / 0 | 0 / 24 | development |
| 33 | `rom_q192` | 8 | 1.3600 | 3.4908 | 3.9087 | 3.9087 | — | 417.752 | — | 0 / 0 | 0 / 24 | development |
| 33 | `rom_q64` | 8 | 4.9359 | 8.9108 | 9.4908 | 9.4908 | — | 267.228 | — | 0 / 0 | 0 / 24 | development |
| 33 | `unet3d` | 8 | 1.6239 | 1.9129 | 1.9129 | 0.0000 | — | 2.242 | — | 0 / 0 | 0 / 24 | development |

A missing physical-error or total-time cell means unmeasured, not zero. Physical errors require the attempt's separate reference-refinement qualification. A passing numerical audit verifies the recorded experiment; it does not establish good predictive accuracy, convergence of training or a competitive method.

## Heat 3D — tune02

**Provisional:** Audited matched-training development comparison. The spatial bank improved but the nonlinear head has a substantial held-out representation gap; a few initial fits exhaust the declared iteration budget, while evolved solves are stationary. Free-bank/POD controls and neural operators remain stronger than the nonlinear head. Native-grid prediction plus interpolation is distinct from failed direct operator resolution transfer. All sampled quadrature certificates fail. A larger-head candidate, independent training seed and final evaluation remain pending. Finer-grid POD is rebuilt from the same training members and is a mesh-adapted classical control.

Source `4a93e5868acf83cbe07d84e6c451dc50ffbd9ea2`; job `3990698`; GPU `NVIDIA A100 80GB PCIe`. [Invocation data](../worktrees/2026-09-20-paper-h3d/experiments/paper-h3d/runs/tune02/archive/out/result.json) and [independent audit](../worktrees/2026-09-20-paper-h3d/experiments/paper-h3d/runs/tune02/audit-local.json).

Errors use each reference field's current norm. Evolved errors exclude the initial state. Total timing includes host transfers; GPU timing includes initialization, evolution and dense output. Mesh size counts intervals per axis.

| Mesh | Method | Cases | Error median (%) | Error worst (%) | All-times worst (%) | Initial worst (%) | Physical worst (%) | GPU median (ms) | Total median (ms) | Nonfinite / nonstationary cases | Timing outliers / calls | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 32 | `dst_exact` | 16 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5119 | 1.319 | 2.694 | 0 / 0 | 0 / 48 | development |
| 32 | `fno3d_w16_m6` | 16 | 0.2449 | 0.3947 | 0.3947 | 0.0000 | 0.6332 | 7.242 | 8.236 | 0 / 0 | 0 / 48 | development |
| 32 | `linear_bank_cn` | 16 | 0.2591 | 1.3315 | 3.0636 | 3.0636 | 1.3507 | 0.504 | 2.786 | 0 / 0 | 5 / 48 | development |
| 32 | `linear_bank_exact` | 16 | 0.2549 | 1.3277 | 3.0636 | 3.0636 | 1.3655 | 0.568 | 2.743 | 0 / 0 | 7 / 48 | development |
| 32 | `linear_bank_galerkin_exact` | 16 | 0.2568 | 1.3324 | 3.0588 | 3.0588 | 1.3607 | 0.157 | 2.354 | 0 / 0 | 5 / 48 | development |
| 32 | `nmrom_K16_q0_dense` | 16 | 1.0782 | 5.8855 | 11.0119 | 11.0119 | 5.8118 | 17.226 | 18.218 | 0 / 2 | 6 / 48 | stopping failures |
| 32 | `nmrom_K16_q0_eq` | 16 | 1.7106 | 7.3453 | 12.2349 | 12.2349 | 7.1286 | 17.620 | 18.797 | 0 / 1 | 5 / 48 | failed EQ certificate; stopping failures |
| 32 | `nmrom_K16_q32_dense` | 16 | 0.8696 | 4.9328 | 9.2331 | 9.2331 | 4.8607 | 18.406 | 19.573 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K16_q32_eq` | 16 | 1.7638 | 6.5738 | 11.9167 | 11.9167 | 6.3634 | 18.497 | 19.550 | 0 / 1 | 3 / 48 | failed EQ certificate; stopping failures |
| 32 | `nmrom_K16_q64_dense` | 16 | 0.5815 | 4.0310 | 7.5745 | 7.5745 | 3.9835 | 19.531 | 20.541 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K16_q64_eq` | 16 | 1.8174 | 6.1609 | 11.9241 | 11.9241 | 5.9453 | 20.952 | 22.045 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 32 | `nmrom_K16_q8_dense` | 16 | 1.0381 | 5.3133 | 10.2925 | 10.2925 | 5.2396 | 18.065 | 19.188 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K16_q8_eq` | 16 | 1.7395 | 6.7804 | 11.4408 | 11.4408 | 6.5705 | 18.703 | 19.720 | 0 / 1 | 3 / 48 | failed EQ certificate; stopping failures |
| 32 | `nmrom_K16_q96_dense` | 16 | 0.3673 | 2.9899 | 5.4636 | 5.4636 | 2.9665 | 23.817 | 24.901 | 0 / 1 | 3 / 48 | stopping failures |
| 32 | `nmrom_K16_q96_dense_dt_half` | 16 | 0.3648 | 2.9260 | 5.4636 | 5.4636 | 2.9091 | 36.514 | 37.688 | 0 / 1 | 0 / 48 | stopping failures |
| 32 | `nmrom_K16_q96_eq` | 16 | 1.8085 | 5.4484 | 11.8278 | 11.8278 | 5.2358 | 23.846 | 25.012 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 32 | `nmrom_K8_q0_dense` | 16 | 1.3934 | 6.1977 | 11.0875 | 11.0875 | 6.1354 | 12.967 | 13.982 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K8_q0_eq` | 16 | 1.9458 | 7.3560 | 12.6302 | 12.6302 | 7.1537 | 12.794 | 13.996 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 32 | `nmrom_K8_q32_dense` | 16 | 0.9327 | 4.9721 | 9.3116 | 9.3116 | 4.9193 | 14.352 | 15.462 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K8_q32_eq` | 16 | 1.8236 | 6.4978 | 11.8833 | 11.8833 | 6.2981 | 13.828 | 15.019 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 32 | `nmrom_K8_q64_dense` | 16 | 0.6970 | 3.7752 | 7.2769 | 7.2769 | 3.7427 | 14.452 | 15.485 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K8_q64_eq` | 16 | 1.8554 | 5.9447 | 11.7466 | 11.7466 | 5.7428 | 14.174 | 15.220 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 32 | `nmrom_K8_q8_dense` | 16 | 1.1850 | 6.0419 | 10.8848 | 10.8848 | 5.9761 | 13.513 | 14.579 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K8_q8_eq` | 16 | 1.8845 | 7.2247 | 12.4862 | 12.4862 | 7.0177 | 13.526 | 14.621 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 32 | `nmrom_K8_q96_dense` | 16 | 0.4060 | 2.2781 | 4.6850 | 4.6850 | 2.2724 | 15.397 | 16.471 | 0 / 1 | 3 / 48 | stopping failures |
| 32 | `nmrom_K8_q96_dense_dt_half` | 16 | 0.4053 | 2.2742 | 4.6850 | 4.6850 | 2.2770 | 24.010 | 25.150 | 0 / 1 | 0 / 48 | stopping failures |
| 32 | `nmrom_K8_q96_eq` | 16 | 1.7987 | 5.4736 | 11.4307 | 11.4307 | 5.2615 | 16.707 | 17.734 | 0 / 0 | 3 / 48 | failed EQ certificate |
| 32 | `pod104_exact` | 16 | 0.2697 | 1.6482 | 3.3966 | 3.3966 | 1.6743 | 0.153 | 2.255 | 0 / 0 | 5 / 48 | development |
| 32 | `pod112_exact` | 16 | 0.2097 | 1.3665 | 3.0837 | 3.0837 | 1.3938 | 0.152 | 2.221 | 0 / 0 | 7 / 48 | development |
| 32 | `pod128_exact` | 16 | 0.1431 | 1.2441 | 2.6741 | 2.6741 | 1.2806 | 0.154 | 2.250 | 0 / 0 | 5 / 48 | development |
| 32 | `pod16_exact` | 16 | 8.6384 | 25.2986 | 36.4964 | 36.4964 | 25.1430 | 0.130 | 2.277 | 0 / 0 | 8 / 48 | development |
| 32 | `pod24_exact` | 16 | 5.5732 | 11.6555 | 19.6057 | 19.6057 | 11.6993 | 0.131 | 2.214 | 0 / 0 | 8 / 48 | development |
| 32 | `pod40_exact` | 16 | 2.2006 | 7.5846 | 13.3221 | 13.3221 | 7.5508 | 0.134 | 2.300 | 0 / 0 | 8 / 48 | development |
| 32 | `pod48_exact` | 16 | 1.3691 | 4.8625 | 9.8946 | 9.8946 | 4.8286 | 0.131 | 2.183 | 0 / 0 | 4 / 48 | development |
| 32 | `pod72_exact` | 16 | 0.6934 | 3.2741 | 6.1329 | 6.1329 | 3.2533 | 0.138 | 2.362 | 0 / 0 | 5 / 48 | development |
| 32 | `pod80_exact` | 16 | 0.4948 | 2.6899 | 5.4338 | 5.4338 | 2.6768 | 0.145 | 2.296 | 0 / 0 | 8 / 48 | development |
| 32 | `pod8_exact` | 16 | 19.9122 | 39.3251 | 50.6560 | 50.6560 | 39.1660 | 0.131 | 2.215 | 0 / 0 | 7 / 48 | development |
| 32 | `unet3d_w8` | 16 | 0.1697 | 0.3185 | 0.3185 | 0.0000 | 0.5321 | 3.563 | 4.590 | 0 / 0 | 0 / 48 | development |
| 64 | `dst_exact` | 16 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1273 | 1.329 | 4.708 | 0 / 0 | 1 / 48 | development |
| 64 | `fno3d_w16_m6` | 16 | 15.1234 | 17.6641 | 17.6641 | 0.0000 | 17.7688 | 10.981 | 14.020 | 0 / 0 | 0 / 48 | development |
| 64 | `fno3d_w16_m6_native_grid_interpolated` | 16 | 0.7669 | 0.9891 | 0.9891 | 0.0000 | 0.9391 | 7.140 | 10.252 | 0 / 0 | 0 / 48 | development |
| 64 | `linear_bank_cn` | 16 | 0.2562 | 1.3155 | 3.0639 | 3.0639 | 1.3079 | 1.301 | 5.144 | 0 / 0 | 1 / 48 | development |
| 64 | `linear_bank_exact` | 16 | 0.2542 | 1.3113 | 3.0639 | 3.0639 | 1.3087 | 1.421 | 5.382 | 0 / 0 | 0 / 48 | development |
| 64 | `linear_bank_galerkin_exact` | 16 | 0.2563 | 1.3166 | 3.0591 | 3.0591 | 1.3115 | 0.514 | 4.369 | 0 / 0 | 3 / 48 | development |
| 64 | `nmrom_K16_q0_dense` | 16 | 1.0804 | 5.8427 | 11.0119 | 11.0119 | 5.8223 | 18.392 | 21.669 | 0 / 2 | 6 / 48 | stopping failures |
| 64 | `nmrom_K16_q0_eq` | 16 | 3.3932 | 6.6841 | 11.2389 | 11.2389 | 6.6503 | 18.262 | 21.567 | 0 / 0 | 3 / 48 | failed EQ certificate |
| 64 | `nmrom_K16_q32_dense` | 16 | 0.8752 | 4.8852 | 9.2332 | 9.2332 | 4.8649 | 19.608 | 22.856 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K16_q32_eq` | 16 | 3.3169 | 6.0002 | 9.9017 | 9.9017 | 5.9690 | 18.941 | 22.156 | 0 / 1 | 3 / 48 | failed EQ certificate; stopping failures |
| 64 | `nmrom_K16_q64_dense` | 16 | 0.5892 | 3.9930 | 7.5746 | 7.5746 | 3.9779 | 20.528 | 23.742 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K16_q64_eq` | 16 | 3.1830 | 5.8451 | 9.4854 | 9.4854 | 5.8145 | 22.122 | 25.414 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 64 | `nmrom_K16_q8_dense` | 16 | 1.0426 | 5.2535 | 10.2925 | 10.2925 | 5.2332 | 18.294 | 21.583 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K16_q8_eq` | 16 | 3.3969 | 6.2816 | 10.6440 | 10.6440 | 6.2508 | 18.284 | 21.818 | 0 / 2 | 9 / 48 | failed EQ certificate; stopping failures |
| 64 | `nmrom_K16_q96_dense` | 16 | 0.3697 | 2.9628 | 5.4639 | 5.4639 | 2.9521 | 24.581 | 27.871 | 0 / 1 | 3 / 48 | stopping failures |
| 64 | `nmrom_K16_q96_dense_dt_half` | 16 | 0.3670 | 2.8982 | 5.4639 | 5.4639 | 2.8892 | 37.969 | 41.157 | 0 / 1 | 0 / 48 | stopping failures |
| 64 | `nmrom_K16_q96_eq` | 16 | 3.2429 | 5.0434 | 9.8065 | 9.8065 | 5.0161 | 24.898 | 28.202 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 64 | `nmrom_K8_q0_dense` | 16 | 1.3770 | 6.1474 | 11.0875 | 11.0875 | 6.1300 | 13.892 | 17.096 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K8_q0_eq` | 16 | 3.2476 | 7.0657 | 11.5082 | 11.5082 | 7.0346 | 13.238 | 16.510 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 64 | `nmrom_K8_q32_dense` | 16 | 0.9234 | 4.9261 | 9.3117 | 9.3117 | 4.9105 | 15.160 | 18.539 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K8_q32_eq` | 16 | 3.1334 | 6.3777 | 10.7137 | 10.7137 | 6.3468 | 14.529 | 17.865 | 0 / 1 | 3 / 48 | failed EQ certificate; stopping failures |
| 64 | `nmrom_K8_q64_dense` | 16 | 0.6920 | 3.7381 | 7.2770 | 7.2770 | 3.7263 | 15.455 | 18.727 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K8_q64_eq` | 16 | 3.1646 | 5.7227 | 9.7992 | 9.7992 | 5.6971 | 15.372 | 18.742 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 64 | `nmrom_K8_q8_dense` | 16 | 1.1719 | 5.9918 | 10.8848 | 10.8848 | 5.9734 | 14.317 | 17.628 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K8_q8_eq` | 16 | 3.2001 | 6.9863 | 11.3739 | 11.3739 | 6.9545 | 13.700 | 17.058 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 64 | `nmrom_K8_q96_dense` | 16 | 0.4037 | 2.2625 | 4.6852 | 4.6852 | 2.2541 | 16.567 | 19.843 | 0 / 1 | 3 / 48 | stopping failures |
| 64 | `nmrom_K8_q96_dense_dt_half` | 16 | 0.4030 | 2.2582 | 4.6852 | 4.6852 | 2.2520 | 25.735 | 28.927 | 0 / 1 | 0 / 48 | stopping failures |
| 64 | `nmrom_K8_q96_eq` | 16 | 3.2287 | 5.2020 | 9.9205 | 9.9205 | 5.1782 | 17.956 | 21.348 | 0 / 0 | 3 / 48 | failed EQ certificate |
| 64 | `pod104_exact` | 16 | 0.2691 | 1.6623 | 3.3620 | 3.3620 | 1.6589 | 0.456 | 4.367 | 0 / 0 | 6 / 48 | development |
| 64 | `pod112_exact` | 16 | 0.2090 | 1.3512 | 3.0908 | 3.0908 | 1.3465 | 0.479 | 4.256 | 0 / 0 | 5 / 48 | development |
| 64 | `pod128_exact` | 16 | 0.1414 | 1.2338 | 2.6795 | 2.6795 | 1.2298 | 0.507 | 4.427 | 0 / 0 | 3 / 48 | development |
| 64 | `pod16_exact` | 16 | 8.6633 | 25.1877 | 36.5082 | 36.5082 | 25.1487 | 0.248 | 4.064 | 0 / 0 | 12 / 48 | development |
| 64 | `pod24_exact` | 16 | 5.5947 | 11.7352 | 19.6233 | 19.6233 | 11.7442 | 0.263 | 4.073 | 0 / 0 | 7 / 48 | development |
| 64 | `pod40_exact` | 16 | 2.1906 | 7.5604 | 13.4307 | 13.4307 | 7.5504 | 0.312 | 4.105 | 0 / 0 | 8 / 48 | development |
| 64 | `pod48_exact` | 16 | 1.3733 | 4.8685 | 9.9616 | 9.9616 | 4.8571 | 0.331 | 4.139 | 0 / 0 | 6 / 48 | development |
| 64 | `pod72_exact` | 16 | 0.6842 | 3.2199 | 6.1130 | 6.1130 | 3.2108 | 0.386 | 4.223 | 0 / 0 | 8 / 48 | development |
| 64 | `pod80_exact` | 16 | 0.4886 | 2.6697 | 5.4395 | 5.4395 | 2.6607 | 0.396 | 4.293 | 0 / 0 | 7 / 48 | development |
| 64 | `pod8_exact` | 16 | 20.0080 | 39.1939 | 50.6608 | 50.6608 | 39.1541 | 0.238 | 4.118 | 0 / 0 | 12 / 48 | development |
| 64 | `unet3d_w8` | 16 | 75.7115 | 92.6756 | 92.6756 | 0.0000 | 92.8275 | 7.047 | 10.016 | 0 / 0 | 0 / 48 | development |
| 64 | `unet3d_w8_native_grid_interpolated` | 16 | 0.7512 | 0.9468 | 0.9468 | 0.0000 | 0.8996 | 3.319 | 6.382 | 0 / 0 | 0 / 48 | development |

A missing physical-error or total-time cell means unmeasured, not zero. Physical errors require the attempt's separate reference-refinement qualification. A passing numerical audit verifies the recorded experiment; it does not establish good predictive accuracy, convergence of training or a competitive method.

## Navier–Stokes 3D — comparison02

**Provisional:** Audited development comparison with the fixed three-dimensional interacting velocity family and identical training membership. The learned bank and nonlinear head miss the physical accuracy target; these negative results motivate the larger-bank/head experiment. Timed results cover the declared first eight development cases, not the full sixteen used in training validation. Galerkin bank/POD controls solve a different reduced projection from the weak least-squares correction ladder. Initial fitting and dense input/output are charged. No latent state history was saved in this attempt, so evolution stopping records are checked for consistency only. A CUDA delay-kernel warning during operator training preceded timing burn-ins and is retained. Final data remain unopened.

Source `23932ac4eca33075fc43b4e10e9d3ca24ee2a388`; job `3993021`; GPU `NVIDIA A100-PCIE-40GB, 40960 MiB`. [Invocation data](../worktrees/2026-09-20-paper-ns3d/experiments/ns3d/runs/comparison02/collected/output/result.json) and [independent audit](../worktrees/2026-09-20-paper-ns3d/experiments/ns3d/runs/comparison02/audit.json).

Errors use the initial velocity-field norm, with all three components combined. The evolved metric excludes time zero. Physical error uses Fourier interpolation to the independently refined grid. Total timing includes host transfers; device timing includes initialization, evolution and every requested dense velocity field. Mesh size counts periodic points per axis. The timed cohort is a declared subset of the larger validation cohort; training-validation summaries are not substituted for its measured query errors. Stopping records lack saved latent histories and support an internal-consistency check only.

| Mesh | Method | Cases | Error median (%) | Error worst (%) | All-times worst (%) | Initial worst (%) | Physical worst (%) | GPU median (ms) | Total median (ms) | Nonfinite / nonstationary cases | Timing outliers / calls | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 32 | `fno3d_projected` | 8 | 1.0974 | 1.5121 | 1.5121 | 0.0000 | 1.5183 | 9.596 | 10.738 | 0 / 0 | 0 / 24 | development |
| 32 | `fno3d_raw` | 8 | 1.3422 | 1.7664 | 1.7664 | 0.0000 | 1.7604 | 9.554 | 10.678 | 0 / 0 | 0 / 24 | development |
| 32 | `fom_dt0.001` | 8 | 0.0031 | 0.0051 | 0.0051 | 0.0000 | 0.3651 | 25.763 | 26.964 | 0 / 0 | 0 / 24 | development |
| 32 | `fom_dt0.002` | 8 | 0.0130 | 0.0213 | 0.0213 | 0.0000 | 0.3663 | 13.268 | 14.397 | 0 / 0 | 0 / 24 | development |
| 32 | `fom_dt0.004` | 8 | 0.0524 | 0.0866 | 0.0866 | 0.0000 | 0.3745 | 7.056 | 8.144 | 0 / 0 | 0 / 24 | development |
| 32 | `fom_dt0.008` | 8 | 0.2125 | 0.3598 | 0.3598 | 0.0000 | 0.4711 | 3.936 | 5.020 | 0 / 0 | 0 / 24 | development |
| 32 | `free_bank_galerkin` | 8 | 13.4503 | 24.2696 | 24.2696 | 17.7985 | 24.2689 | 49.405 | 50.631 | 0 / 0 | 0 / 24 | development |
| 32 | `nmrom_q0` | 8 | 28.5319 | 50.4758 | 53.7445 | 53.7445 | 50.4758 | 901.362 | 903.963 | 0 / 0 | 0 / 24 | development |
| 32 | `nmrom_q128` | 8 | 25.2664 | 45.3348 | 46.7275 | 46.7275 | 45.3345 | 2226.148 | 2228.664 | 0 / 0 | 0 / 24 | development |
| 32 | `nmrom_q16` | 8 | 28.2539 | 49.9583 | 53.1214 | 53.1214 | 49.9583 | 969.484 | 972.076 | 0 / 0 | 0 / 24 | development |
| 32 | `nmrom_q32` | 8 | 27.8899 | 49.2381 | 52.3195 | 52.3195 | 49.2381 | 1092.342 | 1094.953 | 0 / 0 | 0 / 24 | development |
| 32 | `nmrom_q64` | 8 | 27.0931 | 47.6297 | 50.5176 | 50.5176 | 47.6297 | 1588.925 | 1591.515 | 0 / 0 | 0 / 24 | development |
| 32 | `pod_galerkin_128` | 8 | 34.5481 | 47.3334 | 50.1174 | 50.1174 | 47.3334 | 5.889 | 6.986 | 0 / 0 | 0 / 24 | development |
| 32 | `pod_galerkin_256` | 8 | 24.8410 | 36.0594 | 36.0594 | 34.7360 | 36.0592 | 12.233 | 13.401 | 0 / 0 | 0 / 24 | development |
| 32 | `pod_galerkin_512` | 8 | 15.2837 | 27.7537 | 27.7537 | 21.6223 | 27.7535 | 49.327 | 50.571 | 0 / 0 | 0 / 24 | development |
| 32 | `pod_weak_16` | 8 | 87.7996 | 91.9857 | 94.8670 | 94.8670 | 91.9857 | 24.465 | 26.554 | 0 / 0 | 0 / 24 | development |
| 32 | `pod_weak_32` | 8 | 78.1203 | 85.5263 | 89.3108 | 89.3108 | 85.5263 | 27.381 | 29.381 | 0 / 0 | 0 / 24 | development |
| 32 | `pod_weak_48` | 8 | 67.3545 | 76.1070 | 82.4515 | 82.4515 | 76.1069 | 35.558 | 37.642 | 0 / 0 | 0 / 24 | development |
| 32 | `unet3d_projected` | 8 | 3.0414 | 4.2830 | 4.2830 | 0.0000 | 4.2809 | 2.128 | 3.236 | 0 / 0 | 1 / 24 | development |
| 32 | `unet3d_raw` | 8 | 3.4540 | 4.6468 | 4.6468 | 0.0000 | 4.6463 | 2.019 | 3.063 | 0 / 0 | 0 / 24 | development |

A missing physical-error or total-time cell means unmeasured, not zero. Physical errors require the attempt's separate reference-refinement qualification. A passing numerical audit verifies the recorded experiment; it does not establish good predictive accuracy, convergence of training or a competitive method.

## Poisson 3D — extra03

**Provisional:** Audited development comparison with all four operator families and matched training membership. All static weights reside on the GPU during setup. The dense K16 correction ladder satisfies its recorded stopping checks and improves accuracy, but stronger POD and unrestricted linear-bank controls remain faster and more accurate. DeepONet/Transolver training still improves near the declared budget; longer common schedules and independent seeds are running. Native-grid interpolation and failed direct mesh-transfer variants remain distinct; fixed FNO padding changes the physical padded domain and a corrected-domain control is pending. Failed quadrature certificates remain diagnostics. Final cases are unopened. Selected latent states were not saved; stationarity is supported by the recorded stopping values and internal consistency only, not an independent weak-gradient reconstruction. Finer-grid POD is rebuilt from the same training members and is a mesh-adapted classical control.

Source `a50ce0977373977688d73f82700108a6138b433b`; job `3995104`; GPU `NVIDIA A100 80GB PCIe`. [Invocation data](../worktrees/2026-09-20-paper-p3d/experiments/paper-p3d/runs/extra03/archive/out/result.json) and [independent audit](../worktrees/2026-09-20-paper-p3d/experiments/paper-p3d/runs/extra03/audit-local.json).

Errors use the reference solution norm. There is one stationary output field; evolved, initial and all-times terminology does not apply. Total timing includes host transfers. Mesh size counts intervals per axis.

| Mesh | Method | Cases | Error median (%) | Error worst (%) | All-times worst (%) | Initial worst (%) | Physical worst (%) | GPU median (ms) | Total median (ms) | Nonfinite / nonstationary cases | Timing outliers / calls | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 32 | `deeponet3d_r128_w16` | 16 | 2.1923 | 4.3270 | — | — | 4.2567 | 2.935 | 3.707 | 0 / 0 | 0 / 48 | development |
| 32 | `dst_exact` | 16 | 0.0000 | 0.0000 | — | — | 0.2759 | 0.184 | 2.039 | 0 / 0 | 9 / 48 | development |
| 32 | `fno3d_w24_m8` | 16 | 0.1502 | 0.2394 | — | — | 0.3379 | 11.331 | 12.209 | 0 / 0 | 0 / 48 | development |
| 32 | `linear_bank_galerkin` | 16 | 0.0505 | 0.1471 | — | — | 0.3114 | 0.215 | 2.072 | 0 / 0 | 2 / 48 | development |
| 32 | `linear_bank_weak_qR` | 16 | 0.0497 | 0.1450 | — | — | 0.3110 | 0.263 | 2.075 | 0 / 0 | 4 / 48 | development |
| 32 | `nmrom_K16_q0_dense` | 16 | 0.2070 | 0.5892 | — | — | 0.6145 | 2.740 | 3.573 | 0 / 0 | 1 / 48 | development |
| 32 | `nmrom_K16_q0_eq` | 16 | 0.2221 | 0.5939 | — | — | 0.6269 | 2.727 | 3.569 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 32 | `nmrom_K16_q16_dense` | 16 | 0.1427 | 0.4603 | — | — | 0.5314 | 2.828 | 3.651 | 0 / 0 | 1 / 48 | development |
| 32 | `nmrom_K16_q16_eq` | 16 | 0.2145 | 0.5074 | — | — | 0.5764 | 2.648 | 3.454 | 0 / 0 | 1 / 48 | failed EQ certificate |
| 32 | `nmrom_K16_q32_dense` | 16 | 0.1112 | 0.3284 | — | — | 0.3993 | 2.686 | 3.501 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K16_q32_eq` | 16 | 0.1932 | 0.3947 | — | — | 0.4854 | 2.743 | 3.623 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 32 | `nmrom_K16_q64_dense` | 16 | 0.0669 | 0.2018 | — | — | 0.3393 | 2.776 | 3.605 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K16_q64_eq` | 16 | 0.2181 | 0.4824 | — | — | 0.5642 | 2.945 | 3.860 | 0 / 0 | 3 / 48 | failed EQ certificate |
| 32 | `nmrom_K16_q96_dense` | 16 | 0.0513 | 0.1605 | — | — | 0.3182 | 2.854 | 3.667 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K16_q96_eq` | 16 | 0.2684 | 0.5880 | — | — | 0.6552 | 3.914 | 4.778 | 0 / 0 | 9 / 48 | failed EQ certificate |
| 32 | `nmrom_K8_q0_dense` | 16 | 0.5633 | 1.7154 | — | — | 1.6730 | 2.426 | 3.330 | 0 / 0 | 3 / 48 | development |
| 32 | `nmrom_K8_q0_eq` | 16 | 0.5636 | 1.7161 | — | — | 1.6748 | 2.550 | 3.424 | 0 / 0 | 3 / 48 | failed EQ certificate |
| 32 | `nmrom_K8_q16_dense` | 16 | 0.3087 | 0.9815 | — | — | 0.9964 | 2.365 | 3.328 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K8_q16_eq` | 16 | 0.3266 | 0.9936 | — | — | 1.0109 | 2.487 | 3.329 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 32 | `nmrom_K8_q32_dense` | 16 | 0.1936 | 0.6150 | — | — | 0.6603 | 2.443 | 3.344 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K8_q32_eq` | 16 | 0.2356 | 0.6634 | — | — | 0.7060 | 2.524 | 3.339 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 32 | `nmrom_K8_q64_dense` | 16 | 0.0838 | 0.3434 | — | — | 0.4326 | 2.522 | 3.440 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K8_q64_eq` | 16 | 0.1783 | 0.5207 | — | — | 0.5883 | 2.559 | 3.410 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 32 | `nmrom_K8_q96_dense` | 16 | 0.0560 | 0.2094 | — | — | 0.3433 | 2.620 | 3.488 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K8_q96_eq` | 16 | 0.2361 | 0.6410 | — | — | 0.7018 | 2.472 | 3.340 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 32 | `pod104_galerkin` | 16 | 0.0154 | 0.0776 | — | — | 0.2864 | 0.212 | 2.178 | 0 / 0 | 3 / 48 | development |
| 32 | `pod112_galerkin` | 16 | 0.0117 | 0.0685 | — | — | 0.2839 | 0.213 | 2.024 | 0 / 0 | 4 / 48 | development |
| 32 | `pod128_galerkin` | 16 | 0.0067 | 0.0370 | — | — | 0.2779 | 0.217 | 2.129 | 0 / 0 | 4 / 48 | development |
| 32 | `pod16_galerkin` | 16 | 1.8426 | 5.3174 | — | — | 5.2258 | 0.177 | 2.041 | 0 / 0 | 4 / 48 | development |
| 32 | `pod24_galerkin` | 16 | 0.6949 | 2.9930 | — | — | 2.9595 | 0.174 | 2.074 | 0 / 0 | 10 / 48 | development |
| 32 | `pod32_galerkin` | 16 | 0.4246 | 1.7517 | — | — | 1.7524 | 0.182 | 2.057 | 0 / 0 | 8 / 48 | development |
| 32 | `pod40_galerkin` | 16 | 0.2412 | 0.8478 | — | — | 0.8874 | 0.185 | 2.069 | 0 / 0 | 4 / 48 | development |
| 32 | `pod48_galerkin` | 16 | 0.1506 | 0.5672 | — | — | 0.6383 | 0.186 | 2.003 | 0 / 0 | 3 / 48 | development |
| 32 | `pod72_galerkin` | 16 | 0.0445 | 0.2366 | — | — | 0.3666 | 0.201 | 2.068 | 0 / 0 | 10 / 48 | development |
| 32 | `pod80_galerkin` | 16 | 0.0363 | 0.1980 | — | — | 0.3425 | 0.200 | 2.048 | 0 / 0 | 10 / 48 | development |
| 32 | `pod8_galerkin` | 16 | 4.8920 | 8.8900 | — | — | 8.8274 | 0.174 | 2.056 | 0 / 0 | 7 / 48 | development |
| 32 | `transolver3d_w48_s32` | 16 | 0.7736 | 1.1941 | — | — | 1.2517 | 4.492 | 5.321 | 0 / 0 | 0 / 48 | development |
| 32 | `unet3d_w16` | 16 | 0.1326 | 0.1932 | — | — | 0.3357 | 4.159 | 5.013 | 0 / 0 | 0 / 48 | development |
| 64 | `deeponet3d_r128_w16` | 16 | 16.7244 | 26.8325 | — | — | 26.8588 | 7.221 | 8.519 | 0 / 0 | 0 / 48 | development |
| 64 | `deeponet3d_r128_w16_native32_interpolate` | 16 | 2.1141 | 4.2024 | — | — | 4.1808 | 3.016 | 4.336 | 0 / 0 | 0 / 48 | development |
| 64 | `dst_exact` | 16 | 0.0000 | 0.0000 | — | — | 0.0687 | 0.220 | 2.652 | 0 / 0 | 6 / 48 | development |
| 64 | `fno3d_w24_m8` | 16 | 16.0680 | 17.1866 | — | — | 17.2392 | 17.107 | 18.503 | 0 / 0 | 0 / 48 | development |
| 64 | `fno3d_w24_m8_native32_interpolate` | 16 | 0.2852 | 0.3656 | — | — | 0.3525 | 11.197 | 12.611 | 0 / 0 | 0 / 48 | development |
| 64 | `linear_bank_galerkin` | 16 | 0.0530 | 0.1474 | — | — | 0.1624 | 0.479 | 2.880 | 0 / 0 | 3 / 48 | development |
| 64 | `linear_bank_weak_qR` | 16 | 0.0520 | 0.1444 | — | — | 0.1599 | 0.944 | 3.290 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K16_q0_dense` | 16 | 0.2149 | 0.5726 | — | — | 0.5707 | 3.390 | 4.784 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K16_q0_eq` | 16 | 0.2610 | 0.6521 | — | — | 0.6523 | 2.867 | 4.299 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 64 | `nmrom_K16_q16_dense` | 16 | 0.1492 | 0.4580 | — | — | 0.4629 | 3.435 | 4.860 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K16_q16_eq` | 16 | 0.2748 | 0.6267 | — | — | 0.6302 | 2.982 | 4.432 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 64 | `nmrom_K16_q32_dense` | 16 | 0.1159 | 0.3215 | — | — | 0.3252 | 3.301 | 4.625 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K16_q32_eq` | 16 | 0.2804 | 0.7197 | — | — | 0.7245 | 2.910 | 4.304 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 64 | `nmrom_K16_q64_dense` | 16 | 0.0687 | 0.2001 | — | — | 0.2113 | 3.465 | 4.906 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K16_q64_eq` | 16 | 0.3607 | 0.8852 | — | — | 0.8906 | 3.070 | 4.504 | 0 / 0 | 9 / 48 | failed EQ certificate |
| 64 | `nmrom_K16_q96_dense` | 16 | 0.0539 | 0.1598 | — | — | 0.1739 | 3.566 | 4.966 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K16_q96_eq` | 16 | 0.4959 | 1.2060 | — | — | 1.2104 | 6.107 | 7.510 | 0 / 0 | 8 / 48 | failed EQ certificate |
| 64 | `nmrom_K8_q0_dense` | 16 | 0.5622 | 1.6683 | — | — | 1.6544 | 3.236 | 4.639 | 0 / 0 | 3 / 48 | development |
| 64 | `nmrom_K8_q0_eq` | 16 | 0.5653 | 1.6689 | — | — | 1.6553 | 2.514 | 3.972 | 0 / 0 | 3 / 48 | failed EQ certificate |
| 64 | `nmrom_K8_q16_dense` | 16 | 0.3133 | 0.9645 | — | — | 0.9617 | 3.218 | 4.584 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K8_q16_eq` | 16 | 0.3441 | 0.9913 | — | — | 0.9893 | 2.639 | 4.102 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 64 | `nmrom_K8_q32_dense` | 16 | 0.1969 | 0.6047 | — | — | 0.6055 | 3.044 | 4.446 | 0 / 0 | 1 / 48 | development |
| 64 | `nmrom_K8_q32_eq` | 16 | 0.2998 | 0.6795 | — | — | 0.6865 | 2.597 | 3.981 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 64 | `nmrom_K8_q64_dense` | 16 | 0.0860 | 0.3366 | — | — | 0.3415 | 3.222 | 4.555 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K8_q64_eq` | 16 | 0.3466 | 0.8552 | — | — | 0.8617 | 2.584 | 4.002 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 64 | `nmrom_K8_q96_dense` | 16 | 0.0566 | 0.2067 | — | — | 0.2170 | 3.154 | 4.546 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K8_q96_eq` | 16 | 0.4525 | 1.1390 | — | — | 1.1440 | 2.633 | 3.999 | 0 / 0 | 0 / 48 | failed EQ certificate |
| 64 | `pod104_galerkin` | 16 | 0.0150 | 0.0761 | — | — | 0.1024 | 0.422 | 2.801 | 0 / 0 | 5 / 48 | development |
| 64 | `pod112_galerkin` | 16 | 0.0115 | 0.0672 | — | — | 0.0958 | 0.440 | 2.866 | 0 / 0 | 5 / 48 | development |
| 64 | `pod128_galerkin` | 16 | 0.0066 | 0.0360 | — | — | 0.0771 | 0.480 | 2.869 | 0 / 0 | 4 / 48 | development |
| 64 | `pod16_galerkin` | 16 | 1.8304 | 5.2702 | — | — | 5.2463 | 0.203 | 2.689 | 0 / 0 | 5 / 48 | development |
| 64 | `pod24_galerkin` | 16 | 0.6890 | 2.9688 | — | — | 2.9582 | 0.227 | 2.664 | 0 / 0 | 3 / 48 | development |
| 64 | `pod32_galerkin` | 16 | 0.4204 | 1.7372 | — | — | 1.7334 | 0.247 | 2.608 | 0 / 0 | 2 / 48 | development |
| 64 | `pod40_galerkin` | 16 | 0.2383 | 0.8384 | — | — | 0.8402 | 0.270 | 2.785 | 0 / 0 | 3 / 48 | development |
| 64 | `pod48_galerkin` | 16 | 0.1484 | 0.5576 | — | — | 0.5639 | 0.287 | 2.702 | 0 / 0 | 2 / 48 | development |
| 64 | `pod72_galerkin` | 16 | 0.0438 | 0.2329 | — | — | 0.2440 | 0.346 | 2.789 | 0 / 0 | 3 / 48 | development |
| 64 | `pod80_galerkin` | 16 | 0.0357 | 0.1951 | — | — | 0.2080 | 0.361 | 2.827 | 0 / 0 | 3 / 48 | development |
| 64 | `pod8_galerkin` | 16 | 4.8633 | 8.8440 | — | — | 8.8280 | 0.196 | 2.656 | 0 / 0 | 5 / 48 | development |
| 64 | `transolver3d_w48_s32` | 16 | 16.9674 | 22.0653 | — | — | 22.0984 | 12.719 | 14.096 | 0 / 0 | 0 / 48 | development |
| 64 | `transolver3d_w48_s32_native32_interpolate` | 16 | 0.7201 | 1.0698 | — | — | 1.0700 | 4.493 | 5.854 | 0 / 0 | 0 / 48 | development |
| 64 | `unet3d_w16` | 16 | 168.0982 | 219.8928 | — | — | 220.0031 | 12.157 | 13.560 | 0 / 0 | 0 / 48 | development |
| 64 | `unet3d_w16_native32_interpolate` | 16 | 0.2828 | 0.3769 | — | — | 0.3562 | 4.262 | 5.631 | 0 / 0 | 0 / 48 | development |

A missing physical-error or total-time cell means unmeasured, not zero. Physical errors require the attempt's separate reference-refinement qualification. A passing numerical audit verifies the recorded experiment; it does not establish good predictive accuracy, convergence of training or a competitive method.

## Heat 3D — extra03

**Provisional:** Audited development comparison with all four operator families and identical training membership. Larger initial-fit budgets resolve the earlier stopping failures; every recorded reduced fit and evolved solve is stationary. The larger head improves the high-correction endpoint while a substantial held-out head gap remains. POD, unrestricted linear-bank and native FNO/U-Net controls remain stronger. DeepONet is still inaccurate at its first training budget, and longer operator training plus matched head-initialization controls are in preparation. Failed direct transfer variants remain explicit, and prior failed quadrature certificates are not promoted. Final cases remain unopened. Finer-grid POD is rebuilt from the same training members and is a mesh-adapted classical control.

Source `4ac8b16455f5b71bcdd560df3432cac43b8816d9`; job `3995709`; GPU `NVIDIA A100-PCIE-40GB`. [Invocation data](../worktrees/2026-09-20-paper-h3d/experiments/paper-h3d/runs/extra03/archive/out/result.json) and [independent audit](../worktrees/2026-09-20-paper-h3d/experiments/paper-h3d/runs/extra03/audit-local.json).

Errors use each reference field's current norm. Evolved errors exclude the initial state. Total timing includes host transfers; GPU timing includes initialization, evolution and dense output. Mesh size counts intervals per axis.

| Mesh | Method | Cases | Error median (%) | Error worst (%) | All-times worst (%) | Initial worst (%) | Physical worst (%) | GPU median (ms) | Total median (ms) | Nonfinite / nonstationary cases | Timing outliers / calls | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 32 | `deeponet3d_r128_w16` | 16 | 7.1651 | 21.7692 | 21.7692 | 0.0000 | 21.5236 | 3.889 | 5.107 | 0 / 0 | 0 / 48 | development |
| 32 | `dst_exact` | 16 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.5119 | 1.428 | 2.816 | 0 / 0 | 0 / 48 | development |
| 32 | `fno3d_w16_m6` | 16 | 0.2449 | 0.3947 | 0.3947 | 0.0000 | 0.6332 | 10.638 | 11.818 | 0 / 0 | 0 / 48 | development |
| 32 | `linear_bank_cn` | 16 | 0.2591 | 1.3315 | 3.0636 | 3.0636 | 1.3507 | 0.641 | 2.939 | 0 / 0 | 4 / 48 | development |
| 32 | `linear_bank_exact` | 16 | 0.2549 | 1.3277 | 3.0636 | 3.0636 | 1.3655 | 0.628 | 2.880 | 0 / 0 | 8 / 48 | development |
| 32 | `linear_bank_galerkin_exact` | 16 | 0.2568 | 1.3324 | 3.0588 | 3.0588 | 1.3607 | 0.187 | 2.429 | 0 / 0 | 6 / 48 | development |
| 32 | `nmrom_K16_q0_dense` | 16 | 1.0782 | 5.8875 | 11.0118 | 11.0118 | 5.8138 | 27.044 | 28.212 | 0 / 0 | 6 / 48 | development |
| 32 | `nmrom_K16_q32_dense` | 16 | 0.8696 | 4.9327 | 9.2331 | 9.2331 | 4.8606 | 30.205 | 31.403 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K16_q64_dense` | 16 | 0.5815 | 4.0309 | 7.5745 | 7.5745 | 3.9834 | 31.477 | 32.749 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K16_q8_dense` | 16 | 1.0381 | 5.3133 | 10.2925 | 10.2925 | 5.2396 | 28.285 | 29.423 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K16_q96_dense` | 16 | 0.3673 | 2.9899 | 5.4636 | 5.4636 | 2.9665 | 38.169 | 39.275 | 0 / 0 | 3 / 48 | development |
| 32 | `nmrom_K16_q96_dense_dt_half` | 16 | 0.3648 | 2.9260 | 5.4636 | 5.4636 | 2.9091 | 56.154 | 57.420 | 0 / 0 | 3 / 48 | development |
| 32 | `nmrom_K32_q0_dense` | 16 | 1.1486 | 6.0134 | 10.6852 | 10.6852 | 5.9374 | 34.771 | 35.907 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K32_q32_dense` | 16 | 0.8215 | 4.4533 | 8.1773 | 8.1773 | 4.3894 | 38.699 | 39.927 | 0 / 0 | 2 / 48 | development |
| 32 | `nmrom_K32_q64_dense` | 16 | 0.5582 | 3.2137 | 5.9582 | 5.9582 | 3.1793 | 48.712 | 49.977 | 0 / 0 | 4 / 48 | development |
| 32 | `nmrom_K32_q8_dense` | 16 | 1.0209 | 5.2288 | 9.3616 | 9.3616 | 5.1613 | 35.713 | 36.871 | 0 / 0 | 8 / 48 | development |
| 32 | `nmrom_K32_q96_dense` | 16 | 0.3991 | 1.3791 | 3.1001 | 3.1001 | 1.3922 | 114.343 | 115.557 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K32_q96_dense_dt_half` | 16 | 0.3961 | 1.3713 | 3.1001 | 3.1001 | 1.3996 | 147.204 | 148.455 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K8_q0_dense` | 16 | 1.3933 | 6.1977 | 11.0875 | 11.0875 | 6.1354 | 18.938 | 20.062 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K8_q32_dense` | 16 | 0.9327 | 4.9721 | 9.3116 | 9.3116 | 4.9193 | 22.023 | 23.203 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K8_q64_dense` | 16 | 0.6969 | 3.7752 | 7.2769 | 7.2769 | 3.7427 | 21.872 | 23.059 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K8_q8_dense` | 16 | 1.1850 | 6.0419 | 10.8848 | 10.8848 | 5.9761 | 19.733 | 20.981 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K8_q96_dense` | 16 | 0.4060 | 2.2781 | 4.6849 | 4.6849 | 2.2724 | 23.916 | 25.156 | 0 / 0 | 3 / 48 | development |
| 32 | `nmrom_K8_q96_dense_dt_half` | 16 | 0.4053 | 2.2742 | 4.6849 | 4.6849 | 2.2770 | 37.631 | 38.780 | 0 / 0 | 3 / 48 | development |
| 32 | `pod104_exact` | 16 | 0.2697 | 1.6482 | 3.3966 | 3.3966 | 1.6743 | 0.185 | 2.539 | 0 / 0 | 4 / 48 | development |
| 32 | `pod112_exact` | 16 | 0.2097 | 1.3665 | 3.0837 | 3.0837 | 1.3938 | 0.182 | 2.446 | 0 / 0 | 7 / 48 | development |
| 32 | `pod128_exact` | 16 | 0.1431 | 1.2441 | 2.6741 | 2.6741 | 1.2806 | 0.189 | 2.485 | 0 / 0 | 8 / 48 | development |
| 32 | `pod16_exact` | 16 | 8.6384 | 25.2986 | 36.4964 | 36.4964 | 25.1430 | 0.143 | 2.425 | 0 / 0 | 8 / 48 | development |
| 32 | `pod24_exact` | 16 | 5.5732 | 11.6555 | 19.6057 | 19.6057 | 11.6993 | 0.148 | 2.491 | 0 / 0 | 6 / 48 | development |
| 32 | `pod32_exact` | 16 | 3.0942 | 9.4366 | 16.6338 | 16.6338 | 9.3574 | 0.161 | 2.502 | 0 / 0 | 8 / 48 | development |
| 32 | `pod40_exact` | 16 | 2.2006 | 7.5846 | 13.3221 | 13.3221 | 7.5508 | 0.156 | 2.426 | 0 / 0 | 4 / 48 | development |
| 32 | `pod48_exact` | 16 | 1.3691 | 4.8625 | 9.8946 | 9.8946 | 4.8286 | 0.158 | 2.534 | 0 / 0 | 5 / 48 | development |
| 32 | `pod64_exact` | 16 | 0.8393 | 3.5288 | 6.8345 | 6.8345 | 3.5194 | 0.162 | 2.463 | 0 / 0 | 7 / 48 | development |
| 32 | `pod72_exact` | 16 | 0.6934 | 3.2741 | 6.1329 | 6.1329 | 3.2533 | 0.164 | 2.392 | 0 / 0 | 6 / 48 | development |
| 32 | `pod80_exact` | 16 | 0.4948 | 2.6899 | 5.4338 | 5.4338 | 2.6768 | 0.171 | 2.449 | 0 / 0 | 5 / 48 | development |
| 32 | `pod8_exact` | 16 | 19.9122 | 39.3251 | 50.6560 | 50.6560 | 39.1660 | 0.141 | 2.417 | 0 / 0 | 8 / 48 | development |
| 32 | `pod96_exact` | 16 | 0.3322 | 1.7675 | 3.7157 | 3.7157 | 1.7851 | 0.178 | 2.432 | 0 / 0 | 8 / 48 | development |
| 32 | `transolver3d_w48_s32` | 16 | 1.0619 | 1.4826 | 1.4826 | 0.0000 | 1.5419 | 6.282 | 7.527 | 0 / 0 | 0 / 48 | development |
| 32 | `unet3d_w8` | 16 | 0.1697 | 0.3185 | 0.3185 | 0.0000 | 0.5321 | 4.373 | 5.490 | 0 / 0 | 0 / 48 | development |
| 64 | `deeponet3d_r128_w16` | 16 | 33.8829 | 52.5118 | 52.5118 | 0.0000 | 52.4733 | 11.270 | 16.165 | 0 / 0 | 0 / 48 | development |
| 64 | `deeponet3d_r128_w16_native_grid_interpolated` | 16 | 7.0168 | 21.7903 | 21.7903 | 0.0000 | 21.7279 | 4.029 | 8.881 | 0 / 0 | 0 / 48 | development |
| 64 | `dst_exact` | 16 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.1273 | 1.591 | 6.717 | 0 / 0 | 0 / 48 | development |
| 64 | `fno3d_w16_m6` | 16 | 15.1234 | 17.6641 | 17.6641 | 0.0000 | 17.7688 | 16.246 | 21.129 | 0 / 0 | 0 / 48 | development |
| 64 | `fno3d_w16_m6_native_grid_interpolated` | 16 | 0.7669 | 0.9891 | 0.9891 | 0.0000 | 0.9391 | 10.681 | 15.538 | 0 / 0 | 0 / 48 | development |
| 64 | `linear_bank_cn` | 16 | 0.2562 | 1.3155 | 3.0639 | 3.0639 | 1.3079 | 1.452 | 7.166 | 0 / 0 | 0 / 48 | development |
| 64 | `linear_bank_exact` | 16 | 0.2542 | 1.3113 | 3.0639 | 3.0639 | 1.3087 | 1.495 | 7.155 | 0 / 0 | 0 / 48 | development |
| 64 | `linear_bank_galerkin_exact` | 16 | 0.2563 | 1.3166 | 3.0591 | 3.0591 | 1.3115 | 0.564 | 6.273 | 0 / 0 | 1 / 48 | development |
| 64 | `nmrom_K16_q0_dense` | 16 | 1.0804 | 5.8448 | 11.0119 | 11.0119 | 5.8243 | 28.606 | 33.459 | 0 / 0 | 6 / 48 | development |
| 64 | `nmrom_K16_q32_dense` | 16 | 0.8751 | 4.8851 | 9.2332 | 9.2332 | 4.8647 | 31.391 | 36.263 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K16_q64_dense` | 16 | 0.5892 | 3.9929 | 7.5747 | 7.5747 | 3.9778 | 33.687 | 38.640 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K16_q8_dense` | 16 | 1.0426 | 5.2535 | 10.2925 | 10.2925 | 5.2332 | 29.963 | 34.782 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K16_q96_dense` | 16 | 0.3697 | 2.9628 | 5.4639 | 5.4639 | 2.9521 | 40.919 | 45.855 | 0 / 0 | 2 / 48 | development |
| 64 | `nmrom_K16_q96_dense_dt_half` | 16 | 0.3670 | 2.8982 | 5.4639 | 5.4639 | 2.8892 | 64.886 | 69.897 | 0 / 0 | 2 / 48 | development |
| 64 | `nmrom_K32_q0_dense` | 16 | 1.1494 | 5.9683 | 10.6852 | 10.6852 | 5.9471 | 37.143 | 42.045 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K32_q32_dense` | 16 | 0.8224 | 4.4151 | 8.1774 | 8.1774 | 4.3959 | 40.445 | 45.528 | 0 / 0 | 1 / 48 | development |
| 64 | `nmrom_K32_q64_dense` | 16 | 0.5591 | 3.1815 | 5.9583 | 5.9583 | 3.1683 | 52.322 | 57.222 | 0 / 0 | 5 / 48 | development |
| 64 | `nmrom_K32_q8_dense` | 16 | 1.0223 | 5.1921 | 9.3616 | 9.3616 | 5.1727 | 40.126 | 45.088 | 0 / 0 | 5 / 48 | development |
| 64 | `nmrom_K32_q96_dense` | 16 | 0.3903 | 1.3555 | 3.1004 | 3.1004 | 1.3466 | 113.590 | 118.610 | 0 / 0 | 1 / 48 | development |
| 64 | `nmrom_K32_q96_dense_dt_half` | 16 | 0.3869 | 1.3588 | 3.1004 | 3.1004 | 1.3538 | 154.194 | 159.203 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K8_q0_dense` | 16 | 1.3768 | 6.1474 | 11.0875 | 11.0875 | 6.1300 | 20.982 | 26.005 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K8_q32_dense` | 16 | 0.9234 | 4.9261 | 9.3117 | 9.3117 | 4.9105 | 23.500 | 28.484 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K8_q64_dense` | 16 | 0.6919 | 3.7381 | 7.2770 | 7.2770 | 3.7263 | 23.519 | 28.523 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K8_q8_dense` | 16 | 1.1719 | 5.9918 | 10.8848 | 10.8848 | 5.9735 | 22.046 | 26.908 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K8_q96_dense` | 16 | 0.4037 | 2.2625 | 4.6851 | 4.6851 | 2.2541 | 24.658 | 29.569 | 0 / 0 | 3 / 48 | development |
| 64 | `nmrom_K8_q96_dense_dt_half` | 16 | 0.4030 | 2.2582 | 4.6851 | 4.6851 | 2.2520 | 40.888 | 45.769 | 0 / 0 | 2 / 48 | development |
| 64 | `pod104_exact` | 16 | 0.2691 | 1.6623 | 3.3620 | 3.3620 | 1.6589 | 0.533 | 6.117 | 0 / 0 | 3 / 48 | development |
| 64 | `pod112_exact` | 16 | 0.2090 | 1.3512 | 3.0908 | 3.0908 | 1.3465 | 0.553 | 6.068 | 0 / 0 | 6 / 48 | development |
| 64 | `pod128_exact` | 16 | 0.1414 | 1.2338 | 2.6795 | 2.6795 | 1.2298 | 0.565 | 6.149 | 0 / 0 | 0 / 48 | development |
| 64 | `pod16_exact` | 16 | 8.6633 | 25.1877 | 36.5082 | 36.5082 | 25.1487 | 0.244 | 5.904 | 0 / 0 | 10 / 48 | development |
| 64 | `pod24_exact` | 16 | 5.5947 | 11.7352 | 19.6233 | 19.6233 | 11.7442 | 0.298 | 5.906 | 0 / 0 | 13 / 48 | development |
| 64 | `pod32_exact` | 16 | 3.0811 | 9.3730 | 16.6440 | 16.6440 | 9.3524 | 0.311 | 6.035 | 0 / 0 | 9 / 48 | development |
| 64 | `pod40_exact` | 16 | 2.1906 | 7.5604 | 13.4307 | 13.4307 | 7.5504 | 0.345 | 5.891 | 0 / 0 | 12 / 48 | development |
| 64 | `pod48_exact` | 16 | 1.3733 | 4.8685 | 9.9616 | 9.9616 | 4.8571 | 0.354 | 6.025 | 0 / 0 | 11 / 48 | development |
| 64 | `pod64_exact` | 16 | 0.8208 | 3.5334 | 6.8850 | 6.8850 | 3.5268 | 0.395 | 6.081 | 0 / 0 | 7 / 48 | development |
| 64 | `pod72_exact` | 16 | 0.6842 | 3.2199 | 6.1130 | 6.1130 | 3.2108 | 0.472 | 6.044 | 0 / 0 | 6 / 48 | development |
| 64 | `pod80_exact` | 16 | 0.4886 | 2.6697 | 5.4395 | 5.4395 | 2.6607 | 0.462 | 6.064 | 0 / 0 | 7 / 48 | development |
| 64 | `pod8_exact` | 16 | 20.0080 | 39.1939 | 50.6608 | 50.6608 | 39.1541 | 0.210 | 5.855 | 0 / 0 | 6 / 48 | development |
| 64 | `pod96_exact` | 16 | 0.3259 | 1.7569 | 3.7168 | 3.7168 | 1.7521 | 0.499 | 6.156 | 0 / 0 | 3 / 48 | development |
| 64 | `transolver3d_w48_s32` | 16 | 19.8616 | 25.2954 | 25.2954 | 0.0000 | 25.2677 | 20.828 | 25.703 | 0 / 0 | 0 / 48 | development |
| 64 | `transolver3d_w48_s32_native_grid_interpolated` | 16 | 0.9475 | 1.2766 | 1.2766 | 0.0000 | 1.2793 | 6.854 | 11.721 | 0 / 0 | 0 / 48 | development |
| 64 | `unet3d_w8` | 16 | 75.7115 | 92.6756 | 92.6756 | 0.0000 | 92.8275 | 10.960 | 15.834 | 0 / 0 | 0 / 48 | development |
| 64 | `unet3d_w8_native_grid_interpolated` | 16 | 0.7512 | 0.9468 | 0.9468 | 0.0000 | 0.8996 | 4.240 | 9.116 | 0 / 0 | 0 / 48 | development |

A missing physical-error or total-time cell means unmeasured, not zero. Physical errors require the attempt's separate reference-refinement qualification. A passing numerical audit verifies the recorded experiment; it does not establish good predictive accuracy, convergence of training or a competitive method.

## Burgers 3D — b3d004

**Provisional:** Audited development comparison adding DeepONet and Transolver to the matched frozen-bank/head panel. The recorded training budgets are finite and both newer operators receive longer fresh schedules next. Static checkpoint reuse is hash-pinned. Every recorded reduced solve is stationary and sampled NumPy weak-gradient checks pass. Independent initialization, expanded development inputs, cheaper full-order controls and final evaluation are pending.

Source `c026d56b83905293e71268c249b57324c7d01baf`; job `3995688`; GPU `NVIDIA A100 80GB PCIe`. [Invocation data](../worktrees/2026-09-20-paper-b3d/experiments/paper-b3d/runs/b3d004/collected/out/result.json) and [independent audit](../worktrees/2026-09-20-paper-b3d/experiments/paper-b3d/runs/b3d004/collected/out/audit-local.json).

Errors use the initial-field norm. The evolved error excludes the initial compression; all-times and initial errors are also shown. GPU timing begins with the dense input already on device and ends with every requested dense output on device. Host transfers are unmeasured in this attempt. Mesh size counts nodes per axis.

| Mesh | Method | Cases | Error median (%) | Error worst (%) | All-times worst (%) | Initial worst (%) | Physical worst (%) | GPU median (ms) | Total median (ms) | Nonfinite / nonstationary cases | Timing outliers / calls | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 33 | `deeponet3d` | 8 | 7.9145 | 21.1988 | 21.1988 | 0.0000 | — | 2.534 | — | 0 / 0 | 0 / 24 | development |
| 33 | `fno3d` | 8 | 1.6413 | 1.9979 | 1.9979 | 0.0000 | — | 5.862 | — | 0 / 0 | 0 / 24 | development |
| 33 | `fom_nt1e-02_lt5e-01` | 8 | 1.2739 | 1.6185 | 1.6185 | 0.0000 | — | 16.068 | — | 0 / 0 | 0 / 24 | development |
| 33 | `fom_nt1e-02_lt5e-03` | 8 | 0.0382 | 0.6543 | 0.6543 | 0.0000 | — | 17.940 | — | 0 / 0 | 0 / 24 | development |
| 33 | `fom_nt1e-03_lt5e-01` | 8 | 0.4692 | 0.5248 | 0.5248 | 0.0000 | — | 19.753 | — | 0 / 0 | 0 / 24 | development |
| 33 | `fom_nt1e-04_lt1e-06` | 8 | 0.0188 | 0.0207 | 0.0207 | 0.0000 | — | 37.394 | — | 0 / 0 | 0 / 24 | development |
| 33 | `fom_nt1e-04_lt5e-01` | 8 | 0.0267 | 0.0303 | 0.0303 | 0.0000 | — | 32.494 | — | 0 / 0 | 0 / 24 | development |
| 33 | `fom_nt1e-06_lt1e-01` | 8 | 0.0002 | 0.0003 | 0.0003 | 0.0000 | — | 48.156 | — | 0 / 0 | 0 / 24 | development |
| 33 | `free_R256` | 8 | 0.8437 | 2.4345 | 2.8901 | 2.8901 | — | 316.964 | — | 0 / 0 | 0 / 24 | development |
| 33 | `pod_160` | 8 | 0.7675 | 2.5057 | 3.0153 | 3.0153 | — | 224.346 | — | 0 / 0 | 0 / 24 | development |
| 33 | `pod_224` | 8 | 0.3988 | 1.2100 | 1.5306 | 1.5306 | — | 284.540 | — | 0 / 0 | 0 / 24 | development |
| 33 | `pod_256` | 8 | 0.3292 | 1.0363 | 1.2631 | 1.2631 | — | 316.567 | — | 0 / 0 | 0 / 24 | development |
| 33 | `pod_32` | 8 | 9.5224 | 22.6947 | 24.9424 | 24.9424 | — | 95.738 | — | 0 / 0 | 0 / 24 | development |
| 33 | `pod_96` | 8 | 1.9855 | 5.5785 | 6.6983 | 6.6983 | — | 151.273 | — | 0 / 0 | 0 / 24 | development |
| 33 | `rom_q0` | 8 | 5.7071 | 10.8780 | 11.7177 | 11.7177 | — | 186.364 | — | 0 / 0 | 0 / 24 | development |
| 33 | `rom_q128` | 8 | 3.3258 | 6.6491 | 7.1187 | 7.1187 | — | 353.542 | — | 0 / 0 | 0 / 24 | development |
| 33 | `rom_q192` | 8 | 1.3600 | 3.4908 | 3.9087 | 3.9087 | — | 418.712 | — | 0 / 0 | 0 / 24 | development |
| 33 | `rom_q64` | 8 | 4.9359 | 8.9108 | 9.4908 | 9.4908 | — | 267.817 | — | 0 / 0 | 0 / 24 | development |
| 33 | `transolver3d` | 8 | 1.8700 | 3.7391 | 3.7391 | 0.0000 | — | 3.545 | — | 0 / 0 | 0 / 24 | development |
| 33 | `unet3d` | 8 | 1.6239 | 1.9129 | 1.9129 | 0.0000 | — | 2.212 | — | 0 / 0 | 0 / 24 | development |

A missing physical-error or total-time cell means unmeasured, not zero. Physical errors require the attempt's separate reference-refinement qualification. A passing numerical audit verifies the recorded experiment; it does not establish good predictive accuracy, convergence of training or a competitive method.

## Burgers 3D — b3d004-head64

**Provisional:** Audited larger-head development comparison on exactly the same learned spatial bank, training membership and GPU allocation as b3d004. The correction endpoint reaches the unrestricted bank solution, while the compressed head retains a substantial held-out gap. All recorded reduced solves are stationary, with independently sampled Jacobian/gradient and transversality checks. Operators were measured separately in the main b3d004 record on this same allocation, not remeasured here. No full-cohort physical refinement or final acceptance is claimed.

Source `c026d56b83905293e71268c249b57324c7d01baf`; job `3995688`; GPU `NVIDIA A100 80GB PCIe`. [Invocation data](../worktrees/2026-09-20-paper-b3d/experiments/paper-b3d/runs/b3d004/collected/out/head64/result.json) and [independent audit](../worktrees/2026-09-20-paper-b3d/experiments/paper-b3d/runs/b3d004/collected/out/head64/audit-local.json).

Errors use the initial-field norm. The evolved error excludes the initial compression; all-times and initial errors are also shown. GPU timing begins with the dense input already on device and ends with every requested dense output on device. Host transfers are unmeasured in this attempt. Mesh size counts nodes per axis.

| Mesh | Method | Cases | Error median (%) | Error worst (%) | All-times worst (%) | Initial worst (%) | Physical worst (%) | GPU median (ms) | Total median (ms) | Nonfinite / nonstationary cases | Timing outliers / calls | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 33 | `fom_nt1e-02_lt5e-01` | 8 | 1.2739 | 1.6185 | 1.6185 | 0.0000 | — | 16.355 | — | 0 / 0 | 0 / 24 | development |
| 33 | `fom_nt1e-02_lt5e-03` | 8 | 0.0382 | 0.6543 | 0.6543 | 0.0000 | — | 18.485 | — | 0 / 0 | 0 / 24 | development |
| 33 | `fom_nt1e-03_lt5e-01` | 8 | 0.4692 | 0.5248 | 0.5248 | 0.0000 | — | 20.118 | — | 0 / 0 | 0 / 24 | development |
| 33 | `fom_nt1e-04_lt1e-06` | 8 | 0.0188 | 0.0207 | 0.0207 | 0.0000 | — | 37.846 | — | 0 / 0 | 0 / 24 | development |
| 33 | `fom_nt1e-04_lt5e-01` | 8 | 0.0267 | 0.0303 | 0.0303 | 0.0000 | — | 33.015 | — | 0 / 0 | 0 / 24 | development |
| 33 | `fom_nt1e-06_lt1e-01` | 8 | 0.0002 | 0.0003 | 0.0003 | 0.0000 | — | 49.690 | — | 0 / 0 | 0 / 24 | development |
| 33 | `free_R256` | 8 | 0.8437 | 2.4345 | 2.8901 | 2.8901 | — | 316.996 | — | 0 / 0 | 0 / 24 | development |
| 33 | `pod_128` | 8 | 1.1931 | 3.2911 | 3.9585 | 3.9585 | — | 181.896 | — | 0 / 0 | 0 / 24 | development |
| 33 | `pod_192` | 8 | 0.5316 | 1.5671 | 1.9458 | 1.9458 | — | 245.275 | — | 0 / 0 | 0 / 24 | development |
| 33 | `pod_256` | 8 | 0.3292 | 1.0363 | 1.2631 | 1.2631 | — | 316.447 | — | 0 / 0 | 0 / 24 | development |
| 33 | `pod_64` | 8 | 3.6615 | 9.2416 | 10.8059 | 10.8059 | — | 124.724 | — | 0 / 0 | 0 / 24 | development |
| 33 | `rom_q0` | 8 | 2.6471 | 8.9875 | 9.6063 | 9.6063 | — | 225.324 | — | 0 / 0 | 0 / 24 | development |
| 33 | `rom_q128` | 8 | 1.9470 | 6.8452 | 7.3625 | 7.3625 | — | 377.360 | — | 0 / 0 | 0 / 24 | development |
| 33 | `rom_q192` | 8 | 0.8437 | 2.4345 | 2.8901 | 2.8901 | — | 391.488 | — | 0 / 0 | 0 / 24 | development |
| 33 | `rom_q64` | 8 | 2.3985 | 8.2287 | 8.8661 | 8.8661 | — | 301.864 | — | 0 / 0 | 0 / 24 | development |

A missing physical-error or total-time cell means unmeasured, not zero. Physical errors require the attempt's separate reference-refinement qualification. A passing numerical audit verifies the recorded experiment; it does not establish good predictive accuracy, convergence of training or a competitive method.

## Navier–Stokes 3D — extra03

**Provisional:** Audited new paired development measurements with all four operator families and identical training membership. Independent saved-field, source, reference and sampled latent-history weak-gradient checks pass. The original strict frozen-field replay gate failed and remains failed; no cross-run numerical equivalence or implementation-speed ratio is claimed. The existing learned bank and head remain inaccurate; the larger-bank representation screen also misses its target. These negative results motivate the separately running matched augmentation and capacity experiment. Timed queries cover the declared first eight development cases. Final evaluation remains unopened.

Source `8ba0a11ee83aaeb36bb486add6517a51aa921a87`; job `3995695`; GPU `NVIDIA A100 80GB PCIe, 81920 MiB`. [Invocation data](../worktrees/2026-09-20-paper-ns3d/experiments/ns3d/runs/extra03/collected/output/result.json) and [independent audit](../worktrees/2026-09-20-paper-ns3d/experiments/ns3d/runs/extra03/numerical_validity.json).

Errors use the initial velocity-field norm, with all three components combined. The evolved metric excludes time zero. Physical error uses Fourier interpolation to the independently refined grid. Total timing includes host transfers; device timing includes initialization, evolution and every requested dense velocity field. Mesh size counts periodic points per axis. The timed cohort is a declared subset of the larger validation cohort; training-validation summaries are not substituted for its measured query errors. Saved latent histories support the linked independently sampled weak-gradient checks.

| Mesh | Method | Cases | Error median (%) | Error worst (%) | All-times worst (%) | Initial worst (%) | Physical worst (%) | GPU median (ms) | Total median (ms) | Nonfinite / nonstationary cases | Timing outliers / calls | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 32 | `deeponet3d_projected` | 8 | 35.8894 | 44.8450 | 44.8450 | 0.0000 | 44.8449 | 4.372 | 5.240 | 0 / 0 | 0 / 24 | development |
| 32 | `deeponet3d_raw` | 8 | 36.8223 | 45.3970 | 45.3970 | 0.0000 | 45.3965 | 4.124 | 4.983 | 0 / 0 | 0 / 24 | development |
| 32 | `fno3d_projected` | 8 | 1.0974 | 1.5121 | 1.5121 | 0.0000 | 1.5183 | 8.970 | 9.867 | 0 / 0 | 1 / 24 | development |
| 32 | `fno3d_raw` | 8 | 1.3422 | 1.7664 | 1.7664 | 0.0000 | 1.7604 | 8.787 | 9.676 | 0 / 0 | 0 / 24 | development |
| 32 | `fom_dt0.001` | 8 | 0.0031 | 0.0051 | 0.0051 | 0.0000 | 0.3651 | 24.712 | 25.737 | 0 / 0 | 0 / 24 | development |
| 32 | `fom_dt0.002` | 8 | 0.0130 | 0.0213 | 0.0213 | 0.0000 | 0.3663 | 12.810 | 13.823 | 0 / 0 | 0 / 24 | development |
| 32 | `fom_dt0.004` | 8 | 0.0524 | 0.0866 | 0.0866 | 0.0000 | 0.3745 | 6.886 | 7.822 | 0 / 0 | 0 / 24 | development |
| 32 | `fom_dt0.008` | 8 | 0.2125 | 0.3598 | 0.3598 | 0.0000 | 0.4711 | 3.893 | 4.760 | 0 / 0 | 0 / 24 | development |
| 32 | `free_bank_galerkin` | 8 | 13.4503 | 24.2696 | 24.2696 | 17.7985 | 24.2689 | 43.878 | 44.829 | 0 / 0 | 0 / 24 | development |
| 32 | `nmrom_q0` | 8 | 28.5319 | 50.4758 | 53.7445 | 53.7445 | 50.4758 | 779.231 | 781.336 | 0 / 0 | 0 / 24 | development |
| 32 | `nmrom_q128` | 8 | 25.2664 | 45.3348 | 46.7275 | 46.7275 | 45.3345 | 1865.999 | 1868.080 | 0 / 0 | 0 / 24 | development |
| 32 | `nmrom_q16` | 8 | 28.2539 | 49.9583 | 53.1214 | 53.1214 | 49.9583 | 810.785 | 812.945 | 0 / 0 | 0 / 24 | development |
| 32 | `nmrom_q32` | 8 | 27.8899 | 49.2381 | 52.3195 | 52.3195 | 49.2381 | 970.197 | 972.605 | 0 / 0 | 0 / 24 | development |
| 32 | `nmrom_q64` | 8 | 27.0931 | 47.6297 | 50.5176 | 50.5176 | 47.6297 | 1297.298 | 1299.700 | 0 / 0 | 0 / 24 | development |
| 32 | `pod_galerkin_128` | 8 | 34.5481 | 47.3334 | 50.1174 | 50.1174 | 47.3334 | 5.609 | 6.471 | 0 / 0 | 0 / 24 | development |
| 32 | `pod_galerkin_256` | 8 | 24.8410 | 36.0594 | 36.0594 | 34.7360 | 36.0592 | 11.294 | 12.129 | 0 / 0 | 0 / 24 | development |
| 32 | `pod_galerkin_512` | 8 | 15.2837 | 27.7537 | 27.7537 | 21.6223 | 27.7535 | 43.895 | 44.868 | 0 / 0 | 0 / 24 | development |
| 32 | `pod_weak_16` | 8 | 87.7996 | 91.9857 | 94.8670 | 94.8670 | 91.9857 | 25.084 | 26.861 | 0 / 0 | 0 / 24 | development |
| 32 | `pod_weak_32` | 8 | 78.1203 | 85.5263 | 89.3108 | 89.3108 | 85.5263 | 27.511 | 29.203 | 0 / 0 | 0 / 24 | development |
| 32 | `pod_weak_48` | 8 | 67.3545 | 76.1070 | 82.4515 | 82.4515 | 76.1069 | 35.081 | 36.888 | 0 / 0 | 0 / 24 | development |
| 32 | `transolver3d_projected` | 8 | 1.3773 | 2.0366 | 2.0366 | 0.0000 | 2.0439 | 3.398 | 4.311 | 0 / 0 | 1 / 24 | development |
| 32 | `transolver3d_raw` | 8 | 2.2017 | 2.9715 | 2.9715 | 0.0000 | 2.9295 | 3.317 | 4.214 | 0 / 0 | 1 / 24 | development |
| 32 | `unet3d_projected` | 8 | 3.0414 | 4.2830 | 4.2830 | 0.0000 | 4.2809 | 2.094 | 3.006 | 0 / 0 | 0 / 24 | development |
| 32 | `unet3d_raw` | 8 | 3.4540 | 4.6468 | 4.6468 | 0.0000 | 4.6463 | 1.982 | 2.841 | 0 / 0 | 0 / 24 | development |

A missing physical-error or total-time cell means unmeasured, not zero. Physical errors require the attempt's separate reference-refinement qualification. A passing numerical audit verifies the recorded experiment; it does not establish good predictive accuracy, convergence of training or a competitive method.

## Poisson 3D — seed04

**Provisional:** Audited independent-initialization development comparison. The same bank/head recipe is retrained; FNO/U-Net retain their earlier budgets, while DeepONet/Transolver receive common longer schedules. All methods share training membership. The uncorrected head varies with initialization; the high-correction endpoint remains accurate. All observed stopping checks pass, but selected latent states were not retained for independent gradient reconstruction in this source revision. Finer-grid POD is an offline mesh-adapted control; frozen neural transfer and native-interpolation variants remain distinct. Physical-domain-preserving FNO padding and native-sensor continuous-trunk DeepONet are separately named controls. Further training-recipe comparison and frozen final testing remain pending.

Source `99befa2a4a7152df74b3086255bc2f47efbb5b70`; job `3996479`; GPU `NVIDIA A100 80GB PCIe`. [Invocation data](../worktrees/2026-09-20-paper-p3d/experiments/paper-p3d/runs/seed04/archive/out/result.json) and [independent audit](../worktrees/2026-09-20-paper-p3d/experiments/paper-p3d/runs/seed04/audit-local.json).

Errors use the reference solution norm. There is one stationary output field; evolved, initial and all-times terminology does not apply. Total timing includes host transfers. Mesh size counts intervals per axis.

| Mesh | Method | Cases | Error median (%) | Error worst (%) | All-times worst (%) | Initial worst (%) | Physical worst (%) | GPU median (ms) | Total median (ms) | Nonfinite / nonstationary cases | Timing outliers / calls | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 32 | `deeponet3d_r128_w16` | 16 | 1.3221 | 3.5683 | — | — | 3.4966 | 2.778 | 3.573 | 0 / 0 | 0 / 48 | development |
| 32 | `dst_exact` | 16 | 0.0000 | 0.0000 | — | — | 0.2759 | 0.181 | 1.917 | 0 / 0 | 5 / 48 | development |
| 32 | `fno3d_w24_m8` | 16 | 0.1349 | 0.2366 | — | — | 0.4144 | 11.209 | 12.096 | 0 / 0 | 0 / 48 | development |
| 32 | `linear_bank_galerkin` | 16 | 0.0463 | 0.1443 | — | — | 0.3083 | 0.218 | 2.018 | 0 / 0 | 6 / 48 | development |
| 32 | `linear_bank_weak_qR` | 16 | 0.0457 | 0.1421 | — | — | 0.3081 | 0.255 | 1.989 | 0 / 0 | 6 / 48 | development |
| 32 | `nmrom_K16_q0_dense` | 16 | 0.2281 | 0.9704 | — | — | 0.9909 | 2.487 | 3.319 | 0 / 0 | 3 / 48 | development |
| 32 | `nmrom_K16_q32_dense` | 16 | 0.1021 | 0.6571 | — | — | 0.6968 | 2.670 | 3.537 | 0 / 0 | 0 / 48 | development |
| 32 | `nmrom_K16_q96_dense` | 16 | 0.0483 | 0.1765 | — | — | 0.3244 | 2.589 | 3.470 | 0 / 0 | 0 / 48 | development |
| 32 | `pod112_galerkin` | 16 | 0.0117 | 0.0685 | — | — | 0.2839 | 0.210 | 1.981 | 0 / 0 | 5 / 48 | development |
| 32 | `pod128_galerkin` | 16 | 0.0067 | 0.0370 | — | — | 0.2779 | 0.215 | 1.877 | 0 / 0 | 6 / 48 | development |
| 32 | `pod16_galerkin` | 16 | 1.8426 | 5.3174 | — | — | 5.2258 | 0.181 | 1.891 | 0 / 0 | 9 / 48 | development |
| 32 | `pod48_galerkin` | 16 | 0.1506 | 0.5672 | — | — | 0.6383 | 0.191 | 2.066 | 0 / 0 | 10 / 48 | development |
| 32 | `transolver3d_w48_s32` | 16 | 0.5089 | 0.7639 | — | — | 0.7735 | 4.162 | 5.065 | 0 / 0 | 0 / 48 | development |
| 32 | `unet3d_w16` | 16 | 0.1644 | 0.2310 | — | — | 0.3761 | 3.871 | 4.789 | 0 / 0 | 0 / 48 | development |
| 64 | `deeponet3d_r128_w16` | 16 | 25.7708 | 42.8642 | — | — | 42.9014 | 6.996 | 8.332 | 0 / 0 | 0 / 48 | development |
| 64 | `deeponet3d_r128_w16_native32_interpolate` | 16 | 1.2770 | 3.5146 | — | — | 3.4922 | 2.583 | 3.956 | 0 / 0 | 0 / 48 | development |
| 64 | `dst_exact` | 16 | 0.0000 | 0.0000 | — | — | 0.0687 | 0.221 | 2.616 | 0 / 0 | 8 / 48 | development |
| 64 | `fno3d_w24_m8` | 16 | 15.7351 | 16.7537 | — | — | 16.8054 | 17.353 | 18.721 | 0 / 0 | 0 / 48 | development |
| 64 | `fno3d_w24_m8_native32_interpolate` | 16 | 0.2755 | 0.3679 | — | — | 0.3541 | 11.430 | 12.854 | 0 / 0 | 0 / 48 | development |
| 64 | `fno3d_w24_m8_physical_padding17` | 16 | 0.9433 | 1.0835 | — | — | 1.1082 | 19.794 | 21.170 | 0 / 0 | 0 / 48 | development |
| 64 | `linear_bank_galerkin` | 16 | 0.0482 | 0.1409 | — | — | 0.1558 | 0.478 | 2.751 | 0 / 0 | 4 / 48 | development |
| 64 | `linear_bank_weak_qR` | 16 | 0.0474 | 0.1385 | — | — | 0.1539 | 0.942 | 3.327 | 0 / 0 | 2 / 48 | development |
| 64 | `nmrom_K16_q0_dense` | 16 | 0.2358 | 0.9596 | — | — | 0.9596 | 3.144 | 4.573 | 0 / 0 | 1 / 48 | development |
| 64 | `nmrom_K16_q32_dense` | 16 | 0.1054 | 0.6450 | — | — | 0.6451 | 3.217 | 4.611 | 0 / 0 | 0 / 48 | development |
| 64 | `nmrom_K16_q96_dense` | 16 | 0.0498 | 0.1726 | — | — | 0.1849 | 3.371 | 4.809 | 0 / 0 | 0 / 48 | development |
| 64 | `pod112_galerkin` | 16 | 0.0115 | 0.0672 | — | — | 0.0958 | 0.441 | 2.760 | 0 / 0 | 2 / 48 | development |
| 64 | `pod128_galerkin` | 16 | 0.0066 | 0.0360 | — | — | 0.0771 | 0.476 | 2.864 | 0 / 0 | 4 / 48 | development |
| 64 | `pod16_galerkin` | 16 | 1.8304 | 5.2702 | — | — | 5.2463 | 0.204 | 2.682 | 0 / 0 | 8 / 48 | development |
| 64 | `pod48_galerkin` | 16 | 0.1484 | 0.5576 | — | — | 0.5639 | 0.285 | 2.720 | 0 / 0 | 7 / 48 | development |
| 64 | `transolver3d_w48_s32` | 16 | 5.5317 | 7.2965 | — | — | 7.2610 | 12.442 | 13.839 | 0 / 0 | 0 / 48 | development |
| 64 | `transolver3d_w48_s32_native32_interpolate` | 16 | 0.5088 | 0.7859 | — | — | 0.7616 | 4.337 | 5.731 | 0 / 0 | 0 / 48 | development |
| 64 | `unet3d_w16` | 16 | 293.1485 | 532.3332 | — | — | 532.5369 | 11.889 | 13.294 | 0 / 0 | 0 / 48 | development |
| 64 | `unet3d_w16_native32_interpolate` | 16 | 0.2949 | 0.3791 | — | — | 0.3629 | 4.016 | 5.460 | 0 / 0 | 0 / 48 | development |

A missing physical-error or total-time cell means unmeasured, not zero. Physical errors require the attempt's separate reference-refinement qualification. A passing numerical audit verifies the recorded experiment; it does not establish good predictive accuracy, convergence of training or a competitive method.

## Glossary

- **NM-ROM / ROM:** a neural-manifold reduced model / a model solving for a smaller state.
- **Bank, head, rank:** learned spatial functions, their nonlinear coefficient map, and the number of spatial functions.
- **q / K / R:** correction rank / nonlinear latent dimension / full learned-bank rank; values in method names identify the saved configuration.
- **POD:** a linear reduced basis computed from training snapshots.
- **FOM / DST:** the full-grid numerical solver / a discrete sine-transform direct solver.
- **Dense / EQ:** full-grid contractions / sampled empirical quadrature; an EQ name alone does not mean its accuracy certificate passed.
- **Free bank / Galerkin / weak:** unrestricted bank coefficients / projection against the basis / residual projection against smooth tests.
- **Mesh / cases / calls:** grid size per spatial axis under the stated convention / distinct inputs / timed solver invocations including repetitions.
- **Error median / worst:** median or maximum over distinct cases of each case's largest evolved error, or stationary solution error for Poisson; the largest value across repeated calls is retained.
- **All-times / initial:** maximum including time zero / error from compressing the initial field.
- **Physical error:** discrepancy against the independently refined reference; a refinement test is empirical, not a proved continuum bound.
- **GPU / total median:** median elapsed milliseconds on the device / including recorded host transfers. Timings may only be compared within a job and matching output contract.
- **Nonfinite / nonstationary cases:** inputs with an invalid output in any repeat / a failed declared numerical stopping check in any repeat.
- **Timing outliers:** calls slower than one and a half times that method's median; every measured time remains in the machine-readable output.
- **Status / certificate:** a row's remaining qualification / the declared held-out check that sampled quadrature reproduces the required moments accurately enough.
- **Representation / snapshot / projection:** the fields a model can express / one saved state at one time / the nearest field in a linear basis under the stated norm. A best-found neural fit uses numerical optimization and is not a proof of global optimality.
- **Snapshots above declared target:** saved states exceeding the representation accuracy threshold fixed in that experiment's design; these are not independent trajectory counts.
- **Provisional / development / final cohort:** not accepted as a final paper claim / data available during selection / independent data reserved until configurations freeze.
- **Source / job / audit:** pinned scientific code revision / cluster allocation identifier / independent validation record.
