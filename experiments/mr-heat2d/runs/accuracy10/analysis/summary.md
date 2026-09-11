# Heat accuracy from fixed-bank initial and tail training

These audited development results compare matched training procedures while retaining the nonlinear decoder and weak heat solver. They are provisional for publication because only one training seed and a restricted family have been tested; final cases remain unopened.

Source `c7cc18d4a12820f471076fcbb5e95cce3c6c75c4`, GPU job `3563072`, NVIDIA A100 80GB PCIe. All 1152 timed invocations are retained, with cost and field error from the same invocation.

The initial-plus-tail arm is the predeclared primary. The initial-only arm isolates the effect of emphasizing the initial field; it is not a newly selected primary. The frozen model on the older opened cohort must be compared separately from its newly added development cases.

## Opened development

| N | Method | GPU ms | Host ms | Median / worst error % | Tight / loose CG GPU ratio | Nonstationary initial / steps | Qualifies at 5% |
|---|---|---|---|---|---|---|---|
| 64 | nmrom_frozen | 12.048164 | 12.697080 | 3.474489 / 4.559260 | 0.432292 / 0.191707 | 0 / 0 | True |
| 64 | nmrom_uniform | 11.847643 | 12.340171 | 2.394971 / 3.511178 | 0.439608 / 0.194951 | 3 / 0 | False |
| 64 | nmrom_initial_only | 11.229900 | 11.663054 | 2.391783 / 3.306601 | 0.463791 / 0.205675 | 0 / 0 | True |
| 64 | nmrom_initial_tail | 11.127344 | 11.532207 | 2.355837 / 3.598146 | 0.468065 / 0.207571 | 0 / 0 | True |
| 64 | fom_cg_cn | 5.208323 | 5.554547 | 0.052478 / 0.067379 | 1.000000 / 0.443466 | 0 / 0 | True |
| 64 | fom_cg_cn_tol1e2 | 2.309714 | 2.646217 | 1.273798 / 1.586031 | 2.254964 / 1.000000 | 0 / 0 | True |
| 64 | linear_weak_exact | 0.113402 | 0.452571 | 0.838723 / 1.675754 | 45.927953 / 20.367485 | 0 / 0 | True |
| 64 | fom_same_grid | 0.169710 | 0.490932 | 0.064494 / 0.089749 | 30.689543 / 13.609768 | 0 / 0 | True |
| 256 | nmrom_frozen | 11.826391 | 12.972996 | 3.472565 / 4.555681 | 1.713760 / 0.553782 | 0 / 0 | True |
| 256 | nmrom_uniform | 11.469316 | 12.466889 | 2.403580 / 3.511185 | 1.767114 / 0.571023 | 3 / 0 | False |
| 256 | nmrom_initial_only | 10.978715 | 11.927652 | 2.391812 / 3.306614 | 1.846081 / 0.596540 | 0 / 0 | True |
| 256 | nmrom_initial_tail | 10.866611 | 11.872177 | 2.353668 / 3.598163 | 1.865126 / 0.602694 | 0 / 0 | True |
| 256 | fom_cg_cn | 20.267594 | 21.205003 | 0.014679 / 0.031261 | 1.000000 / 0.323139 | 0 / 0 | True |
| 256 | fom_cg_cn_tol1e2 | 6.549245 | 7.374055 | 0.755698 / 0.925464 | 3.094646 / 1.000000 | 0 / 0 | True |
| 256 | linear_weak_exact | 0.129862 | 0.980233 | 0.838760 / 1.675830 | 156.070252 / 50.432347 | 0 / 0 | True |
| 256 | fom_same_grid | 0.205205 | 1.025718 | 0.004028 / 0.005602 | 98.767516 / 31.915612 | 0 / 0 | True |
| 1024 | nmrom_frozen | 11.951280 | 26.835650 | 3.472489 / 4.555479 | 16.211262 / 4.521411 | 0 / 0 | True |
| 1024 | nmrom_uniform | 12.038750 | 27.498759 | 2.404151 / 3.511185 | 16.093476 / 4.488560 | 3 / 0 | False |
| 1024 | nmrom_initial_only | 11.603191 | 27.382593 | 2.391813 / 3.306614 | 16.697591 / 4.657051 | 0 / 0 | True |
| 1024 | nmrom_initial_tail | 11.837538 | 27.388241 | 2.353578 / 3.598163 | 16.367029 / 4.564856 | 0 / 0 | True |
| 1024 | fom_cg_cn | 193.745335 | 209.217314 | 0.017574 / 0.034909 | 1.000000 / 0.278906 | 0 / 0 | True |
| 1024 | fom_cg_cn_tol1e2 | 54.036655 | 69.677957 | 0.562778 / 0.770685 | 3.585443 / 1.000000 | 0 / 0 | True |
| 1024 | linear_weak_exact | 0.526776 | 16.080830 | 0.838760 / 1.675830 | 367.794487 / 102.579935 | 0 / 0 | True |
| 1024 | fom_same_grid | 0.893139 | 16.434310 | 0.000252 / 0.000350 | 216.926313 / 60.501959 | 0 / 0 | True |

## Confirmation development

| N | Method | GPU ms | Host ms | Median / worst error % | Tight / loose CG GPU ratio | Nonstationary initial / steps | Qualifies at 5% |
|---|---|---|---|---|---|---|---|
| 64 | nmrom_frozen | 11.786808 | 12.407935 | 3.605441 / 7.595341 | 0.432731 / 0.194598 | 0 / 0 | False |
| 64 | nmrom_uniform | 10.979868 | 11.420182 | 2.565570 / 5.335268 | 0.464533 / 0.208899 | 0 / 0 | False |
| 64 | nmrom_initial_only | 10.549965 | 11.040690 | 2.285200 / 4.867408 | 0.483463 / 0.217412 | 0 / 0 | True |
| 64 | nmrom_initial_tail | 11.178499 | 11.639858 | 2.410386 / 4.762500 | 0.456279 / 0.205187 | 0 / 0 | True |
| 64 | fom_cg_cn | 5.100515 | 5.441573 | 0.057502 / 0.069093 | 1.000000 / 0.449697 | 0 / 0 | True |
| 64 | fom_cg_cn_tol1e2 | 2.293687 | 2.624223 | 1.291785 / 1.471030 | 2.223719 / 1.000000 | 0 / 0 | True |
| 64 | linear_weak_exact | 0.112829 | 0.435940 | 0.833008 / 1.121597 | 45.205701 / 20.328873 | 0 / 0 | True |
| 64 | fom_same_grid | 0.182436 | 0.476823 | 0.073519 / 0.093826 | 27.957840 / 12.572560 | 0 / 0 | True |
| 256 | nmrom_frozen | 11.463344 | 12.508564 | 3.605465 / 7.595346 | 1.691687 / 0.573926 | 0 / 0 | False |
| 256 | nmrom_uniform | 10.774811 | 11.848215 | 2.565602 / 5.335281 | 1.799790 / 0.610601 | 0 / 0 | False |
| 256 | nmrom_initial_only | 10.431563 | 11.383845 | 2.282123 / 4.867423 | 1.859011 / 0.630692 | 0 / 0 | True |
| 256 | nmrom_initial_tail | 10.532825 | 11.516118 | 2.408542 / 4.762515 | 1.841139 / 0.624629 | 0 / 0 | True |
| 256 | fom_cg_cn | 19.392394 | 20.373069 | 0.019820 / 0.035160 | 1.000000 / 0.339262 | 0 / 0 | True |
| 256 | fom_cg_cn_tol1e2 | 6.579107 | 7.414749 | 0.786803 / 0.931950 | 2.947572 / 1.000000 | 0 / 0 | True |
| 256 | linear_weak_exact | 0.132433 | 0.952663 | 0.833123 / 1.121648 | 146.431711 / 49.678750 | 0 / 0 | True |
| 256 | fom_same_grid | 0.209601 | 0.997290 | 0.004590 / 0.005856 | 92.520517 / 31.388718 | 0 / 0 | True |
| 1024 | nmrom_frozen | 11.971563 | 26.552387 | 3.605465 / 7.595346 | 14.943161 / 4.445712 | 0 / 0 | False |
| 1024 | nmrom_uniform | 11.374651 | 27.047991 | 2.565602 / 5.335282 | 15.727339 / 4.679011 | 0 / 0 | False |
| 1024 | nmrom_initial_only | 11.109028 | 26.888075 | 2.282036 / 4.867423 | 16.103390 / 4.790889 | 0 / 0 | True |
| 1024 | nmrom_initial_tail | 11.485037 | 27.192750 | 2.408453 / 4.762515 | 15.576180 / 4.634040 | 0 / 0 | True |
| 1024 | fom_cg_cn | 178.893005 | 194.371108 | 0.023276 / 0.039098 | 1.000000 / 0.297508 | 0 / 0 | True |
| 1024 | fom_cg_cn_tol1e2 | 53.222122 | 68.807083 | 0.631952 / 0.790111 | 3.361253 / 1.000000 | 0 / 0 | True |
| 1024 | linear_weak_exact | 0.529914 | 16.140076 | 0.833123 / 1.121648 | 337.588795 / 100.435409 | 0 / 0 | True |
| 1024 | fom_same_grid | 0.905514 | 16.629342 | 0.000287 / 0.000366 | 197.559523 / 58.775563 | 0 / 0 | True |

## All development

| N | Method | GPU ms | Host ms | Median / worst error % | Tight / loose CG GPU ratio | Nonstationary initial / steps | Qualifies at 5% |
|---|---|---|---|---|---|---|---|
| 64 | nmrom_frozen | 11.942480 | 12.624216 | 3.474489 / 7.595341 | 0.434864 / 0.192706 | 0 / 0 | False |
| 64 | nmrom_uniform | 11.339868 | 11.832074 | 2.394971 / 5.335268 | 0.457973 / 0.202946 | 3 / 0 | False |
| 64 | nmrom_initial_only | 10.973598 | 11.610159 | 2.391783 / 4.867408 | 0.473259 / 0.209720 | 0 / 0 | True |
| 64 | nmrom_initial_tail | 11.127344 | 11.532207 | 2.355837 / 4.762500 | 0.466720 / 0.206822 | 0 / 0 | True |
| 64 | fom_cg_cn | 5.193358 | 5.540019 | 0.053076 / 0.069093 | 1.000000 / 0.443140 | 0 / 0 | True |
| 64 | fom_cg_cn_tol1e2 | 2.301383 | 2.632502 | 1.291785 / 1.586031 | 2.256625 / 1.000000 | 0 / 0 | True |
| 64 | linear_weak_exact | 0.112989 | 0.445599 | 0.838723 / 1.675754 | 45.963353 / 20.368186 | 0 / 0 | True |
| 64 | fom_same_grid | 0.173806 | 0.489892 | 0.065996 / 0.093826 | 29.880284 / 13.241140 | 0 / 0 | True |
| 256 | nmrom_frozen | 11.594060 | 12.821892 | 3.472565 / 7.595346 | 1.738233 / 0.565306 | 0 / 0 | False |
| 256 | nmrom_uniform | 11.265571 | 12.233312 | 2.403580 / 5.335281 | 1.788917 / 0.581790 | 3 / 0 | False |
| 256 | nmrom_initial_only | 10.825906 | 11.826781 | 2.391812 / 4.867423 | 1.861569 / 0.605418 | 0 / 0 | True |
| 256 | nmrom_initial_tail | 10.847778 | 11.804435 | 2.353668 / 4.762515 | 1.857816 / 0.604197 | 0 / 0 | True |
| 256 | fom_cg_cn | 20.153173 | 21.050890 | 0.015407 / 0.035160 | 1.000000 / 0.325219 | 0 / 0 | True |
| 256 | fom_cg_cn_tol1e2 | 6.554194 | 7.389063 | 0.766264 / 0.931950 | 3.074851 / 1.000000 | 0 / 0 | True |
| 256 | linear_weak_exact | 0.130083 | 0.979336 | 0.838760 / 1.675830 | 154.925007 / 50.384552 | 0 / 0 | True |
| 256 | fom_same_grid | 0.208149 | 1.010324 | 0.004121 / 0.005856 | 96.820699 / 31.487929 | 0 / 0 | True |
| 1024 | nmrom_frozen | 11.970928 | 26.835650 | 3.472489 / 7.595346 | 15.984565 / 4.498764 | 0 / 0 | False |
| 1024 | nmrom_uniform | 11.724363 | 27.155451 | 2.404151 / 5.335282 | 16.320722 / 4.593373 | 3 / 0 | False |
| 1024 | nmrom_initial_only | 11.371227 | 27.103460 | 2.391813 / 4.867423 | 16.827566 / 4.736021 | 0 / 0 | True |
| 1024 | nmrom_initial_tail | 11.623313 | 27.322126 | 2.353578 / 4.762515 | 16.462611 / 4.633307 | 0 / 0 | True |
| 1024 | fom_cg_cn | 191.350069 | 207.142028 | 0.018386 / 0.039098 | 1.000000 / 0.281444 | 0 / 0 | True |
| 1024 | fom_cg_cn_tol1e2 | 53.854375 | 69.577017 | 0.576822 / 0.790111 | 3.553102 / 1.000000 | 0 / 0 | True |
| 1024 | linear_weak_exact | 0.528061 | 16.094075 | 0.838760 / 1.675830 | 362.363526 / 101.985127 | 0 / 0 | True |
| 1024 | fom_same_grid | 0.895605 | 16.483799 | 0.000258 / 0.000366 | 213.654544 / 60.131841 | 0 / 0 | True |

## Training and representation

| Model | Full / initial / later training relative MSE | Worst initial / later diagnostic error % | Selected diagnostic nonstationary fits |
|---|---|---|---|
| frozen | 0.000343334528629 / 0.000936065997777 / 0.0002247882348 | 7.595337 / 3.451458 | 2 |
| uniform | 0.000166550012798 / 0.000507268284762 / 9.84063584058e-05 | 5.335266 / 2.753779 | 1 |
| initial_only | 0.000172471388331 / 0.000362181850929 / 0.000134529295812 | 4.867403 / 2.726243 | 0 |
| initial_tail | 0.000193319352991 / 0.00036155410927 / 0.000159672401735 | 4.762499 / 2.705043 | 0 |

The unchanged bank's worst training-mesh development projection error is 1.675754%. Fine-mesh per-case full-field projection and every attempted fit's gradient are retained in `analysis/panel.json` and the raw result. Diagnostic best-found fits are not certified global minima.

GPU times are pooled repetition medians. Tight CG uses tolerance 1e-6; loose CG uses 1e-2, both with the matched Crank–Nicolson time step. Ratios above one favor the indicated method. Every repetition is retained; an outlier exceeds 1.5 times the median of its own case/method/mesh group. Exact outlier counts, solver exits and paired ratios are in `analysis/panel.json`.

## Plain-language glossary

- **N:** intervals along each spatial axis; full output contains the interior grid.
- **GPU / host ms:** blocked full GPU input-to-output query time / the same invocation including input and output transfers.
- **Median / worst error:** median or largest case error, where each case uses its worst output time and all retained repetitions.
- **CG ratio:** the named full-order conjugate-gradient solver time divided by this method time.
- **Stationarity / qualifies:** sufficiently small reduced objective gradient / all attempted solves stationary and every field meeting the empirical physical target.
- **Bank / head / MSE:** learned spatial functions / nonlinear coefficient map / mean squared relative reconstruction error.
- **Opened / confirmation development:** previous development cases / new draws fixed before this training round; neither is the final paper cohort.
- **Primary / ablation:** setting declared before results / controlled alternative isolating a training change.
