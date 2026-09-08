# Frozen heat transfer and efficient full-output cost

These bounded development results are checked against preserved full fields. They use both frozen expanded-coverage heads and unchanged initializer libraries on original and fresh inputs from the restricted single-bump family; final confirmation remains unopened.

Across the union cohort and all requested meshes, the largest head current-relative physical error was 4.560009%. At the empirically adjusted full-grid 5% target, a head qualified at 5 meshes and beat the selected efficient FOM at 0 of those meshes. This describes this restricted development family only.

The native independent NumPy audit checked 1152 invocations and 528 field files; maximum metric disagreement was 5.26245713672e-14. The empirical spectral refinement discrepancy was 1.62810884586e-12. This is observed convergence evidence, with no rigorous physical error certificate.

Independent SciPy propagation and NumPy physical interpolation reproduce the stored FOM fields to relative discrepancy 6.44690924177e-16, and the continuum-spectral reference trajectories to 1.1044037433e-15.

[Accuracy and complete-query figure](heat-transfer.png) · [Input, device and output cost figure](heat-transfer-components.png)

Every FOM returns the exact supplied initial field. Host coarse restriction, GPU propagation, physically aligned interpolation and complete contiguous host output construction are charged. ROM outputs include its actual fitted initial state; full input projection, nonlinear fitting, evolution and full readout are charged. Both use the same requested host output grids and times.

The run took 10.786947 minutes. The audit found 0 selected nonstationary initial fits and 0 nonstationary evolution steps across all repetitions. Failures remain in raw records and cannot qualify for target selection.

Costs below are medians of per-case timing medians. All errors are maxima over every case, repetition and output time in the named cohort. Each head remains separately visible; this experiment changes neither its representation nor its training coverage.

| Cohort | Output intervals | Method | Query ms | Full current error % | Common current error % | Initial full error % | Invalid cases |
|---|---:|---|---:|---:|---:|---:|---:|
| original | 64 | expanded_seed790714 | 12.331279 | 3.953113 | 3.953113 | 3.953113 | 0 |
| original | 64 | expanded_seed790715 | 11.736978 | 4.030347 | 4.030347 | 4.030347 | 0 |
| original | 64 | fom_dst_16 | 1.140582 | 2.230146 | 2.230146 | 0.000000 | 0 |
| original | 64 | fom_dst_32 | 0.521205 | 0.543078 | 0.543078 | 0.000000 | 0 |
| original | 64 | fom_dst_64 | 0.517002 | 0.070921 | 0.070921 | 0.000000 | 0 |
| original | 128 | expanded_seed790714 | 12.060446 | 3.953116 | 3.953113 | 3.953116 | 0 |
| original | 128 | expanded_seed790715 | 11.775822 | 4.030350 | 4.030347 | 4.030350 | 0 |
| original | 128 | fom_dst_128 | 0.682671 | 0.017718 | 0.017718 | 0.000000 | 0 |
| original | 128 | fom_dst_16 | 1.313582 | 2.233559 | 2.230146 | 0.000000 | 0 |
| original | 128 | fom_dst_32 | 0.693559 | 0.557332 | 0.543078 | 0.000000 | 0 |
| original | 128 | fom_dst_64 | 0.664286 | 0.135743 | 0.070921 | 0.000000 | 0 |
| original | 256 | expanded_seed790714 | 12.762246 | 3.953116 | 3.953113 | 3.953116 | 0 |
| original | 256 | expanded_seed790715 | 12.407728 | 4.030350 | 4.030347 | 4.030350 | 0 |
| original | 256 | fom_dst_128 | 1.174383 | 0.033934 | 0.017718 | 0.000000 | 0 |
| original | 256 | fom_dst_16 | 1.936381 | 2.233763 | 2.230146 | 0.000000 | 0 |
| original | 256 | fom_dst_256 | 1.216128 | 0.004429 | 0.004429 | 0.000000 | 0 |
| original | 256 | fom_dst_32 | 1.208174 | 0.558155 | 0.543078 | 0.000000 | 0 |
| original | 256 | fom_dst_64 | 1.174177 | 0.139316 | 0.070921 | 0.000000 | 0 |
| original | 512 | expanded_seed790714 | 14.921848 | 3.953116 | 3.953113 | 3.953116 | 0 |
| original | 512 | expanded_seed790715 | 15.577164 | 4.030350 | 4.030347 | 4.030350 | 0 |
| original | 512 | fom_dst_128 | 5.020113 | 0.034828 | 0.017718 | 0.000000 | 0 |
| original | 512 | fom_dst_16 | 4.534181 | 2.233774 | 2.230146 | 0.000000 | 0 |
| original | 512 | fom_dst_32 | 5.127869 | 0.558193 | 0.543078 | 0.000000 | 0 |
| original | 512 | fom_dst_512 | 5.340955 | 0.001107 | 0.001107 | 0.000000 | 0 |
| original | 512 | fom_dst_64 | 5.011191 | 0.139520 | 0.070921 | 0.000000 | 0 |
| original | 1024 | expanded_seed790714 | 28.148306 | 3.953116 | 3.953113 | 3.953116 | 0 |
| original | 1024 | expanded_seed790715 | 27.151294 | 4.030350 | 4.030347 | 4.030350 | 0 |
| original | 1024 | fom_dst_1024 | 24.795638 | 0.000277 | 0.000277 | 0.000000 | 0 |
| original | 1024 | fom_dst_128 | 23.348211 | 0.034879 | 0.017718 | 0.000000 | 0 |
| original | 1024 | fom_dst_16 | 20.318986 | 2.233774 | 2.230146 | 0.000000 | 0 |
| original | 1024 | fom_dst_32 | 23.029349 | 0.558192 | 0.543078 | 0.000000 | 0 |
| original | 1024 | fom_dst_64 | 23.165786 | 0.139529 | 0.070921 | 0.000000 | 0 |
| fresh | 64 | expanded_seed790714 | 12.770320 | 4.560001 | 4.560001 | 4.560001 | 0 |
| fresh | 64 | expanded_seed790715 | 12.752228 | 4.559260 | 4.559260 | 4.545174 | 0 |
| fresh | 64 | fom_dst_16 | 1.137556 | 2.585355 | 2.585355 | 0.000000 | 0 |
| fresh | 64 | fom_dst_32 | 0.547381 | 0.635784 | 0.635784 | 0.000000 | 0 |
| fresh | 64 | fom_dst_64 | 0.509903 | 0.089749 | 0.089749 | 0.000000 | 0 |
| fresh | 128 | expanded_seed790714 | 12.439358 | 4.560008 | 4.560001 | 4.560008 | 0 |
| fresh | 128 | expanded_seed790715 | 12.518827 | 4.556344 | 4.556334 | 4.545182 | 0 |
| fresh | 128 | fom_dst_128 | 0.667090 | 0.022414 | 0.022414 | 0.000000 | 0 |
| fresh | 128 | fom_dst_16 | 1.550141 | 2.580599 | 2.585355 | 0.000000 | 0 |
| fresh | 128 | fom_dst_32 | 0.714539 | 0.644137 | 0.635784 | 0.000000 | 0 |
| fresh | 128 | fom_dst_64 | 0.681631 | 0.158804 | 0.089749 | 0.000000 | 0 |
| fresh | 256 | expanded_seed790714 | 13.283650 | 4.560009 | 4.560001 | 4.560009 | 0 |
| fresh | 256 | expanded_seed790715 | 13.733003 | 4.555681 | 4.555670 | 4.545182 | 0 |
| fresh | 256 | fom_dst_128 | 1.180990 | 0.039692 | 0.022414 | 0.000000 | 0 |
| fresh | 256 | fom_dst_16 | 2.016193 | 2.578601 | 2.585355 | 0.000000 | 0 |
| fresh | 256 | fom_dst_256 | 1.209482 | 0.005602 | 0.005602 | 0.000000 | 0 |
| fresh | 256 | fom_dst_32 | 1.210508 | 0.642842 | 0.635784 | 0.000000 | 0 |
| fresh | 256 | fom_dst_64 | 1.157405 | 0.160895 | 0.089749 | 0.000000 | 0 |
| fresh | 512 | expanded_seed790714 | 15.055508 | 4.560009 | 4.560001 | 4.560009 | 0 |
| fresh | 512 | expanded_seed790715 | 16.210593 | 4.555520 | 4.555508 | 4.545182 | 0 |
| fresh | 512 | fom_dst_128 | 5.075152 | 0.040215 | 0.022414 | 0.000000 | 0 |
| fresh | 512 | fom_dst_16 | 4.508266 | 2.578051 | 2.585355 | 0.000000 | 0 |
| fresh | 512 | fom_dst_32 | 5.182241 | 0.642307 | 0.635784 | 0.000000 | 0 |
| fresh | 512 | fom_dst_512 | 5.024727 | 0.001400 | 0.001400 | 0.000000 | 0 |
| fresh | 512 | fom_dst_64 | 5.002105 | 0.160565 | 0.089749 | 0.000000 | 0 |
| fresh | 1024 | expanded_seed790714 | 28.177594 | 4.560009 | 4.560001 | 4.560009 | 0 |
| fresh | 1024 | expanded_seed790715 | 27.855190 | 4.555479 | 4.555468 | 4.545182 | 0 |
| fresh | 1024 | fom_dst_1024 | 24.937231 | 0.000350 | 0.000350 | 0.000000 | 0 |
| fresh | 1024 | fom_dst_128 | 23.240902 | 0.040132 | 0.022414 | 0.000000 | 0 |
| fresh | 1024 | fom_dst_16 | 21.410562 | 2.577910 | 2.585355 | 0.000000 | 0 |
| fresh | 1024 | fom_dst_32 | 23.229804 | 0.642160 | 0.635784 | 0.000000 | 0 |
| fresh | 1024 | fom_dst_64 | 23.376217 | 0.160429 | 0.089749 | 0.000000 | 0 |
| union | 64 | expanded_seed790714 | 12.592115 | 4.560001 | 4.560001 | 4.560001 | 0 |
| union | 64 | expanded_seed790715 | 12.565909 | 4.559260 | 4.559260 | 4.545174 | 0 |
| union | 64 | fom_dst_16 | 1.140582 | 2.585355 | 2.585355 | 0.000000 | 0 |
| union | 64 | fom_dst_32 | 0.540558 | 0.635784 | 0.635784 | 0.000000 | 0 |
| union | 64 | fom_dst_64 | 0.509903 | 0.089749 | 0.089749 | 0.000000 | 0 |
| union | 128 | expanded_seed790714 | 12.260174 | 4.560008 | 4.560001 | 4.560008 | 0 |
| union | 128 | expanded_seed790715 | 12.310351 | 4.556344 | 4.556334 | 4.545182 | 0 |
| union | 128 | fom_dst_128 | 0.680490 | 0.022414 | 0.022414 | 0.000000 | 0 |
| union | 128 | fom_dst_16 | 1.462293 | 2.580599 | 2.585355 | 0.000000 | 0 |
| union | 128 | fom_dst_32 | 0.703910 | 0.644137 | 0.635784 | 0.000000 | 0 |
| union | 128 | fom_dst_64 | 0.673200 | 0.158804 | 0.089749 | 0.000000 | 0 |
| union | 256 | expanded_seed790714 | 13.207777 | 4.560009 | 4.560001 | 4.560009 | 0 |
| union | 256 | expanded_seed790715 | 13.351068 | 4.555681 | 4.555670 | 4.545182 | 0 |
| union | 256 | fom_dst_128 | 1.178230 | 0.039692 | 0.022414 | 0.000000 | 0 |
| union | 256 | fom_dst_16 | 1.957491 | 2.578601 | 2.585355 | 0.000000 | 0 |
| union | 256 | fom_dst_256 | 1.214582 | 0.005602 | 0.005602 | 0.000000 | 0 |
| union | 256 | fom_dst_32 | 1.210508 | 0.642842 | 0.635784 | 0.000000 | 0 |
| union | 256 | fom_dst_64 | 1.164132 | 0.160895 | 0.089749 | 0.000000 | 0 |
| union | 512 | expanded_seed790714 | 14.986074 | 4.560009 | 4.560001 | 4.560009 | 0 |
| union | 512 | expanded_seed790715 | 16.071136 | 4.555520 | 4.555508 | 4.545182 | 0 |
| union | 512 | fom_dst_128 | 5.055869 | 0.040215 | 0.022414 | 0.000000 | 0 |
| union | 512 | fom_dst_16 | 4.515824 | 2.578051 | 2.585355 | 0.000000 | 0 |
| union | 512 | fom_dst_32 | 5.182241 | 0.642307 | 0.635784 | 0.000000 | 0 |
| union | 512 | fom_dst_512 | 5.338911 | 0.001400 | 0.001400 | 0.000000 | 0 |
| union | 512 | fom_dst_64 | 5.002105 | 0.160565 | 0.089749 | 0.000000 | 0 |
| union | 1024 | expanded_seed790714 | 28.160363 | 4.560009 | 4.560001 | 4.560009 | 0 |
| union | 1024 | expanded_seed790715 | 27.732991 | 4.555479 | 4.555468 | 4.545182 | 0 |
| union | 1024 | fom_dst_1024 | 24.919446 | 0.000350 | 0.000350 | 0.000000 | 0 |
| union | 1024 | fom_dst_128 | 23.278861 | 0.040132 | 0.022414 | 0.000000 | 0 |
| union | 1024 | fom_dst_16 | 21.410562 | 2.577910 | 2.585355 | 0.000000 | 0 |
| union | 1024 | fom_dst_32 | 23.144724 | 0.642160 | 0.635784 | 0.000000 | 0 |
| union | 1024 | fom_dst_64 | 23.296839 | 0.160429 | 0.089749 | 0.000000 | 0 |

Selection uses every repetition's validity and empirical adjusted physical error $(e+\delta)/(1-\delta)$, where $e$ is observed current-relative error and $\delta$ is the paired reference-refinement discrepancy in the same norm. Full requested-grid and common-grid selections remain separate. The ratio is the median of per-case FOM/ROM ratios of case timing medians; values above one would favor ROM. A missing ROM means no head qualified, not zero runtime.

| Cohort | Output | Norm | Target % | Selected FOM | Selected ROM | FOM ms | ROM ms | Paired FOM/ROM |
|---|---:|---|---:|---|---|---:|---:|---:|
| original | 64 | full | 10 | fom_dst_64 | expanded_seed790715 | 0.517002 | 11.736978 | 0.043802 |
| original | 64 | full | 5 | fom_dst_64 | expanded_seed790715 | 0.517002 | 11.736978 | 0.043802 |
| original | 64 | full | 1 | fom_dst_64 | — | 0.517002 | — | — |
| original | 64 | full | 0.1 | fom_dst_64 | — | 0.517002 | — | — |
| original | 64 | common | 10 | fom_dst_64 | expanded_seed790715 | 0.517002 | 11.736978 | 0.043802 |
| original | 64 | common | 5 | fom_dst_64 | expanded_seed790715 | 0.517002 | 11.736978 | 0.043802 |
| original | 64 | common | 1 | fom_dst_64 | — | 0.517002 | — | — |
| original | 64 | common | 0.1 | fom_dst_64 | — | 0.517002 | — | — |
| original | 128 | full | 10 | fom_dst_64 | expanded_seed790715 | 0.664286 | 11.775822 | 0.056944 |
| original | 128 | full | 5 | fom_dst_64 | expanded_seed790715 | 0.664286 | 11.775822 | 0.056944 |
| original | 128 | full | 1 | fom_dst_64 | — | 0.664286 | — | — |
| original | 128 | full | 0.1 | fom_dst_128 | — | 0.682671 | — | — |
| original | 128 | common | 10 | fom_dst_64 | expanded_seed790715 | 0.664286 | 11.775822 | 0.056944 |
| original | 128 | common | 5 | fom_dst_64 | expanded_seed790715 | 0.664286 | 11.775822 | 0.056944 |
| original | 128 | common | 1 | fom_dst_64 | — | 0.664286 | — | — |
| original | 128 | common | 0.1 | fom_dst_64 | — | 0.664286 | — | — |
| original | 256 | full | 10 | fom_dst_64 | expanded_seed790715 | 1.174177 | 12.407728 | 0.093984 |
| original | 256 | full | 5 | fom_dst_64 | expanded_seed790715 | 1.174177 | 12.407728 | 0.093984 |
| original | 256 | full | 1 | fom_dst_64 | — | 1.174177 | — | — |
| original | 256 | full | 0.1 | fom_dst_128 | — | 1.174383 | — | — |
| original | 256 | common | 10 | fom_dst_64 | expanded_seed790715 | 1.174177 | 12.407728 | 0.093984 |
| original | 256 | common | 5 | fom_dst_64 | expanded_seed790715 | 1.174177 | 12.407728 | 0.093984 |
| original | 256 | common | 1 | fom_dst_64 | — | 1.174177 | — | — |
| original | 256 | common | 0.1 | fom_dst_64 | — | 1.174177 | — | — |
| original | 512 | full | 10 | fom_dst_16 | expanded_seed790714 | 4.534181 | 14.921848 | 0.298009 |
| original | 512 | full | 5 | fom_dst_16 | expanded_seed790714 | 4.534181 | 14.921848 | 0.298009 |
| original | 512 | full | 1 | fom_dst_64 | — | 5.011191 | — | — |
| original | 512 | full | 0.1 | fom_dst_128 | — | 5.020113 | — | — |
| original | 512 | common | 10 | fom_dst_16 | expanded_seed790714 | 4.534181 | 14.921848 | 0.298009 |
| original | 512 | common | 5 | fom_dst_16 | expanded_seed790714 | 4.534181 | 14.921848 | 0.298009 |
| original | 512 | common | 1 | fom_dst_64 | — | 5.011191 | — | — |
| original | 512 | common | 0.1 | fom_dst_64 | — | 5.011191 | — | — |
| original | 1024 | full | 10 | fom_dst_16 | expanded_seed790715 | 20.318986 | 27.151294 | 0.746163 |
| original | 1024 | full | 5 | fom_dst_16 | expanded_seed790715 | 20.318986 | 27.151294 | 0.746163 |
| original | 1024 | full | 1 | fom_dst_32 | — | 23.029349 | — | — |
| original | 1024 | full | 0.1 | fom_dst_128 | — | 23.348211 | — | — |
| original | 1024 | common | 10 | fom_dst_16 | expanded_seed790715 | 20.318986 | 27.151294 | 0.746163 |
| original | 1024 | common | 5 | fom_dst_16 | expanded_seed790715 | 20.318986 | 27.151294 | 0.746163 |
| original | 1024 | common | 1 | fom_dst_32 | — | 23.029349 | — | — |
| original | 1024 | common | 0.1 | fom_dst_64 | — | 23.165786 | — | — |
| fresh | 64 | full | 10 | fom_dst_64 | expanded_seed790715 | 0.509903 | 12.752228 | 0.040218 |
| fresh | 64 | full | 5 | fom_dst_64 | expanded_seed790715 | 0.509903 | 12.752228 | 0.040218 |
| fresh | 64 | full | 1 | fom_dst_64 | — | 0.509903 | — | — |
| fresh | 64 | full | 0.1 | fom_dst_64 | — | 0.509903 | — | — |
| fresh | 64 | common | 10 | fom_dst_64 | expanded_seed790715 | 0.509903 | 12.752228 | 0.040218 |
| fresh | 64 | common | 5 | fom_dst_64 | expanded_seed790715 | 0.509903 | 12.752228 | 0.040218 |
| fresh | 64 | common | 1 | fom_dst_64 | — | 0.509903 | — | — |
| fresh | 64 | common | 0.1 | fom_dst_64 | — | 0.509903 | — | — |
| fresh | 128 | full | 10 | fom_dst_128 | expanded_seed790714 | 0.667090 | 12.439358 | 0.054700 |
| fresh | 128 | full | 5 | fom_dst_128 | expanded_seed790714 | 0.667090 | 12.439358 | 0.054700 |
| fresh | 128 | full | 1 | fom_dst_128 | — | 0.667090 | — | — |
| fresh | 128 | full | 0.1 | fom_dst_128 | — | 0.667090 | — | — |
| fresh | 128 | common | 10 | fom_dst_128 | expanded_seed790714 | 0.667090 | 12.439358 | 0.054700 |
| fresh | 128 | common | 5 | fom_dst_128 | expanded_seed790714 | 0.667090 | 12.439358 | 0.054700 |
| fresh | 128 | common | 1 | fom_dst_128 | — | 0.667090 | — | — |
| fresh | 128 | common | 0.1 | fom_dst_128 | — | 0.667090 | — | — |
| fresh | 256 | full | 10 | fom_dst_64 | expanded_seed790714 | 1.157405 | 13.283650 | 0.087477 |
| fresh | 256 | full | 5 | fom_dst_64 | expanded_seed790714 | 1.157405 | 13.283650 | 0.087477 |
| fresh | 256 | full | 1 | fom_dst_64 | — | 1.157405 | — | — |
| fresh | 256 | full | 0.1 | fom_dst_128 | — | 1.180990 | — | — |
| fresh | 256 | common | 10 | fom_dst_64 | expanded_seed790714 | 1.157405 | 13.283650 | 0.087477 |
| fresh | 256 | common | 5 | fom_dst_64 | expanded_seed790714 | 1.157405 | 13.283650 | 0.087477 |
| fresh | 256 | common | 1 | fom_dst_64 | — | 1.157405 | — | — |
| fresh | 256 | common | 0.1 | fom_dst_64 | — | 1.157405 | — | — |
| fresh | 512 | full | 10 | fom_dst_16 | expanded_seed790714 | 4.508266 | 15.055508 | 0.283780 |
| fresh | 512 | full | 5 | fom_dst_16 | expanded_seed790714 | 4.508266 | 15.055508 | 0.283780 |
| fresh | 512 | full | 1 | fom_dst_64 | — | 5.002105 | — | — |
| fresh | 512 | full | 0.1 | fom_dst_512 | — | 5.024727 | — | — |
| fresh | 512 | common | 10 | fom_dst_16 | expanded_seed790714 | 4.508266 | 15.055508 | 0.283780 |
| fresh | 512 | common | 5 | fom_dst_16 | expanded_seed790714 | 4.508266 | 15.055508 | 0.283780 |
| fresh | 512 | common | 1 | fom_dst_64 | — | 5.002105 | — | — |
| fresh | 512 | common | 0.1 | fom_dst_64 | — | 5.002105 | — | — |
| fresh | 1024 | full | 10 | fom_dst_16 | expanded_seed790715 | 21.410562 | 27.855190 | 0.766730 |
| fresh | 1024 | full | 5 | fom_dst_16 | expanded_seed790715 | 21.410562 | 27.855190 | 0.766730 |
| fresh | 1024 | full | 1 | fom_dst_32 | — | 23.229804 | — | — |
| fresh | 1024 | full | 0.1 | fom_dst_128 | — | 23.240902 | — | — |
| fresh | 1024 | common | 10 | fom_dst_16 | expanded_seed790715 | 21.410562 | 27.855190 | 0.766730 |
| fresh | 1024 | common | 5 | fom_dst_16 | expanded_seed790715 | 21.410562 | 27.855190 | 0.766730 |
| fresh | 1024 | common | 1 | fom_dst_32 | — | 23.229804 | — | — |
| fresh | 1024 | common | 0.1 | fom_dst_128 | — | 23.240902 | — | — |
| union | 64 | full | 10 | fom_dst_64 | expanded_seed790715 | 0.509903 | 12.565909 | 0.040695 |
| union | 64 | full | 5 | fom_dst_64 | expanded_seed790715 | 0.509903 | 12.565909 | 0.040695 |
| union | 64 | full | 1 | fom_dst_64 | — | 0.509903 | — | — |
| union | 64 | full | 0.1 | fom_dst_64 | — | 0.509903 | — | — |
| union | 64 | common | 10 | fom_dst_64 | expanded_seed790715 | 0.509903 | 12.565909 | 0.040695 |
| union | 64 | common | 5 | fom_dst_64 | expanded_seed790715 | 0.509903 | 12.565909 | 0.040695 |
| union | 64 | common | 1 | fom_dst_64 | — | 0.509903 | — | — |
| union | 64 | common | 0.1 | fom_dst_64 | — | 0.509903 | — | — |
| union | 128 | full | 10 | fom_dst_64 | expanded_seed790714 | 0.673200 | 12.260174 | 0.056624 |
| union | 128 | full | 5 | fom_dst_64 | expanded_seed790714 | 0.673200 | 12.260174 | 0.056624 |
| union | 128 | full | 1 | fom_dst_64 | — | 0.673200 | — | — |
| union | 128 | full | 0.1 | fom_dst_128 | — | 0.680490 | — | — |
| union | 128 | common | 10 | fom_dst_64 | expanded_seed790714 | 0.673200 | 12.260174 | 0.056624 |
| union | 128 | common | 5 | fom_dst_64 | expanded_seed790714 | 0.673200 | 12.260174 | 0.056624 |
| union | 128 | common | 1 | fom_dst_64 | — | 0.673200 | — | — |
| union | 128 | common | 0.1 | fom_dst_64 | — | 0.673200 | — | — |
| union | 256 | full | 10 | fom_dst_64 | expanded_seed790714 | 1.164132 | 13.207777 | 0.087957 |
| union | 256 | full | 5 | fom_dst_64 | expanded_seed790714 | 1.164132 | 13.207777 | 0.087957 |
| union | 256 | full | 1 | fom_dst_64 | — | 1.164132 | — | — |
| union | 256 | full | 0.1 | fom_dst_128 | — | 1.178230 | — | — |
| union | 256 | common | 10 | fom_dst_64 | expanded_seed790714 | 1.164132 | 13.207777 | 0.087957 |
| union | 256 | common | 5 | fom_dst_64 | expanded_seed790714 | 1.164132 | 13.207777 | 0.087957 |
| union | 256 | common | 1 | fom_dst_64 | — | 1.164132 | — | — |
| union | 256 | common | 0.1 | fom_dst_64 | — | 1.164132 | — | — |
| union | 512 | full | 10 | fom_dst_16 | expanded_seed790714 | 4.515824 | 14.986074 | 0.291339 |
| union | 512 | full | 5 | fom_dst_16 | expanded_seed790714 | 4.515824 | 14.986074 | 0.291339 |
| union | 512 | full | 1 | fom_dst_64 | — | 5.002105 | — | — |
| union | 512 | full | 0.1 | fom_dst_128 | — | 5.055869 | — | — |
| union | 512 | common | 10 | fom_dst_16 | expanded_seed790714 | 4.515824 | 14.986074 | 0.291339 |
| union | 512 | common | 5 | fom_dst_16 | expanded_seed790714 | 4.515824 | 14.986074 | 0.291339 |
| union | 512 | common | 1 | fom_dst_64 | — | 5.002105 | — | — |
| union | 512 | common | 0.1 | fom_dst_64 | — | 5.002105 | — | — |
| union | 1024 | full | 10 | fom_dst_16 | expanded_seed790715 | 21.410562 | 27.732991 | 0.766730 |
| union | 1024 | full | 5 | fom_dst_16 | expanded_seed790715 | 21.410562 | 27.732991 | 0.766730 |
| union | 1024 | full | 1 | fom_dst_32 | — | 23.144724 | — | — |
| union | 1024 | full | 0.1 | fom_dst_128 | — | 23.278861 | — | — |
| union | 1024 | common | 10 | fom_dst_16 | expanded_seed790715 | 21.410562 | 27.732991 | 0.766730 |
| union | 1024 | common | 5 | fom_dst_16 | expanded_seed790715 | 21.410562 | 27.732991 | 0.766730 |
| union | 1024 | common | 1 | fom_dst_32 | — | 23.144724 | — | — |
| union | 1024 | common | 0.1 | fom_dst_128 | — | 23.278861 | — | — |

The coordinate bank, full-input QR projection and weak operator were rebuilt at every requested mesh. The following setup durations are observed wall times including first compilation and host work; they are outside paired online query cost and do not include the separate operator-verification work. Bank plus projection bytes are array storage, not peak device allocation.

| Output intervals | Interior unknowns | Bank rank | Bank + projection MiB | Bank evaluation s | QR/operator s | Operator discrepancy | Jacobian discrepancy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 3969 | 32 | 1.937988 | 1.809521 | 3.411462 | 1.59332e-15 | 1.48596e-15 |
| 128 | 16129 | 32 | 7.875488 | 1.905673 | 1.912033 | 1.78998e-15 | 2.05353e-15 |
| 256 | 65025 | 32 | 31.750488 | 1.924909 | 1.999318 | 2.67282e-15 | 3.90324e-15 |
| 512 | 261121 | 32 | 127.500488 | 1.872459 | 2.040899 | 8.8814e-15 | 7.29205e-15 |
| 1024 | 1046529 | 32 | 511.000488 | 1.988250 | 2.169679 | 1.13106e-14 | 1.50413e-14 |

The reference and supplied initial field are preserved at complete requested resolution. Same-grid semidiscrete discrepancies are separate from discrete-versus-spectral spatial errors in the raw results. ROM-versus-discrete error includes representation, nonlinear solve and Crank–Nicolson time error; it is not a pure solver error. Prior timestep validation is retained, but this transfer study does not independently separate those three contributions. Current and initial normalization, absolute errors, state advancement and energy/decay diagnostics remain in each raw invocation.

Mesh setup, compilation and reference generation are outside online query cost. GPU capacity figures in the planning config are live-array estimates; peak device allocation was not profiled, and the configured JAX allocator reserves a fraction of device memory. Detailed Slurm host-memory accounting is preserved separately. All checkpoints and code libraries are byte-identical to the declared inputs. Large extracted fields can be restored from the checked tracked archive chunks using `restore_transfer.py`; metadata and generated audit are also tracked directly.

## Plain-language glossary

- **Cohort / original / fresh / union:** a group of physical inputs / repeated earlier development inputs / new independently seeded development inputs / both groups together.
- **Output intervals / common grid:** cells per axis on the fully returned grid / fixed shared physical observation nodes for mesh comparison.
- **Method / head / FOM / ROM:** solver configuration / frozen latent-to-coefficient neural mapping / full-order heat solver / reduced nonlinear solver.
- **DST / semidiscrete / spectral:** sine-transform propagation / exact time solution of fixed-grid equations / continuum sine-series physical reference.
- **Query ms / case median / paired FOM/ROM:** charged full-input-to-full-output milliseconds / middle repeated time for one input / median across inputs of full-model time divided by reduced-model time.
- **Full current error / common current error / initial full error:** discrepancy divided by the current reference norm on all returned nodes / the same on shared observation nodes / actual fitted initial discrepancy on all returned nodes.
- **Initial-normalized / absolute / decay:** discrepancy divided by the initial truth norm / area-weighted unnormalized discrepancy / the reference field becoming smaller in time.
- **Invalid cases / stationarity / multistart:** inputs with any failed validity check / the configured small-gradient stopping condition / fitting from both a nearby library code and the mean code.
- **Empirical adjustment / target / selected envelope:** reference allowance inferred from observed refinement / required physical error / cheapest configuration satisfying the same cohort and target.
- **QR / weak operator / Crank–Nicolson:** exact full-field least-squares compression / heat equations tested against smooth functions / fixed second-order time formula.
- **Interior unknowns / bank rank / MiB / operator and Jacobian discrepancies:** scalar field values excluding known boundaries / number of independent spatial bank columns / binary megabytes of array storage / relative differences from the explicit discrete stencil and its derivative.
- **Frozen / checkpoint / code library / final:** unchanged model parameters / complete saved model / unchanged training starting codes / reserved independent evaluation not used here.
