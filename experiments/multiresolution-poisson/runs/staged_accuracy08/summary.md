# Audited fixed-rank Poisson staged-training comparison

All scheduled endpoints are retained. These are opened development results; final cases remain unopened. No trained arm passes the complete-cohort physical target.

| Cohort | Intervals | Model | GPU ms | Host ms | Worst field error (%) | Bank projection error (%) | Invalid solves | Target passes all |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| all | 64 | original_relative | 2.419104450 | 3.423970542 | 7.281746791 | 6.668704333 | 0 | False |
| all | 64 | joint_matched | 2.321338048 | 3.336758120 | 7.572733697 | 7.122133955 | 0 | False |
| all | 64 | staged_head | 2.386875451 | 3.420413006 | 7.758832178 | 7.168591979 | 0 | False |
| all | 64 | staged_joint | 2.384635969 | 3.426286625 | 7.661793954 | 7.152710767 | 0 | False |
| all | 256 | original_relative | 2.276121057 | 3.419827437 | 7.280264410 | 6.499382715 | 0 | False |
| all | 256 | joint_matched | 2.200240036 | 3.321343451 | 7.571224001 | 6.947000669 | 0 | False |
| all | 256 | staged_head | 2.257173532 | 3.455823520 | 7.757488769 | 6.991931060 | 0 | False |
| all | 256 | staged_joint | 2.190783969 | 3.406237578 | 7.660349295 | 6.977186401 | 0 | False |
| all | 1024 | original_relative | 2.613090910 | 6.211456028 | 7.280247642 | 6.489096125 | 0 | False |
| all | 1024 | joint_matched | 2.545561409 | 6.116467994 | 7.571198615 | 6.936344320 | 0 | False |
| all | 1024 | staged_head | 2.597750048 | 6.231383537 | 7.757471671 | 6.981179171 | 0 | False |
| all | 1024 | staged_joint | 2.565345028 | 6.163463928 | 7.660328205 | 6.966506025 | 0 | False |
| existing_development | 64 | original_relative | 2.430920023 | 3.431391087 | 6.802558107 | 5.661080913 | 0 | False |
| existing_development | 64 | joint_matched | 2.336070989 | 3.347537131 | 6.606667873 | 5.790823218 | 0 | False |
| existing_development | 64 | staged_head | 2.390562906 | 3.440328059 | 7.353979687 | 6.325825099 | 0 | False |
| existing_development | 64 | staged_joint | 2.383254934 | 3.415670595 | 6.843793863 | 6.094774910 | 0 | False |
| existing_development | 256 | original_relative | 2.318772487 | 3.443000023 | 6.801564318 | 5.509782655 | 0 | False |
| existing_development | 256 | joint_matched | 2.212747466 | 3.347531892 | 6.605694323 | 5.655844907 | 0 | False |
| existing_development | 256 | staged_head | 2.289671567 | 3.443434951 | 7.353080770 | 6.186279625 | 0 | False |
| existing_development | 256 | staged_joint | 2.228788449 | 3.360477975 | 6.842726589 | 5.956735875 | 0 | False |
| existing_development | 1024 | original_relative | 2.638045582 | 6.222271477 | 6.801556288 | 5.500597428 | 0 | False |
| existing_development | 1024 | joint_matched | 2.531181555 | 6.128605921 | 6.605687017 | 5.647628742 | 0 | False |
| existing_development | 1024 | staged_head | 2.622753032 | 6.282633520 | 7.353072989 | 6.177771841 | 0 | False |
| existing_development | 1024 | staged_joint | 2.613186021 | 6.211774074 | 6.842717057 | 5.948325627 | 0 | False |
| new_development | 64 | original_relative | 2.363739070 | 3.402997041 | 7.281746791 | 6.668704333 | 0 | False |
| new_development | 64 | joint_matched | 2.252888517 | 3.326006932 | 7.572733697 | 7.122133955 | 0 | False |
| new_development | 64 | staged_head | 2.348024515 | 3.409847035 | 7.758832178 | 7.168591979 | 0 | False |
| new_development | 64 | staged_joint | 2.405298990 | 3.450765857 | 7.661793954 | 7.152710767 | 0 | False |
| new_development | 256 | original_relative | 2.216343535 | 3.383993055 | 7.280264410 | 6.499382715 | 0 | False |
| new_development | 256 | joint_matched | 2.076764824 | 3.153876052 | 7.571224001 | 6.947000669 | 0 | False |
| new_development | 256 | staged_head | 2.162296441 | 3.469624557 | 7.757488769 | 6.991931060 | 0 | False |
| new_development | 256 | staged_joint | 2.159079071 | 3.430087469 | 7.660349295 | 6.977186401 | 0 | False |
| new_development | 1024 | original_relative | 2.572860569 | 6.042668596 | 7.280247642 | 6.489096125 | 0 | False |
| new_development | 1024 | joint_matched | 2.565196948 | 6.094126846 | 7.571198615 | 6.936344320 | 0 | False |
| new_development | 1024 | staged_head | 2.535607899 | 6.145584513 | 7.757471671 | 6.981179171 | 0 | False |
| new_development | 1024 | staged_joint | 2.498483984 | 6.064752932 | 7.660328205 | 6.966506025 | 0 | False |

All paired CG/DST controls, raw repetitions, stationarity/exit counts, refinement-adjusted errors, training costs and complete head diagnostics are in `panel.json` and `result.json`.

Audited 3024 invocations and 1848 distinct full fields; restored 2058 archive members from 104 tracked parts. Exact remote deletion is verified.

The bank column projects into each unrestricted learned spatial span on the same mesh. Head diagnostics minimize full-field reconstruction at the recorded latent dimension; they are best-found stationary fits, not global optima. POD64/128 diagnostics construct the span from normalized training snapshots only and do not certify minimax lower bounds.

Glossary: cohort identifies the original or new development cases; intervals counts mesh subdivisions per axis; GPU/host ms are pooled paired-invocation medians; worst field error is relative L2 error versus restricted fine truth; bank projection error is relative same-grid unrestricted projection error; invalid solves fail recorded numerical gates; target passes all requires numerical, physical-error and reference-refinement gates for every case.
