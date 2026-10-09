**The numerical tables check out. “No useful winner” is supported, but several statements need qualification, and the frontier PNG does not match the current generator.** No files were modified.

I independently recomputed 4,140 rho summaries, all 216 m* values, all 36 tail fits, spectrum classifications, and timing medians. I checked coverage and exits for 4,742 recorded rollouts.

| Item | Verdict | Finding |
|---|---|---|
| R0 | **CORRECT; reporting incomplete** | All four training fingerprints match within \(8.7\times10^{-15}\) relative, below \(10^{-9}\). The report omits the explicit pass. |
| R1 | **CORRECT** | Checked against the actual 2D-lane summary, not just generator constants. Gauss-64 lat64 rho: acc **0.018554000267682662**, fast **0.023299740400774736**. These and all four gref/lat64 ST errors reproduce to floating-point roundoff. |
| R2a | **NEEDS-RESTATEMENT** | Step 1 passes. Step 5,000 **fails**: 0.00300262 versus 0.002125, approximately **41.3% higher**. The report prints these numbers without naming the failure. A2.5 explicitly makes this checkpoint reporting-only, so it does **not** invalidate evaluation. |
| R2b | **NEEDS-RESTATEMENT** | Recorded parity passes with deterministic XLA settings; default-autotuning parity failed. Report this qualification rather than suggesting unconditional bitwise replication. |
| C1 | **CORRECT; omitted** | Gauss-8 fails strongly for every bank: own-population worst rho **14.56–15.03 acc**, **7.04–7.09 fast**. A2 requires reporting this stress arm; the report omits it. |
| C2/C3/C5 | **CORRECT** | All six task records agree: bump bandwidths **20 < 64**, both geometric; kink algebraic; C5b error **\(1.69\times10^{-13}\)**. Every bank passes C5a, minimum errors **\(1.25–2.35\times10^{-7}\)**. |
| C4 | **CORRECT in saved records** | Present for **all three populations**, every bank/setting. Largest check/flux discrepancies are **\(2.642\times10^{-6}\)/\(2.847\times10^{-6}\)**, both sob01/acc/common, below \(10^{-5}\). |
| Rollout convergence and C6 | **CORRECT** | Maximum Gauss-768 distance **\(1.873\times10^{-10}\)** versus \(10^{-6}\); tight-solver distance **\(1.705\times10^{-5}\)** versus \(10^{-3}\). All recorded rollouts finite; **zero non-accepted exits**. |
| Completeness/eligibility | **CORRECT at saved-metric level** | Required arms have exactly 38 cases; sensitivity checks exactly six. Populations contain 1,900/1,900/228 states. Saved target norms are finite and positive; rotation diagnostics pass. |

**Independent ladder and tail checks**

All recomputed values match. Representative own-population Gauss-64 worst rho: base/acc **0.0309390**, sob01/acc **0.0208241**, sig1/fast **0.0192488**.

Monotone confirmation is implemented correctly and materially changes results:

- base/fast, \(b=0.06\): first crossing **1,600**, confirmed **3,136**; Gauss-48 rebounds to **0.0708506**.
- sob01/acc/common, \(b=0.01\): first crossing **25,600**, confirmed **65,536**.
- **No recorded m* is censored**. Censoring logic therefore has no effect on these results.

Tail fits correctly use A1.4’s worst-rho \(<0.5\), median-rho \(>10^{-12}\) window. Examples: base/acc **1.0221177**, sig1/acc **1.0322294**, sob01/fast **1.0185090**. **CORRECT as descriptive fits**, not analyticity certificates.

**Verdicts recomputed from the rules**

| Claim | Verdict | Independent reading |
|---|---|---|
| H1 fails | **CORRECT** | sob01’s common-support D2/D4 reductions are **70.35%/69.68% acc**, but **20.20%/17.77% fast**. Fast D4 misses 20%; the result is stencil-sensitive at the threshold. Its acc rollout error also **increases 1.72%**. |
| Report’s H1 gradient explanation | **NEEDS-RESTATEMENT** | The printed Δe∇ uses **all-interior D2**, whereas the verdict uses **common-support D2 and D4**. Show the deciding quantities; otherwise the large acc derivative improvement is obscured. |
| H2 “fail” | **NEEDS-RESTATEMENT** | sig1/sig2 satisfy the descriptive \(b=0.06\) reduction: fast **3,136→1,600**, factor **1.96**, acc unchanged, common ladders no worse. But base/frozen_lane fast m* differ, so the preregistered verdict is **unresolved**, not evidence against H2. |
| sob01 H2 | **CORRECT: no pass** | Besides unresolved noise, its acc/common \(b=0.06\) m* worsens **6,400→9,216**. |
| No useful winner | **CORRECT under the amended noise gate** | All treatments satisfy the ±10% S/ST accuracy allowance, but none has an accepted H1/H2 pass. Do not describe this as failure of the accuracy allowance. |

The acc \(E_S\) noise yardstick is indeed **33.9155%**, requiring an improvement exceeding **67.8311%** to count as resolved. sig2’s **13.784%** and sig1’s **8.824%** improvements cannot pass. This is a prescribed comparator discrepancy, **not an estimated statistical standard deviation**.

**Presentation risks**

- **WRONG — generator/plot consistency.** The existing [frontier PNG](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-smooth-bank/experiments/jcp-smooth-bank/plots/frontier.png) shows squares for confirmed 0.06 and circles for **interpolated 0.01**. Current `make_report.py` generates confirmed **0.06/0.116** markers instead. The PNG labels interpolation, but it is not the current generator’s output and cannot establish discrete H2 reductions.
- **NEEDS-RESTATEMENT — rotation effect.** Fast gref S error changes **3.451%→4.510%** between frozen/deployed and frozen/lane: **1.059 percentage points, about 30.7% worse**. The same-bank rotation/pipeline distinction matters substantially. Retraining comparisons must use frozen_lane.
- **NEEDS-RESTATEMENT — first 320 tests.** These are correctly recomputed diagnostics, with a separately restricted denominator. They are neither the actual \(M=512/1536\) solver nor a matched replication of Hari’s bank. Smaller bracketed rho does not establish equivalent rollout accuracy.
- **NEEDS-RESTATEMENT — spectra.** Every \(10^{-8}\) bandwidth is unresolved: **64/64 states for both u and f**, every bank/setting. Print “unresolved,” not unexplained dashes. The \(10^{-4}\) medians use different subsets—e.g. acc base **45/64**, sob01 **40/64**, sig1 **60/64**—so their ranking is conditional.
- **CORRECT — timing arithmetic.** All **504 invocations**, 36 per subject, reproduce the reported medians; maximum coefficient discrepancy is **\(1.84\times10^{-15}\)**. These support measured query costs, not a resolved recipe-level speedup.

**Supported:** numerical replication, valid recorded solver checks, stronger observed sig1 tail decay, substantial sob01 acc projection/common-support derivative improvements, and no promotable winner under the preregistered rules.

**Not supported:** “smoothing has no effect,” “Sobolev training improves rollout accuracy,” a resolved H2 win, physical-gradient accuracy, certified analyticity, or recipe/generalization claims beyond these provisional single-seed cohorts.

Audit limit: C4 vector discrepancies and rollout-field distances were checked from saved JSON records; the listed arrays do not contain their underlying vectors/fields. The 512-point spectra were independently checked, but 256-point envelopes are not saved.