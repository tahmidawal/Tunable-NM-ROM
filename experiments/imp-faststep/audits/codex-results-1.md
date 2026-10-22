**Overall verdict: NEEDS-RESTATEMENT before reporting.** The principal result is correct: **E5 is not fixed at 1024²**. All four new arms fail the worst-error condition; G-CN and G-R1 also fail the 3× evolution-speed condition. I found one small F5 aggregation error and reporting qualifications that must remain visible.

Audit scope: local saved coefficients, certificate arrays, timing records, control witnesses and E0 references. No cluster access, jobs, file changes or scratch files. Archived `F_*` fields were not inspected.

**1. Independent E0 errors — CORRECT**

I decoded 33 bank-arm samples with `npcheck.decode257`: `dev6|0`, `val32|0`, and the worst case for all 11 configurations. I also decoded nine samples from the three selected POD configurations using `Phi_lat`, padding the zero boundaries.

All sampled errors agree with `result.json` within approximately \(5\times10^{-17}\) absolute. Examples:

| Configuration | Case | Independently recomputed error |
|---|---|---:|
| L-CNR, r=128 | val32\|19 | 0.0520036243878051 |
| G-E4a, r=384 | val32\|19 | 0.0181524834416277 |
| G-E4b, r=384 | val32\|19 | 0.0190812019581434 |
| POD r=384, κ=4, Δt=.0125 | val32\|26 | 0.0106456812299179 |
| POD r=128, κ=8, Δt=.025 | val32\|5 | 0.0544673682758085 |

These remain **conditional E0 accuracy estimates**.

**2. Stop predicates — CORRECT**

All 38 cases are E0-eligible here, so the all-38 and eligible-subset worst-error predicates coincide.

| Arm | Worst error r=128 | Worst error r=384 | r=384 evolution speed-up | Conditions firing |
|---|---:|---:|---:|---|
| Incumbent L-CNR | 5.200362% | 1.775269% | — | — |
| G-CN | 5.651121% | 1.850974% | 0.726166× | Error, speed |
| G-E4a | 5.634568% | 1.815248% | 14.277634× | Error |
| G-E4b | 5.485333% | 1.908120% | 22.849436× | Error |
| G-R1 | 5.692457% | 1.926565% | 1.583933× | Error, speed |

The evolution statistics were recomputed from the bracketing times in the **E records**, taking the median of three ratios per case, then the median across six cases.

No production invalidity or instability was found for the new bank arms. Saved trajectories are finite; their maximum monitored norm ratios are **1.0**. No classification was required, consistent with the absence of `CLS_*` files.

Every subject has all 18 required timing repetitions. Across **198 E, 774 F and 234 D records**, there are zero invalid final triplets and zero recorded identity failures, including discarded attempts.

**3. Q1/Q2 labels — CORRECT, with comparator qualifications essential**

All **11 bank configurations pass Q1 and Q2**, consistent with the saved arrays:

- LSPG: zero nonstationary production steps; tight/tighter gamma maxima remain below their respective thresholds.
- G-CN: maximum fresh residual ratios are \(9.013\times10^{-11}\) and \(9.452\times10^{-11}\), below \(10^{-10}\); tighter anchors also pass.
- ETDRK4: maximum reconstruction discrepancy is \(1.1511\times10^{-14}\).
- G-R1: maximum Jacobian and increment discrepancies are \(2.9570\times10^{-15}\) and \(2.5842\times10^{-13}\).
- Reconstructed output schedules agree within \(4.01\times10^{-15}\) relative; LSPG schedules agree exactly.
- Independently recomputed Q2 rollout distances range from \(6.8352\times10^{-5}\) to \(2.1386\times10^{-4}\), below \(2.5\times10^{-4}\). Maximum stored force ratio is **0.0539464**, below 0.116.

For POD, all 30 ETDRK4 members pass Q1. Q2 passes only for **κ=8 at r=256 and r=384**, at both timesteps; the six full-node members are HR-free. The remaining 20 ECSW members are HR-unqualified.

Crucially, **every selected matched competitor is HR-unqualified**:

| Selected competitor | Q2 rollout distance | Maximum force ratio |
|---|---:|---:|
| r=128, κ=8, Δt=.025 | 0.00247729 | 0.0921497 |
| r=384, κ=4, Δt=.0125 | 0.00320757 | 0.846417 |
| r=384, κ=4, Δt=.025 | 0.00322302 | 0.846417 |

Selection permits this under the registered validity/stability rule, but reporting must show the qualification labels.

Native ECSW’s descriptive anchor failure is also real: tighter-anchor maximum gamma **\(4.21218\times10^{-7}\)** exceeds \(10^{-8}\). Native Q-DEIM’s anchors pass.

These checks evaluate saved certificate evidence; I did not regenerate every original-table RHS/Jacobian certificate.

**4. Competitor selection — CORRECT; F5 aggregation — WRONG at small numerical scale**

All 11 competitor selections reproduce independently using the screening records and registered 1% tie rule.

Below, ratios are **competitor full-query time / our full-query time**, recomputed from direct triplets:

| Our arm | r=128 ratio | r=384 ratio |
|---|---:|---:|
| L-CNR | 0.102375 | 0.036341 |
| G-CN | 0.060609 | 0.032220 |
| G-E4a | 0.852579 | 0.623042 |
| G-E4b | 1.094748 | 0.796270 |
| G-R1 | 0.126346 | 0.069904 |

All r=128 picks are POD κ=8, Δt=.025. At r=384, G-CN/G-E4a/G-R1 select κ=4, Δt=.0125; the others select κ=4, Δt=.025. Arm 6’s ratio is **0.042429**.

Every comparison matches on dev6, val32 and all38 **except L-CNR r=128**, which matches on dev6 only.

F5:

| Rank | Recomputed median slope | Largest reference/error ratio | Correct median gain over G-CN |
|---|---:|---:|---:|
| 128 | 2.000998411 | \(1.42885\times10^{-5}\) | 2.141323175× |
| 384 | 2.003967504 | \(1.96426\times10^{-4}\) | 2.186762535× |

The checker reports **2.141320105× / 2.186762442×** by inverting the aggregate median. With six cases, that is not exactly the median of the per-case gains. The discrepancy is tiny and changes no decision.

F5’s order and speed conditions pass, but its error condition fails at both ranks: G-R1’s worst errors exceed G-CN’s. **F5 is rejected.**

**5. Controls — CORRECT under A1–A6**

The recorded witnesses support the required passes/firings:

- C1 equivalence: at most \(1.198\times10^{-13}\); C2 distance: at most \(1.632\times10^{-13}\); C3 linear exactness: at most \(1.134\times10^{-15}\).
- C2m/C16: **50/50** unconverged steps at both ranks.
- C3m mutation discrepancies: **0.00105977 / 0.00142328**; positive slopes independently recompute to **3.750213 / 2.027717**.
- C4m records nonfinite outcomes; C5’s Euler mutant has slope **1.016538** at r=128 and nonfinite error at r=384.
- C6m Jacobian discrepancies: **0.0548554 / 0.0749738**.
- C7m force ratios: **0.127000 / 0.845250**; both fire.
- C9 self-timing: **0.995122146 / 0.995122487**, within the required band.
- C11 random-weight error ratios independently recompute to **4,732× / 31,021× / 60,967×**; aggregate-residual agreement is within \(1.01\times10^{-13}\) relative.
- C8, C12–C15 and C18 have the required recorded witnesses. C10’s doubled worst error fires; the unchanged baseline does not. C17’s missing population is detected.
- C19/C21 successor fixtures and C20/C24 population-match fixtures reproduce; C23 returns incomplete/identity-failed as required. C22’s boundary labels agree.

The raw r=384 control summary says `all_ok=false` because its C3m slope is below 3.5. **A6 explicitly exempts that larger-rank positive-order bar**; this is not a production rejection. It must nevertheless be disclosed as observed finite-range order near two.

**6. Potentially misleading reporting — NEEDS-RESTATEMENT**

- None of the eight main worst-error exceedances is within \(2\varepsilon_{\rm ref}\). All maxima occur at `val32|19`, where \(2\varepsilon_{\rm ref}=1.20303\times10^{-5}\); increases range from **0.000399792 to 0.004920948**.
- Most cases improve despite the failed worst-error gate. Better/worse/tied counts are **26/12/0** at r=128 for every new arm; at r=384 they are **36/2/0**, except G-E4b’s **34/4/0**.
- The **14.28× and 22.85×** results describe evolution against the incumbent, not full-query gains against POD.
- The only new-arm direct ratio above one is G-E4b r=128’s **1.09475×**, against an **HR-unqualified** tested competitor.
- `accepted=true` means the evidence passes the acceptance checks; it does **not** mean E5 succeeds.
- This audit supports conclusions for **f1024 only**, not the full two-mesh campaign.

The defensible result is: **numerically qualified faster ETDRK4 evolution, but no arm satisfies E5’s registered worst-error requirement at 1024².**