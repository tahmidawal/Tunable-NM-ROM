**Overall verdict: NEEDS-RESTATEMENT before publication.** The audited numerical results are correct, including the repaired F5 aggregation. **E5 fails on the completed scope at both meshes.** Remaining issues concern reporting, particularly the broken POD table and the omitted outcome of f4096’s frozen dev6 selection.

Read-only audit completed: no file changes, jobs, SSH, scratch files, or LAB-LOG writes. Archived `F_*` fields were unavailable locally. All available files covered by the output manifests matched their hashes: 45 for f1024 and 7 for f4096.

**1. Independent E0 errors — CORRECT**

Recomputed 48 bank samples using `npcheck.decode257`: dev6|0, val32|0 and the worst case for each configuration. Also recomputed nine selected-POD samples using `Phi_lat`, with zero boundary padding. Scoring used the reference fields directly and the registered norm.

Maximum absolute disagreement: **\(4.86\times10^{-17}\)**.

| f4096 configuration, r=384 | Recomputed worst error |
|---|---:|
| L-CNR | 0.0177468326550048 |
| G-CN | 0.0185031962546226 |
| G-E4a | 0.0181450620952352 |
| G-E4b | 0.0190756434793673 |
| G-R1 | 0.0192584957026959 |

All these maxima occur at **val32|19**. Errors remain conditional E0 accuracy estimates.

**2. Stop predicates — CORRECT**

All 38 cases are eligible; therefore the all-38 and eligible-subset predicates coincide. Both fire for every new arm at every completed rank/mesh.

| Mesh | Arm | Worst error r=128 | Worst error r=384 | r=384 evolve speed-up | Conditions firing |
|---|---|---:|---:|---:|---|
| 1024² | G-CN | 5.651121% | 1.850974% | 0.726166× | Error, speed |
| 1024² | G-E4a | 5.634568% | 1.815248% | 14.277634× | Error |
| 1024² | G-E4b | 5.485333% | 1.908120% | 22.849436× | Error |
| 1024² | G-R1 | 5.692457% | 1.926565% | 1.583933× | Error, speed |
| 4096² | G-CN | Deferred | 1.850320% | 0.732675× | Error, speed |
| 4096² | G-E4a | Deferred | 1.814506% | 14.974450× | Error |
| 4096² | G-E4b | Deferred | 1.907564% | 24.149574× | Error |
| 4096² | G-R1 | Deferred | 1.925850% | 1.592867× | Error, speed |

Incumbent worst errors are **5.200362% / 1.775269%** at 1024² and **1.774683%** at 4096².

The speed statistics were independently rebuilt from E-record bracketing times: median of three ratios per case, then median across six cases.

No new-arm invalidity or instability was found. At f4096, all saved trajectories are finite and maximum monitored growth is **1.0**. There are no tripped runs requiring `CLS_*` evidence.

Timing coverage is complete:

| Job | E records | F records | D records | Invalid final triplets | Recorded identity failures, including retries |
|---|---:|---:|---:|---:|---:|
| f1024 | 198 | 774 | 234 | 0 | 0 |
| f4096 | 90 | 90 | 18 | 0 | 0 |

**3. Q1/Q2 labels — CORRECT within the saved-evidence scope**

All five f4096 bank configurations pass.

| f4096 arm | Independently decoded maximum Q2 distance |
|---|---:|
| L-CNR | \(1.4214263\times10^{-4}\) |
| G-CN | \(7.0015223\times10^{-5}\) |
| G-E4a | \(6.8046584\times10^{-5}\) |
| G-E4b | \(7.6940985\times10^{-5}\) |
| G-R1 | \(7.2124044\times10^{-5}\) |

All are below \(2.5\times10^{-4}\); maximum stored force ratio is **0.0483471**, below 0.116.

The f4096 certificate arrays support Q1:

- LSPG production/tight/tighter gamma maxima: **\(9.92083\times10^{-4}\), \(9.96498\times10^{-8}\), \(9.75667\times10^{-9}\)**.
- CN production/tight/tighter residual ratios: **\(9.29976\times10^{-11}\), \(9.46000\times10^{-13}\), \(9.33690\times10^{-14}\)**.
- ETDRK4 maximum reconstruction discrepancy: **\(1.21130\times10^{-14}\)**.
- G-R1 Jacobian/increment discrepancies: **\(2.90804\times10^{-15}\) / \(2.20806\times10^{-13}\)**.
- Reconstructed Galerkin output schedules agree within **\(1.18\times10^{-15}\)** relative; initial coefficients match the incumbent bitwise.
- Saved twin-distance witnesses are below **\(4.90\times10^{-16}\)**.

The f1024 restatements remain correct: all 11 bank configurations qualify. POD successor distances recomputed from coefficients reproduce **four Q2 passes, six HR-free members and 20 HR-unqualified members**. All selected matched competitors are HR-unqualified.

Native ECSW’s tighter-anchor failure remains real: **\(4.21218\times10^{-7}>10^{-8}\)**. Native Q-DEIM’s anchors pass.

These are checks of saved certificates and decoded trajectories; I did not regenerate every original-table RHS/Jacobian or force-ratio certificate.

**4. Selection, direct ratios and F5 — CORRECT; previous aggregation error fixed**

All 11 f1024 competitor selections reproduce using the screening records and 1% tie rule.

Direct ratios below are **competitor full-query time / our full-query time**:

| Arm | r=128 | r=384 |
|---|---:|---:|
| L-CNR | 0.102375 | 0.036341 |
| G-CN | 0.060609 | 0.032220 |
| G-E4a | 0.852579 | 0.623042 |
| G-E4b | 1.094748 | 0.796270 |
| G-R1 | 0.126346 | 0.069904 |

Arm 6’s ratio is **0.042429**. All comparisons match on dev6, val32 and all38 **except L-CNR r=128**, which matches only on dev6.

There is **no POD competitor comparison at 4096²**. Its 18 direct records measure F5.

| Mesh / rank | Recomputed median slope | Maximum reference/error ratio | Correct direct F5 gain |
|---|---:|---:|---:|
| 1024² / 128 | 2.000998411 | \(1.42885\times10^{-5}\) | 2.141323175× |
| 1024² / 384 | 2.003967504 | \(1.96426\times10^{-4}\) | 2.186762535× |
| 4096² / 384 | 2.003978277 | \(1.99052\times10^{-4}\) | 1.963014702× |

These gains now agree with the checker. **F5 fails at all three cells** because G-R1’s worst error exceeds G-CN’s, despite passing order and speed.

**5. Controls — CORRECT, with f4096 explicitly inheriting smoke evidence**

The f4096 `controls_from` hash exactly matches `checks/smk1.json`. f4096 did **not** rerun those in-job controls; their witnesses come from smk1.

The inspected witnesses support every required pass/firing:

- C1/C2/C3 positive discrepancies are at most **\(1.20\times10^{-13}\), \(1.64\times10^{-13}\), \(1.14\times10^{-15}\)**.
- C1m reconstruction discrepancy is approximately **0.0348–0.0349**.
- C2m/C16 expose **50/50** unconverged steps.
- C3m positive slopes independently recompute to **3.750213 / 2.027717**; mutant discrepancies are **0.00105977 / 0.00142328**. A6 permits the larger-rank positive slope.
- C4m produces nonfinite outcomes; C5’s Euler mutant has slope **1.016538** at r=128 and a nonfinite error at r=384.
- C6m detects **50/50** nonstationary mutant steps.
- Smoke C7m fires after escalation at r=128: **ρ=0.868557**; at r=384, **ρ=0.810526**.
- C9 self-timing is **0.994954982** at f4096, within the required band.
- C11 mutant/reference ratios recompute to **4,732.295× / 31,021.466× / 60,967.404×**.
- C8, C12–C15 and C18 contain the required identity/guard/mismatch witnesses.
- C10/C17 detect the error perturbation/missing population; C19/C21 successor rules, C20/C24 population matching, C22 boundary classification and C23 incomplete/identity-failure outcomes agree with the required behavior.

**6. Reporting — NEEDS-RESTATEMENT**

The substantive f1024 qualifications have been added, but these issues remain in [report.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-faststep/experiments/imp-faststep/report.md):

- **The POD table is malformed.** Unescaped `|` characters in configuration identifiers create extra columns, displacing or hiding qualification labels, match statuses and ratios. Fix the generator before publishing.
- **Explicitly report the failed frozen selection.** f4096 selected **G-E4a on dev6**: worst error **0.003423850**, versus incumbent **0.004077301**. It then fails validation: **0.018145062 > 0.017746833**. This is the failed frozen selection, not a successful best stepper. f1024 selected none.
- **State “F5 rejected” explicitly.** The current table exposes the error failure but labels only the order condition “pass.”
- **Restrict the upper-bound language by population.** L-CNR r=128 is dev-matched only; the blanket paragraph must not imply a validation-population upper-bound claim for that row.
- Label the near-two ETDRK4 slope as the **observed smoke/control finite-range slope**, rather than implying it was independently measured across all production cases and both meshes.

No worst-error failure is a reference-resolution tie. At f4096, exceedances are **0.000398229–0.001511663**, while \(2\varepsilon_{\rm ref}\) at the common worst case is **0.0000120303**. The f1024 exceedances likewise remain resolved.

**The supported result is qualified, much faster ETDRK4 evolution, but failure of E5’s worst-error requirement throughout the completed scope.** f4096 r=128 and M1152 remain deferred; no 4096² H-base conclusion is supported.