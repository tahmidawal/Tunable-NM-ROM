**JOB GATE: HOLD both `c3budA` and `c3budW1024`.** Their resource certificates pass, but the frozen **fast-equal-accuracy selection violates the registered selection rule**.

| Audit item | Verdict | Finding |
|---|---|---|
| G-score bound to this result | **CORRECT** | Result SHA matches; archive checksums pass, including `W_grid.npz`. All 157,440 comparisons pass; worst discrepancy \(2.78\times10^{-16}\). Negative controls discriminate. |
| H4a/H4b | **CORRECT** | **H4a fails and H4b passes at all five settings.** The H4b witness is CN, \(K=1\), \(h=0.01\) throughout. |
| Comparator population | **CORRECT** | M2_512 uses 63/64 cases, excluding unresolved case 12; all W settings use 64/64. Unresolved candidate cases retain the conservative \(3U_{\rm anc}\) bound. |
| H4 solver eligibility | **CORRECT** | Recomputed flags agree with the per-case solver-distance and tighter-probe requirements. CN \(K=2,h=0.025\) remains ineligible, including the seven failing cases at every W setting. |
| Timing panels | **CORRECT** | All six panels have complete invocation counts. Recomputed ratios agree exactly; output hashes match archived trajectories. Subject-median drift ranges from 0.982824 to 1.013095, within the registered band. |
| Accurate-equal-cost selections | **CORRECT** | Independently recomputed eligible minima reproduce all five selections. H4b’s allowed ratio \(\le1.1\) is correctly distinguished from this selection’s \(\le1.0\). |
| Fast-equal-accuracy selections | **WRONG** | The implementation selects from a broader feasible set than “H4a’s cheapest”; details below. |
| H6 under first-order reference | **CORRECT** | All transitions are labelled **reference-limited (descriptive)**, with raw bars evaluated separately. They are not confirmation of dial extension. |
| Frozen budget arms and witnesses | **CORRECT mechanically** | Both configs exactly match the frozen arm specifications, roles and selected flags; frozen/source hashes match. They inherit the fast-selection defect. |
| POST HOC rule label | **CORRECT** | Carried through result, scores, analysis, frozen rule selection and both budget configs. |
| `budget3d.py` fix | **CORRECT; description NEEDS-RESTATEMENT** | The replaced access is in the **G-score case-swap/time-shift controls**, not the B0 fixture. The smoke analysis matches the current analyzer hash and passes its applicable gates. |

The blocking discrepancy is explicit: [DESIGN §6](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/DESIGN.md:230) defines fast-equal-accuracy as **“H4a’s cheapest.”** However, [analyze3d.py:153](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/analyze3d.py:153) drops H4a’s step and \(0.6\) timing-ratio constraints when constructing that selection.

| Setting | Frozen fast arm | Paired ratio |
|---|---|---:|
| M2_512 | BDF2, \(h=0.0125,K=1\) | 0.873887 |
| W256 | CN, \(h=0.0125,K=1\) | 0.847940 |
| W512 | BDF2, \(h=0.0125,K=1\) | 0.862532 |
| W768 | BDF2, \(h=1/60,K=2\) | 0.961504 |
| W1024 | BDF2, \(h=1/60,K=2\) | 0.955818 |

Every ratio exceeds 0.6. Under the registered definition, **all five fast selections are unavailable**. These arms can remain useful diagnostics, but retaining them under a broader selection rule requires a separately disclosed post-hoc change. The existing **POST HOC rule** label covers quadrature selection, not this solver-selection change.

Independent numerical sample: I decoded W256 cases 0, 5 and 63 over the full scoring lattice using `einsum`. Sampled anchor discrepancies agreed within \(4.17\times10^{-17}\), anchor uncertainties within \(1.75\times10^{-17}\); the sampled `einsum`/BLAS field difference was \(1.25\times10^{-16}\). Case 5’s CN \(K=2,h=0.025\) solver distance was \(3.88157\times10^{-4}\), independently confirming rejection.

| Job | Resource gate | Overall |
|---|---|---|
| `c3budA` | **CORRECT:** H200/48 h/200G; reproduced projection **20.6663 h** including margin, **104.7625 GB** peak | **HOLD** |
| `c3budW1024` | **CORRECT:** H200/48 h/200G; reproduced projection **42.1884 h** including margin, **103.7275 GB** peak | **HOLD** |

Both certificates verify against committed sources and the bound smokes. Deployed/converged/finer rules are correct: M2_512 `gl32/gl56/gl64`; W256 `gl24/gl56/gl64`; W512 `lat32768/gl56/gl64`; W768 `lat32768/gl64/gl80`; W1024 `lat65536/gl80/gl96`.

Before submission, resolve the fast-selection contract, preserve the original audit trail, and update the affected freeze/config/certificate bindings. No files were written or jobs submitted.