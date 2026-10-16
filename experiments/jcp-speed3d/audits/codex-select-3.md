**No blocking defects remain in the reviewed analysis/selection scope under A5.** Both surviving round-1 fixtures were rerun against the current code in a copied lane. Additional CPU checks exercised the analysis entry point and downstream claim gates.

| Round-1 item | Disposition | Evidence |
|---|---|---|
| Complete selection population | **RESOLVED** | Missing-configuration and duplicate-input fixtures disable selection and prevent H1.4 scientific verdicts. |
| H1.2/H1.3 evidence classification | **RESOLVED** | Unresolved eligibility evidence remains unavailable rather than scientifically ineligible. |
| False H1.4 rescue | **RESOLVED** | Unresolved FS-K2 returns `unavailable (FS-K2 unavailable (evidence))`. |
| H1.4 3% cost tie | **RESOLVED** | Picker chooses the more accurate candidate costing 1.02 over 1.00. |
| H2 comparator admissibility | **RESOLVED** | Blocked comparator produces no winner or ratio; analysis printing now completes successfully. |
| Reference-eligible ranked cases | **RESOLVED under A5** | Explicit case IDs align comparisons; solver eligibility still covers the full cohort. Honest score producers pin the reference version/status. |
| Missing B* decomposition | **RESOLVED** | Fixture retains the winner and explicitly marks decomposition unavailable. |
| K9 gating | **RESOLVED** | Failed-K9 entry-point fixture blocks H2 and H1.4 claims. |
| FOM case alignment and η | **RESOLVED under A5** | Fixture excludes an unrankable case with enormous error, reports 63 ranked cases, and checks aligned η. |
| K10 convergence control | **RESOLVED** | Synthetic raw evidence is rejected by the shared validator for nonconvergence. |
| K10 near-reference-floor control | **RESOLVED** | At error `1e-5`, η `1e-4`, the real candidate matches and the inaccurate control does not. |
| No match versus unavailable | **RESOLVED** | Empty admissible family explicitly reports unavailable; failed joint matching does not assert accuracy dominance. |
| Comparator/winner label collision | **RESOLVED** | C_W768 winning against C_old retains canonical label `C_W768`, without crashing. |
| Direct B*–W* confirmation | **RESOLVED** | Distinct-winner fixture emits `['B_C_old', 'W_C_old']`. |
| Winner equals comparator | **RESOLVED** | Self-selection fixture records explicit ratio-one identities. |
| Unqualified ROMs exported for matching | **RESOLVED** | FOM CLI rejects `selectable=False` ROMs; mesh export excludes them. K3 failure also clears QN selectability. |
| Matching the actual timed ROM | **RESOLVED by A6** | `match|<label>` times the selected n=65 mesh-path configuration whose errors supplied the match. |
| p1/m1 resource certificates | **RESOLVED as executable estimates** | Both projection functions execute against actual smoke/pilot records and generated fixture configurations. |

A6 is consistent with H4.3/A1.9/A4.4 **as an explicit scope amendment**. It retains cross-resolution FOM selection and the every-mesh timing inventory, while restricting the primary matched claim to the selected n=65 mesh-path ROM. Larger-mesh panels require their own accuracy qualification before being called matched. Final analysis must preserve H4.3’s comparison direction: **FOM errors tie-or-beat ROM errors**. That final qualification/reporting remains deferred work, not a result established by this review.

Two resource concerns remain **NOT RESOLVED technically, but nonblocking under A5**:

- The K2 lookup still uses a K1 proxy; tighter-ADT and FOM calibration/compilation costs remain approximations. The output now correctly disclaims demonstrated upper bounds.
- `project_mesh()` does not explicitly inventory the added `match|<label>` panel. Its aggregate headroom is not evidence that this omitted cost is covered. This limits allocation confidence; it does not demonstrate a false scientific verdict.

Reproductions: [round-1 fixture rerun](/tmp/speed3d-review-round3-nu5vik92/check.py), [claim-gate checks](/tmp/speed3d-review-round3-nu5vik92/gates.py). No repository or git changes, job submissions, or GPU use occurred.

**READY.**