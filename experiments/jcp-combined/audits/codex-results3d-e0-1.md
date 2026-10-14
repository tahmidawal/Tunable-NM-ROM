**Overall: NEEDS-RESTATEMENT.** The rescoring and A13 H6 calculations are correct. The interpretation must distinguish a selected-configuration dial from a bank-width effect, and reporting needs the qualifications below. No files were modified or created.

**1. Independent E0 rescoring — CORRECT.**

I independently decoded archived coefficients through the NumPy bank implementation on CPU, in float64, at all \(63^3\) scoring nodes. Sample: cases **0, 5, 33, 63**, all five settings, covering the selected accurate-equal-cost arm, BE baseline and CN-at-\(\Delta t_0\): **56 arm–case trajectories**.

Manual errors, `refstore.score`, and archived per-time/worst errors agreed within **\(1.11\times10^{-16}\)**. Normalisation was \(\|R(0)\|_2\); maxima excluded \(t=0\). Bank hashes, E0 store version, result hash and frozen-selection hash matched. The certified first-order gate passed all **157,440** comparisons. All **26,240** score entries also matched the E0 analysis summaries.

**2. H6 under E0 — CORRECT.**

All **64 val64/ST case-kinds are eligible**; registry uncertainties match the loader. Independently applying \(|e_2-e_1|>2\eta\) reproduces:

| Transition | Median change | Resolved / 64 | Improved / resolved | Ties | Verdict |
|---|---:|---:|---:|---:|---|
| W256→W512 | +8.923% | 63/64 | 33/63 = 52.38% | 1 | Does not extend |
| W512→W768 | −60.176% | 64/64 | 63/64 = 98.44% | 0 | Dial extends |
| W768→W1024 | −52.910% | 64/64 | 64/64 = 100% | 0 | Dial extends |

The bars are correctly applied: at least half resolved, median decrease ≥10%, and ≥60% improvement **among resolved cases**.

These compare **CN/0.0125/K1 → BDF2/0.0125/K1 → CN/0.016666667/K2 → CN/0.01/K1**, respectively. “Equal cost” is relative to each setting’s own BE comparator.

**3. H4 and frozen selections — CORRECT.**

Every reference-free arm diagnostic, anchor uncertainty, timing ratio and selection is unchanged. Reconstructing the decisions reproduces **H4a fail and H4b pass at all five settings**, with no fast-equal-accuracy selection. Comparison populations remain 63/64 for M2_512 and 64/64 elsewhere.

One serialization difference: E0 omits `cheapest_equal_accuracy_diagnostic`, present in the first-order H4 objects. Thus H4’s substantive results are unchanged, but the objects are not literally identical.

**4. Changed conclusions — NEEDS-RESTATEMENT.**

| Transition | First-order descriptive result | E0 result |
|---|---|---|
| W256→W512 | −21.22%; bars met | +8.92%; fails |
| W512→W768 | +20.96%; bars failed | −60.18%; passes |
| W768→W1024 | −12.75%; bars met | −52.91%; passes |

The first two conclusions reverse; the last retains its direction and now passes the conditional uncertainty-aware test. The old results were explicitly reference-limited descriptions, not established rankings.

Selected-arm median/worst errors, expressed as percentages:

| Setting | First-order median / worst | E0 median / worst |
|---|---:|---:|
| W256 | 0.9701 / 6.3652 | 0.5886 / 6.3095 |
| W512 | 0.7642 / 5.5187 | 0.6411 / 5.3742 |
| W768 | 0.9244 / 7.0749 | 0.2553 / 6.6984 |
| W1024 | 0.8065 / 6.0704 | 0.1202 / 5.5721 |

Supported wording: **the last two transitions of the frozen selected-configuration dial meet H6 under E0**. This does **not** establish a continuously improving full dial or isolate improvement caused by increasing bank width.

In particular, **W512→W768 worsens the worst error by 24.64%**, despite passing H6’s median/fraction bars. Case 5 supplies every displayed worst error. Neither uniform improvement nor improved worst-case accuracy is supported for that transition. Budget attribution remains separate.

**5. Labels — NEEDS-RESTATEMENT.**

- **POST HOC:** correctly carried through scores and analysis as “DESIGN A10, A11 extension.”
- **E0 conditional accuracy:** correctly qualified as an order-verified reference with conditional Richardson uncertainty, not a rigorous bound; the post-data cross-check amendment is disclosed.
- The shared accuracy-label suffix about \(R_S\) being time-matched to BE at \(\Delta t_0\) is inappropriate for this 3D output. **Only \(R_{ST}\) is used; 3D \(R_S\) does not match the ROM’s 0.01 step.**
- The current `report.md` 3D section still presents the first-order H6 results. It is correctly labelled historical/provisional, but does not yet communicate the E0 reversals required by A13’s primary-scoring convention.