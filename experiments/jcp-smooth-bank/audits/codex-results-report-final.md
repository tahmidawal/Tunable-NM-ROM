**The round-2 numbers and “no useful winner” verdict are supported. The report still needs corrections.**

Checked 180 round-2 thresholds, 88 projection/derivative medians, 4,072 rollout records, all 76 FOM arrays and 540 timing invocations. The report reproduces byte-for-byte.

| Item | Verdict | Finding / concrete fix |
|---|---|---|
| Seed comparison | **CORRECT** | `base_s1` changes worst S error by **−3.3598% acc / −0.09818% fast**, and common-support D2 median by **+128.7007% acc**. Both confirmed \(m^*(0.06)\) values are **3136**. Call this a **two-seed difference**, rather than an estimated variance. |
| Re-evaluation | **CORRECT, narrowly scoped** | Largest relative difference across the checked headline metrics is **\(7.55\times10^{-15}\)**. Say “headline accuracy/projection/derivative metrics”; this does not describe timing reproduction. |
| λ ladder | **CORRECT** | Recomputed acc D2 / projection / S changes: λ=.01 **−53.019 / −23.448 / +0.962%**; λ=.1 **−70.348 / −38.207 / +1.720%**; λ=1 **−74.445 / −40.642 / +62.297%**. Printed thresholds match raw rho arrays. |
| Derivative statement | **NEEDS-RESTATEMENT** | The stated ranges and **3.24×** value-only spread are correct. Explicitly say **median common-support D2 error against mesh targets**. The ordering is descriptive; it establishes neither physical-gradient accuracy nor a seed-replicated treatment effect. Explain failure using the prescribed comparator gate, rather than implying the across-recipe range itself defines that gate. |
| Coarse-control table | **CORRECT** | Recomputed values below agree at printed precision. FOM errors independently recomputed from fields and references agree within **\(1.4\times10^{-17}\)**. |
| Coarse interpretation | **CORRECT, limited** | Supports the existing narrow statement: finer training data are not necessary for this observed gain over the 129-node FOM. It does **not** isolate smoothness, off-mesh sampling, or training resolution as the cause; the ROM also changes the numerical operator/procedure. Fast coarse **does not beat the 257-node FOM against S**. |
| Round-2 gates | **CORRECT in saved evidence** | Complete populations and case coverage; zero nonaccepted exits. Maximum C4 discrepancy **\(2.56\times10^{-7}\)**; tight-solver distance **\(1.70\times10^{-5}\)**; Gauss-768 distance **\(3.92\times10^{-10}\)**. C5a and rotation diagnostics pass. |
| `base_ev2r` eligibility | **WRONG** | “Rotation diagnostics missing” is a generator lookup bug: the renamed evaluation label has no corresponding training key. Its rotation hash equals `base`’s. Resolve training provenance through the original bank identity/hash; report eligibility **ok**. |
| FOM acceptance | **CORRECT; report omission** | All **76/76** cases have finite fields/residuals; largest residual **\(9.99\times10^{-13}<10^{-8}\)**. Print this explicit acceptance result beside the coarse table. |
| ev2r timing | **CORRECT** | Exact coverage: **15 subjects × 36 invocations**; every median reproduced. Burn-in, synchronization, f64/highest precision and GPU preflights present. Maximum coefficient discrepancy **\(1.79\times10^{-15}\)**. Valid within-job costs; no cross-job speedup inference. |
| Headline cost claim | **WRONG** | “Every bank’s \(m^*\) … same rung” is now false: acc sob1 needs **4096**, coarse **16384**. Restrict **67.8–68.2 ms** explicitly to ev1. In ev2r, acc selected-rule costs include base **65.3**, sob1 **79.6**, coarse **196.1 ms**. |
| Job provenance | **WRONG** | Combined rows assign round-1 IDs/hardware to round 2. Split jobs: tr2 **5027733**; ev2r **5037650**, **A100 80GB PCIe**. Cite amendments **A1–A7**, and replace “round 2 when present” with completed status. |

Coarse table, independently checked; **ST / S, percentages on the common 129-node restriction**:

| Model | acc | fast |
|---|---:|---:|
| coarse ROM | 2.749649 / 0.832359 | 4.940753 / 4.513604 |
| base ROM | 2.755661 / 0.876029 | 4.854310 / 4.357627 |
| FOM 129 | 9.580672 / 7.575506 | same |
| FOM 257 | 6.169585 / 4.117145 | same |

**Verdicts: CORRECT.** With `base_s1`, acc D2/D4 resolution thresholds become **112.55% / 112.19%**, so none of the derivative reductions clears them. Acc S requires **67.83%**; no treatment clears it. H1 is not met; sig1/sig2 H2 remain **UNRESOLVED**; all Sobolev H2 verdicts are **not met**. No promotion or `comb`. Sob1 additionally fails the useful-winner accuracy allowance.

| Remaining report/plot item | Verdict | Concrete fix |
|---|---|---|
| Prior H2 wording, onset, R1 provisional label, LM description | **CORRECT** | Main requested corrections are present. End the H2 brief with “sig1/sig2 remain unresolved” to remove residual ambiguity. |
| “Every number generated” | **WRONG, partly repaired** | R1 targets now come from structured evidence. Configuration/population numbers remain literals; derive or validate them against inputs. |
| Glossary | **NEEDS-RESTATEMENT** | Still missing the **228-state, initialization-included** projection/derivative aggregation distinction. Convert remaining prose equations to LaTeX. |
| Frontier | **NEEDS-RESTATEMENT** | Markers and annotations still overlap. Use facets or readable grouped annotations. |
| Frontier/ladders/spectra colors | **WRONG identification encoding** | Eleven labels cycle through ten colors: `coarse` and `frozen_deployed` share blue. Give them distinct encodings; identify `base_ev2r` as a repeated evaluation, not another trained bank. |
| E2E plot | **NEEDS-RESTATEMENT** | Accuracy shows both rounds; cost shows **ev1 only**. Label that explicitly and add a separate ev2r panel. The timing caption contains literal `\n`, overlaps and clips; use an actual newline. |
| Derivatives plot | **CORRECT** | Values, common-support labels and units are appropriate. |

Audit limitation: C4 vector discrepancies and solver-distance checks were verified from saved records, not regenerated vectors.

Repository content is unchanged and `git status` is clean. During the in-memory reproduction check, an unmocked plotting call inadvertently rewrote `e2e.png` with identical content; its timestamp changed.