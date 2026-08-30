The FOM is still correct, but the gate harness is not ready as-is: items 1–3 and 5 are confirmed; item 4 is mathematically wrong; item 6 has the right scaling observation but an arbitrary proposed constant.

The operators and solver did not change. The diff from `6ca89db` only appends the generic MMS and consistency machinery after the original code; the JSON’s REF gate remains exactly zero.

## Item verdicts

1. **GENERIC MMS — CONFIRMED**

The implementation is independent of the published numerical anchors:

- `psi/u/p/f` are encoded analytically in [stk2d_common.py](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-30-stokes-vector/experiments/separable-decoder/stk2d_common.py:509).
- The published values are hardcoded only later as regression anchors in [stk2d_fom_gates.py](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-30-stokes-vector/experiments/separable-decoder/stk2d_fom_gates.py:103); they do not feed the fields, forcing, or solve.

I derived \(u,v,\Delta u,\Delta v,\nabla p\) afresh with SymPy from \(\psi_g,p_g\). Maximum relative disagreement with their formulas was \(1.69\times10^{-16}\). Using that independently generated forcing gave:

| N | err_u | err_p | cosine |
|---:|---:|---:|---:|
| 32 | 1.541713180702e-2 | 1.833939203346e-1 | 0.9106740518 |
| 64 | 3.820960253511e-3 | 4.577325785566e-2 | 0.9122181644 |
| 128 | 9.531799713310e-4 | 1.144216294892e-2 | 0.9125989704 |

Orders: velocity 2.012527 / 2.003115; pressure 2.002369 / 2.000145.

The cosine assertion \(<0.99\) is an appropriate non-degeneracy guard when combined with the anchor and order assertions. It is not independently a correctness test, but it reliably excludes the frozen family’s parallel-error pathology.

2. **S3 REPAIR — CONFIRMED**

My separate construction at \(N=16,32,64,128,256\) found:

- control metric \(0.125000000000\), to at worst \(1.3\times10^{-14}\) absolute;
- matched cosine \(0.999999999999996\) down to \(0.999999999999949\);
- solenoidal Frobenius maximum \(1.51\times10^{-16}\);
- solenoidal cosine maximum \(1.48\times10^{-15}\).

The tiny difference from the JSON’s \(1.386\times10^{-16}\) and \(1.453\times10^{-15}\) is floating-point multiplication order.

The retraction is accurate. The metric contains the \(1/\sqrt M\) aggregation but no remaining \(h\)-factor. The original decay came from projecting grid-white pressure noise onto a fixed, smooth, low-frequency control space—not from an impossible normalization.

3. **ASSERTIONS — CONFIRMED for the frozen invocation**

All requested checks now have executable assertions:

- S0: f64, JAX x64, `highest`, GPU;
- S-FOM: orders and anchors;
- dense ranks and indirect rank witnesses;
- S-ADJ negative control, measured \(0.7071067812\);
- all repaired S-PRESS metrics;
- S-EXACT field and prediction checks;
- S-FREESLIP deliberately required to fail: minimum velocity error \(1.27317\ge0.5\), maximum absolute order \(0.049526\le0.5\).

`complete=true` is assigned only after those assertions at [stk2d_fom_gates.py](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-30-stokes-vector/experiments/separable-decoder/stk2d_fom_gates.py:823). A normal failed run leaves the previously saved JSON at `complete=false`.

Carry-forward caveat: this guarantee assumes `ALLOW_CPU=0`, the frozen nonempty rank/free-slip ladders, and Python without `-O`. The environment permits explicitly skipping/freeing some of those requirements, although such overrides are recorded in the JSON.

4. **S-EXACT THRESHOLD FIX — WRONG**

Let:

- \(z\) be the sampled continuous solution;
- \(y=a z\) the exact discrete solution;
- \(x\) the computed solution;
- \(\epsilon=a-1\);
- \(\rho=\|x-y\|/\|y\|\), which is the code’s `exact_rel`.

The triangle inequality gives

\[
\frac{\left|\|x-z\|/\|z\|-\epsilon\right|}{\epsilon}
\le \frac{a\rho}{\epsilon},
\]

not \(\rho/\epsilon\). Their bound is missing the factor \(a=1+\epsilon\).

More importantly, \(\rho\) is measured from the same numerical error the check is supposedly constraining. Adding the missing factor would make the inequality a direct consequence of the reverse triangle inequality—hence unfalsifiable for every \(x\).

Concrete tests using the \(N=256\) pressure value \(\epsilon=2.5100143\times10^{-5}\):

- Parallel perturbation with field error \(9.9900\times10^{-9}\), inside the \(10^{-8}\) field tolerance: their prediction margin is \(1.0000251\), so it **fails**.
- Orthogonal perturbation of 10%: their prediction margin is \(0.999774\), so the prediction gate **passes a grossly broken field**. The separate field assertion catches it.
- Orthogonal perturbation just inside the field tolerance: prediction margin \(1.99\times10^{-4}\), easily passing.

Thus the current bound is technically capable of failing, but only because it is incorrectly too tight and direction-dependent. It is not a valid error bound or useful discriminator. The reported worst margin \(0.107594\) is arithmetically correct but carries no certification value.

The right repair is to retain the meaningful closed-form field assertion and either:

- demote `pred_dev` to a diagnostic; or
- compare it against a genuinely independent, pre-frozen rounding/backward-error model.

Simply inserting the missing \(a\) factor would create a tautological gate.

5. **MMSF — CONFIRMED, with quantified resolution**

The fourth-order FD comparison is adequate for ordinary sign and coefficient transcription errors. Independent symbolic differentiation confirms the formulas exactly.

For a single generic-MMS coefficient, a relative coefficient error of approximately \(1.0\times10^{-6}\) to \(2.8\times10^{-5}\)—about 1–28 ppm depending on the term—could slip beneath the \(10^{-6}\) gate. Sign flips produced relative discrepancies from \(7.10\times10^{-2}\) to \(1.99\), at least 70,000 times the tolerance.

So the gate does not detect arbitrarily tiny coefficient errors, but it decisively catches any scientifically meaningful sign, missing-term, or ordinary decimal/coefficient slip. The wording that every coefficient error is \(O(1)\) is loose; the actual gate remains adequate.

6. **\(1/\sqrt M\) FORWARD FLAG — OVERSTATED**

The scaling observation is correct:

\[
\frac1{\sqrt M}\ge10^{-2}
\quad\Longleftrightarrow\quad
M\le10^4.
\]

The constant floor first rejects a correct aligned control for \(M>10{,}000\).

However, \(0.5/\sqrt M\) is a conservative choice, not a derived “correct” constant. Since the matched cosine is separately required to exceed 0.99, that condition already implies approximately \(0.99/\sqrt M\) for the aligned contribution. A cleaner statement is to gate the dimensionless quantity

\[
\sqrt M\,(\text{control metric})\ge c
\]

with \(c\) explicitly justified—0.5 for coarse discrimination, or near 0.99 for agreement with the matched-control requirement.

Phase 2’s frozen contract only uses \(M\in\{32,64,128\}\) [STOKES-DESIGN.md](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-08-30-stokes-vector/experiments/separable-decoder/STOKES-DESIGN.md:231). The expected values are 0.17678, 0.125, and 0.08839, all safely above \(10^{-2}\). No phase-2 threshold change is currently necessary.

## Overall verdict

**PROCEED AFTER ONE REQUIRED HARNESS CORRECTION.**

The FOM, operators, generic MMS, repaired S3 control, MMSF gate, and assertion aggregation are accepted. Phase 2 should not proceed under the claim that the current S-EXACT prediction assertion is valid. Before phase 2:

1. Remove/demote the circular `pred_dev <= exact_rel/predicted_err` assertion or replace it with an independently frozen criterion.
2. Regenerate a `complete=true` phase-1 JSON under the frozen configuration with `ALLOW_CPU=0`.
3. Keep \(M\in\{32,64,128\}\); the \(1/\sqrt M\) issue is only a future scaling note.

No repository files were modified; verification was CPU-only, with scratch confined to the permitted directory.