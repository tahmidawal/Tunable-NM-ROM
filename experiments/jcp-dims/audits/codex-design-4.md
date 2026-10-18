**Overall: NEEDS-RESTATEMENT.** A3 resolves the QN identity and zero-scale recovery blockers. It fixes rollout-versus-frozen-state precedence, but still leaves a false-positive path in fixture establishment.

Read-only audit completed: no files changed, jobs submitted, scratch created, or arrays copied.

References below: **D** = [DESIGN.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/DESIGN.md); **S/** = the predecessor’s `experiments/jcp-speed3d/`; **L/** = this lane’s `experiments/jcp-dims/`.

| Round-3 blocker | Verdict on A3 | Evidence |
|---|---|---|
| QN identity payload | **CORRECT** | D:656–663 replaces the nonexistent `it`/`rej` requirement with every returned per-step record, specifies serialization boundaries, and requires synchronization before stopping the timer. |
| Cross-family gate precedence | **NEEDS-RESTATEMENT** | D:665–672 correctly separates component (b) from rollout Q1. However, its nonfinite convention is not separated from A2.2’s fixture-admission predicate at D:623–629. |
| Zero-scale G-cert recovery | **CORRECT** | D:674–678 uses the actual denominator from L/p1core.py:213,218 and rejects nonfinite recovery. |

**1. QN identity: the named records really exist.**

[S/qn3.py:228](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d/qn3.py:228) constructs these per-step records:

```text
n_ref, n_exec, q, n_bt, n_clip, n_mu,
mu, refresh, pred, gam, rn, reason
```

Its return at line 240 includes all of them plus `W_all`. S/s3common.py:183–189 forwards this dictionary; lines 204–212 add output coefficients and decoded fields. Thus A3’s named QN keys exist, and “every other per-step key” covers `mu`, `pred`, `gam`, and `rn`.

The FS list also matches L/fs3.py:165–167. Adaptive records match L/p1core.py:217–221, provided the recording buffers are requested.

This is **design-level acceptance**, not implementation acceptance: S/s3ctx.py:229–231 still returns only output coefficients from its timing subject. The prospective wrapper required by A3 must replace that behavior and preserve the full-field query contract.

**2. Gate precedence: repaired, but fixture admission can still succeed on computational failure.**

The central precedence repair is sound:

- Component (b) can flag even when the bad rule’s rollout fails Q1.
- Component (a) retains A1.5’s rollout prerequisites.
- If (b) passes but (a) is unavailable, the family cannot pass.

The remaining defect is the interaction between two rules:

- **D:625–626:** admit the fixture iff every state has \(\rho_{\rm tgt}\ge1\).
- **D:666–668:** map nonfinite evaluations or ratios to \(+\infty\), which flags component (b).

There is no explicit requirement that **establishment** use finite vectors, finite norms, a positive target norm, and a finite ratio. Consequently, applying A3’s convention during establishment admits six failed evaluations:

```text
all(+inf >= 1)  → admitted
any(+inf > .116) → control fired
```

I checked this predicate counterexample with a small CPU-only calculation. It could certify the negative control without establishing any finite bad-vector discrepancy. Independent recomputation does not repair the logical predicate if both computations fail.

**Required correction:** separate establishment from operational gate handling. Fixture admission must require finite inputs and computed vectors/norms, strictly positive target norms, and finite ratios ≥1 at every frozen state. Otherwise label the fixture unavailable under A2.2. Production component (b) may continue to flag nonfinite results conservatively.

Also, D:668’s assertion that \(0/0\) is impossible for a nonzero state is unsupported: a nonzero coefficient vector does not itself prove a nonzero tested-advection vector. The conservative flag is acceptable; the impossibility claim should be removed.

**3. G-cert recovery: correct denominator, with ordinary floating-point rounding.**

[L/p1core.py:218](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims/p1core.py:218) records

\[
\mathrm{tolratio}=\frac{r_n}{\mathrm{tol}+10^{-300}}.
\]

A3 therefore gives the correct algebraic inverse. I checked zero residual/zero tolerance, nonzero residual/zero tolerance, and positive tolerance; all satisfied its comparison tolerance.

“Exactly inverting” should be understood algebraically, not as guaranteed bitwise recovery after floating-point division. This is **not a blocker**. For GAL, use its actual per-step tolerance, based on \(\|Ac_n\|\) at L/p1core.py:192. The existing L/qcert.py:84–95 remains an implementation-update item.

I re-scanned sections 0–11 and A1–A3, applying later amendments over superseded text. **No additional independent WRONG item rose to blocker level.** Previously resolved issues are not reopened. Fixture numerical admission and production implementation remain unverified.

**Ranked remaining blockers**

1. **Require finite, positive-denominator evidence for bad-vector fixture establishment.** Keep computational failure distinct from an established finite discrepancy; otherwise the control can falsely “fire.”

**Overall: NEEDS-RESTATEMENT.**