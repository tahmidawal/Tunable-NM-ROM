**Overall verdict: NEEDS-RESTATEMENT, with WRONG implementation and disclosure items.** The recomputed PS data support A10’s numerical observations. The Richardson modification is defensible conditionally, but the round-off exception needs a stronger qualification, and the independent auditor contradicts it. No files were modified.

I recomputed the differences and orders directly from all eight NPZs using NumPy and an independent bisection solve on \([0.25,8]\). Below, differences are normalized by the stored canonical \(n_0\); maxima and order ranges cover \(t_1,\ldots,t_5\).

| Case | Observed \(q\) | \(\max d_{12}/n_0\) | \(\max d_{23}/n_0\) | A10 \(s_x\) | \(s_t\) |
|---|---:|---:|---:|---:|---:|
| 2D/0 | 1.02158–1.02497 | 5.41494e−9 | 2.68368e−9 | 7.82536e−9 | 6.71628e−9 |
| 2D/1 | undefined at \(t_1\); otherwise 2.79990–3.85227 | 7.85797e−15 | 1.39689e−15 | 1e−12 | 3.09304e−13 |
| 2D/2 | 1.00539–1.00818 | 1.66046e−9 | 8.27831e−10 | 2.46030e−9 | 3.40307e−9 |
| 2D/3 | 3.90753–4.23705 | 9.44367e−13 | 1.48296e−13 | 1e−12 | 2.96228e−13 |
| 3D/0 | 5.70451–5.86281 | 1.12994e−7 | 9.42000e−9 | 9.42000e−9 | 1.34269e−12 |
| 3D/1 | 5.55492–5.89088 | 2.17393e−9 | 1.85199e−10 | 1.85199e−10 | 1.02241e−12 |
| 3D/2 | 5.84783–5.89806 | 1.72462e−9 | 1.44601e−10 | 1.44601e−10 | 7.53269e−13 |
| 3D/3 | 5.80939–5.89083 | 1.06402e−8 | 8.77606e−10 | 8.77606e−10 | 1.56656e−13 |

**1. What was seen and when — NEEDS-RESTATEMENT.**

The narrow chronology is supported:

- `303eecb2d`, October 13 at 19:17:07 EDT, records PS3D and **FD3D dev4** outputs.
- `122ca4d5e`, 19:18:35 EDT, introduces A10 and records PS2D.
- The available `runs/` contains no new FD reference outputs for the **val32/val64 cross-check cases**. The stored summary marks all their cross-checks `MISSING`.

However, **“no FD reference of these cases existed” is literally false without “new second-order.”** Older first-order references for these same cases exist in the sibling worktrees, and `collect.py` explicitly loads them. Their history predates A10: 2D val32 in `81519a889` on October 2; 3D val64 in `2805fe949` on October 2. A9 also already reports a 2D FD calibration solve.

Thus the defensible disclosure is: *after inspecting PS self-convergence, before any recorded comparison with the new second-order FD references for these validation cases*. Repository history supports that statement; it cannot prove nobody computed an unrecorded comparison against the older references. I found no evidence establishing such a comparison.

**2. Indicator mathematics — CORRECT above the floor; NEEDS-RESTATEMENT at the floor.**

Under the assumed common leading error \(u_N=u_*+C N^{-q}\), the finest-level error multiplier is

\[
\frac{1}{(N_3/N_2)^q-1}.
\]

Taking its maximum with 1 guarantees an indicator at least as large as the raw difference. The floor branch also exceeds its raw difference. This is an asymptotic conditional estimate, not a guaranteed error bound; three levels do not establish the model by themselves. [NASA’s grid-convergence guidance](https://www.grc.nasa.gov/www/wind/valid/tutorial/spatconv.html) makes the same distinction.

**A \(q\) just above 0.5 is not a loophole.** On the fixed \(4/3\) ladder, the multiplier is bounded above by approximately **6.4641**, and on the escalated \(3/2\) ladder by **4.4495**. Increasing it makes the resolution gate harder to pass; it does not enlarge the discrepancy threshold \(2\epsilon_{\rm ref}\).

The **floor exception does introduce a loophole in the error-model argument**. For example, a perfectly valid single-power model with \(q=0.50001\) and \(d_{23}/n_0=9\times10^{-13}\) has Richardson error approximately \(5.82\times10^{-12}\), but A10 assigns \(10^{-12}\). Crossing just below the threshold can abruptly reduce the indicator. Undefined or arbitrarily slow convergence is also exempted there.

For these data, \(10^{-12}\) is a plausible empirical numerical allowance: case 1’s spatial differences are around \(10^{-15}\), while its time-step difference is \(3.09\times10^{-13}\). But case 3 shows coherent fourth-order differences below the floor—**below-floor does not mean demonstrated round-off**. The 3D time differences occasionally exceed \(10^{-12}\), although those are separately included in \(s_t\).

The floor therefore needs an explicit additional assumption: *remaining spatial error is negligible at this absolute scale*. Neither small self-differences nor FD Newton residual floors establish that assumption for PS.

**3. Agreement of implementations — WRONG.**

The per-time logic in [verdict.py:160](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/verdict.py:160) matches A10.1 for valid finite inputs. It correctly permits undefined order only at floor times and rejects undefined order at a non-floor time.

The independent auditor implements that exception, then cancels it: [audit_ref.py:297](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/audit_ref.py:297) puts every \(q\) into `nums` and requires every value to be non-null and finite.

I reproduced the contradiction entirely in memory, using the **actual case-1 PS arrays** with a synthetic accepted FD field equal to PS and \(\epsilon_{\rm ref}=10^{-6}\):

- `verdict.crosscheck`: **PASS**, \(\sigma=1.30930\times10^{-12}\).
- `audit_ref.audit_crosscheck`: **UNRESOLVED-PENDING-ESCALATION**, `match=False`.

The same bug affects mixed floor/non-floor times whenever a floor time has `q=None`.

Other edge findings:

- NaN `d23` is rejected; NaN `d12` at a non-floor time is rejected.
- At a floor time, NaN `d12` is exempted by the helper. Normally `ps['finite']=False` prevents acceptance. However, `crosscheck` trusts that flag: with a stale `True`, a coarse-array NaN can produce **PASS**. The independent auditor checks the arrays directly and rejects it.
- `eps_ref=NaN` produces `FD-INVALID` in `crosscheck`; the auditor lacks the corresponding finite-\(\epsilon\) validity check.

Thus the implementations neither agree on all requested edges nor implement the full policy identically.

**4. First-order convergence and the IC — NEEDS-RESTATEMENT; no demonstrated PS defect.**

The finest PS \(t_0\) discrepancies from nodal \(t_0\) are **0.00190243** and **0.00158395** for cases 0 and 2. They shrink approximately as \(N^{-1}\).

Crucially, **the discrepancy is not literally on the wall nodes**: both stored fields are zero there, and their wall-node difference is exactly zero. The Gaussian’s nonzero limiting wall values make its sine extension incompatible/singular; finite sine reconstruction differs at interior lattice points. Loss of spectral convergence from nonsmooth extensions is expected. [Trefethen’s discussion](https://epubs.siam.org/doi/10.1137/1.9780898719598.ch5) explains this mechanism.

That makes algebraic convergence unsurprising, but does **not prove that the precise evolved first-order behavior comes solely from IC projection**. Startup and space–time coupling remain possible. Here \(s_t\) is comparable to \(s_x\), so that qualification matters. Conversely, the spatial difference vectors align well: their normalized leading-vector defects are approximately 0.00245–0.00346, supporting the local Richardson model.

The observations do not invalidate PS’s independence from FD. They do limit a PASS to **conditional agreement on the scoring lattice at the five positive output times**. A PASS does not validate the \(t_0\) representation, early-time behavior, spectral-order accuracy, or absence of a bias shared by the PS resolutions. Nor does it prove the FD uncertainty is a rigorous bound.

**5. `old_rule_verdict` disclosure — WRONG/incomplete.**

The intended disclosure is appropriate, but it is not delivered fully:

- [verdict.py:206](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/verdict.py:206) sets `old_rule_verdict` only after several early returns. FD-invalid, gate-failed, spatial-order-failed and stability-failed records omit it.
- It computes the old verdict **before** checking finite discrepancy/time-error values. A synthetic NaN discrepancy yields old **FAIL**, while the current verdict correctly returns **UNRESOLVED**.
- `audit_ref.py` does not independently verify the old verdict.
- [make_report.py:260](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/make_report.py:260) does not print it. Its “X2 (old ref)” column concerns the **old reference field**, not the old acceptance rule.
- A verdict on the selected triple must be distinguished from the counterfactual **full old procedure**, which might require an escalation A10 avoids.

For the supplied initial ladders, the old spatial rule rejects 2D cases **0, 1 and 2**; A10 makes them eligible for subsequent checks. That substantive, post-data change must remain visible. **I would not accept A10 and its code as audited until the auditor contradiction and disclosure defects are corrected and the chronology/floor claims are narrowed.**