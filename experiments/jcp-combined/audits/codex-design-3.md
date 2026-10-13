**A2 is not yet ready.** It resolves most operational issues, but leaves gaps in finalist qualification, selection semantics, H3 uncertainty handling, and mandatory negative controls.

Reviewed the [design and amendments](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/DESIGN.md) against [audit 2](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined/audits/codex-design-2.md), with A2 governing. “RESOLVED” below means resolved as a design contract, not implementation-validated. No files were modified, including the lab log.

**Previously PARTIALLY resolved items**

| Item | A2 disposition | Assessment and remaining fix |
|---|---|---|
| P5 | **RESOLVED** | A2.1 explicitly permits exact reproduction or approved equivalence, requires matching eligibility and verified-exit counts, and specifies stopping otherwise. |
| P7 | **RESOLVED** | A2.2 assigns mutations to discriminating schemes, specifies affected steps, and requires independent residual expectations. |
| P13 | **RESOLVED** | A2.3 replaces the interaction attribution with context sensitivity, adds a resolution floor, and requires verified, explicitly inventoried corners. |
| P14 | **PARTIALLY** | The 2D claim restriction and 32-case 3D budget checks are sound. However, selected-arm quadrature qualification extends to 64 cases without extending its actual-solver finer-rule check beyond 32. **Fix:** qualify each finalist’s reference under that finalist’s scheme, step and solver on all 64 cases, or restrict its qualification claim accordingly. |
| P15 | **RESOLVED** | A2.5 separates selection and adaptive-diagnostic statuses and withdraws the unsupported quantitative amplification inference. |
| P16 | **PARTIALLY** | Finalist distances now cover all 64 cases, and fallback identities and status are specified. Their reference qualification still has the P14 population gap. **Fix:** add the missing finer-rule runs and make failed reference qualification an explicit unavailable outcome. |
| P17 | **RESOLVED** | A2.6 gives symmetric heuristic thresholds and propagates E0 intervals through median/max endpoints. |
| P19 | **PARTIALLY** | Global winner points, missing-arm sums and unavailable settings are defined. The new generic “step choice” rule conflicts with the cheapest fast-equal-accuracy selection and does not explicitly define a tied set for nontransitive ties. **Fix:** specify separate algorithms for the two selections; see finding F2. |
| P20 | **PARTIALLY** | The fixed population, 52-case requirement, candidate censoring and separate H4b/equal-cost rules resolve the main issues. The BE comparator’s adaptive diagnostic is merely “reported”; unlike candidate probes, its verification and tighter-replay acceptance are unspecified. **Fix:** qualify that probe too, or mark its solver-sensitivity diagnostic unavailable and explicitly retain H4 solely as a comparison against deployed fixed BE. |
| P21 | **PARTIALLY** | G-FS1 is reconciled and alignment is separated from B0. But the replacement B0 constructions still do not guarantee a nonzero defect for the stated arbitrary inputs. **Fix:** pin discriminating fixtures and their analytic defects; see F4. |
| P23 | **RESOLVED** | A2.10 separates lossless score reproduction from float32 archives and explicitly permits reference-dependent H5 rematching over frozen inventories. |
| P24 | **RESOLVED** | A2.11 times all potentially selectable arms before selection, provides setting-relative panels and defines the per-invocation H5 statistic. |
| P26 | **RESOLVED** | A2.12 inventories conditional runs, qualification, fallbacks, timing and decodes, and gates admission on measured time and projected peak memory. |

**Previous findings N1–N13**

| Finding | A2 disposition | Assessment and remaining fix |
|---|---|---|
| N1 | **RESOLVED** | All potentially selectable arms enter the common timing panel before offline selection. |
| N2 | **RESOLVED** | Fallback lists are frozen; failed original qualification remains visible; a fallback cannot silently replace the selected method. Full qualification remains necessary for any promoted fallback. |
| N3 | **RESOLVED** | A1’s tight-solver form chain survives A2. Separate LSPG/GAL verification remains required. |
| N4 | **RESOLVED** | A2.3 explicitly inventories and verifies the four corners while preserving initialization and nonvaried ingredients. |
| N5 | **PARTIALLY** | Finer-neighbour checks, actual budget trajectory types and `M2_512` are added. Finalist reference qualification still stops at 32 cases although finalist acceptance uses 64. **Fix:** extend the actual-finalist finer check and define failure handling; F5. |
| N6 | **RESOLVED** | Selection/adaptive statuses and operator denominator-zero handling are explicit; amplification wording is withdrawn. |
| N7 | **PARTIALLY** | Normalization, time ties, ST/TX choices, zero vectors and conclusion scope are fixed. Scalar variation of \(e_\infty\) does not account for the simultaneous changes in \(\delta_{\rm BE}\) and the angle. **Fix:** use a vector-consistent uncertainty guard; F3. |
| N8 | **PARTIALLY** | Candidate qualification and comparison populations are fixed. The BE adaptive diagnostic still lacks an explicit verification/replay contract. **Fix:** qualify it or withhold that diagnostic, as in P20. |
| N9 | **PARTIALLY** | Aggregate symmetry, propagated intervals and global winner sets are fixed. Step selection and tolerance-based front construction remain ambiguous. **Fix:** use explicit set-based selection and separate front construction from tolerant coverage; F1–F2. |
| N10 | **PARTIALLY** | The invalid shifted-reference identity control is removed. The new oblique/out-of-span controls still need concrete discriminating fixtures. **Fix:** F4. |
| N11 | **RESOLVED** | Strict reproduction, archive precision and reference-dependent decisions now have separate contracts. |
| N12 | **RESOLVED** | Setting-relative timing, invocation-level FOM/ROM aggregation, baseline fallback and drift-failure handling are specified. |
| N13 | **RESOLVED** | `wrong_weight` is assigned to BDF2, where \(\beta_0\ne\beta_1\); the other mutations have appropriate designated schemes. |

**New or newly exposed problems in A2**

**F1 — NEEDS-RESTATEMENT: A2.6’s tolerant “dominance” cannot safely construct the advertised Pareto fronts.**

The relation permits mutual domination at equal cost and is nontransitive. For an error tie threshold \(T>0\), three equal-cost points with errors \(e\), \(e+0.75T\), and \(e+1.5T\) have adjacent ties but different endpoints. Removing every point “dominated” by another can remove all three; sequential pruning can depend on processing order.

This changes H2a’s evaluated front and potentially its verdict.

**Fix:** construct ordinary Pareto fronts using numerical cost/error and at least one strict improvement, collapsing exact duplicates deterministically. Apply the declared tolerance only to the subsequent cross-front coverage test. Alternatively, define H2a directly as coverage of every eligible BE point by an eligible candidate, without tolerant front pruning.

**F2 — NEEDS-RESTATEMENT: A2.6’s step-choice instruction conflates two different objectives.**

“Step choice within a scheme × setting: smallest median \(e_{\rm TX}\)” is unqualified. Applied literally to fast-equal-accuracy selection, it replaces A1.9’s cheapest qualifying step with the most accurate qualifying step.

The phrase “ties → smaller cost” also needs a set definition because tolerance ties are nontransitive.

**Fix:** specify:

- Accurate-equal-cost: restrict to eligible arms with ratio \(\le1\); form the set not beaten in median error by any other eligible arm; choose minimum cost, then largest step.
- Fast-equal-accuracy: restrict to arms satisfying all BE-relative accuracy constraints; choose minimum cost, with a deterministic tie-break.
- Empty feasible set: unavailable.

Apply the same explicit “beaten” relation when E0 intervals replace scalar thresholds.

**F3 — WRONG: A2.7’s scalar uncertainty check is insufficient for its H3 confirmation guard.**

Perturbing the normalized anchor by \(q\) changes both vectors:

\[
\varepsilon_\infty\longmapsto\varepsilon_\infty+q,
\qquad
\delta_{\rm BE}\longmapsto\delta_{\rm BE}-q.
\]

A2 checks changes in \(e_\infty\), but condition (c) also contains \(\|\delta_{\rm BE}\|^2\), and condition (b) depends on both vectors’ directions. An anchor perturbation can therefore reverse confirmation without being detected by the specified scalar test.

**Fix:** test all three conditions under a consistent anchor-perturbation model, using the uncertainty indicator as the stated resolution scale. For example, with \(E=\|\varepsilon_\infty\|\), \(D=\|\delta_{\rm BE}\|\), and perturbation radius \(U\), a conservative sufficient guard for (c) is

\[
\max(0,E-U)^2-e_{\rm BE}^2
\ge \tfrac12(D+U)^2.
\]

Add a corresponding angle guard and withhold confirmation when the angle is unresolved. Keep this explicitly an indicator-based robustness test unless \(U\) is certified as an error bound.

**F4 — WRONG: A2.9’s proposed properties do not themselves guarantee that B0 breaks.**

For an orthogonal \(P\), write \(v=v_\parallel+z\), where \(z\) is out of span. The identity defect is

\[
\|v-y\|^2-\|(I-P)y\|^2-\|v-Py\|^2
=-2\langle z,(I-P)y\rangle.
\]

Thus \(\|z\|=10^{-3}\) provides no positive lower bound: the defect is zero whenever \(y\) is in span, or its out-of-span component is orthogonal to \(z\). An oblique projector likewise need not break the identity for arbitrary \(y\).

**Fix:** pin fixtures with analytically nonzero defects exceeding the inherited \(10^{-6}\) bar. For example, with unit normalization:

- \(P=\operatorname{diag}(1,0)\), \(v=e_1+10^{-3}e_2\), \(y=e_2\): defect \(-2\times10^{-3}\).
- \(P'=\begin{pmatrix}1&1/2\\0&0\end{pmatrix}\), \(v=e_1\), \(y=e_2\): defect \(1/2\).

Specify an absolute-defect acceptance test. Likewise, use a known spatially varying fixture for the coordinate-shift control; a shift need not change an invariant field.

**F5 — NEEDS-RESTATEMENT: A2.4 extends finalist acceptance farther than reference qualification.**

The selected arm’s deployed-versus-\(r_\infty\) distance is evaluated on all 64 cases. Its \(r_\infty\)-versus-finer check under the actual selected algorithm is required only on the 32 budget cases. The all-64 checks under fixed BE and fixed CN do not cover an arbitrary selected BDF2/CN-R arm, different step, or \(K=2\).

A2 also says actual budget trajectories are “re-checked” without specifying what happens when these checks fail after the selection-solvers’ rule has passed.

**Fix:** include the actual finalist’s finer-rule trajectory on all 64 cases. If its reference fails that check, mark quadrature qualification unavailable and withhold conclusions requiring that qualification. Preserve the frozen selection record. Apply the same population requirement to any fallback claimed to be fully qualified; other budget diagnostics can remain restricted to 32 cases.

**NOT READY.**