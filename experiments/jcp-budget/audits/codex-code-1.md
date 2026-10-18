**Overall verdict: NEEDS-FIX.** The numerical stepping changes look sound, but the analyses and acceptance tooling can issue unsupported verdicts or accept incomplete evidence.

Audited HEAD: `51c2690cd70317374e3e0209e9a8eee8782a611c`. No files changed, jobs submitted, GPU used, or scratch created. During the audit, another process changed `ana7.py` to make stability independent of reference availability; I inspected that diff. Findings below remain applicable.

Paths below are relative to `experiments/jcp-budget/`.

**1. Estimands, decisions, and controls — NEEDS-FIX**

| Item | Verdict | Evidence and finding |
|---|---|---|
| Corner inventory and Möbius effects | CORRECT | `b2run.py:44–52` implements 22 pairwise corners or the 32-corner factorial plus six M corners. `ana2d.py:90–102,165–170` implements Möbius inversion and five-factor closure. |
| Dominance candidate universe | CORRECT | `ana2d.py:267–295` includes floor, ideal dynamics, and six main effects. **A2.1 supersedes §4.3’s maximum-pair candidate; A3 adds M.** Omitting pair interactions from this contest is correct. |
| Floor identity and \(U_\star\) | CORRECT | `ana2d.py:143–150,174–181` uses the scoring-interior projector, common normalization, and the sum of three distances before maximizing over time. |
| 2D \(\chi,\chi^\star\) | NEEDS-FIX | Signed extrema are computed correctly at `ana2d.py:184–198`, but the separately labelled numerical-background term is absent. Independently maximized scalars cannot reconstruct its per-time decomposition. |
| 3D interactions | WRONG | `ana3d.py:134–135` takes an absolute maximum and loses the sign of \(\chi^\star\). Ordinary \(\chi\), the form-effect cross terms/background, and pair classification/aggregation are absent. Form Q/T contrasts themselves exist at `ana3d.py:108–119`. |
| Quadrature-failure propagation | WRONG | `ana2d.py:406` checks only the single Q corner for `q_ok`; failed checks for other Q-containing corner types do not invalidate those contrasts. `summarise()` passes neither `q_ok` nor `star_ok` into pair decisions (`311–335`). In 3D, dominance is computed **before** G-B4 results and never uses them (`ana3d.py:259–275`). |
| Incomplete/minimum-population rules | NEEDS-FIX | Dominance checks eligible population before exclusions, then permits decisions on a smaller complete population (`ana2d.py:316–323`; `ana3d.py:262–268`). Pair percentages use available cases rather than the registered eligible-case denominator (`ana2d.py:326–333`). |
| Closure reporting and mesh comparison | NEEDS-FIX | Algebraic closure exists, but the requested \(D_a\) comparison with sum/RSS of main effects, GAL-member closure to the shared ideal endpoint, matched-sub16 mesh-invariance verdicts and bootstrap component ratios are absent from the analysis entry points (`ana2d.py:348–430`; `ana3d.py:215–345`). |
| B7 growth/front order | NEEDS-FIX | `ana7.py:143–166,287–311` implements only W2/W1, irrespective of T. W3/W2 is missing. Growth/front order use minimum eight even at 4096², where A2.4 specifies six. The front summary mixes diagnostic GAL and LSPG members. |
| 3D B7 metrics | NEEDS-FIX | `ana3d.py:302–343` produces qualification distances and some raw errors/energies, but no stability verdict, growth bootstrap/label, front-order result, or G-L1 reproduction gate. `--anaref` is accepted but unused. |
| Sampling adequacy | NEEDS-FIX | The uncertainty-aware per-case inequality and 90%/50% classifier are correct (`ana7.py:205–213,252–261`). Missing: fine-versus-coarse front-ranking adequacy, LSPG-only lane conclusion, and cross-check/G-R2 eligibility enforcement. `main()` also fails to pass the 4096² minimum population of six (`330`). |
| B8 IC diagnostics | NEEDS-FIX | Only deployed fine-route IC error is stored (`b2run.py:401`). Fine-route LS/floor errors, deployed/floor ratios and IC-effect persistence are missing; see DESIGN §6.4 versus `ana7.py:231–249`. |
| C-A and C-E | CORRECT construction; incomplete verification | `b2run.py:402–421` implements the amended alias amplitude and raw balance closed form. However, offline acceptance checks window maxima rather than independently reproducing every fine-route number. |
| C-F | NEEDS-FIX | The twice-orthogonal construction is present (`ana2d.py:221–258`; `ana3d.py:160–192`). The 3D control does not check unchanged ideal dynamics, and the deficient classifier/gating means it cannot establish the full registered decision contract. |
| C-I | CORRECT | `ana2d.py:448–454` tests the A2.2 perturbed-minus-unperturbed IC identity. |
| C-Q/C-S/C-T | NEEDS-FIX | Fixtures and distance thresholds are implemented. But `b2run.py:437–464` discards comparator validity; `ana2d.py:443–447` can count C-S as fired using a nonconverged tight comparator. Smoke acceptance also applies production “more than half” predicates to some two-case controls rather than the registered ≥1 smoke-case demonstration. |
| C-R1 | CORRECT | `anaref.py:149–154` explicitly requires solver/floor/shared-IC prerequisites and failed order. |
| C-score/C-R2 | WRONG | `ana2d.py:43–56` is not the production scoring path: ordinary analyses call `RS.score` or direct norms. The control tests an S/CNR incompatibility, not an ST↔S identity swap on an allowed BE row (`461–472`). Reference SHA, route, horizon and complete time-list identity are not enforced. No numerical shifted-score comparison or 3D C-score exists. |

A small CPU counterexample confirmed the gate-propagation defect: with both Q and ideal-anchor checks failed, `summarise()` returns **“inconclusive (incomplete)” dominance but “interacting” for QT**.

**2. Numerical drivers — mostly CORRECT, with acceptance defects**

- **CORRECT — `budcore` output-count change.** The diff against `p1core.py` is restricted to `nout` plumbing, output allocation and clipping. I ran the manufactured test with its file-writing statements removed in memory: all eight LSPG/GAL × BE/CNR/BDF2/CN combinations passed bitwise equivalence and horizon-prefix checks. Evidence: `budcore.py:118,166,226,340`; `test_budcore.py:41–68`.

- **CORRECT — split initialization and M switch.** `budcommon.py:182–207` preserves the deployed initialization scale while replacing only \(w_0\). `b2run.py:144–148,205–219` builds M=2048 settings with both qinf and qcheck roles.

- **CORRECT — streamed QR mathematics and ordering.** Bank blocks are x-major (`vendor/quad2d/qcore.py:166–182`), matching IC flattening (`budcommon.py:124`). Prefix columns, direct \(Q_b^\mathsf TU_b\), stacked QR and conditioning/orthogonality acceptance match A3.4 (`111–157`). A small synthetic check with a rank-deficient row block matched NumPy LS within `1.12e-16`.

- **CORRECT — 2D production certificates.** LSPG fresh-Jacobian gamma and GAL eta/gcert wiring match their intended roles (`b2run.py:249–287`).

- **WRONG — B7 check-rule acceptance.** 2D qp/qpp runs retain `level='prod'` and do not pass `anchor=True`, so `nfail>0` is allowed (`b2run.py:295–300`; `budcommon.py:203–206`). In 3D, qp/qpp FS/QN runs explicitly use `certify=False` (`b3run.py:101–102`), contrary to solver-specific A3.5 acceptance.

- **WRONG — pooled 3D acceptance.** `ana3d.py:53–64` applies the 1% certificate threshold separately per case, not pooled over the run. A synthetic 1/200 nonstationary-step example—0.5% overall—incorrectly rejected the case containing the one failure.

- **CORRECT — horizon plumbing and cache transition.** `s3common.py:100–103,180–238` sizes steps/output arrays and uses `budcore` for both `ad` and `adg`; `s3ctx.py:142` sizes storage dynamically. `b3run.py:163–165` clears both query and certifier caches before changing horizon-dependent execution. I found no stale-horizon reuse in this driver.

- **CORRECT — reference buffer/generalization edits.** `ref/refcore.py:152–191,207–238,255–278` handles output count, evidence buffers, stages and lattice reshaping. `ref/verdict.py:94–133` judges every evolved output. Per-window judging and separate fine files are implemented in `ref/refrun.py:131–142,204–238,264–272`. A small synthetic 3D stride check confirmed the `s*j-1` interior convention and shared-node equality.

- **CORRECT — B8 numerical formulas.** Fine decode, central-difference dissipation, interior band mask and alias field match the amendments (`budcommon.py:215–253`). Both route errors use the same final FD realization (`b2run.py:383–393`).

- **NEEDS-FIX — reference procedure beyond the solver.** `ref/refrun.py:355–380` implements stability retry but **not** the registered PS spatial/resolution escalation. `anaref.py:155–156` merely copies PS summary flags; it never calls either cross-check verdict function.

**3. Silent-failure risks — WRONG**

1. **Cross-check tiers are absent.** No per-window validated/unvalidated/suspended status is computed or enforced. B7/B8 can therefore issue verdicts without PS validation. Evidence: `anaref.py:155–161`; `ana7.py:143–166,205–261`; `ana3d.py:302–343`.

2. **G-R2 can match superseded attempts.** `anaref.py:47–74` searches all attempts and accepts any matched finest-base attempt; it does not require the relevant final-purpose realization. An initial prod attempt can satisfy the gate while the actual final reference uses unmatched tight/retry data. The T=1 versus T=0.5 prefix comparison is also absent.

3. **Reference metadata is not checked against the requested identity.** `b2run.py:317–341` verifies file hashes but does not assert metadata dimension/cohort/case/horizon/kind/times/cohort hash. `ana7.py:41–48` does not verify a manifest at all. A valid manifest for the wrong data is insufficient.

4. **NaN reduction is not a safe qualification rule.** `ana7.py:83` ignores NaNs in rho without retaining why each was undefined. `ana3d.py:331–340` uses Python `max`, whose result with NaNs depends on ordering. Preserve explicit zero/zero status and reject other nonfinite values.

5. **Reproduction coverage can be partial or empty.** G-B3 passes when any available comparison passes (`ana2d.py:415`; `ana3d.py:278`). Smoke G-L1 filters out members with no matches, allowing `all([])` to pass (`smoke_accept.py:82–83`). Archive selection uses glob/name conventions rather than checked configuration manifests (`ana2d.py:105–120`).

6. **Fine-route offline recomputation is missing.** Despite its docstring, `ana7.py` has no `--fine` argument or recomputation path (`314–335`). It trusts in-job scalar arrays. G-R3 trusts the recorded flag, and missing flags pass `is not False` (`anaref.py:159`). G-D1 checks only ROM walls (`ana7.py:264`), not reference walls.

7. **Missing work can appear as passing gates.** G-B0/G-B2 start at zero, so no evaluated identities can still pass (`ana2d.py:132,414`). The real-data mismatched-normalizer negative fixture is absent. The 3D smoke omits ideal-family G-B4 acceptance, G-B2, G-L1 and C-score (`smoke_accept.py:99–111`).

The manifest’s `.lstrip('./')` is overbroad normalization, but I did **not** find a collision for the currently generated `2d/...` and `3d/...` paths.

**4. Cluster mechanics — NEEDS-FIX**

| Item | Verdict | Evidence |
|---|---|---|
| Unique attempt directories, namespace, committed staging | CORRECT | `cluster/stage.py:53–73`; `cluster/submit.sh:22,29–31`. Git blobs come from HEAD and existing checkout files must match. |
| GPU partition, preflight, precision, paralab outputs | CORRECT | `cluster/stage.py:83–108` sets `gpu`, x64, highest precision, paralab output/cache paths and backend exit-42 preflight. |
| 45-hour reservations | CORRECT for the normal one-GPU submission path | `ledger.py:50–75` charges spent plus outstanding time-limit reservations; ambiguous submissions retain reservations. |
| Lane/campaign caps | NEEDS-FIX | `submit.sh:13,18–24` enforces lane 3 and account 8 at observation time. The lock is lane-local, so simultaneous submissions from different lanes can exceed campaign 8. It also does not enforce one smoke at a time or ≤2 hours per smoke. |
| Reference promotion | WRONG failure handling | `collect.py:44–51` checks destination existence, but inner-loop/subshell failures can be masked by later successful iterations; promotion is neither atomic nor reliably fail-fast. |
| `--no-fine` verification/cleanup | WRONG | `collect.py:34,65` uses unrestricted `--ignore-missing`, allowing missing non-fine files too. More seriously, cleanup deletes the job directory without verifying fine-file promotion. The marker at `68` claims files are in `refs/`, although collection does not promote them. |
| Provenance | NEEDS-FIX | All checked source hashes matched; all recorded destination hashes matched **except `resources.py`**, still declared unchanged at `PROVENANCE.json:49–54`. |

**5. Real-size memory/time — NEEDS-FIX certification; no demonstrated OOM**

Calculated f64 payloads—not measured peaks:

| Allocation | Payload |
|---|---:|
| 4096² rotated bank, r=384 | 47.98 GiB |
| All 38 mesh ICs retained on device | 4.75 GiB |
| LS stacked RHS, 16 cases | 2.00 GiB |
| Fine Bank(1024), r=384 | 2.99 GiB |
| Gauss640² table, M=2048, r=384: Gq/Gs/Psi | 8.59 GiB |
| FD L=8192, T=0.5 output buffer | 5.50 GiB |
| FD L=12288, T=0.5 output buffer | 12.37 GiB |
| FD L=512 in 3D, T=0.5 output buffer | 10.94 GiB |

Additional risks:

- `b2run.py:81–85` builds **all 38 ICs even for dev2/sub16**, retaining host and device copies.
- QR retains the full bank and stacked RHS, plus concatenated block, Q/workspace and residual temporaries (`budcommon.py:124–157`). Its memory is not merely the small final R.
- Off-mesh construction retains chunks while concatenating final tables (`vendor/quad2d/qcore.py:213–228`).
- **The pilot does not execute production’s M-table path:** it builds M=2048 only for `main` (`b2run.py:508–509`), while production builds M=2048 qinf and qcheck (`144–148,208–209`).
- Reference extraction pads whole output arrays before striding (`ref/refcore.py:233–235,273–275`), adding large temporaries.
- 3D rule tables remain cached across members/rules (`s3common.py:122–142`); the test-table builder constructs full m×M arrays (`vendor/quad3d/offmesh.py:58–75`).
- `resources.py:87–104` does not scale 2D B7 cost by horizon and assumes every b2run config has `b2`. A future B7-only T=1 job either fails projection or gets an inappropriate estimate. Certificates omit GPU identity (`124–125`) despite staging allowing different GPU types.

These are reasons to repair the pilot/certificate coverage before using its result as permission for larger jobs—not evidence that H200 must run out of memory.

**6. Other smoke blockers — NEEDS-FIX**

- The optional T=1 FD/PS configs exist, but `make_configs.py:115–117` creates no `b1024t1` ROM config.
- B8 3D is unimplemented. It may be **non-blocking only if explicitly recorded as deferred**, as permitted by DESIGN.
- The smoke acceptance program prints failure but exits successfully (`smoke_accept.py:112–117`), making shell automation liable to treat a failed audit as success.
- Positive CPU checks establish core arithmetic only; they do not establish real-size resource fit or any empirical gate/control outcome.

**Required fixes — blocking**

1. Implement PS cross-checks, escalation, per-window eligibility tiers and their propagation into every protected verdict.
2. Fix G-R2 final-purpose matching and add T=1/T=0.5 prefix checks; enforce complete reference/scoring identity on the actual production paths.
3. Propagate G-B4 and ideal-family failures to all dependent effects, pair labels and dominance; fix exclusion/population denominators.
4. Enforce converged 2D check-rule anchors and certified 3D FS/QN check runs; implement pooled 3D acceptance and explicit nonfinite handling.
5. Complete the registered B2/B7/B8 analyses: signed 3D interactions, closure comparisons, mesh comparisons, growth/front-order coverage and fine-route IC/ranking diagnostics.
6. Make smoke acceptance fail closed on missing coverage; add omitted controls/gates, independent fine-route recomputation, comparator-validity checks and nonzero exit on failure.
7. Make promotion fail-fast and verifiable; restrict missing-file exceptions to named fine files and prohibit cleanup until their retained copies are checksum-verified.
8. Exercise the actual production memory path in the pilot; bind resource certificates to GPU/config/horizon, and enforce sequential ≤2-hour smokes plus coordinated campaign-cap submission.
9. Correct `resources.py` provenance and either implement the optional T=1 ROM/3D B8 paths or explicitly record their authorized deferral.

**Overall verdict: NEEDS-FIX.**