**Overall verdict: WRONG as a ready-to-run pre-registration.** A1 repairs many items, but several blocking errors remain, and its MMS projection and deterministic fault injection introduce new problems.

I reviewed [A1 and §§1–14](experiments/jcp-references/DESIGN.md), the [previous audit](experiments/jcp-references/audits/codex-design-1.md), and [the plan’s K0](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-06-jcp-offmesh-paper-plan.md:133). I ran small, read-only numerical checks; no files were modified.

The following table covers **all 43 previous WRONG or NEEDS-RESTATEMENT items**. “CORRECT” means the specified repair works, not that the entire design is approved.

| Previous item | Verdict | Finding and concrete fix |
|---|---|---|
| **3. Incompatible 2D IC** | **CORRECT** | A1.1 distinguishes the common continuum IC from nodal and spectral representations. Positive-time order is tested rather than presumed. **Keep.** |
| **5. Amplitude bound** | **CORRECT** | A1.2 withdraws the strict bound of 2 and records running maxima. **Keep measured maxima for diagnostics and stability checks.** |
| **7. Rannacher startup guarantee** | **CORRECT** | A1.3 replaces the unsupported theoretical guarantee with per-case positive-time verification. **Keep.** |
| **8. Preconditioner** | **CORRECT** | Explicit \(\theta,\tau\) remove the double-halving ambiguity: both CN and startup BE use \(I+\nu\Delta t\Lambda_h/2\). **Keep.** |
| **10. CFL interpretation** | **CORRECT** | Direction-summed values and the absence of a stability guarantee are explicit. **Keep.** |
| **13. Nonuniform order solve** | **CORRECT** | A1.5 rejects unbracketed, zero and nonfinite differences without clipping. **Apply the same explicit policy to the PS order solve.** |
| **15. Evidence for an asymptotic regime** | **NEEDS-RESTATEMENT** | A1.8 adds useful diagnostics, but arbitrarily inconsistent error vectors can still be accepted because \(c\) is never gated. Separate ladders on four development cases cannot establish absence of cancellation elsewhere. **Call these diagnostics; prescribe what a large \(c\) does to uncertainty eligibility.** |
| **17. Numerical tolerance floor** | **WRONG** | A1.4 measures sensitivity on only two cases per dimension and transfers it to all cases. Worse, it claims an unresolved reference has true error \(\le d_{23}/n_0+2\tau_{\mathrm{tol}}\). Neither level agreement nor tolerance agreement bounds common discretization bias. **Delete that bound; use case-specific sensitivity and explicitly conditional numerical-error estimates.** |
| **18. Remedy for floor failure** | **WRONG** | Tightening before refinement is correct, but “re-measured” after adopting the tight solution has no defined tighter comparator. Comparing tight against production measures the discarded solution’s sensitivity. **Specify a further tolerance level, its acceptance rule, and recomputation of all differences/order; unresolved cases remain unverified without an accuracy claim.** |
| **19. Péclet gate** | **CORRECT** | Running maxima replace output-only maxima; the gate is correctly described as a sanity check with an injected rejection test. **Keep.** |
| **20. Solver failures and retained attempts** | **CORRECT** | A1.5 separates persistent solver failure from order failure and retains attempts. **Keep; the storage schema must actually distinguish attempts, as discussed under item 54.** |
| **21. Meaning of \(2\epsilon_{\mathrm{ref}}\)** | **CORRECT** | A1.9 properly distinguishes the true-error triangle inequality from an estimated resolution threshold. **Keep that qualification throughout, including A1.4 and A1.7.** |
| **23. Tensor-sine de-aliasing** | **CORRECT** | A1.6 correctly replaces exact de-aliasing with projection quadrature. Fourth-order trapezoidal error is appropriate for the smooth, fixed-mode Burgers product. **Keep, but do not extend this order automatically to manufactured forcing or uniformly increasing bandwidth.** |
| **25. Initial coefficient quadrature** | **NEEDS-RESTATEMENT** | Numerical integration plus a doubled-order comparison is a repair, but “relative coefficient difference” does not specify a norm or treatment of zero/tiny coefficients. **Define the norm, absolute safeguard, and application at the escalation cutoff.** |
| **26. Spectral versus canonical \(t_0\)** | **CORRECT** | A1.1 stores the actual spectral initialization separately and fixes normalization using canonical nodal data. **Keep.** |
| **27. ETDRK4/contour accuracy** | **NEEDS-RESTATEMENT** | Contour comparison and a temporal MMS ladder are useful. However, coefficient agreement is not a trajectory-error bound, and nonautonomous forcing evaluation at ETDRK4 stage times is unspecified. **Define coefficient comparison norms and stage-time forcing; separate classical order tests from production stiff-order assumptions.** |
| **28. Assumed PS spatial order** | **NEEDS-RESTATEMENT** | Measuring three-level \(q\) repairs the blanket “about four” prediction. Calling the resulting difference a conservative bound remains unjustified. **Use “conditional spatial-error indicator” and specify the asymptotic/saturation assumptions.** |
| **29. Explicit-advection timestep heuristic** | **NEEDS-RESTATEMENT** | Stage monitoring and casewise halving improve the design, but A1 does not explicitly withdraw §3.3’s “explicit-advection limit” based on \(\lvert u\rvert\le2\). **State that it is only a heuristic and prescribe handling of large measured stage amplitudes or failed timestep checks.** |
| **30. PS temporal-error assessment** | **WRONG** | Every case now gets a halving run, but \(s_t\) underestimates the error of the **coarser timestep solution actually used in \(x_{PS}\)** even in an ideal fourth-order model: its error is \(16s_t/15\). Common temporal bias is also invisible. **Use an appropriately inflated conditional estimate, or compare against the half-step solution with verified temporal saturation.** |
| **32. Cross-check case selection** | **CORRECT** | Ties are resolved and the unsupported worst-case claim is withdrawn. **Keep.** |
| **33. Exhaustive cross-check decisions** | **WRONG** | The old numerical gap is closed for valid finite inputs, but rows overlap unless ordered, and there is no branch for invalid/unaccepted FD references or undefined \(\epsilon_{\mathrm{ref}}\). **Make this an ordered decision procedure with FD validity first, then PS validity/resolution, then discrepancy.** |
| **34. Escalation and terminal disposition** | **NEEDS-RESTATEMENT** | The new triple and terminal UNRESOLVED outcome are clear. Invalid FD uncertainty still lacks a disposition, and solver failure should include failed accuracy gates, not just nonfinite output. **Enumerate these terminal reasons and prohibit PASS on any of them.** |
| **35. Dimension versus case policy** | **CORRECT** | A1.7 explicitly suspends a whole dimension on FAIL and treats terminal UNRESOLVED cases individually. This is a conservative extension of K0. **Keep that distinction; repair eligibility and overall E0 status below.** |
| **38. Restricted scoring lattice** | **NEEDS-RESTATEMENT** | Recomputing \(n_0,p,\epsilon\) and restricting every arm is correct. But `stride` is ambiguous for flattened 3D interior storage, and acceptance/cross-check status must also be reconsidered on the changed metric. **Specify coordinate-wise restriction and recompute eligibility, not just numbers.** |
| **39. Vendor parity** | **CORRECT** | A1.13 fixes timesteps, tolerances, solver limits, initialization and outputs. **Keep.** |
| **40. FD manufactured solution** | **CORRECT** | The proposed \(u^*\) is smooth, vanishes on every wall, has nontrivial spatial derivatives and time dependence, and gives valid independently differentiated forcing. CN/BE forcing quadratures are correct. **Keep for G2; G4 has a separate projection problem.** |
| **41. Spectral linear test** | **NEEDS-RESTATEMENT** | “A finite set of sine modes” still leaves the actual test unspecified. **Freeze mode indices and coefficients, including mixed/high modes, and the precise error norm.** |
| **42. Spectral MMS** | **WRONG** | The MMS forcing generally does **not** vanish on walls. Projecting it with the same trapezoidal/DST rule introduces second-order low-mode quadrature error, incompatible with the prescribed spatial order \(\ge3\). **Integrate forcing coefficients accurately and independently, or choose a boundary-compatible MMS for the claimed projection order.** |
| **43. G5 recomputation/faults** | **WRONG** | Swapping fields **and metadata** also swaps descriptors. Reconstructing \(t_0\) from those swapped descriptors reproduces the swapped \(t_0\), so the advertised identity check passes. **Obtain expected descriptors from an independently pinned `(dimension, cohort, case)` mapping.** The 1% field perturbation is a sensible separate consistency test. |
| **45. X1 first-order rejection** | **WRONG** | X1 itself is repaired: timesteps are fixed and accidental passes are reported. But the new supposedly deterministic X1b can pass on the permitted nonuniform refined triple. **Use a synthetic exact \(O(h)\) sequence or prove a sufficient injection amplitude for every allowed triple.** |
| **46. X2 old-reference discrimination** | **CORRECT** | It now uses the new uncertainty, requires PS resolution, and treats universal failure as a hypothesis. **Keep, subject to repairing the PS uncertainty calculation.** |
| **47. X3 old bias** | **CORRECT** | Reference kinds and non-sealed population are specified; failed resolution is no longer called immateriality. **Keep.** |
| **50. Prefix-stable RNG** | **CORRECT** | Each row uses a generator determined only by `(seed, j)`, so changing cohort length cannot change earlier rows. Historical tables retain their exact counts. A read-only prefix test passed. **Pin the NumPy/bit-generator version for reproducibility.** |
| **51. Sealed storage/access separation** | **WRONG** | Encryption protects against ordinary accidental NPZ reads, but the key remains available to every consuming lane. A1 honestly admits this; it still does not meet independent custody. There are additional cryptographic/protocol gaps below. **Either establish separate custody or explicitly classify this as cooperative, unenforced sealing.** |
| **52. Global freeze/opening** | **NEEDS-RESTATEMENT** | A global manifest is the right repair, but “lists every lane” needs a trusted expected roster and verification of actual committed selections. Mere presence of hashes is insufficient. **Validate completeness, contents, reference version and coordinator authorization; reject duplicate/replayed openings.** |
| **53. QC and seed lifecycle** | **NEEDS-RESTATEMENT** | Seed collision handling and aggregate-only publication are improved. Failed generation attempts and undefined-\(p\) quantile handling remain unspecified; custody remains shared. **Freeze failure records, aggregate denominators/null handling and the permitted publication cadence.** |
| **54. Store/provenance schema** | **WRONG** | `u_L<L>` cannot store every attempt when the same level is rerun at different tolerances. The schema also does not explicitly retain all PS/tolerance-comparison fields needed for G5. **Key by attempt/configuration as well as level; store floor evidence and PS ladders; specify actual per-step residual/count arrays.** |
| **55. Loader/ranking API** | **WRONG** | Separating scoring from pairwise comparison is good, but `eligible = accepted and dimension validated` incorrectly permits an FD-accepted case whose own PS check is terminal UNRESOLVED when six other cases pass. `resolved` also ignores eligibility and metric identity. **Include case cross-check status and bind ranking to eligible scores with matching case, metric and reference version.** |
| **56. Two uncertainty bars** | **CORRECT** | A1.9 distinguishes the \(2\epsilon\) case rule from the stronger campaign claim margin and assigns the latter to consumers. **Keep.** |
| **58. Median old-reference displacement** | **CORRECT** | The maximum supplies the aggregate bound; the median is descriptive only. **Keep.** |
| **59. Historical discrepancy wording** | **CORRECT** | A1.9 distinguishes discrepancies/extrapolation displacement from verified true error and records loose tolerances. **Keep.** |
| **62. Compute estimate** | **CORRECT** | A1.15 withdraws the obsolete PS subtotal and conditions production on a calibrated replacement budget. **Include all added gates, repeats and escalation costs in that calibration.** |
| **63. Memory feasibility** | **CORRECT** | Streaming replaces whole-trajectory retention, host/device memory are distinguished, and real-size device-memory measurement is required. **Keep; the largest escalation configuration must pass that feasibility check.** |

Further adversarial checks of A1 give these results:

| A1-specific issue | Verdict | Finding and concrete fix |
|---|---|---|
| **Does \(q\ge2.5\), ratio \(4/3\), make \(s_x\) conservative?** | **WRONG** | It dominates the **Richardson estimate** under a single leading-error model, not the unknown true error. A shared bias survives all differences. **State the conditional model and call \(s_x\) an indicator; add directional consistency and numerical-floor checks.** |
| **Encryption beyond the shared-account limitation** | **NEEDS-RESTATEMENT** | CBC is unauthenticated; successful padding is not reliable wrong-key detection. OpenSSL explicitly documents this limitation. **Authenticate ciphertext or verify independently committed ciphertext and plaintext hashes before publishing any decrypted artifact.** Specify key encoding: raw random bytes must not inadvertently become a truncated line-based password. [OpenSSL documentation](https://docs.openssl.org/3.0/man1/openssl-enc/) |
| **Opening transaction** | **NEEDS-RESTATEMENT** | Writing an “immutable opening record” before decryption can leave an apparent opening after a wrong-key or partial-decryption failure. **Use an authorized opening-intent record, verified decryption, atomic publication, and a separate completion record.** |
| **A1 versus §10’s E0 verdict** | **NEEDS-RESTATEMENT** | §10 requires all eight PS cases to PASS; A1 allows dimension validation with six. The general precedence clause resolves direct overrides, but does not say whether “validated” now means “E0 passes.” **Define separate overall E0, dimension and case statuses, with an explicit truth table.** |
| **PS cross-check versus \(R_S\)** | **NEEDS-RESTATEMENT** | PS targets the continuous-time PDE; \(R_S\) targets fixed-step BE. Applying the same discrepancy criterion to both would confound BE time error with reference error. **Specify that this PS cross-check compares against \(R_{ST}\), and state exactly what its outcome establishes for \(R_S\).** |
| **Post-opening loader behavior** | **NEEDS-RESTATEMENT** | “The loader cannot read sealed/OOD data at all” conflicts literally with decrypting those cohorts into its store. **Say “before a verified global opening” and define how loads verify the opening record.** |

Four concrete counterexamples explain the most consequential findings:

1. **The spatial threshold arithmetic is right; the bound claim is not.** Under
   \[
   u_N=U+C N^{-q},
   \qquad
   \frac{s_x}{\lVert u_{N_3}-U\rVert}
   =\left(\frac43\right)^q-1,
   \]
   the threshold is \(q\ge\log 2/\log(4/3)=2.40942084\). At \(q=2.5\), the factor is only \(1.05280096\). But replacing the sequence by \(u_N=U+B+C N^{-q}\) leaves the measured order unchanged while allowing arbitrary common bias \(B\). Likewise, small tolerance differences cannot justify A1.4’s asserted true-error bound.

2. **G4’s forcing projection is second-order.** At a wall, \(u^*=u_t^*=0\), but generally \(\Delta u^*\ne0\), hence \(f=-\nu\Delta u^*\ne0\). The endpoint derivative of \(f\sin(k\pi x)\) then produces an \(O(M^{-2})\) trapezoidal error. For the stated 2D MMS at \(t=0.1,\nu=0.05\), a read-only calculation of the unnormalized forcing coefficient \((1,1)\) gave:

   | Projection intervals \(M\) | Absolute quadrature error | Observed order |
   |---:|---:|---:|
   | 48 | \(2.00550\times10^{-5}\) | — |
   | 96 | \(5.12978\times10^{-6}\) | 1.967 |
   | 192 | \(1.28969\times10^{-6}\) | 1.992 |
   | 384 | \(3.22875\times10^{-7}\) | 1.998 |

   This is a projection check, not a solver experiment. It demonstrates why the proposed G4 setup cannot assume spatial order at least three.

3. **X1b is not universally must-fail.** For the allowed refined widths proportional to \((3,1.5,1)\), take a common unit vector \(\phi\) and
   \[
   u_1-u_2=5.4d\phi,\qquad u_2-u_3=-d\phi.
   \]
   The original norm ratio is \(5.4\), giving \(p=2\). Set the injected finest-level magnitude to \(10d\), as prescribed. The new differences become \(20.4d\phi\) and \(4d\phi\), giving ratio \(5.1\) and **\(p=1.904531\)**—still accepted. A1 permits the inconsistent original directions because \(c\) is diagnostic only. On an initial dyadic triple, the common-direction injection is much stronger; the failure is the claimed guarantee across allowed triples.

4. **The loader can rank a K0-affected case.** Suppose six cross-check cases PASS, none FAIL, and case 7 has valid FD order but terminal PS UNRESOLVED. A1.7 makes the dimension `validated` and case 7 individually K0-affected. A1.12 nevertheless returns `eligible=true` because FD acceptance and dimension validation are both true. This is a direct internal contradiction.

**Final verdict: WRONG.** Before implementation, repair the floor claims, PS uncertainty construction and decision preconditions, G4 forcing projection, G5 identity check, X1b guarantee, and loader eligibility. The central/CN-R reference strategy and the revised row-wise RNG remain defensible.