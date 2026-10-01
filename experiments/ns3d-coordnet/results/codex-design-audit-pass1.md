**The lane is not ready for a trustworthy verdict.** The largest correctness bug is that Adam trains the Fourier frequencies, breaking periodicity. Several acceptance gates also fail to enforce their stated conditions.

Read-only review completed; no files changed, GPU jobs run, or pushes made. References below are relative to `experiments/ns3d-coordnet/` unless stated otherwise.

1. **WRONG — periodicity; CORRECT — curl, layout, scaling.**  
   `coordnet.py:72–75` correctly implements
   \((\partial_y\psi_z-\partial_z\psi_y,\partial_z\psi_x-\partial_x\psi_z,\partial_x\psi_y-\partial_y\psi_x)\).
   The `ij` grid and component-first reshape at `90–108` match `ns3d/ns3d_fom.py:129–143`. The \(n^{-3/2}\) factor is consistent with Euclidean POD normalization and tests scaled by \(\sqrt{2/n^3}\) (`ns3d-shift/shift_rom.py:356–386`).

   **But:** `coordnet.py:49,54` puts `B` in the differentiable parameter tree, and `train_bank.py:175–177` updates every leaf. Integer frequencies therefore drift; continuous divergence remains zero, so the divergence check will not catch lost periodicity.  
   **Fix:** exclude `B` from optimization; assert exact agreement with `frequency_set(kmax)` when saving/loading, and check opposite-face values and derivatives.

2. **NEEDS-CHANGE — projection objective is algebraically correct, but certification is incomplete.**  
   `coordnet.py:122–127` computes the stated variable-projection objective when ridge is zero. Positive ridge correctly decreases captured energy and increases loss; it represents a regularized least-squares objective, not the exact orthogonal-projector loss. Forming \(G^\mathsf TG\) squares conditioning; f64 plus a relative \(10^{-12}\) ridge does not establish accuracy for nearly dependent columns.

   Training optimizes **raw** samples (`train_bank.py:175`), whereas deployment projects them (`235–238`). This matches DESIGN, but means training loss can select the worse deployed bank.  
   **Fix:** retain the registered objective if intended, but certify final loss using QR/SVD, record singular values and ridge bias, and report the projected-bank loss separately.

3. **CORRECT — ordering algebra; NEEDS-CHANGE — transfer interpretation.**  
   `coordnet.py:152–158` correctly forms rows \(u_i^\mathsf TQ/\|u_i\|\), obtains \(V\), and solves \(T=R^{-1}V\). Hence \(P_{48}G_{\rm raw,48}T=QV\). A small NumPy check agreed to roundoff.

   Applying the same \(T\) elsewhere is valid, but does **not** guarantee identical columns or orthonormality: \(P_{48}\) retains component frequencies below 16, while \(P_{32}\) retains below \(32/3\) (`ns3d/ns3d_fom.py:23`). SiLU introduces higher harmonics.  
   **Fix:** qualify mesh equivalence by measured projection/aliasing error and enforce a cross-mesh discrepancy threshold, rather than relying only on Gram matrices.

4. **CORRECT — resampling on the intended even meshes.**  
   `bankio.py:24–33` copies exactly \(0,\ldots,m-1,-m+1,\ldots,-1\) on each axis, excluding Nyquist planes. Parseval energy accounting and amplitude preservation with `norm="forward"` are correct. A small sinusoidal round trip agreed to roundoff.

   **Minor fix:** validate even cubic input/output meshes. The same-size shortcut at `22–23` preserves Nyquist planes, contrary to the blanket docstring; clarify that exception.

5. **NEEDS-CHANGE — Löwdin and autodiff algebra are correct; prefix semantics and guards are not sufficient.**  
   `bankio.py:54–81` correctly computes the symmetric orthonormalization. `114–135` correctly aligns component/point chunks; `rom_mesh.py:196` correctly applies `T @ lowdin_S`. The tested derivative comparison is valid up to sampling aliasing, provided periodicity is restored.

   However, Löwdin mixes columns: generally
   \[
   \operatorname{Lowdin}(B)_{:,:r}\ne\operatorname{Lowdin}(B_{:,:r}).
   \]
   Thus truncating `T` to export a selected prefix would change the bank whose floor was measured. Entrywise Gram error also does not tightly bound collective distortion at large rank. `lowdin` has no finite/eigenvalue guard, and the bank job never enforces DESIGN’s 0.05 abort threshold.

   **Fix:** freeze parent-bank identity plus prefix length; always reproduce the same full-bank normalization before slicing. Check finite positive eigenvalues before inversion, enforce the abort in both stages, and gate/report spectral Gram error and actual column changes.

6. **NEEDS-CHANGE — training compression is correct; weighting and references need repair.**  
   Pooling/centering are correct (`train_bank.py:99–117`); `Xw.T @ V` is \(U_KS_K\) (`142`); `ev[R:K].sum()/ev[:K].sum()` is the correct compressed Eckart–Young optimum (`223`). Oracle floors use the parent metric directly (`260`), and cross-mesh scaling \((n/n_0)^{3/2}\) is correct (`311–313`). The uncentred branch genuinely avoids shifting (`113,289–300`).

   Problems:
   - `118` normalizes by the **resampled** initial state, changing the intended original-FOM normalization when energy is discarded. **Fix:** take `mean(frames[0]**2)` before resampling.
   - Same-data POD is resampled and QR’d without \(P_n\) (`268–271,296–299`), unlike coordnet. **Fix:** apply the same discrete-space projection, and distinguish the raw training optimum from the deployed reference.
   - Calibration times unclipped Adam, then trains clipped Adam with a fresh compilation (`188–205`); its allocation excludes calibration and downstream work. **Fix:** calibrate the actual update, reserve overhead, and ensure `steps > warmup`.
   - Parent floors are recorded but never reproduction-gated (`273–288`). **Fix:** compare with pinned parent floor records before accepting stage 1.

7. **NEEDS-CHANGE — ROM recipe matches; acceptance and freezing are incomplete.**  
   POD reconstruction/sign alignment, frozen-head loading, head data/weights/seed, midpoint stepping, and CNAB2 calls match the parent (`rom_mesh.py:141–166,237–287,313–339,391–407`). The development and test reference keys in `make_configs.py:73–94` exist and have the expected case/time dimensions.

   Concrete failures:
   - Reproduction failure merely sets `passed=False` (`361–370`); frame-zero, LM parity, and derivative discrepancies only get recorded (`198–200,373–388`). **Fix:** enforce explicit finite-value and acceptance checks.
   - Timing omits the lower bound on `end_ratio` (`437–438`), and neither timing failure nor output disagreement blocks acceptance (`449–454`). **Fix:** enforce both bounds and an explicit output-agreement tolerance.
   - No supplied code implements bar (b)’s comparator selection. **Fix:** select minimum measured CNAB2 median among finite, stable rows with evolved worst no greater than the arm’s; persist the eligible set and selected row.
   - Test mode reads frozen heads and spans, but still takes bank identity, `k`, timestep, iterations, and damping from mutable config (`116–126,212,308–310`). **Fix:** source or validate all these against the frozen manifest.
   - There is no selected-prefix argument: the head always uses all columns (`171–174`). **Fix:** implement the prefix convention from item 5.
   - `make_sampler` passes points explicitly to its inner JIT (`coordnet.py:103–109`), **but the outer training JIT captures `sample_t` and therefore its points** (`train_bank.py:157,171–175`). Parent `head_floor_errors` also captures candidate arrays (`ns3d-shift-head/head_rom.py:319–326`). **Fix:** pass these arrays explicitly through the outermost compiled functions.

8. **WRONG — the independent verifier can accept incomplete or nonfinite evidence.**  
   The NumPy error formula is independent and correct (`verify_coordnet.py:20–24`). But:
   - It audits whatever files exist, not every expected finite row (`51–53,67`). Missing arm files can pass.
   - It never rejects NaNs explicitly. A small CPU check produced `gap=nan`, `count_ok=True`; `max(0.0, nan)` then leaves the accumulated worst gap at zero (`73–80`).
   - Floors and frame-zero errors are not saved/audited, despite DESIGN’s “every reported error” wording.
   - The perturbation control uses a fixed tolerance rather than the selected file’s acceptance path (`54–64`).

   **Fix:** derive the required file set from the summary; require exact shapes, approved dtypes and finite values throughout; audit all claimed statistics/controls; run corruption through the identical rejection function. The existing perturbation should fire on the parent CNAB2 data, but that does not repair these holes.

9. **WRONG — stop-rule justification; NEEDS-CHANGE — selection and controls.**  
   `DESIGN.md:182–185` treats centroid-shift projection error as a lower bound for a ROM that solves its shift. It is not:
   \[
   \min_{c,a}\|u-\tau_cGa\|
   \le \min_a\|u-\tau_{c_{\rm centroid}}Ga\|.
   \]
   A different shift can beat the reported “floor.” Likewise, nearest-code plus finite-sweep head fits are achievable reconstruction errors, not certified manifold minima.

   **Fix:** call stopping a development-screen decision; report rollouts as **not evaluated**, not “unreachable.”

   Additional design defects:
   - If no bank reaches 0.25% but the rank-64 ratio is at most three, neither selection nor stopping is defined (`178–185`). Prefix ties between larger parent banks are also unresolved. **Fix:** specify exhaustive deterministic outcomes.
   - Eckart–Young does not transfer a historical POD held-out worst error into a lower bound for another training set/bank (`186–190`). A stronger bank might also absorb motion without frame evolution. **Fix:** treat control success as requiring explanation, not automatically a broken setup.
   - Development-selected floors and mesh ratios are selection-conditioned; label them accordingly. The named test cohort is already historically opened (`194–199`), so it is not a newly sealed cohort.
   - The 0.25%, 1.5× and 1.25× bars are understandable engineering thresholds, but no consolidated validator enforces them. Keep fractions (`0.0025`) separate from percentages and from the unrelated 5% outlier-count threshold.

10. **WRONG — cluster failure propagation and cleanup; CORRECT — basic staging/preflight.**  
    GPU partition, f64/highest precision, backend preflight, direct paralab staging, fresh-directory check, and before/after queue checks are present. Both `fd_fom.py` and `shift_pilot.py` are covered by the staged packages (`cluster/submit.sh:11–17`); frozen directories are included.

    Critical fixes:
    - `job.sbatch:49` exits with `RC`, ignoring verifier failure. **Fix:** return nonzero if either execution or verification fails.
    - Fields are deleted even after failed execution/audit (`37–38,45–47`). **Fix:** delete only successfully audited fields; retain failed evidence.
    - `pull.sh:15–24` checks hashes and a GPU line, then deletes the remote directory without checking run completion or audit success. **Fix:** require successful exit records and expected passing verification artifacts before deletion.
    - Bank staging suppresses every error with `|| true` (`submit.sh:20`). **Fix:** resolve required bank files from configs and stage/hash them explicitly.
    - Test manifests hash nested manifest files but do not validate their contents during pull. **Fix:** validate each mesh manifest and expected outputs.
    - `COMMIT.txt` alone cannot identify uncommitted staged code. **Fix:** create a complete staged-input content manifest and preserve it with results.

Other bugs found:

- `make_configs.py:44` omits rank 192, although DESIGN and `bank_r256.json:35` include that candidate. Derive the ladder consistently.
- `configs/bank_ff64.json:18` uses compression \(K=1024\), while DESIGN specifies 512; `bank_pilot.json:107–115` adds an unregistered architecture C. Record prospective amendments and keep pilot selection separate.
- `probe_local.py:70–71` already reads development data; existing probe logs contain measured development floors. Document this exposure when describing preregistration.
- Mesh jobs retain every field set until the final audit (`rom_mesh.py:350,401`), unlike the parent’s streaming audit. On the nearly full share, stream verification and delete only each successfully verified field set.

Python and shell syntax checks passed. Small CPU-only checks confirmed resampling/ordering algebra and reproduced the verifier’s NaN acceptance hole.