# Codex design audit (2026-09-21, before any job)

Codex CLI 0.155.1 cannot run shell commands on this box (bwrap: RTM_NEWADDR not permitted); the first attempt read nothing. This second run had DESIGN.md, the protocol, hires-burgers DESIGN.md (1-200), sep_common.py (60-170) and hires.py inlined in the prompt. Verbatim output below; dispositions are in DESIGN.md section 9.

```
codex
I’ll audit the inlined design and code, separating definite defects from implementation risks that need explicit checks. I won’t run shell commands.
codex
Audit verdict: the design is not ready for a confirmatory held-out claim. The clearest blocker is reuse of an already inspected hold64 to motivate the new model. The POD construction can be mathematically correct, but its orientation and network folding need precise specifications. Several required gates are recorded without being enforced.

This audit uses only the inlined material. The proposed POD builder, bh_hires.py, configuration files, and audit_hires.py were not supplied, so implementation-dependent findings are marked accordingly.

BLOCKER

1. hold64 has already influenced model selection at the campaign level.

DESIGN §1 uses hold64’s incumbent error, projection floor, and optimal-basis floor to motivate replacing the bank. §4 explicitly acknowledges prior hold64 evaluation of cat1024, which is now the selected starting bank.

This is adaptive reuse of a test set, even if this lane never loads hold64 before bh4. Selecting the three POD weight variants on sel32 avoids additional direct leakage, but does not restore hold64’s independence. The assertion “evaluated once” is also false across the campaign.

Fix: classify hold64 as an opened validation/regression cohort. Freeze the complete model and selection procedure, then evaluate a genuinely untouched cohort once. Keep the hold64 results, but do not describe them as independent confirmation. The supplied code does not show direct hold64 use in bh2 selection; the demonstrated leakage is historical and design-level.

MAJOR

2. POD singular-vector orientation is ambiguous and potentially wrong.

Let G have shape n × 1024, H = GᵀG = LLᵀ, and c_i be projected coefficients. Then a_i = Lᵀc_i is correct because

||Gc_i||² = ||Lᵀc_i||².

If the weighted snapshot matrix is A = [w_i a_i], with snapshots as columns, the required W consists of its LEFT singular vectors. Its right singular vectors live in snapshot space and cannot be used in V = sL⁻ᵀW.

If snapshots are rows, A_i,: = w_i a_iᵀ, the right singular vectors are correct.

Fix: specify shapes and orientation explicitly. For column snapshots, use W = U[:, :512] from A = UΣQᵀ. Verify WᵀW = I and VᵀHV = s²I.

The stated weights are correct if the intended objective is sum_i w_i² times squared field error. If w_i denotes the weight on squared error itself, use sqrt(w_i) in A.

3. Exact MLP folding requires more than block-diagonal concatenation.

For block b, write its final affine layer as h_b A_b + d_b and its out_scale as α_b. Partition V vertically into V_1 and V_2. The desired compressed features are

bc_poly(x) [α_1(h_1A_1 + d_1)V_1 + α_2(h_2A_2 + d_2)V_2].

For a merged out_scale β, the final weight must vertically stack

(α_1/β)A_1V_1 and (α_2/β)A_2V_2,

and the final bias must be

(α_1d_1V_1 + α_2d_2V_2)/β.

Both biases and each block’s out_scale must be included. bc_poly is applied exactly once by features(); it must not be folded in again.

There is another concrete trap: concatenating B_1 and B_2 makes features() produce

[sin(B_1), sin(B_2), cos(B_1), cos(B_2)],

whereas a naïve block-diagonal first layer expects

[sin(B_1), cos(B_1), sin(B_2), cos(B_2)].

Fix: explicitly permute first-layer rows, fold both scales and biases, and verify compatible network depths. No activation may be inserted after the compressed final layer. The proposed parity gate can detect these mistakes, provided its reference evaluates the original blocks independently. Add boundary and actual grid/stencil points to the random-point check.

4. Matching Gram trace does not preserve coefficient scales.

With the stated whitening, G′ᵀG′ = s²I. Matching the incumbent trace gives

s² = trace(G_incᵀG_inc)/512,

assuming the same grid and metric. This matches total bank energy, not individual coefficient scales, conditioning, or the incumbent’s anisotropic coefficient geometry.

Fix: replace “coefficient scales are unchanged” with “total Gram trace is matched.” Record s and the old/new Gram spectra. Define whether the extraction identity gate expects I or s²I, and account for any subsequent normalization when fitting the head and directions.

5. Cholesky whitening and the reported floor need numerical qualification.

Concatenated learned banks can have nearly dependent columns. H can be singular or ill-conditioned; Cholesky and one refinement step do not establish an accurate projection. Adding an undocumented ridge would change the POD metric and projection.

An approximate projection residual is an upper bound on the exact projection floor, not necessarily a lower bound on every ROM error. Therefore epsilon_k ≥ computed_f_k − 1e−9 is not justified solely by using Cholesky.

Fix: use rank-revealing QR/SVD, or document conditioning and a justified solver-error bound. Check projection orthogonality and independently validate representative floors. Treat near-threshold floor comparisons as unresolved unless numerical uncertainty is smaller than their margin.

6. The implementation does not enforce several advertised gates.

hires.py records failed parity, failed control certificates, repetition mismatches, and insufficient repetitions, then can still write complete=True and COMPLETE. Uncertified ROM arms remain in subjects and are timed. No supplied code enforces the frozen headline or prevents a failed headline from being replaced.

Some apparent gates are not tests at all:

- weights_frozen=True and final_cohort_unopened=True are unconditional metadata.
- The cohort hash check is bypassed when expected_physical_sha256 is absent.
- required_reps is configurable, so the five-repetition requirement can be weakened.

Fix: separate execution completion from scientific acceptance. Require cohort/model/config hashes, exactly defined acceptance conditions, and at least five repetitions. A failed headline remains a reported failure; it must not trigger replacement using hold64.

No supplied numerical threshold is literally proven impossible to fail. However, ROM error ≥ the exact bank floor is a geometric necessity for every in-span field, including a useless ROM. It checks consistency, not predictive quality. Likewise, a Gram check computed from the same matrix used to construct whitening is not independent validation.

7. FOM selection is underspecified and required comparator reporting is weakened.

Selecting the fastest eligible FOM using hold64 errors is expressly authorized by the stated FOM-selection rule. It is not forbidden ROM tuning. But the procedure must select one setting whose cohort-worst evolved error is no larger than the frozen ROM’s cohort-worst error. It must not choose a different FOM per case, discard inconvenient stalled settings, or use a different accuracy metric.

The design does not define the cohort timing aggregate used to choose “fastest” or calculate S. Median casewise speedup, ratio of median times, and ratio of total times differ.

There are also explicit contract discrepancies:

- fft_tight is untimed on hold64, yet a tight-FOM speedup is promised.
- The inherited addendum requires tight, named relaxed-passing, fastest eligible, and coarse comparator reporting on both scopes.
- Coarse controls are declared excluded from the paper, despite the campaign requiring their comparison.

Fix: freeze the comparator set and aggregation formula before bh4. Select separately for each timing scope if appropriate. Time fft_tight if quoting its speedup; otherwise label lean_tight as the measured comparator and establish its equivalence. Preserve the mandated coarse comparison. Never import a tight-FOM timing from another allocation.

8. The refined-reference comparison is not secured.

The new design promises only one addition to hires.py, but a refined reference depends on configuration and can be omitted, deadline-skipped, or fail while the job still completes. Prior dev6 refined-reference results cannot establish physical error for new hold64 trajectories.

Fix: require the refined-reference phase for the claimed coarse matched-accuracy comparison, with explicit coverage and acceptance criteria. If unavailable, label that comparison incomplete. Describe the primary ≤1% result precisely as evolved error relative to same-grid fft_tight, not error relative to the PDE solution. Nonlinear solver residual tolerance alone does not bound reference solution error.

9. Rung selection and tolerance selection can contradict each other.

The design first chooses a rung satisfying ≤0.8%, then permits gtol=1e−2 if its error is within 2% relative of the tighter arm. A tighter-arm error of 0.795% and looser-arm error of 0.810% meets that tolerance rule but loses the advertised margin.

“Cheapest” is also a fixed ordering, not measured cost. Different q, M, iteration counts, and retries can reverse that ordering. “Most accurate certified rung” leaves tolerance and ties unclear.

Fix: select among complete (q, M, gtol, variant) arms. Require the final selected arm itself to meet the threshold. Specify whether cost means measured bh2 time or a predeclared priority, and define ties and fallback behavior.

10. The certificate’s held-out population is not independent of model training.

Both dense and deployed certificate trajectories come from train_physical. Their fit/cert index split only establishes disjointness within that draw. The bank, head, and directions may already have seen those trajectories; directions explicitly use params_draw(0,128).

This is not demonstrated hold64 leakage, but “held-out reachable states” overstates the certificate’s independence.

Fix: use a separate certificate cohort disjoint from all training and selection data, or explicitly call this a within-training-distribution certificate. Extend overlap checks beyond cfg['train_seed']/cfg['train_trajectories'] to all training draws.

11. Full-grid error auditing is incomplete for most cases.

run_quick saves spatially subsampled fields for every case and full fields only for named audit arms/cases. Subsampled fields cannot independently reproduce full-grid norms or verify the worst hold64 case. New arm names can also silently miss audit_arms.

Fix: retain full fields outside git long enough for an independent streaming audit of every headline case/time and eligible FOM, then delete after verified collection. Assert audit coverage. Include the floor computation in that independent audit.

12. Fixed rank does not guarantee unchanged query cost.

R=512 preserves several array shapes, but changes in conditioning and the fitted head/directions can alter iteration counts, retries, initialization cost, and certificates. Wider coordinate evaluation is legitimately excluded from a frozen-bank steady-state query only if all bank-dependent work is truly outside every query.

The inherited timing is substantially sound: same allocation, randomized order, burn-in, synchronization, retained invocation errors, and dense outputs. However, block_until_ready(v) times the entire returned tuple, not only the six fields, so diagnostic work is included too.

Fix: state the comparison as steady-state query cost with setup excluded; report setup time and memory separately. Verify query-required initialization and decode remain inside timing and account for diagnostic overhead. Do not infer timing parity from dimensions.

MINOR

13. lat16 is a plausible negative control, not a guaranteed failure.

Having 225 nodes for 1088 tests does not prove large error on a finite reachable-state population. Passing the certificate is possible without an implementation bug. The new control also lacks the inherited requirement to demonstrate actual solution regression.

Fix: predeclare both certificate and solution-error outcomes. If it passes, report the negative control as unsuccessful; do not alter it after inspecting hold64.

14. The bh1 stop rule overstates its conclusion.

A selected floor ≥0.6% does not establish that rank 512 cannot reach ≤1%. It fails the chosen margin criterion for the tested construction.

Fix: describe it as a budget/selection stop, not an impossibility result.

15. Host-memory use needs an explicit bound.

At roughly 4096², six f64 fields occupy about 0.75 GiB per case. Keeping truth for 64 cases is roughly 48 GiB, and each retained arm adds roughly another 48 GiB. prune_each_arm only prunes after processing the complete cohort. Several retained neighbors can exhaust a 240 GiB host allocation.

Fix: stream truth and arm fields casewise, bound retained arrays, and verify complete cohort coverage. An OOM-dropped headline or comparator must make the comparison incomplete, not silently reduce the tested panel.
tokens used
28,786
Audit verdict: the design is not ready for a confirmatory held-out claim. The clearest blocker is reuse of an already inspected hold64 to motivate the new model. The POD construction can be mathematically correct, but its orientation and network folding need precise specifications. Several required gates are recorded without being enforced.

This audit uses only the inlined material. The proposed POD builder, bh_hires.py, configuration files, and audit_hires.py were not supplied, so implementation-dependent findings are marked accordingly.

BLOCKER

1. hold64 has already influenced model selection at the campaign level.

DESIGN §1 uses hold64’s incumbent error, projection floor, and optimal-basis floor to motivate replacing the bank. §4 explicitly acknowledges prior hold64 evaluation of cat1024, which is now the selected starting bank.

This is adaptive reuse of a test set, even if this lane never loads hold64 before bh4. Selecting the three POD weight variants on sel32 avoids additional direct leakage, but does not restore hold64’s independence. The assertion “evaluated once” is also false across the campaign.

Fix: classify hold64 as an opened validation/regression cohort. Freeze the complete model and selection procedure, then evaluate a genuinely untouched cohort once. Keep the hold64 results, but do not describe them as independent confirmation. The supplied code does not show direct hold64 use in bh2 selection; the demonstrated leakage is historical and design-level.

MAJOR

2. POD singular-vector orientation is ambiguous and potentially wrong.

Let G have shape n × 1024, H = GᵀG = LLᵀ, and c_i be projected coefficients. Then a_i = Lᵀc_i is correct because

||Gc_i||² = ||Lᵀc_i||².

If the weighted snapshot matrix is A = [w_i a_i], with snapshots as columns, the required W consists of its LEFT singular vectors. Its right singular vectors live in snapshot space and cannot be used in V = sL⁻ᵀW.

If snapshots are rows, A_i,: = w_i a_iᵀ, the right singular vectors are correct.

Fix: specify shapes and orientation explicitly. For column snapshots, use W = U[:, :512] from A = UΣQᵀ. Verify WᵀW = I and VᵀHV = s²I.

The stated weights are correct if the intended objective is sum_i w_i² times squared field error. If w_i denotes the weight on squared error itself, use sqrt(w_i) in A.

3. Exact MLP folding requires more than block-diagonal concatenation.

For block b, write its final affine layer as h_b A_b + d_b and its out_scale as α_b. Partition V vertically into V_1 and V_2. The desired compressed features are

bc_poly(x) [α_1(h_1A_1 + d_1)V_1 + α_2(h_2A_2 + d_2)V_2].

For a merged out_scale β, the final weight must vertically stack

(α_1/β)A_1V_1 and (α_2/β)A_2V_2,

and the final bias must be

(α_1d_1V_1 + α_2d_2V_2)/β.

Both biases and each block’s out_scale must be included. bc_poly is applied exactly once by features(); it must not be folded in again.

There is another concrete trap: concatenating B_1 and B_2 makes features() produce

[sin(B_1), sin(B_2), cos(B_1), cos(B_2)],

whereas a naïve block-diagonal first layer expects

[sin(B_1), cos(B_1), sin(B_2), cos(B_2)].

Fix: explicitly permute first-layer rows, fold both scales and biases, and verify compatible network depths. No activation may be inserted after the compressed final layer. The proposed parity gate can detect these mistakes, provided its reference evaluates the original blocks independently. Add boundary and actual grid/stencil points to the random-point check.

4. Matching Gram trace does not preserve coefficient scales.

With the stated whitening, G′ᵀG′ = s²I. Matching the incumbent trace gives

s² = trace(G_incᵀG_inc)/512,

assuming the same grid and metric. This matches total bank energy, not individual coefficient scales, conditioning, or the incumbent’s anisotropic coefficient geometry.

Fix: replace “coefficient scales are unchanged” with “total Gram trace is matched.” Record s and the old/new Gram spectra. Define whether the extraction identity gate expects I or s²I, and account for any subsequent normalization when fitting the head and directions.

5. Cholesky whitening and the reported floor need numerical qualification.

Concatenated learned banks can have nearly dependent columns. H can be singular or ill-conditioned; Cholesky and one refinement step do not establish an accurate projection. Adding an undocumented ridge would change the POD metric and projection.

An approximate projection residual is an upper bound on the exact projection floor, not necessarily a lower bound on every ROM error. Therefore epsilon_k ≥ computed_f_k − 1e−9 is not justified solely by using Cholesky.

Fix: use rank-revealing QR/SVD, or document conditioning and a justified solver-error bound. Check projection orthogonality and independently validate representative floors. Treat near-threshold floor comparisons as unresolved unless numerical uncertainty is smaller than their margin.

6. The implementation does not enforce several advertised gates.

hires.py records failed parity, failed control certificates, repetition mismatches, and insufficient repetitions, then can still write complete=True and COMPLETE. Uncertified ROM arms remain in subjects and are timed. No supplied code enforces the frozen headline or prevents a failed headline from being replaced.

Some apparent gates are not tests at all:

- weights_frozen=True and final_cohort_unopened=True are unconditional metadata.
- The cohort hash check is bypassed when expected_physical_sha256 is absent.
- required_reps is configurable, so the five-repetition requirement can be weakened.

Fix: separate execution completion from scientific acceptance. Require cohort/model/config hashes, exactly defined acceptance conditions, and at least five repetitions. A failed headline remains a reported failure; it must not trigger replacement using hold64.

No supplied numerical threshold is literally proven impossible to fail. However, ROM error ≥ the exact bank floor is a geometric necessity for every in-span field, including a useless ROM. It checks consistency, not predictive quality. Likewise, a Gram check computed from the same matrix used to construct whitening is not independent validation.

7. FOM selection is underspecified and required comparator reporting is weakened.

Selecting the fastest eligible FOM using hold64 errors is expressly authorized by the stated FOM-selection rule. It is not forbidden ROM tuning. But the procedure must select one setting whose cohort-worst evolved error is no larger than the frozen ROM’s cohort-worst error. It must not choose a different FOM per case, discard inconvenient stalled settings, or use a different accuracy metric.

The design does not define the cohort timing aggregate used to choose “fastest” or calculate S. Median casewise speedup, ratio of median times, and ratio of total times differ.

There are also explicit contract discrepancies:

- fft_tight is untimed on hold64, yet a tight-FOM speedup is promised.
- The inherited addendum requires tight, named relaxed-passing, fastest eligible, and coarse comparator reporting on both scopes.
- Coarse controls are declared excluded from the paper, despite the campaign requiring their comparison.

Fix: freeze the comparator set and aggregation formula before bh4. Select separately for each timing scope if appropriate. Time fft_tight if quoting its speedup; otherwise label lean_tight as the measured comparator and establish its equivalence. Preserve the mandated coarse comparison. Never import a tight-FOM timing from another allocation.

8. The refined-reference comparison is not secured.

The new design promises only one addition to hires.py, but a refined reference depends on configuration and can be omitted, deadline-skipped, or fail while the job still completes. Prior dev6 refined-reference results cannot establish physical error for new hold64 trajectories.

Fix: require the refined-reference phase for the claimed coarse matched-accuracy comparison, with explicit coverage and acceptance criteria. If unavailable, label that comparison incomplete. Describe the primary ≤1% result precisely as evolved error relative to same-grid fft_tight, not error relative to the PDE solution. Nonlinear solver residual tolerance alone does not bound reference solution error.

9. Rung selection and tolerance selection can contradict each other.

The design first chooses a rung satisfying ≤0.8%, then permits gtol=1e−2 if its error is within 2% relative of the tighter arm. A tighter-arm error of 0.795% and looser-arm error of 0.810% meets that tolerance rule but loses the advertised margin.

“Cheapest” is also a fixed ordering, not measured cost. Different q, M, iteration counts, and retries can reverse that ordering. “Most accurate certified rung” leaves tolerance and ties unclear.

Fix: select among complete (q, M, gtol, variant) arms. Require the final selected arm itself to meet the threshold. Specify whether cost means measured bh2 time or a predeclared priority, and define ties and fallback behavior.

10. The certificate’s held-out population is not independent of model training.

Both dense and deployed certificate trajectories come from train_physical. Their fit/cert index split only establishes disjointness within that draw. The bank, head, and directions may already have seen those trajectories; directions explicitly use params_draw(0,128).

This is not demonstrated hold64 leakage, but “held-out reachable states” overstates the certificate’s independence.

Fix: use a separate certificate cohort disjoint from all training and selection data, or explicitly call this a within-training-distribution certificate. Extend overlap checks beyond cfg['train_seed']/cfg['train_trajectories'] to all training draws.

11. Full-grid error auditing is incomplete for most cases.

run_quick saves spatially subsampled fields for every case and full fields only for named audit arms/cases. Subsampled fields cannot independently reproduce full-grid norms or verify the worst hold64 case. New arm names can also silently miss audit_arms.

Fix: retain full fields outside git long enough for an independent streaming audit of every headline case/time and eligible FOM, then delete after verified collection. Assert audit coverage. Include the floor computation in that independent audit.

12. Fixed rank does not guarantee unchanged query cost.

R=512 preserves several array shapes, but changes in conditioning and the fitted head/directions can alter iteration counts, retries, initialization cost, and certificates. Wider coordinate evaluation is legitimately excluded from a frozen-bank steady-state query only if all bank-dependent work is truly outside every query.

The inherited timing is substantially sound: same allocation, randomized order, burn-in, synchronization, retained invocation errors, and dense outputs. However, block_until_ready(v) times the entire returned tuple, not only the six fields, so diagnostic work is included too.

Fix: state the comparison as steady-state query cost with setup excluded; report setup time and memory separately. Verify query-required initialization and decode remain inside timing and account for diagnostic overhead. Do not infer timing parity from dimensions.

MINOR

13. lat16 is a plausible negative control, not a guaranteed failure.

Having 225 nodes for 1088 tests does not prove large error on a finite reachable-state population. Passing the certificate is possible without an implementation bug. The new control also lacks the inherited requirement to demonstrate actual solution regression.

Fix: predeclare both certificate and solution-error outcomes. If it passes, report the negative control as unsuccessful; do not alter it after inspecting hold64.

14. The bh1 stop rule overstates its conclusion.

A selected floor ≥0.6% does not establish that rank 512 cannot reach ≤1%. It fails the chosen margin criterion for the tested construction.

Fix: describe it as a budget/selection stop, not an impossibility result.

15. Host-memory use needs an explicit bound.

At roughly 4096², six f64 fields occupy about 0.75 GiB per case. Keeping truth for 64 cases is roughly 48 GiB, and each retained arm adds roughly another 48 GiB. prune_each_arm only prunes after processing the complete cohort. Several retained neighbors can exhaust a 240 GiB host allocation.

Fix: stream truth and arm fields casewise, bound retained arrays, and verify complete cohort coverage. An OOM-dropped headline or comparator must make the comparison incomplete, not silently reduce the tested panel.
```
