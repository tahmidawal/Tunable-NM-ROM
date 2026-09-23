# heat-compare-hires — comparison with other methods, heat 2D at $1024^2$ / $2048^2$ / $4096^2$

Pre-registration, written and committed before any cluster job of this lane. Amendments are
appended as §A1, §A2, … and nothing above them is edited afterwards.

Branch `exp/2026-09-23-heat-compare-hires`, forked from `exp/2026-09-20-hires-heat` @ `4fb12a6d`
(frozen wide-bank model `experiments/hires-heat/inputs/wide2d`, its `core.py`, the CN–CG FOM).
Cluster namespace `/cluster/tufts/paralab/tawal01/hcmp_20260923/<job>/`, one directory per job.

## 1. Question

The paper's Table 1 heat rows (wide bank, $K=8$, $R=128$, $q=32$, sealed cohort) report
0.49 % worst all-times error and a speedup over the fastest tested CN–CG at least as accurate
of 13.9× / 60.0× / 103× at $1024^2$ / $2048^2$ / $4096^2$ (batched fit). The Burgers table at
$256^2$ prices POD-LSPG, a quadratic manifold and four neural operators in the same allocation.
This lane builds the same comparison for heat at high resolution: every method, one allocation
per mesh, every speedup a ratio of two times measured in that allocation.

Heat is linear. Linear-subspace methods (POD, the linear span of our own bank) are expected to be
at least as accurate as the NM-ROM and possibly faster. That is reported plainly if it happens.

## 2. Problem, cohorts and metric (inherited, unchanged)

Family `mr2d` (hires-heat `core.family`), $\nu=0.02$, outputs $t\in\{0,0.1,\dots,0.5\}$,
zero-Dirichlet unit square, $n$ intervals per axis, $(n-1)^2$ unknowns.

* **Training** (everything any baseline is fitted on): seed 791000, 512 draws — the NM-ROM's own
  training set. **Validation** (every baseline choice is made on it or on a training split):
  seed 791001, 16 draws — the NM-ROM's validation set.
* **Evaluation**: the hires-heat **sealed cohort, seed 791099, 16 draws**, the cohort of the
  paper's heat rows (hires-heat `configs/h2d-final04.json`, `cohort_names` "sealed_opened_once";
  DESIGN addendum 1 of that lane). It was opened once by hires-heat for the NM-ROM. This lane
  evaluates every method, frozen, on the same 16 draws so the comparison matches Table 1; this
  is a second opening of that cohort and is recorded as such. **No choice in this lane is made on
  it**: all ranks, ridge weights, operator configurations and checkpoints are fixed before the
  panel jobs, from training/validation data only.
* **Error**: current-relative $L^2$ error against the exact same-grid semidiscrete flow (DST),
  per output time; per arm we report the **worst** over cases of the maximum over all six times
  (incl. $t=0$), and the **median** over cases of that per-case maximum. Evolved-only and
  continuum-reference errors are stored beside it.
* **Time**: median GPU milliseconds of the query "supplied full initial field on the GPU → six
  full fields on the GPU", synchronised, after burn-in; offline assembly/compilation excluded
  and recorded. The same contract for every subject (hires-heat's `query_contract`).
* **Speedup of an arm** = time of the **fastest tested CN–CG setting whose worst error is ≤ the
  arm's worst error**, divided by the arm's time, both from the same job. If no tested CN–CG is
  as accurate, the arm is reported against the most accurate tested CN–CG and flagged
  (a lower bound on the true ratio). The named FOM (Δt 0.025, rtol 1e-6) ratio is stored beside it.

## 3. Arms (per mesh)

**NM-ROM** (frozen `wide2d`, $K=8$, $R=128$, $M=256$ sine tests, no retraining):
`nmrom_q0_cn`, `nmrom_q32_cn`, `nmrom_q0_field_direct_tol1e-4_chol`,
`nmrom_q32_field_direct_tol1e-4_chol` — exactly the arms of the paper's heat rows (hires-heat
`h2d-final04`), same code (`hires-heat/core.py`).

**Linear-bank baseline** (the $q=R$ top rung, not the NM-ROM): hires-heat's
`linear_bank_{field,moments}_BASELINE` (free bank coefficients, exact reduced weak evolution) and
a weak-CN twin `linear_bank_moments_cn_BASELINE` (same $M=256$ weak CN step the NM-ROM `cn` arm
uses, Δt 0.025).

**POD** (classical, uncentred, orthonormal $V_r$ = leading POD modes of the $512\times6=3072$
training snapshots **computed at the target mesh**), $r\in\{8,32,128,256\}$:
* `pod{r}_galerkin_cn`: Galerkin projection of the FOM's own Crank–Nicolson step (Δt 0.025),
  $a_{k+1}=(I+\tfrac{\Delta t\nu}{2}K_r)^{-1}(I-\tfrac{\Delta t\nu}{2}K_r)a_k$, $K_r=V_r^\top L_hV_r$;
* `pod{r}_lspg_cn`: LSPG on the same CN residual, $a_{k+1}=\arg\min_a\|A V_r a - B V_r a_k\|$,
  $A=I+\tfrac{\Delta t\nu}{2}L_h$, $B=I-\tfrac{\Delta t\nu}{2}L_h$;
* initial state $a_0=V_r^\top u_0$ (both), decode $V_r a$ at the six outputs.
The step matrices are assembled offline; online cost is the $O(Nr)$ encode, 20 $r\times r$ steps
and the $O(6Nr)$ decode. How $V_r$ is computed: every training snapshot is separable
($u=\text{amp}\,g_x\otimes g_y$), so the snapshot matrix is $S=(Q_y\otimes Q_x)C$ with $Q_{x,y}$
orthonormal bases of the 1D factors (SVD, truncated at $10^{-14}$ relative) and a small core $C$;
the SVD of $C$ gives the POD of $S$ exactly (no Gram matrix, no $10^{-8}$ Gram floor). The
reconstruction of $S$ from the factors is gated in-job ($\le10^{-12}$ relative) and the
orthonormality of $V_r$ is gated ($\le10^{-10}$).

**Quadratic manifold** (Geelen–Wright–Willcox / Barnett–Farhat), $r\in\{8,16,32\}$:
$u=u_{\rm ref}+V_ra+W\,\mathrm{vech}(aa^\top)$, $u_{\rm ref}$ = training-snapshot mean, $V_r$ =
centred POD of the training snapshots, $W$ = ridge solution
$W=E\Pi^\top(\Pi\Pi^\top+\gamma sI)^{-1}$, $s=\mathrm{tr}(\Pi\Pi^\top)/P$, all in the factor
coordinates above (exact). **γ is chosen by trajectory-split holdout**: the 512 training
trajectories are split 80/20 by trajectory (seed 20260923; all six snapshots of a trajectory on
the same side — never a column split), $u_{\rm ref}$, $V_r$ and $W$ are fitted on the 80 %, and
γ minimises the relative Frobenius reconstruction error of the 20 % over the grid
$\{0,10^{-10},10^{-8},10^{-6},10^{-4},10^{-2},1\}$; the winner is refitted on all 512. Online the
columns $[u_{\rm ref}\mid V_r\mid W]$ are the "bank" and $\eta(a)=[1,a,\mathrm{vech}(aa^\top)]$ the
"head" of the unchanged hires-heat solver (`core.make_stages`): the same $M=256$ weak tests, LM
driver, budgets and output contract as the NM-ROM, so only the trial map differs.
Arms `qm{r}_cn` (moments initial fit, CN Δt 0.025, tol 1e-6 — the NM-ROM `cn` contract) and
`qm{r}_field_direct_tol1e-4_chol` (the NM-ROM batched-fit contract).

**Neural operators** FNO / U-Net / Transolver / DeepONet, trained at the target mesh
($1024^2$ and $2048^2$ only, see §6) with the Burgers panel's published configurations
(`fno-large`, `unet-medium`, `transolver-large`, `deeponet-small`; exp/2026-09-22-ops-tune-grid
@ 52d1b573), batch 8, AdamW, a **3000 s wall budget per arm** (the Burgers budget), on the
NM-ROM's 512 training draws, checkpoint = best validation (seed 791001) mean-case-max error.
Declared deviations: heat contract (no parameter channel, current-relative loss), wall-time cosine
learning rate, gradient accumulation where batch 8 does not fit, Transolver patch $n/64$
(token grid 65², as at Burgers 256²). One configuration per family; no selection across
configurations. Timed query: supplied interior field on the GPU → pad to the nodal grid →
network → mask → six interior fields on the GPU, as in the Burgers panel.

**Full-order candidate grid** (same-grid CN + unpreconditioned warm-started CG, hires-heat
`core.make_cg`): Δt 0.1 × rtol {1e-2, 1e-3}; Δt 0.05 × {1e-2, 1e-3, 1e-4}; Δt 0.025 ×
{1e-2, 1e-3, 1e-4, 1e-6 = NAMED}; Δt 0.0125 × 1e-6; and the tight reference Δt 0.00625 × 1e-8.
**Labelled controls** (printed, never "FOM chosen"): exact DST propagation
(`dst_exact_CONTROL`) and the 64-interval coarse-grid CN–CG + interpolation.

## 4. Timing protocol and the order-effect gate

Known trap (other lane): a very slow arm in a randomised panel slowed the next arm by up to 50 %.
Design: **arm-major blocks.** Each arm is built, warmed on every case (untimed; errors and
audit fields saved), then timed: for each case, 5 repetitions, each preceded by a 0.15 s GPU
burn-in. Fast arms first, then the FOM grid (slowest last), with 1 s idle + burn-in between
blocks. Between every two blocks a **sentinel** (the `dst_exact_CONTROL` query on case 0, 7 reps)
is timed.

Gate **G-order** (must pass): every sentinel block median is within ±10 % of the median of all
sentinel blocks; in particular the sentinels immediately after the FOM blocks. Positive control
(must FAIL the same ±10 % test): a "contaminated sentinel" timed once per FOM block by
dispatching the FOM query without synchronising and then timing the sentinel, so the sentinel
waits for the FOM. If the control does not fail, the gate is not trusted.

## 5. Gates (all must pass before a number is reported)

`jax_backend=gpu` (exit 42 otherwise); f64 everywhere except the f32 operator networks (as in
Burgers); every field finite; POD factor reconstruction ≤1e-12, $V_r$ orthonormality ≤1e-10;
weak-matrix DST identity where affordable (inherited); NM-ROM arms reproduce hires-heat
`h2d-final04` sealed worst errors to 1e-6 relative (0.4876 % for $q=32$, 1.3616 % for $q=0$);
CG failures counted, an arm with failures is not eligible as comparator; independent NumPy/SciPy
audit recomputes every reported error from saved fields (strided sub-grid for every arm and
case, full field for two cases at $1024^2$, a random 100 000-node sample for two cases at every
mesh) against a SciPy reference; the summary and the table are generated only from the job JSON
and the audit JSON; the G-order gate and its positive control.

## 6. Scope limits declared now

* Operators at $4096^2$: not attempted (f64 FNO activations at $4097^2$ with batch 8 and the
  3000 s budget are not feasible in this lane's time); the $4096^2$ table has no operator rows
  and says so.
* No retraining of the NM-ROM; no new NM-ROM arm.
* One seed per operator; no hyperparameter search (the Burgers panel's published configurations).

## 7. Jobs

`tr1024`, `tr2048` (operator training, H200), then `pn1024`, `pn2048`, `pn4096` (panels,
H200, 240 G). At most 2 of this lane's jobs at once.

## A1 (2026-09-23, before any panel job; after the Codex code review `checks/codex-code-review-2026-09-23.md`)

Codex (read-only, unsandboxed because the sandbox cannot start here) confirmed the POD factor
construction, both POD CN step matrices, the quadratic-manifold fit and its use of the solver,
the JAX/PyTorch synchronisation and the operator training hygiene. Dispositions of its findings:

1. *Gates not enforced in reporting* — accepted. `summarize.py` now checks the complete expected
   arm set, the h2d-final04 reproduction, the audit and the recompute agreement, and stamps
   `status: final` only if all pass; otherwise `PROVISIONAL (diagnostic only)`, carried into the report.
2. *Full-grid errors not recomputed exactly* — partly accepted. Full fields cannot be kept
   (0.8 GB per arm-case at 4096²). The random-node sample (50 000 nodes) is now saved for
   **every** case of every arm (was: 2 cases; §5 said 100 000 nodes and full fields at 1024² —
   superseded), so every per-case full-grid error is cross-checked by an independent estimate
   and the table statistics are re-derived from it; sub-grid errors are recomputed exactly; the
   audit is NaN-safe and checks coverage (cases, repetitions, samples, sentinels). The report
   calls this a restricted audit.
3. *Positive control does not use the FOM* — not accepted as stated: a JAX while-loop FOM call
   blocks at dispatch (measured locally: the "FOM-contaminated" sentinel was not contaminated),
   so the control uses queued dense matmuls to prove the sentinel detects lingering GPU work; the
   actual slow-arm carry-over is measured directly by the sentinel after every FOM block (that
   is the gate). Sentinel medians are now recomputed from the raw repetitions in the audit.
4. *Checkpoint provenance* — accepted: the panel asserts checkpoint mesh = panel mesh, family,
   and training/validation seeds from the training job's `provenance.json` (verified to refuse
   a 64-trained checkpoint at 256 and a checkpoint trained on a different training set).
5. *"Lower bound" wording* — accepted: unmatched-accuracy ratios are labelled as such, not as
   bounds; controls get no "FOM chosen"; the named FOM must have no failed solves.
6. *FNO precision* — clarification: as in the Burgers panel, the FNO is float64/complex128 and
   U-Net / Transolver / DeepONet are float32 (§5's "f32 operator networks" meant those three).
7. *Timed outputs not checked* — accepted: every timed repetition's output is compared with the
   audited warm output outside the timed region (≤1e-9 relative for JAX arms, ≤1e-4 for the
   operators' float32 kernels); stored per case.
8. *Ridge normal equations* — accepted: the ridge is solved as an augmented least-squares problem.

Operational: `tr1024` (4196056) was cancelled by me while PENDING (no H200 free) and resubmitted
unchanged as `tr1024a` (4196355), which landed on an A100-PCIE-40GB; FNO epochs there take ~470 s,
so operators get few epochs within the 3000 s budget at both meshes (reported per arm).

## A2 (2026-09-23 ~06:55 EDT, after pn1024a, before pn1024b / pn2048b / pn4096)

`pn1024a` (job 4205399, A100-80G, complete, all arms) FAILED the pre-registered v1 order gate:
max per-block sentinel deviation 36.8 % > 10 %. Diagnosis from its raw data (not a threshold
change after the fact): the sentinel is a ~1.5 ms kernel whose own repetitions scatter
1.0–3.1 ms *within* single blocks, so the per-block median of 7 reps cannot resolve ±10 %; the
sentinels after the 11 full-order blocks are not slower (pooled median ratio post-FOM / rest
= 0.94). The v1 test measures sentinel noise, not carry-over. `pn1024a` therefore stays a
**provisional, diagnostic** run and is not the 1024² table.

Amended gate for every later panel (the v1 statistic is still computed and reported):
(a) **carry-over**: pooled sentinel repetitions (15 per block) after full-order blocks vs after
all other blocks, median ratio within ±10 %; (b) **re-time**: three real arms
(`nmrom_q32_field_direct_tol1e-4_chol`, `pod128_galerkin_cn`, `op_unet` where operators exist)
are rebuilt and timed again on all 16 cases **after** the full-order phase; each re-timed
median must be within ±10 % of its own block median; (c) the positive control must still fail.
The 1024² panel is re-run unchanged otherwise as `pn1024b`. `pn2048` (4203979) was cancelled while
PENDING (it had staged the v1 code) and resubmitted as `pn2048b`. `pn1024` (4203972) had been
cancelled while PENDING because no H200 was free; 1024² runs on an A100-80G (the lane rule
requires H200 only at ≥2048²); every ratio is within one job, but absolute times at 1024² are
A100 times, not the H200 times of the paper's Table 1 row.

## A3 (2026-09-23 ~07:20 EDT): memory fix after pn2048b, before pn2048c / pn4096b

`pn2048b` (job 4206387, H200) died at the quadratic-manifold r=32 set-up with a GPU OOM:
the `B^T B` gate made XLA materialise a transposed 17.5 GB copy of the 561-column bank under
the default 75 % allocator cap (blocks up to `qm16` had completed; no number from it is used).
Fix, no change to any timed code path: bases are built row-block by row-block into one donated
buffer, every offline `V^T X` / `B^T B` product is streamed over row chunks, device memory in
use is logged after every block, and the allocator cap is 0.92 (the job owns the GPU). Verified
locally at 256²: every error identical to the previous smoke to ≤2e-15; memory returns to
baseline after each block (no leak); the compiled encode `B^T u` and decode `c B^T` contain no
transposed copy of the bank. `pn4096` (4207404) was cancelled while PENDING (old code).

## A4 (2026-09-23 ~08:40 EDT): operator memory at 2048², after pn2048c, before pn2048d

`pn2048c` (job 4207537, H200) completed every JAX arm (27 blocks; JAX peak in use 39 GB) and died
at the first operator (FNO) with a PyTorch OOM: JAX's allocator pool had grown to ~129 GB and
never returns memory, leaving PyTorch 2 GB. Its partial results are kept in
`runs/pn2048c/failed/` as a diagnostic and are not used. Fix: at 2048² the operator block is
timed **first** (`operators_first`), while the GPU is still empty, and the JAX pool is capped at
0.80 so the late U-Net re-time has room. Block order otherwise unchanged; the order gate (A2)
covers the reordering. 1024² (`pn1024b`, operators after the quadratic manifold) is unchanged.

## A5 (2026-09-23 ~08:45 EDT): 4096² allocation order, after pn4096b, before pn4096c

`pn4096b` (job 4207540, H200) completed the NM-ROM, linear-bank, POD and quadratic-manifold
r = 8, 16 blocks and died allocating the 70 GiB r = 32 quadratic-manifold bank: the JAX pool was
already fragmented by the earlier 17–34 GB bases, so no contiguous region was left. Partial
results kept in `runs/pn4096b/failed/` (diagnostic, not used). Fix: at 4096² the
quadratic-manifold section runs **first**, largest rank first (r = 32, 16, 8), on a clean pool
(`qm_first`); the factor model is now always built before the NM-ROM section (host-side,
untimed). No timed code path changes; the A2 order gate covers the reordering.

## A6 (2026-09-23 ~09:25 EDT): 4096² quadratic manifold r = 32, after pn4096c, before pn4096d

`pn4096c` (job 4211639, H200) failed 50 s in, compiling the first timed query of the r = 32
quadratic manifold: XLA's GEMM autotuner needs an operand-sized scratch copy of the 70 GiB
(16.8 M × 561, f64) bank for the decode, which cannot fit beside the bank itself. (Not the 2³¹-
element Triton limit: the 34 GB POD-256 basis compiled and ran in pn4096b.) No number produced.
Fix, 4096² only: the r = 32 query is compiled from the **same** function with
`compiler_options={'xla_gpu_autotune_level': 0}` and labelled `…_noautotune`; everything else is
compiled as before. To measure what disabling autotuning costs, the r = 16 quadratic manifold
is timed **both** ways in the same job (`qm16_*` and `qm16_*_noautotune`); the report prints both.

## A7 (2026-09-23 ~11:10 EDT, after all panels): independent results audit and audit tightening

Codex (read-only) recomputed all 132 raw arms of pn1024b / pn2048d / pn4096d from `results.json`
and found **no mismatch** in any printed number, FOM choice or speedup, and confirmed the A2 gate
statistics and the operator-training table (`checks/codex-results-audit-2026-09-23.md`).
Accepted from it, without any new job: the report now states that the audit is *restricted*
(full-grid errors are cross-checked by per-case random samples and exactly recomputed only on the
strided sub-grid), no longer says the operators were "still improving" (the 1024² Transolver's
best epoch is 198 of 227), prints the exact-DST control beside the NM-ROM at every mesh (it is
faster and exact, so every speedup is over CN–CG only), and notes that the r = 16 autotuning
calibration does not bound the r = 32 effect. The audit now also requires every configured
re-time arm, a sentinel after every block with the configured repetition count, the full
unique random-sample size, and finite positive timings; re-run on the saved data of all three
panels: passed, 0 failures (the corrupted-field negative control still fails). Not fixable after
the fact and recorded: `panel.py` accumulated the timed-vs-warm parity with Python `max`, which
would have hidden a NaN; all stored parities are finite and 0.
