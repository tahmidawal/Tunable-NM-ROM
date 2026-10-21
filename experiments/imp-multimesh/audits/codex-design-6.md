**Not CLEAN. A5 closes the selection-accounting and NNLS-memory findings, but compilation remains blocking.**

Read-only audit: no files changed, jobs submitted, scratch files created, or writes to main or `LAB-LOG.md`. GPU controls were inspected, not executed.

1. **Compilation — WRONG as a complete deployment-cost measurement.**

   [A5.1](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/DESIGN.md:602) improves the measurement boundary: a fresh process, persistent caching disabled, and the first compilation of one deployed query. The queries constructed by [mmcompile.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/mmcompile.py:26) use the appropriate recording path for ours and ECSW query for competitors.

   **The remaining defect is extrapolating those measurements to other executables.** Four probes per mesh do not cover the registered configuration space:

   - R256 has no fresh-process measurement; it is equally near R128 and R384.
   - Competitor rank and actual rule-support size change query argument shapes. An $r=128,\kappa=4$ compile does not measure $r=128,\kappa=8$, nor either $r=256$ configuration.
   - Competitor trust radii are captured by the query closure and vary with the basis. Sharing mesh, side and rank does not establish executable equivalence across CP variants.

   BE versus CNR alone is **not** the problem: `p1core` passes scheme coefficients through traced schedule arguments. Deduplication is reasonable where executable equivalence is established.

   Taking
   $$
   \max(t_{\mathrm{in\ process}},t_{\mathrm{nearest\ rank}})
   $$
   neither measures nor bounds the missing executable’s compilation cost. If the first term is retrieval time, the second remains a different query’s cost. Overcharging ours can manufacture MATCH; overcharging competitors can manufacture NO-MATCH. Compile-zero—and the inherited compile-three sensitivity—does not bound this error.

   **Required:** measure each distinct deployed executable in a fresh process, permitting deduplication only with demonstrated equivalence. Alternatively, explicitly label substituted costs as projections and withhold unconditional deployment-cost verdicts that depend on them.

2. **A_sel — CORRECT at the design level.**

   [A5.2](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/DESIGN.md:610) now specifies the complete selection-work union and subtraction of items already charged in A_dep. It includes arm preparation, rejected construction work and codes, qualification, and compilation inside qualification invocations.

   The stated `run_config` wall boundary covers more than the inherited query-only timer: it encompasses production, tight/tighter calls, certification and scoring. This closes the previous coverage finding.

   The subtraction must be implemented as written: deployment artifacts already included inside those aggregate timers must not survive again in the incremental selection charge. Merely refraining from adding the separate compile probe would not perform that subtraction. That is an implementation obligation, not a missing design rule now.

3. **NNLS memory — CORRECT.**

   [A5.3](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/imp-multimesh/DESIGN.md:619) withdraws the unsupported allocation certificate. The tiled pilot supplies empirical evidence about its executed path; it no longer purports to bound every refactorization/deletion path.

   Memory failures remain UNRESOLVED, with a budget-dependent retry. That preserves the decision semantics without pretending that feasibility has been established. An unresolved alternative does not invalidate an independently measured MATCH witness.

The requested cross-checks remain **CORRECT**:

- **Phase-wise bank fit.** Computed from the specified [A100 log](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-20-jcp-consolidated/experiments/separable-decoder/runs/push_r3a/logs/2835788.out:16):
  $$
  \rho_{\rm sub}=\frac{409-144}{10000}=0.0265,\qquad
  \rho_{\rm full}=\frac{10681-9190}{5000}=0.2982,
  $$
  $$
  \tau_{\rm tr}=9190-7683-5000(0.2982)=16\ \mathrm{s},
  $$
  $$
  T_{\rm fit}=12+289999(0.0265)+16+10000(0.2982)+4
  =10698.9735\ \mathrm{s}.
  $$
  This is **0.1340568% above 10,684.65 s**, passing the registered 5% check. The transition residual is empirical overhead, not an isolated compilation measurement.

- **Coefficient transformations.** Orthonormal DST-I prolongation requires $c_L=(L/s)c_s$. Restriction followed by $\widetilde\Phi_L=QR$ requires $c_L=(L/s)Rc_s$, representing the low-pass projected field. Independent small NumPy checks gave maximum discrepancies of **$7.22\times10^{-16}$** at common prolongation nodes and **$2.78\times10^{-17}$** for restriction reconstruction. The draft `transform_codes` agrees.

- **LSPG dependencies.** It depends indirectly on **both `h` and `Z_tr`**: [`trust_linear`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/jcp-p1-scaling/vendor/quad2d/qcore.py:152) evaluates the head on stored codes, then supplies the resulting trust radius through `Bank.cold_trust`.

- **`q2_cases='all38'`.** Supported: [`p1acc.py`](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-21-imp-multimesh/experiments/jcp-p1-scaling/p1acc.py:129) reads it per configuration through `P.select`, which implements `all38`. Accuracy populations and successor/check rules must also be supplied as registered.

| Item | Sixth-round verdict |
|---|---|
| A5.1 compilation coverage and charging | **WRONG** |
| Fresh-process measurement of the specific probed query | **CORRECT in approach**, pending execution |
| A5.2 selection-work accounting | **CORRECT** |
| A5.3 empirical memory feasibility and failure handling | **CORRECT** |
| Bank-fit formula | **CORRECT** |
| Coefficient transformations | **CORRECT** |
| Head/`Z_tr` dependency and `all38` support | **CORRECT** |

Remaining required change:

- Replace nearest-rank compilation substitution with verified measurements covering each distinct deployed executable, or make affected cost verdicts explicitly conditional on the compilation-cost projection.