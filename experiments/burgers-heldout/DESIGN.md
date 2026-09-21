# burgers-heldout — DESIGN (pre-registered 2026-09-21, before any GPU job)

Lane of the 2026-09-20 speed-and-accuracy campaign; binding contract
`reports/2026-09-20-speed-accuracy-campaign-protocol.md` (on main). Worktree
`worktrees/2026-09-21-burgers-heldout`, branch `exp/2026-09-21-burgers-heldout`, forked from
`exp/2026-09-20-hires-burgers` @ `0ab60014`. Cluster namespace
`/cluster/tufts/paralab/tawal01/bheld_20260921/`, one directory per job. Budget: at most 2 running
and 8 total GPU jobs for this lane; wait if the account already has 6 running.

## 1. Question

The frozen Burgers 2D NM-ROM (`sep_hfit_dense_mid_N256_dense.pkl`: bank $R=512$, head $K=16$,
directions `directions_qtd02.npz`, lattice rule `lat64`) reaches 0.60 % worst evolved error on the six
development cases (dev6) at $4096^2$ but **1.33 %** on the 64 held-out cases hold64 =
`params_draw(20260916, 64)`, at 5.38× the fastest Newton–BiCGStab setting at least as accurate
(hires-burgers hb4kh64). The bank-floor lane showed the held-out projection floor of that bank is the
binding term (hold64 all-times floor 2.24 % at $256^2$; its full-bank solve is 1.70 % on an 8-case
confirmation cohort), and that an optimal rank-512 basis of the same data has a held-out floor of
0.38 %.

**Can a new frozen model, built from a better learned (mesh-free) bank at the SAME rank $R=512$, bring
the hold64 worst evolved error at $4096^2$ to $\le 1\,\%$ while keeping the speedup $\ge 5\times$
against the named Newton–BiCGStab FOM?**

## 2. Bar (copied from the lane brief and the protocol; not adjustable)

- Error metric (inherited unchanged from hires-burgers): $\epsilon_k=\lVert u_k-u^{\rm tight}_k\rVert_2/\lVert u_0\rVert_2$
  on the full $L$-grid, $u^{\rm tight}$ = same-job `fft_tight`. Evolved = $\max_{k\ge1}$ over the six
  output times; worst = max over the cohort. All-times printed beside it.
- **Pass:** at $4096^2$, the accurate setting's hold64 worst evolved error $\le 1\,\%$ **and**
  $S=T_{\rm FOM}/T_{\rm ROM}\ge 5$ where $T_{\rm FOM}$ is the fastest tested same-mesh Newton–BiCGStab
  setting in the same allocation whose hold64 worst evolved error is $\le$ the ROM's (the paper's
  FOM-selection rule). GPU-query scope is the headline; host-inclusive printed beside it.
- The fast setting ($q=0$) is reported beside it. Speedups against the tight FOM and the coarse-grid
  FOM (labelled control, excluded from the paper) are printed too.

## 3. The new model (what changes, and what does not)

Nothing about the ROM's equations, solver, output contract or timing contract changes. Only the
frozen spatial bank $G$, and consequently the head $h_\theta$, the codes and the directions $C$, are
rebuilt — each with the incumbent's own committed recipe.

1. **Bank `cpod512` (mesh-free, rank 512).** Start from bank-floor's learned `cat1024`
   (`bfb03/burgers2d_cat1024.pkl`, sha256 `a722b29e…`: two coordinate-network blocks of 512 columns;
   trained on the first 1024 trajectories of the incumbent draw). Compress its span to rank 512 by a
   field-metric POD of the $256^2$ training states projected onto it:
   $c_i=\arg\min_c\lVert G_{\rm cat}c-u_i\rVert$, whitened $a_i=L^\top c_i$ with $LL^\top=G_{\rm cat}^\top G_{\rm cat}$,
   top-512 right singular vectors $W$ of the weighted matrix $[w_i a_i]$, and
   $G'=G_{\rm cat}\,V$, $V=s\,L^{-\top}W$ ($s$ matches the incumbent bank's total Gram trace;
   individual coefficient scales are NOT preserved — see §9 A-4). $G'(x)$ is a fixed linear combination of mesh-free
   coordinate features, so it evaluates at any mesh exactly like the incumbent. It is written as ONE
   separable-decoder parameter set (block-diagonal MLP, last layer folded with $V$), so every existing
   module (`sep_common`, `arms.CoordBank`, `hops`, `hfast`) runs unchanged. **Parity gate:** merged
   features vs concatenated block features times $V$ on 4096 random points, relative $\le10^{-12}$.
   Training states = exactly the head's training states (the incumbent extraction: 4608
   trajectories, seeds 0 and 1000, 131072 picked states). Weight variants $w_i$: `raw` (1),
   `state` ($1/\lVert u_i\rVert$), `u0` ($1/\lVert u_{0,j(i)}\rVert$, the error metric's
   normalisation). **Selection (pre-registered):** the variant with the smallest worst evolved floor,
   in the $\epsilon$ metric, on the selection cohort sel32 at $256^2$. Floors of the rank ladder
   $R'\in\{256,384,512,640,768,1024\}$ and of the incumbent bank are reported for context only.
2. **Head.** `sep_coeff_extract.py` + `sep_hfit_run.py`, byte-identical to the incumbent's job
   (`run_dn256b.sbatch`: N=256 K=16 MAX_SNAPS=131072 T_EARLY=5 SEED0=0 LOOSE=1 EXTRA_SEED=1000
   EXTRA_TRAJ=4032; ARMS=mid STEPS=200000 BATCH=4096 LR=1e-3 TIME_CAP=1500 SEED0=0 EMIT=mid), with
   CKPT = the `cpod512` bank. The `mid` arm trains head AND codes from scratch, so no incumbent
   parameter leaks in. The same data-generation truth gate ($\le10^{-8}$) and Gram-identity gate apply.
   Fidelity reference: the head's fresh-test oracle is compared with the incumbent's own run
   (`hfit_dn256b_full.json`, same eight TEST_SEED trajectories).
3. **Directions $C$.** `cheap-corrections/directions.py::audited` (q-trajdirs `qtd02`'s rule "field-metric
   POD of $\eta-h_\theta(z^\star)$"), with qtd02's configuration: 128 trajectories of
   `params_draw(0,128)` at $256^2$, stride 2, 1024 residual snapshots, 4 starts, budget 200, seed
   20260915.
4. **Quadrature.** The uniform lattice `lat64` (no fit, bank-agnostic), certified at each mesh by
   held-out $\rho\le0.116$ on the dense-query population AND on the deployed arm's own states
   (hires-burgers §4, unchanged). The incumbent's $q=0$ node rule is bank-specific, so $q=0$ uses
   `lat64` too. **Control that must fail:** `lat16` (15×15 nodes) at $q=256$, $M=1088$ must fail the
   certificate; if it passes, the certificate is not discriminating and that is reported.

Memory and cost: $R=512$ keeps the $4096^2$ bank at 64 GiB f64 and every per-iteration cost identical
in shape to the incumbent; only the (untimed) feature evaluation is wider. No precision change.

## 4. Cohorts and their roles

| name | definition | role |
|---|---|---|
| training | 4608 trajectories (`sample_params(0,576)`+`(1000,4032)`), `params_draw(0,128)` | bank POD, head, directions, rule populations |
| dev6 | `params_draw(7090702,4)`+`params_draw(911702,2)` | opened development; selection |
| sel32 | `params_draw(20260927,32)` (new, this lane) | selection only |
| hold64 | `params_draw(20260916,64)` | the brief's target; evaluated once with frozen settings, never selects in this lane — but OPENED before this lane (see §9, A-1) |
| fresh64 | `params_draw(20260929,64)` (new) | untouched confirmation cohort, evaluated once with the same frozen settings |

Disjointness of sel32 from training, dev6, hold64, Table-2 validation-32 and the sealed cohort is
checked before the first job (`checks/cohort_disjoint.py`, same tests as
`reports/checks/2026-09-21-burgers-train-eval-overlap.py`). The bank-floor lane already reported
hold64 floors of `cat1024`; this lane never computes anything on hold64 before job 4.

## 5. Jobs (≤ 8; planned 4)

| job | mesh | GPU | what |
|---|---|---|---|
| bh1 build | $256^2$ | A100/H100 | extraction on `cat1024` (R=1024) → POD variants → floors on dev6/sel32 → select → extraction on `cpod512` → head fit → directions |
| bh2 select | $1024^2$ | A100/H100 | dev6 + sel32 (38 cases), rung ladder, certificates, floors at the evaluation mesh, 1 rep |
| bh3 dev6+hold64 | $4096^2$ | H200, 320G | frozen arms, dev6 (6) + hold64 (64) in one allocation, 5 reps, paired FOM grid, profile |
| bh4 fresh64 | $4096^2$ | H200, 320G | the same frozen arms on fresh64, 5 reps |

bh3 and bh4 run concurrently (they read nothing from each other). bh2 runs 3 timed repetitions so
its cost ranking is not a single sample.

**Rung ladder (bh2):** $q=0$ ($M=64$); $q=128$ ($M=576$); $q=256$ ($M\in\{544,1088,2176\}$);
$q=384$ ($M=1600$); variant `chol+clip+lamcarry+pred2` (hires-burgers' kept speed recipe), gtol
$\{10^{-3},10^{-2}\}$; control `lat16` at $q=256,M=1088$.

**Selection rule (fixed now, applied on bh2's dev6 ∪ sel32 only; amended per audit finding 9):**
candidates are complete arms (q, M, gtol, variant) that are certified (both populations) and not
controls. The accurate setting is the candidate with the smallest bh2 median GPU time among those
whose OWN worst evolved error over dev6 ∪ sel32 is $\le 0.8\,\%$ (margin for unseen cases); ties
→ smaller q, then smaller M, then tighter gtol. If none qualifies, the candidate with the smallest
worst error (ties → smaller median time). bh3/bh4 run that arm as the headline,
plus $q=0$ (fast) and the neighbouring rungs as a ladder. The headline arm is named in a DESIGN
amendment before bh3/bh4 are submitted; hold64 numbers are then reported whatever they are.

**FOM grid at $4096^2$ (same as hb4k04, no omission):** `fft_tight` (reference, untimed on hold64),
`lean_tight`, `lean_nt1e-3_l1e-3_dt005`, `lean_nt3e-3_l3e-3_dt005`, `lean_nt1e-3_l1e-3_dt01`,
`lean_nt1e-2_l1e-2_dt005`; coarse-grid controls `c2048_nt1e-4_dt005`, `c1024_nt1e-4_dt005`.

## 6. Added measurement: the bank floor at the evaluation mesh

`bh_hires.py` = hires-burgers' `hires.py` plus one phase: for every case and output time, the
projection floor $f_k=\lVert u^{\rm tight}_k-P_Gu^{\rm tight}_k\rVert/\lVert u_0\rVert$ onto the
bank's span at the evaluation mesh (Cholesky of the blocked Gram with one refinement step; residual
formed through $G$). It is a lower bound on every corrected ROM's error ($u_{\rm ROM}\in{\rm span}\,G$);
the audit checks $\epsilon_k\ge f_k-10^{-9}$ for every ROM row. Nothing else in `hires.py` changes.

## 7. Controls and gates (each must be able to fail)

- merged-bank parity $\le10^{-12}$; Gram identity (extraction, unchanged); FOM truth $\le10^{-8}$.
- `lat16` control fails the $\rho$ certificate; ROM error $\ge$ floor everywhere.
- The incumbent bank's floor on sel32 is computed in bh1: the new bank must be below it, or the
  premise is wrong and the lane stops.
- hires-burgers' audit (`audit_hires.py`, NumPy recomputation from saved fields, full-grid case-0
  recheck) on every evaluation job; failed gates reported, thresholds not loosened.

## 8. Stop rules

- bh1: if the selected `cpod512` worst evolved floor on sel32 is not below 0.6 %, the rank-512 route
  cannot deliver ≤ 1 % with margin; stop, report, and propose (not run) a wider-bank arm.
- bh2: if no certified rung is $\le 1\,\%$ on dev6 ∪ sel32, do not spend H200 jobs; report.
- Any failed parity/identity/truth gate stops the chain until explained.

## 9. Independent audit and dispositions (Codex, 2026-09-21, before any job)

Record: `audits/codex-design-audit-2026-09-21.md` (files inlined; Codex cannot run a shell here).

- **A-1 (blocker) hold64 is not an untouched test set.** Accepted as a fact and stated: the incumbent's
  hold64 error, and bank-floor's hold64 floors (which rank `cat1024` among learned banks), were seen
  before this lane. This lane selects nothing on hold64, but the choice of `cat1024` as the source bank
  was informed by hold64 floors in an earlier lane. Fix: a new cohort **fresh64**
  (`params_draw(20260929,64)`, disjointness checked) is evaluated once in bh4 with the same frozen
  settings; hold64 is reported as the brief's target and labelled "opened"; fresh64 is the untouched
  confirmation. Both are reported whatever they show.
- **A-2 POD orientation.** Snapshots are ROWS ($A_{i,:}=w_ia_i^\top$, $A=CL$); right singular vectors
  are correct; the weights multiply rows, so the objective is $\sum_i w_i^2\lVert\cdot\rVert^2$
  (`state` = per-state relative, `u0` = the error metric). The compressed bank's Gram is asserted to be
  $s^2I$ at the fit grid ($\le10^{-8}$).
- **A-3 fold.** Implemented as specified (sin/cos row permutation of the first layer, per-block
  out_scale and biases folded, merged out_scale 1, `bc_poly` applied once by `features`). The parity
  reference evaluates each block independently; the parity points now include random interior points,
  the fit-grid nodes and boundary points.
- **A-4 scale wording.** Corrected: the total Gram trace is matched, not individual coefficient
  scales; $s$ and both spectra are recorded.
- **A-5 floor numerics.** Floors at $256^2$ use a twice-orthonormalised thin QR. At the evaluation mesh
  the blocked-Gram Cholesky with one refinement is used and its residual orthogonality
  $\max_j|g_j^\top r|/(\lVert g_j\rVert\lVert r\rVert)$ is recorded; the floor-vs-error check is a
  consistency check, stated as such.
- **A-6 acceptance vs completion.** The lane's verdict is computed from the audited summary with
  explicit acceptance conditions (backend gpu, f64/highest, cohort hashes, model/directions hashes,
  ≥5 retained reps everywhere, repetition-identical outputs, parity and control outcomes); a job
  that completes with a failed condition is reported as such. `required_reps` is 5 in bh3/bh4.
- **A-7 comparator.** $S$ = (median GPU ms of the FOM setting over all its cohort invocations) /
  (median GPU ms of the ROM arm over all its cohort invocations), the hires-burgers convention, same
  allocation; the FOM setting is the single fastest (by that median) whose cohort-worst evolved error
  is ≤ the ROM arm's cohort-worst; host-inclusive printed beside it. `lean_tight` is timed (it is the
  tight discretisation without the extra diagnostic Jacobian product); `fft_tight` is the untimed
  reference. Coarse-grid FOMs are run and reported as protocol controls; the paper excludes them by
  the user's decision.
- **A-8 refined reference.** Not run on hold64/fresh64. All errors are same-grid errors against
  `fft_tight`, labelled so; the coarse-FOM comparison on those cohorts is labelled incomplete.
- **A-9 selection.** Rule rewritten above (complete arms, own error ≤ 0.8 %, measured cost).
- **A-10 certificate population.** Inherited from hires-burgers; it is a *within-training-distribution*
  held-out certificate (`params_draw(0,128)` trajectories 8–15 are not used by the rule but are
  training trajectories of the directions), stated so.
- **A-11 full-field audit.** Besides case 0, each audit arm's WORST case (by in-job error) and the
  matching `fft_tight` truth are saved at full resolution, so the headline number is recomputed from
  full fields in NumPy.
- **A-12 cost.** Setup (bank build, tables) is reported separately and excluded, as for the incumbent.
- **A-13/14** wording accepted: `lat16` is a plausible negative control, and the bh1 stop rule is a
  budget/margin stop, not an impossibility result.
- **A-15 host memory.** 70 cases × six 4097² f64 fields of truth ≈ 56 GB; `prune_each_arm` keeps one
  arm's fields at a time; `--mem 320G`.

## Amendment A2 (2026-09-21 ~19:10 EDT, after bh2, before bh2b/bh2c are submitted)

**bh2 outcome (job 4142941, audited `checks/bh2-summary.json`).** New model at $1024^2$, worst evolved over
dev6 ∪ sel32 (floor 0.259 %): q256/M1088 1.73 % (dev6 0.80 %), q256/M2176 1.32 % (uncertified, ρ 0.166),
q384/M1600 0.93 % (uncertified, ρ 0.128), q128 2.26 %, q0 4.62 %. The lat16 control fails the certificate
(ρ 1.99) and regresses (20.4 %) as required. **Pre-registered result: no certified arm ≤ 0.8 %; the rule picks
q256/M1088 g1e-2 (1.73 %), and the §8 stop rule (no certified rung ≤ 1 %) forbids spending H200 jobs on it.**
Error is 3–7× the bank floor and falls with the test count M (1.73 → 1.32 % from M=1088 to 2176) and with
q, so the binding term is now the weak-residual test space / quadrature and the correction rank, not the bank.

**A2 (new arms, same cohorts, same selection rule, same stop rule):**
- **bh2b** (new model, `config-1024-bh2b.json`): lattice `lat128` (127² nodes, no fit) so larger M and q can
  certify: q256/M1088, q256/M2176, q384/M1600, q448/M1856, q512/M2112 (q = R: every bank direction), plus
  q256/M1088 lat64 (continuity) and q0; gtol $10^{-2}$ only (bh2: within 0.3 % relative of $10^{-3}$ on every
  rung); a dense exact-residual twin of q384/M1600 on dev6 case 0 and the three hardest sel32 cases
  (global indices 11, 25, 32) to separate quadrature from subspace error. 3 reps. Selection by the §5 rule on
  bh2b's arms; bh3/bh4 are submitted only if the selected arm is ≤ 1 % on dev6 ∪ sel32 (stop rule unchanged).
- **bh2c** (attribution control, `config-1024-bh2c.json`): the INCUMBENT model (inc512, its head, qtd02
  directions) on the same cohorts and rungs (lat64/lat128). Not a candidate; it answers whether the new bank
  changes the corrected error at all. 1 rep.

## Amendment A3 (2026-09-21 ~21:00 EDT, after bh2b, before bh3/bh4 are submitted)

**bh2b outcome (job 4144023, `checks/bh2b-summary.json`), dev6 ∪ sel32 at $1024^2$:** every `lat128` arm FAILS
the ρ certificate (ρ_max 0.37–0.63, set by single outlier states; p95 ≤ 0.005, lower than lat64's), so the
certified set is still {q0, q256/M1088 lat64} and the pre-registered rule again picks q256/M1088 lat64 (1.73 %).
The uncertified arms are accurate — q384/M1600 0.91 %, q448/M1856 0.65 %, **q512/M2112 0.41 %** — but slow:
302–635 ms at $1024^2$ against 109 ms for q256/lat64 and 70 ms for the relaxed FOM (lat128 has 16129 nodes).
The dense exact-residual twin of q384/M1600 reproduces its EQ error on the hard cases (0.906/0.912 % vs
0.914 %), so the quadrature is not what limits accuracy; the certificate failure is an outlier-state property
of the ρ metric, reported, not waived.

**Verdict under the pre-registered rules: the bar cannot be met** (the only certified candidate is 1.73 % on the
selection cohorts; the stop rule forbids H200 jobs for it). **A3 (labelled exploratory, nothing selected on
held-out data):** bh3 (dev6 + hold64) and bh4 (fresh64) at $4096^2$ measure, in one allocation each with the
paired FOM grid and 5 reps: the pre-registered headline q256/M1088 lat64 (certified), the fast q0, and the
uncertified accurate arms q448/M1856 and q512/M2112 lat128, **labelled uncertified**. The "exploratory accurate
arm" is fixed now as q512/M2112 lat128 (most accurate on dev6 ∪ sel32 in bh2b), before any held-out number.
Their purpose is to state, for the paper, what held-out accuracy costs at $4096^2$; no bar verdict is claimed
from an uncertified arm.
