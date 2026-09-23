# poisson-bank-knob — design and pre-registration (written before the cluster jobs returned)

**Question.** On the linear Poisson problem the correction weights $y$ are eliminated in closed form, so the
accurate ($q=256$) and fast ($q=0$) settings of the frozen model cost the same. Hypothesis: at high
resolution the query is dominated by the reconstruction $u = Gc$, which costs $O(nR)$ and is the same for every $q$.
If so, a deployment-time **nested bank truncation** $R' < R$ is a real cost knob. No retraining.

**Model.** The frozen p-linear `primary_K32` checkpoint ($K=32$, $R=512$), as used by `hires-poisson` and paper
Table 1. Cohort: the twelve opened development sources (`eval_seed` 7090703 ×6 and `fresh_seed` 7090732 ×6),
the cohort of `hires-poisson`, used unchanged.

## Construction (offline, training data only)

Let $G = Q_G R_G$ at the training mesh (255 intervals). Stack the training samples
$a_i = R_G c_i / \lVert u_i \rVert$, taking both $c_i$ = the bank projection of training field $u_i$ (the $q=R$ limit) and
$c_i = h(z_i)$ at the stored code (the $q=0$ prediction). The fit split only; never the development sources. Take the SVD
$A = U S V_s^\top$ and set $T = R_G^{-1} V_s$ and $L = V_s^\top R_G = T^{-1}$. This is the SVD of $G\Sigma^{1/2}$ (POD of the
training decoded fields inside $\mathrm{span}(G)$).

At truncation $R'$: the decode is $u = G T_{:,1:R'}\,(L_{1:R',:}\,c)$ and the weak operator is $B P_{R'}$ with
$P_{R'} = T_{:,1:R'} L_{1:R',:}$. The head, the corrections $C_q$ (the first $q$ nested directions), the nearest-code
start and the LM kernel are the parent's, unchanged. The ladder is $q \in \{0, \min(256, R'-K)\}$, plus the
$q=R'$ linear rung (a thin-QR solve in the truncated rotated bank). At $R'=R$ the truncation is the identity, so $P = I$
exactly. The rotated bank is stored as row chunks × nested column blocks at the $R'$ edges, so each decode reads
exactly $R'$ columns. The rotation is pinned: `runs/prep.npz`, built locally on the GB10.

## Gates (every one must pass for the job's numbers to be used)

- **Parity:** at $R'=R$, the rotated arms against the unrotated parent `hp_core.make_lean` fields, worst relative
  $\le 10^{-10}$ (1024², 2048²; the 4096² original bank does not fit beside the rotated one). **Recorded before
  the fix:** with $P = TL$ instead of $I$ at $R'=R$, $q=256$ differed by $5.9\times10^{-10}$, because
  $\mathrm{cond}(R_G) = 2.6\times10^{7}$ and $\lVert LT-I\rVert = 1.1\times10^{-9}$. Below $R$ this error sits many orders
  under the reported errors.
- **Determinism:** every repetition produces a byte-identical field.
- **CG converged:** every CG call.
- **Neighbour gate:** each ROM arm is re-timed immediately after a CG $10^{-3}$ solve on cases 0–2. Its median must
  be $\le 1.10\times$ its main-phase median on the same cases. **Known:** this gate is noise on the shared GB10
  (smoke ratios 0.07–2.5 at 1 sample), so only the H200 value counts.
- **Profile consistency:** the stage-split fields equal the fused fields to $10^{-10}$.
- **Independent NumPy audit** (`pbk_audit_np.py`, run on the cluster before fields are deleted): it recomputes every
  error against a dense-sine-matrix DST-I truth, to relative $10^{-8}$, and re-hashes every field. Two controls must be
  detected: truth from a swapped case, and a perturbed recorded error.

## Timing contract

GPU-query time = `fused_device_seconds` (the device work of one query, between the synchronised host→device source
copy and the device→host field copy), the scope of paper Table 1. Complete-query `total_seconds` is also recorded.
All arms are randomised within each case, with 0.1 s GPU burn-in before every invocation. There are 5 retained
repetitions × 12 sources. The CG tolerances $10^{-3}$ and $10^{-4}$ run in their own phase, with 3 repetitions. One
H200 per job, UUID-guarded. 1024² and 2048² run sequentially in one allocation; 4096² runs in another.

## Verdict rule (fixed now)

- **Profile (step 0):** the hypothesis holds at a mesh if the reconstruction stage is the largest stage of the
  accurate ($q=256$) and fast ($q=0$) queries.
- **Reference arm:** the "model's most accurate arm" is the lowest worst-error ROM arm of any family in the job (expected: the $R'=512$ linear rung at the bank floor).
- **Knob (step 5):** "yes" requires both of the following, from the one frozen model:
  - the worst error is monotone non-decreasing as $R'$ decreases along the chosen family (NM-ROM $q_{\max}$
    arms; the linear rung separately);
  - the GPU-query time falls by $\ge 2\times$ at 4096² between the most accurate and the cheapest arm of the family.

  Speedups use the fastest tested CG whose worst same-grid error is $\le$ the worst error of the model's most accurate
  arm, from the same job. They are also given against the paper's named `cg_0.01`.

## Amendment A1 (2026-09-23, before any cluster result) — Table-1 scope and settings rule

The coordinator relayed a scope change from the user: $R'$ replaces $q$ as the paper's tunability knob, and the
Table 1 rows are regenerated from the frozen models. Changes, all made before any job ran:

- The first submissions, 4196680 (`pbkA`) and 4196682 (`pbkB`), were cancelled by me while still PENDING and never
  started. Their remote directories were deleted and the attempt names are retired.
- Meshes are now **256², 1024², 2048², 4096²**. Job `pbkC` runs 256², 1024² and 2048², each as its own driver run with
  its own full arm and CG set, in one allocation. Job `pbkD` runs 4096².
- The CG loose grid is widened to rtol $\{0.7, 0.5, 0.4, 0.3, 0.2, 0.1, 0.03, 0.01\}$, so that a CG setting can
  match the cheaper, less accurate arms. The tight phase is $\{10^{-3}, 10^{-4}\}$.
- **Pre-registered Table-1 setting rule** (fixed by the coordinator before results):
  - **accurate** = the most accurate arm at that mesh;
  - **fast** = the cheapest arm whose worst error $\le$ that of the current paper fast setting ($q=0$, $R=512$,
    arm `R512_q0`) at that mesh;
  - **Table-1 speedup** = (the fastest tested CG with worst error $\le$ the accurate arm's) / (arm GPU-query time),
    from the same job; one FOM per row.

  Every arm is also compared with the fastest CG at least as accurate as itself. Parent (unrotated) arms are parity
  baselines and are excluded from the selection, because they duplicate `R512_*`.

## Amendment A2 (2026-09-23, before any result) — pbkC moved to A100

`pbkC` (job 4197114) was cancelled by me while PENDING (Slurm start estimate 08:04, waiting for an H200). Its remote
directory was deleted and the name retired. The same three meshes run as `pbkE` on an **A100**: the 2048² original
plus rotated banks are about 34 GB. 4096² (`pbkD`, job 4197115) stays on an H200, because its 69 GB rotated bank does
not fit an 80 GB card. All speedups are within-job ratios, so the GPU-type difference between the 4096² row and the
other rows affects only absolute milliseconds, which are labelled with their GPU.

## Amendment A3 (2026-09-23, before any 4096² result) — 4096² on any card of at least 80 GB

`pbkD` (job 4197115, H200, Slurm start estimate 08:04) was cancelled by me while PENDING and its remote directory
deleted. The earlier "must be an H200" reasoning was a units error on my part: an 80 GB A100/H100 has 81920 MiB =
85.9e9 bytes, and the rotated f64 4096² bank is 68.7e9 bytes. Nothing else large is kept at 4096² (no original bank,
no f32 copy), which leaves about 13e9 bytes of headroom at memory fraction 0.95. The job is restaged as `pbkF` with
`--constraint="a100-80G|h100-80G|h200-141G"`. Separately, `pbkE` landed on an **A100-PCIE-40GB**. Its 2048² pass
keeps both banks (34.4e9 bytes) against a 38.6e9-byte XLA cap. That pass is at risk; if it runs out of memory it is
rerun without the original bank, with the parity then carried by 256² and 1024².
