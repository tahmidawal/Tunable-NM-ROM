# poisson-bank-knob-3d — design and pre-registration

This file was written before any cluster job of this lane was submitted. It covers the L-shaped
Poisson model and the three-dimensional Poisson model. The sibling lane
`exp/2026-09-23-poisson-bank-knob` handles the square 2D model; its construction is reused here unchanged.

**Question.** Is the nested bank truncation $R'$ a real accuracy/cost knob on these two frozen models, with
no retraining? The answer supplies the numbers for the Table 1 rows: L-shape at $256^2$, $512^2$, $1024^2$
and $2048^2$; cube at $32^3$, $64^3$, $128^3$ and $256^3$.

## Models (frozen)

| problem | checkpoint | $K$ | $R$ | weak tests $M$ | training states used for the rotation |
|---|---|---:|---:|---:|---|
| L-shape | `hires-poisson/lshape/head_sdf_R512_K16` (sha `d43082e2…`) | 16 | 512 | 257 L-shape eigenmodes | fit split, 2611 of 3072, at 256 intervals |
| cube | `paper-p3d/runs/final08/checkpoints/{bank,head_K16}.pkl` (sha `6c229877…`, `258d2845…`) | 16 | 128 | 512 sine modes | all 512, at $32^3$ |

## Construction (offline, training data only)

The sibling's rotation is used unchanged. Let $G = Q_G R_G$ at the training mesh. The rows are
$a_i = [Q_G^\top u_i;\ R_G h(z_i)] / \lVert u_i \rVert$. Take the SVD $A = U S V_s^\top$ and set
$T = R_G^{-1} V_s$ and $L = V_s^\top R_G$. A query at truncation $R'$ decodes $u = (GT)_{:,1:R'}\,(L_{1:R'} c)$.
Its residual uses $B P_{R'}$ with $P_{R'} = T_{:,1:R'} L_{1:R'}$, and $P = I$ exactly at $R'=R$. The rotations
are built on the local GB10 by `pbk3_prep.py` and pinned as `runs/prep_{lshape,cube}.npz`.

## Arms (every mesh)

- **NM-ROM** at each $R'$: $q \in$ the parent's ladder $\cap\ [0, R'-K]$. For the L-shape the ladder is
  $\{0,32,64,128\}$, and $K+q < M$ is also required. For the cube it is $\{0,32,96\}$.
- **Linear rung** $q = R'$ (head dropped; thin QR of $B T_{:,1:R'}$; decode from the nested blocks). It needs
  $M > R'$. For the L-shape ($M=257$) it is therefore **not constructible at $R' \in \{384, 512\}$**. Recorded
  in the result as `not_constructible`. Computing more L-shape eigenmodes was ruled out: 1024 modes at
  $256^2$ took over 120 s locally, against 11 s for 257; the parent needed 816 s for 257 modes at $2048^2$.
- **Parent** unrotated arms at full $R$ (L-shape $q \in \{0,64,128\}$; cube $q \in \{0,96\}$) for the parity
  gate. On the cube at $n \le 64$, the unchanged `poisson.engine` is also included: its dense projection is
  the variant behind the Table-1 rows at $32^3$ and $64^3$.
- **CG** grid, in the same allocation. L-shape: rtol $\{0.3,0.2,0.1,0.03,0.01,0.003\}$ in the main phase,
  $\{10^{-3},10^{-4}\}$ in the slow phase; matrix-free GPU CG (`lsh_core.make_gpu_cg`). Cube: rtol
  $\{0.3,0.1,0.03,0.01,0.003\}$ main, $\{10^{-3},10^{-4}\}$ slow; `iterative_cg.engine(retain_history=False)`, the
  efficient CG of the accepted rows.

## Cohorts

- L-shape: the 32 development sources of `hpl32` / the lshape lane (seed 20260917) at every mesh.
- Cube $128^3$, $256^3$: paper-p3d validation cohort (seed 920411, all 16; the earlier hires rows used the
  first 12), development.
- Cube $32^3$, $64^3$: the same development cohort first. The settings chosen there under the rule below are
  written to `frozen-N{32,64}.json` and committed. Only then is the reserved final cohort (seed 920499,
  64 cases) evaluated, once per mesh, in a separate job. The final job runs the whole ladder, but its row
  settings are the frozen development choices; the final numbers never select an arm.

## Setting rule (fixed by the lane brief, applied by `make_tables.py`)

- **accurate** = the arm with the lowest worst same-grid error at that mesh (all ROM families).
- **fast** = the cheapest arm whose worst error is $\le$ the worst error of the current paper fast setting
  ($q=0$, full $R$: `orig_q0`) at that mesh, in the same job.
- **Table-1 speedup** = (median time of the fastest tested CG whose worst error is $\le$ the accurate arm's) /
  (arm median time). One FOM per row; both arms are divided by it. Each arm is also compared against the fastest CG at
  least as accurate as itself.
- **Time scope:** the scope of the existing Table-1 series. L-shape: complete query (`total_seconds`). Cube:
  GPU query (`fused_device_seconds`). The other scope is also reported.
- Medians over all retained repetitions × cases. CG error = worst over cases (fields are checked to be
  identical across repetitions).

## Gates (a job's numbers are used only if all pass)

- **Parity:** rotated $R'=R$ arms vs unrotated parent, worst relative field difference $\le 10^{-10}$.
  On the cube at $n \le 64$, the lean DST parent vs `poisson.engine`, also $\le 10^{-10}$.
- **Determinism:** every repetition produces a byte-identical field.
- **CG converged** on every call.
- **Parent stationarity:** every full-$R$ NM-ROM call ends stationary. Truncated arms report their
  stationary counts, and this is not a gate.
- **Order-effect (neighbour) gate:** on six cases, each ROM arm is re-timed right after a CG $10^{-3}$ solve.
  Its median, in the Table-1 scope, must be $\le 1.10\times$ its main-phase median on the same cases.
- **Profile consistency:** the stage-split field equals the fused field to $10^{-10}$.
- **Independent NumPy audit** (`pbk3_audit_np.py`, run on the cluster before the fields are deleted), with an
  independent truth:
  - L-shape: Kronecker principal-submatrix assembly and SuperLU. Cube: a dense sine-matrix solve.
  - Every error is recomputed, every field re-hashed, and determinism checked.
  - On small meshes, one linear-rung field is recomputed end to end in NumPy.
  - Controls that must be detected: truth from a swapped case, a perturbed recorded error, and the
    linear-rung recompute at the wrong $R'$.

## Timing contract

- Randomised order within each case and repetition.
- 0.1 s GPU burn-in before every invocation.
- 5 retained repetitions (main), 3 for the slow CG phase.
- One GPU per job, UUID-guarded before every invocation.
- One allocation per mesh. H200 with 240G at $2048^2$ and at $\ge 128^3$. A100 or H200 for the smaller meshes.
- Cost profile: four separately jitted stages of each ROM arm — (1) source projection and nearest-code start,
  (2) LM, (3) $y$ elimination and the coefficient map, (4) reconstruction $u = G'a$ with scatter. Each stage is
  synchronised, and all are reported at the largest mesh.
