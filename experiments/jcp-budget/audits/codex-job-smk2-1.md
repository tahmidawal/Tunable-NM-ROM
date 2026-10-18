**FIX-FIRST.** Audited staged `smk2` against HEAD `c68c6f02a7e97078b14653b93857843bfa554c54`. No files changed, SSH used, jobs submitted, or scratch files created.

Paths below are relative to `experiments/jcp-budget` unless stated otherwise.

| Item | Assessment and evidence |
|---|---|
| **1. Submission script** | **CORRECT.** One H200, `gpu`, two-hour limit, 240 GB host memory, absolute cluster Python, x64/highest, backend preflight, and outputs/cache/tmp under the lane namespace: `runs/smk2/run.sbatch:3–34`. Reservation agrees: `runs/smk2/RESERVATION.json:2–19`. |
| **1. Staged provenance** | **CORRECT.** Independently checked all **148 provenance entries** against Git HEAD blobs and their SHA-256 values; every manifest entry also matches. Staging reads committed contents: `cluster/stage.py:61–76`. |
| **1. Submission safeguards** | **CORRECT in the wrapper.** Queue checks, duplicate prevention, one-smoke-at-a-time check, reservation, and shared campaign submission lock are present: `cluster/submit.sh:18–37,50–51`. Live queue and remote disk availability remain unverified under the no-SSH restriction. |
| **2. Predicate coverage** | **CORRECT configuration; NEEDS-FIX acceptance completeness.** Every requested predicate has a configured execution path. G-B3 can nevertheless pass with missing comparisons, as detailed below. |
| **3. Horizons and output contract** | **CORRECT.** B2, stars, and controls run at 0.25; B7 switches to 0.5 and clears both horizon-dependent caches: `b3run.py:157–175`. |
| **4. Runtime and memory** | **CORRECT bounded smoke allocation; NEEDS-FIX memory evidence.** Two hours is plausible, but not demonstrated for the expensive adaptive panel. The driver does not persist its GPU memory peak. |
| **5. Acceptance reliability** | **NEEDS-FIX.** G-B3’s aggregate predicate does not enforce comparison coverage for every configured member/case. |

**Predicate and archive coverage**

- **G-R2, G-R3, judge agreement, C-R1:** evaluable on **reference case 0**, as configured. The reference and first-order ladders are defined at `ref/refrun.py:41–46`; offline recomputation is at `anaref.py:59–118,175–208`. Case 1 is intentionally absent from these reference checks.
- **Coverage/inventory, G-B0, G-B2, G-B4 including stars:** configured for cases **0–1**. An in-memory execution of the driver’s scheduling functions produced exactly the independently defined B2/star/control inventory: **76 arrays, 152 case-rollouts, 21,980 scheduled steps**. Including B7, acceptance expects **91 arrays and 179 case-rollouts**. See `b3run.py:51–84,129–175` and `ana3d.py:86–151`.
- **G-B3/G-L1 archive lookup:** **all three members are present**, including both cases:
  - W256 gl24 FS CN 0.0125: `q3d3`.
  - W256 gl24 FS CN 0.01: `q3d2`.
  - W512 lat16384 QN BDF2 0.025: `q3d3`.

  I inspected the actual `W_q2.npz` keys and case-index arrays. Each contains cases 0–63. The lookup pattern matches them: `ana3d.py:233–242`.
- **C-Q, C-T, C-S, C-F, C-score, G-Q:** execution and acceptance paths exist for this population: `b3run.py:129–146`; `ana3d.py:325–326,580–618`. Whether numerical controls actually fire remains a smoke result.
- **Rule availability:** **CORRECT.** Both rule archives contain the requested rules. `gl56`, `gl64`, both successor chains, and W256’s `lat256`/`lat8192` controls are covered by `configs/smk2_b3.json:11–89`. Availability does **not** establish convergence; G-B4 properly checks that numerically.
- The missing case-1 long reference does **not** prevent G-L1, Q1/Q2, or B2 evaluation. It prevents case-1 long-reference error/growth evaluation. Missing references return `None`, and those window metrics are skipped: `ana7.py:68–74`; `ana3d.py:474–482`. This is consistent with the single-reference smoke scope.

**Horizon, GLAT, and solver handling**

`S.nsteps()` and `S.nout()` derive lengths from the active horizon; adaptive `ad`/`adg` queries use `budcore` with explicit output count and schedule horizon: `s3common.py:42–45,100–103,208–219`. Thus B2 produces six outputs and B7 eleven. Clearing query/certification caches prevents reuse of the shorter compiled schedule.

`adg` receives GAL form data and bypasses the fixed-step certifier: `b3run.py:57–58,65–66,79–81`; `s3common.py:136–141`. GLAT files are written for both configured settings, and `mode="b3"` produces `W_b3.npz` with matching case keys: `b3run.py:176–178`; `s3ctx.py:93–95,168–169`.

**Runtime and memory assessment**

The existing E0 **H200** case-0 metadata records approximately **148 seconds** for the 0.25 reference ladder, including approximately **131 seconds at L512**, with a **25.0 GiB** recorded peak. Source: the references worktree’s `store/3d/val64/ST/c000.json`, fields `attempts`, `job`, and `seconds`. This supports a several-minute nominal reference estimate at 0.5, not a guaranteed bound.

The new L512 eleven-output buffer alone occupies **10.94 GiB**. Automatic L768 refinement raises that buffer to **36.98 GiB**, before solver vectors and FFT workspaces (`ref/refcore.py:135–168`; `ref/refrun.py:196–203`). Refinement is the principal reference-memory risk.

For W512, assuming \(M=2048\), retained `B`, `D`, `Bw`, and `P` occupy approximately **4.69 GiB at gl56** and **7.00 GiB at gl64**. W256 adds approximately **5.84 GiB** across those two rules. These exclude other rules, mesh banks, construction temporaries, and executable workspaces. Tables remain cached: `s3common.py:122–141`; construction is at `vendor/quad3d/offmesh.py:41–76`.

The dense adaptive Jacobian multiplication repeats throughout the 21,980-step B2/control panel; individual adaptive steps permit up to 600 iterations (`c3common.py:145–147`; `budcore.py:118`). **Two hours is a reasonable pilot ceiling, not a defensible completion guarantee.** I found no demonstrated inevitable OOM, but neither an exact peak bound nor recorded B3 peak telemetry.

**Blocking fixes only — FIX-FIRST**

1. **Make G-B3 require every configured member/case comparison.** `ana3d.py:329–335` silently skips invalid deployed corners without incrementing missing coverage; `ana3d.py:571–572` accepts any nonempty surviving comparison set. Require all **six** expected comparisons here, finite and within tolerance, and explicitly fail missing/invalid deployed corners. G-L1 compares a separate B7 invocation and does not repair this omission.

2. **Persist B3 peak GPU-memory evidence before running this smoke.** `DESIGN.md:690–691` explicitly requires measuring construction peaks, including W512 gl56/gl64. `s3ctx.py:39–42` and `b3run.py:176–181` record no peak; the initial `nvidia-smi` reports capacity only. Record allocator peak/current usage after synchronized setup and expensive table/solve phases, retaining the overall peak for the subsequent resource assessment.