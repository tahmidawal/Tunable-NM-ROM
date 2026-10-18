# Priority-1 lane design, restated by the coordinator from PLAN v2 + v3 + v4 (2026-10-17)

**Status:** coordinator restatement for Codex audit, before the `jcp-p1-scaling` lane is launched. It is
the brief the lane's own `DESIGN.md` must implement; it is not itself the pre-registration (the lane's
`DESIGN.md` is, after its own Codex design audit). Sources: `PLAN-v2.md` (V2), `PLAN-v3-amendments.md`
(V3), `PLAN-v4-amendments.md` (V4), `codex-plan-audit-3.md` (A3). Where they differ, V4 > V3 > V2.

Lane: `worktrees/2026-10-17-jcp-p1-scaling` (branch `exp/2026-10-17-jcp-p1-scaling`, forked at
`21c175a1b`, setup commit `8bd0d20b6`), namespace `/cluster/tufts/paralab/tawal01/jcp-p1-scaling/`,
concurrent-job cap 4 (campaign-wide cap 8; jcp-speed3d holds 3).

## 0. The question
Do the reduced model's **cost** and **accuracy** depend on the bank width $r = R'$ and **not** on the
mesh size $N$? Hypotheses H-cost-N, H-cost-r, H-acc-N, H-acc-r, each with pass / fail / inconclusive.
"k" is read as $r=R'$ (campaign default); the head-size reading (K) is covered by C5 in priority 3.

## 1. Items owned (P1a–P1g plus Q3)

| Item | Ranks $r$ | Meshes | Cases | Schemes | Output |
|---|---|---|---|---|---|
| P1a kernel 2D | 32, 64, 128, 256, 384, 512 | 1024² | 12 fixtures (V4-3) | n/a | component times; three panels |
| P1b kernel 3D | 64, 128, 256, 512, 768, 1024 (W1024 prefixes) | 65³ | 12 fixtures (V4-3) | n/a | same components; fixed-(M,m) + nominal panels; actual shell M |
| P1c mesh-cost rollout 2D | 128, 256, 384 (+R512 diag at 1024² only) | 256², 1024², 4096² | dev6, 3 repeats, A–B–A | LSPG-BE Δt₀, LSPG-CN-R Δt₀ | solve (evolution), init, decode, full query timed separately |
| P1d accuracy 2D | 128, 256, 384 | 256² (38: dev6+val32), 4096² (16: dev6+val32[0:10], labelled subset); 1024² reused from combined `c2grid` | as stated | LSPG-BE Δt₀, LSPG-CN-R Δt₀ | E0 ST error; Q1, Q2 per configuration |
| P1e quadrature slice | 2D r=256; 3D r=512 | 1024² / 65³ | dev6 / 6 cases of 923801 | BE | failing / deployed / finer rule on own reached states, both bars |
| P1f memory + offline | all above | all above | — | — | measured peak device memory per panel; offline-cost inventory from logs (sunk vs new) |
| P1g frozen-mode recheck | 384 (M=1536) | 256², 4096² | 38 each | BE, CN-R | how much earlier "flat accuracy" cross-mesh numbers move under the frozen 1024² mode list |
| Q3 conditioning | every P1 rollout configuration | all | — | — | singular values/cond of A, weighted J, $J^\top J$ vs r |

3D cost-vs-mesh (H-cost-N 3D) and 3D Q1 for the g-grid arms come from `jcp-speed3d` (m1, g1–g3) after its
audits, under the V3-4 map and V4-6 restrictions; this lane does not duplicate them. The coordinator
combines them in the final report; the lane may cite them only if speed3d's results audit has passed.

## 2. Frozen configurations
- **Bank:** 2D R512 checkpoint + rotation used by combined (`T2.CKPT`, `qcore.build_rotated_bank`),
  ordered prefixes. Hashes recorded in a development manifest (item 0 below).
- **Frozen rule per rank (2D):** R128 `fib4181`; R256 `gauss128`; R384 `gauss128`; R512 `gauss192`.
  Each is re-certified (Q1+Q2) at every mesh. Small prefixes r=32, 64 are **kernel-only**.
- **Mode lists (V4-1).**
  - 2D: one explicit hashed list per M, taken from the builder at 1024², reused at every mesh. Because
    the frozen list equals the builder list at 1024², the 1024² results reused from combined are the same
    configuration (verify by hash). For M ≤ 1024 (r ≤ 256 at M=4r) the lists are identical across meshes
    (verify and record); they differ at M=1536 (r=384) and M=2048 (r=512).
  - 3D (P1b): the frozen 65³ shell list per r.
  - **Primary** P1c/P1d arms use the frozen lists. A **nominal-policy** arm (builder list per mesh) runs
    only where the lists differ (r=384 at 256²/4096²) and is labelled "deployment policy (mode set varies
    with mesh)"; report the symmetric difference for every (M, mesh).
- **Initialisation contract (V4-4):** the existing `t2core.make_query` contract is kept — full-field IC
  sampled at the cold quadrature points, reduced projection solve, full-field decode on the mesh. Labelled
  "full-field input sampled at quadrature points; full-field decode".
- **Kernel fixtures (V4-3):** P1a: output indices 1 and 5 of the LSPG-CN-R Δt₀ trajectories of R384 at
  1024² for dev6 cases 0–5 (`c2grid/.../W_R384.npz`), truncated to each r. P1b: BE states, output indices
  1 and 5, cases 0–5 of 923801 from `c3grid/.../W_grid.npz` (W1024), truncated. Labelled "kernel
  fixtures, not reached states of the smaller-r solver".
- **P1a panels:** (i) fixed (M, m) = (1536 frozen list, gauss96 = 9216 points); (ii) fixed m = 9216 with
  M = 4r; (iii) nominal M = 4r with the frozen rule per r (r ≥ 128 only).
- **P1b panels:** (i) fixed (M, m) = (frozen 65³ list for r=512, Gauss 24³ = 13,824); (ii) nominal M≈4r
  with the shell-completed list and actual M recorded.

## 3. Decision rules (frozen before data)

**H-cost-N (2D, P1c; V3-3, V4-4).** Timed quantity: B/A ratio of solve (evolution-only) time, where A =
fixed reference configuration (r=384, 1024², LSPG-CN-R, frozen list) interleaved A–B–A in the same job on
the same GPU. Per (r, scheme) and mesh: median over 6 cases × 3 repeats. Statistic
$R = \max_N / \min_N$ of those medians. 95 % percentile interval from a case-level bootstrap (10,000
resamples, fixed seed; each case resampled **jointly** across meshes, repeats and configurations; the
full max/min statistic recomputed per resample). **Pass** if the upper bound ≤ 1.25; **fail** if the
lower bound > 1.25; otherwise **inconclusive** (expected to be possible with 6 cases). Full query, IC
init and decode are reported separately and are **expected** to grow with N (not part of the verdict).
If 4096² is dropped: "two-mesh evidence, registered three-mesh verdict not reached".
- Must-fail control: multiply the **largest-median mesh's** timings for one configuration by 2.0 (frozen
  target and factor, chosen so the injected contrast alone exceeds 1.25 even if the original spread is
  1.0) — the procedure must return **fail** for that configuration. Also a doubled-work arm (two evolutions
  per call) must show B/A ≈ 2 within 10 %.

**H-cost-r (P1a, P1b; V4-4).** Per component $t(r) = a + b\,r^{p}$, nonlinear least squares over r = 32–512
(2D) / 64–1024 (3D), bootstrap over the 12 fixtures grouped by case (10,000). Pre-declared p at fixed
(M, m): nonlinear residual ≈ 1 (the m·r part; the m·M part is r-independent), tested Jacobian 1,
normal-equation formation 2, Cholesky 3. **Disagreement** = pre-declared p outside the 95 % interval.
Report a as fitted fixed cost, cross-checked against the r=32 kernel. Report FLOP and byte counts per
component; a disagreement is classified as **operation-model failure** only if the FLOP/byte counts also
disagree, otherwise as **hardware scaling** (small-matrix inefficiency). Verdict per component and panel:
consistent / disagrees (op-model) / disagrees (hardware).

**H-acc-N (P1d; V4-5).** Per (r, scheme), on E0-eligible cases (per case-kind from the store):
- per-case slope of error vs log N over the 3 meshes on the 16-case subset; Wilcoxon signed-rank test of
  the slopes (two-sided, α = 0.05);
- TOST on paired differences $e(N) - e(1024^2)$ for N = 256² (38 cases) and N = 4096² (16 cases), margin
  $\delta = \max(2\varepsilon_{\rm ref}, 0.1\,e(1024^2))$ per case (relative-stability margin; absolute
  errors reported beside it), α = 0.05 each side; and the fraction of cases within margin.
- **Pass** if TOST establishes equivalence for both meshes **and** ≥ 90 % of cases are within margin
  (on the 16-case subset that means ≥ 15/16). **Fail** if the Wilcoxon test rejects (p < 0.05) **and**
  TOST does not establish equivalence for at least one mesh. Otherwise **inconclusive**.
- Primary on frozen mode lists; nominal-policy arm reported beside it.

**H-acc-r (P1d; V4-5).** For each registered transition 128→256, 256→384 (and 384→512 at 1024² from
combined, diagnostic), per mesh and scheme: count qualified cases improving / worsening / tied (ties
within 2ε_ref). A transition counts only if resolved coverage ≥ 50 % of eligible cases. "Improves" iff
> 50 % of resolved cases improve **and** the median paired change < 0. "Monotone over the registered
range" iff every transition improves. **Report every transition**, with tie/failure counts, effect sizes
and the worst-case change. The Pareto front is reported separately and is not a monotonicity test.

**Q1 (V4-2)** exactly as in the common brief §3, computed in-job at every step for **every** P1 rollout
configuration (BE included): adaptive tight solves for all 324 new P1d rollouts; tighter solves on a
6-case subset per configuration. **Q2:** for every (r, mesh, scheme), the next finer rule in the same
family (R128 fib4181 → fib6765; gauss128 → gauss160; gauss192 → gauss256; lane to confirm availability
and memory) on the arm's own reached states; bars: rollout distance ≤ 2.5e-4 on every case and the
combined lane's δ_quad finalist bar for ρ. Unqualified configurations are reported, never ranked.

**Three labels (V4-2):** numerical qualification / role qualification / frontier usefulness — never merged.

## 4. Controls (V3-7)
- Positive gates (must pass): split-query equivalence (init+evolve+decode == fused query to 1e-12 and
  identical iteration counts); frozen-mode-list reuse check (hash equality at 1024² with combined); timed
  outputs == scored outputs; reproduce combined's 1024² R384 CN-R errors from the reused arrays (scoring
  path check); manufactured temporal-order check for CN-R (order ≈ 2).
- Negative controls (must fail/fire): injected timing fault (above); doubled-work arm; wrong-mesh table
  rejection (a 256² operator table fed to a 1024² bank must raise); under-resolved rule (e.g. gauss32 at
  r=384) must fail Q2; wrong-gradient solver (perturbed Jacobian) must fail Q1 stationarity; case swap and
  time swap in scoring must be detected; must-fail for Wilcoxon/TOST: a synthetic trend (e × (N/1024)^0.2)
  must give **fail**.
- Sensitivity probes (reported only): GPU type, repeat seed.
Each control is shown to fire in the real-size smoke at one real mesh (1024²) before any verdict.

## 5. Resources
- **Hard cap 20 GPU-h including contingency**, cumulative ledger, smoke-derived cost certificate per job
  (speed3d `resources.py` pattern). Submit only if spent + certified projection ≤ 20.
- **4096² memory pilot first** on H200 (actual device bytes 143,771 MiB). If peak > 120 GB (decimal) the
  4096² arm switches to a **prefix-retaining** builder (only the needed prefix r ≤ 384 kept; ≈ 48 GiB),
  labelled so. No 4096² run before the pilot.
- **Defer order** (first dropped first): P1f-offline → P1e 3D → P1b → R512 diagnostic → P1d 4096²
  subset → P1c 4096². **Never-dropped core:** P1a, P1c at 256²/1024², P1d at 256², P1g. If the core cannot
  finish inside the cap the outcome is **"P1 incomplete"** (no verdict; shortfall reported).
- Run inventory: P1d 324 new rollouts (+228 reused) + 324 tight anchors + tighter subsets + Q2 finer-rule
  replays; P1c 324 timed B rollouts (+36 R512 diag, +72 nominal-policy r=384) with A controls and warmups;
  P1g 152; kernels seconds each.

## 6. Missing code (write → Codex code audit → smoke)
1. kernel harness (separate component timers with `block_until_ready`, fixed-(M,m) control, drift and
   sync checks); 2. split query (`init`, `evolve`, `decode` separately callable) with equivalence check;
3. cross-mesh orchestration (one job per mesh/arm dir); 4. E0 scoring wired to the run path (combined's
`make_e0_registry.py` / `score.py`, CPU); 5. own-state Q2 indicator; 6. in-job fresh-Jacobian Q1
certifier; 7. frozen mode-list builder + hashes; 8. prefix-retaining 4096² builder (if the pilot needs it);
9. the statistics module (bootstrap, NLS fit, Wilcoxon, TOST) with the must-fail synthetic tests.

## 7. Deliverables
`DESIGN.md`, audits, `runs/GPU-LEDGER.json`, `runs/EXPORT.json` (machine-readable verdicts and key
numbers), `make_report.py` → `report.md` with plots: cost vs r per component (2D/3D), solve time vs N per
r, error vs r per mesh, error vs N per r, P1g before/after, memory vs r; hand-back per the common brief.
