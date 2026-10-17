**Overall verdict: NEEDS-RESTATEMENT.** V3 resolves several factual errors, but still has a concrete 2D test-space confound, an incorrect rollout count, incomplete qualification inheritance, and degenerate statistical rules. A bounded pilot looks plausible; completion of all P1a–P1f within 20 GPU-hours is not established.

I made no filesystem changes, ran no experiments, and did not access sealed solutions. Checks used source files, public QC summaries, saved development/validation arrays, and local job records.

For compact citations, these prefixes identify exact repository directories:

| Key | Directory/file |
|---|---|
| V3 | [PLAN-v3-amendments.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/PLAN-v3-amendments.md) |
| V2 | [PLAN-v2.md](/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-17-full-experiment-plan/PLAN-v2.md) |
| C | [Combined lane](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-13-jcp-combined/experiments/jcp-combined) |
| S | [Speed3d lane](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d) |
| W | [Wide-bank lane](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank) |
| T | [Time2 lane](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-08-jcp-time2/experiments/jcp-time2) |
| R | [References lane](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references) |

Thus `C/c2common.py:121` specifies that file and line under C.

**1. Disposition of audit 2’s six ranked blockers**

| Blocker | Verdict | Evidence |
|---|---|---|
| 1. Scoring and qualification contract | **PARTIALLY** | Physical S timestep, per-case-kind eligibility and configuration keys are corrected, V3:9–18. But combined’s “finalist qualified” is not evidence of passing the newly imported speed3d stationarity gate. V3 also omits explicit validity and tight/tighter requirements; see S/DESIGN.md:306–313. |
| 2. Concrete, resource-bounded priority 1 | **PARTIALLY** | Matrix, adaptations, memory pilot, certificate and defer policy now exist, V3:26–49. The rollout count is wrong; rules/state selection and qualification workload remain unspecified. Per-job certificates do not alone enforce an aggregate 20-hour cap. |
| 3. Scaling estimands | **PARTIALLY** | Fixed-(M,m), repeats, query distinction and interval decisions are added. However **fixed M does not preserve the 2D mode set either**, and the log–log intercept is not additive overhead. V3:53,60 need correction. |
| 4. Speed3d reuse versus Q | **PARTIALLY** | V3:65–73 correctly separates mesh-free measurements and requires additional Q2 checks. It must explicitly inherit amended mesh-specific eligibility/matching restrictions, S/DESIGN.md:457–462,537–543. The g-grid alone does not provide the separately timed component profile. |
| 5. Falsifiable accuracy/competition claims | **PARTIALLY** | Fixed rank ranges, per-cell competition and crossing-front outcomes improve matters. But the three-mesh sign test cannot reject as written; interval overlap does not establish matching; selecting only passing rank steps does not establish monotonicity. |
| 6. Freeze roster/dependency accounting | **RESOLVED** at campaign-plan level | V3:97–108 specifies the roster, conditional exclusions, longer-time/finer-observation work, sunk costs and corrected totals. Exact immutable lane identities and implementation costs remain DESIGN-stage work. |

**2. Disposition of audit 2’s ten “new problems”**

| # | Problem | Verdict | Evidence |
|---|---|---|---|
| 1 | Blanket 3D S matching | **RESOLVED** | V3:9 agrees with C/make_e0_registry.py:49–51. |
| 2 | Qualification confused with utility | **PARTIALLY** | V3:13–18 separates them in principle, but imports a combined fallback label that itself includes the original cost/accuracy objective; details below. |
| 3 | Timing specification only promised | **RESOLVED** narrowly | Six cases, three repeats, bootstrap and three-way verdict are now specified, V3:55–59. Bootstrap implementation and A-baseline remain to be frozen. |
| 4 | “Any departure” without tolerance | **RESOLVED** narrowly | V3:20 supplies \(2.5\times10^{-4}\). Full Q1 inheritance remains incomplete. |
| 5 | Fixed M versus shell completion | **PARTIALLY** | Explicit 3D mode-list reuse fixes the known issue. V3 incorrectly asserts that 2D is already exact. |
| 6 | Reuse changes query contract | **RESOLVED** conceptually | V3:54,69 explicitly separates full-field and mesh-free queries. Its description of the existing 2D initialization is nevertheless inaccurate. |
| 7 | P1 omitted fixed-(M,m) and Q3 | **RESOLVED** as scope | Both are explicitly included, V3:22,32–33. P1b’s numerical fixed pair remains unspecified. |
| 8 | B7/B8 references unbudgeted | **RESOLVED** as scope | V3:100 adds the dependency and 20–40 GPU-hours. This is an estimate, not a measured certificate. |
| 9 | Fit time versus total training job | **RESOLVED** | V3:101 matches W/report.md:111. |
| 10 | Optional lanes outside freeze graph | **RESOLVED** | V3:97–99 explicitly excludes unfrozen D2–D5/C4 and requires fresh cohorts for later entrants. |

**3. Repository verification of v3’s factual claims**

| Claim | Finding and evidence |
|---|---|
| S means BE at physical 0.005 | **Correct.** R/refjob.py:43–44 sets this in both dimensions. C/c2common.py:24 gives 2D DT0=0.005; C/c3common.py:35 gives 3D DT0=0.01. C/make_e0_registry.py:36–51 documents the matching distinction. |
| Historical 13 = 12 S + 1 ST | **Correct.** R/report.md:171 lists the 13. The sole ST failure is test64/36; the other twelve are S. These are not thirteen exclusions from the current dev6+val32 ST panel. |
| Sealed 21 S + 5 ST failures | **Correct.** R/report.md:245–252 gives S failures \(2+5+8+6=21\), ST failures \(1+1+0+3=5\). Only published counts were inspected. |
| W512 M=2052 versus 2049 | **Correct**, for the W1024-bank prefix at meshes n=65 and n=129. W/report.md:194,200. |
| W1024 job 1.85 h; bank fit 1.14 h | **Correct.** W/report.md:111. |
| Scoring arrays 257²/63³, horizon 0.25 | **Correct.** C/score.py:59–62. Longer-time and finer-observation work is genuinely additional. |
| H200 capacity | The archived device log reports **143771 MiB**, C/runs/c3grid/archive/logs/5219449.out:2. V3’s “141 GB” is an approximate hardware label; use actual device bytes for the 120-GB trigger. |
| Speed3d mapping | Broadly correct, with qualifications below. S/s3common.py:8–11 says decode is always on the 63³ lattice; at n=65 that equals the interior mesh. Larger-mesh matching remains configuration-specific, S/DESIGN.md:537–543. |

The combined qualification table verifies the following **specific configurations**, not entire ranks:

| Configuration | Original selected arms | First fallback |
|---|---|---|
| R128, fib4181 | CNR at DT0 and 2DT0: yes | — |
| R384, gauss128 | CNR at DT0 and 2DT0: yes | — |
| R512, gauss192 | CNR at DT0 and 2DT0: yes | — |
| R256, gauss96 | Both: no | gauss128: fast-equal-accuracy yes; accurate-equal-cost no |
| acc, gauss96 | Both: no | gauss128: fast-equal-accuracy yes; accurate-equal-cost no |
| fast, fib1597 | Both: no | fib4181: fast-equal-accuracy yes; accurate-equal-cost no |

Sources: C/configs/c2grid.json:9–39; C/report.md:242–267,348–355.

Two distinctions matter:

- BE rows have “—” in the finalist column. They are **not certified by that table**.
- All six fallback rows pass the finalist numerical check. The three accurate-equal-cost fallbacks receive “fallback qualified: no” because they miss the original cost criterion—cost ratios 1.31, 1.56 and 1.49—not because the finalist quadrature check fails. V3 must separate **numerical qualification**, **role qualification**, and **frontier usefulness**.

For speed3d Q1, the complete contract is:

- Every run valid: finite, no non-finite step, no damping saturation.
- Fresh-Jacobian stationarity ratio
  \[
  \gamma=\frac{\|J^\top r\|}{\|J\|_F\|r\|+10^{-300}};
  \]
  non-stationary when \(\gamma>10^{-3}\) or non-finite.
- Non-stationary steps ≤1% **pooled over steps**, not ≤1% separately per case.
- Distance to the corresponding adaptive tight solve ≤\(2.5\times10^{-4}\) on every case.
- Tight/tighter anchor agreement ≤\(2.5\times10^{-5}\) on the prescribed subset.

Sources: S/DESIGN.md:96–98,122–125,306–313. **Later amendment A2.3 adopts scoring-lattice distance**, lines 457–462, so v3’s metric wording is compatible. My preliminary concern about a metric mismatch was withdrawn after checking that amendment.

**The requested reusable results and states do exist—with limits.**

I inspected C/runs/c2grid/archive/output/W_R128.npz, W_R256.npz and W_R384.npz directly:

- Shapes are respectively `(38,39,6,128)`, `(38,39,6,256)`, `(38,39,6,384)`.
- Each includes `LSPG|BE|0.005|prod` and `LSPG|CNR|0.005|prod`.
- Cases are dev6/0–5 and val32/0–31.
- All 38 corresponding result rows for each scheme/rank are marked verified.
- The run mesh is 1024.

The configuration and save path are documented at C/configs/c2grid.json:4–38 and C/c2grid.py:110–161. Thus **“1024² reused” is factually true**, provided bank, modes, rule, solver and metric remain identical. It does not establish newly required Q1 certification.

For P1b, C/runs/c3grid/archive/output/W_grid.npz contains finite `(64,6,r)` BE states for W256, W512, W768 and W1024, with case identifiers; saving is at C/c3run.py:193–198.

However:

- There are no corresponding small-prefix trajectories in those grid assets for 2D r=32/64 or 3D r=64/128.
- Truncating a wider coefficient vector gives a usable **kernel input**, but not a state reached by that smaller-r solver.
- Neither panel specifies which scheme, two times, six 3D case IDs, or source trajectory is used.
- Changing M also changes the dynamics. Historical states are legitimate kernel fixtures, but should not be called own reached states of the changed configuration.

**The fixed 2D (M,m)=(1536,9216) panel is constructible for every requested rank.** Gauss96 supplies \(96^2=9216\) points; the bank accepts arbitrary prefixes; independent M exists in wide-bank. Evidence: C/vendor/quad2d/qcore.py:83–100,141–146,185–227; its vendor/hari_quadrature.py:81–84; W/w2d.py:310–323. Here M/r ranges from 48 to 3, so M>r throughout.

It still needs the new harness: combined’s current `base`, `blocks` and `Setting` hard-code M=4r, C/c2common.py:120–151. Constructibility is not quadrature qualification.

**4. The new 2D mode-set failure**

V3:53 says fixed modes are exact in 2D. The implementation instead sorts **mesh-dependent discrete Laplacian eigenvalues**, C/vendor/quad2d/vendor/hops.py:35–41.

I recomputed that ordering with NumPy:

| M | Symmetric difference between 256² and 1024² mode sets |
|---:|---:|
| 512 | 0 |
| 1024 | 0 |
| 1536 | **4 modes** |
| 2048 | **8 modes** |

At M=1536:

- 256²-only: `(1,45)`, `(44,9)`.
- 1024²-only: `(28,35)`, `(35,28)`.

Therefore **P1c’s r=384 cross-mesh comparison changes its test space under the existing builder**. Freeze explicit mode lists in **both dimensions**. Reusing the 1024² list at the other meshes is one concrete option; the changed configurations then need qualification.

**5. Arithmetic and resource feasibility**

| Item | Correct calculation | Verdict |
|---|---|---|
| New P1d accuracy trajectories | \(3\times2\times38+3\times2\times16=324\) | **V3’s 552 is wrong for new work.** |
| Reused 1024² trajectories | \(3\times2\times38=228\) | 324+228=552 across new **and reused** accuracy results. |
| P1c timed B invocations | \((3\times3\times2+1\times1\times2)\times6\times3=360\) | Excludes A controls and warmups. |
| A–B–A invocations | \(360\times3=1080\) | Assuming each B gets two A calls; component panels may add more. |
| R512 bank | \(4095^2\times512\times8/2^{30}=63.968754\) GiB | ≈64 GiB correct. |
| R384 bank | Same with 384 =47.976565 GiB | ≈48 GiB correct. |
| Base budget | \(20+60+80+120+30\) to \(20+110+160+220+50\) | **310–560**, correct. |
| Contingency | \(20+1.25(290\text{–}540)\) | **382.5–695**, correctly rounded to 383–695. |
| Occupancy | \(382.5/96\), \(695/72\) | **3.984–9.653 days**, correctly rounded to 4.0–9.7. |

The bank arithmetic describes retained storage, not peak memory. The builder already uses blocks, C/vendor/quad2d/qcore.py:166–182. The new requirement is **retaining only the needed prefix**: combined currently forces the full R512 edge, C/c2common.py:105–108. Simply requesting “chunking” does not fix that.

Historical durations support feasibility of a reduced pilot, but not the full promise:

| Historical work | Logged elapsed time |
|---|---:|
| Wide-bank 2D grid | 2.28 h |
| Wide-bank 3D dial | 2.64 h |
| Time2 fast / acc / wide | 0.28 / 1.18 / 1.03 h |
| Combined c2grid | 3.758 h |
| Combined c2budA / B / C | 5.859 / 5.227 / 0.941 h |
| Combined R256 fallback budget | 1.336 h |
| Combined c3grid | 4.414 h |
| Combined W1024 budget rerun | 9.5 h; preceding failed attempt consumed 3.6 h |

Sources: W/report.md:16–23; T/report.md:9–13; C/runs/{c2grid,c2budA,c2budB,c2budC,c2fbbud,c3grid}/archive/output/result.json, respective `elapsed_seconds` fields; C/runs/JOB-LEDGER.json:167–173,195–200.

These are differing workloads and GPUs, not speed comparisons or linear predictors.

**Is P1 concrete enough to write a DESIGN.md?** Enough to begin drafting, yes. Enough to derive one without resolving scientific choices, **no**.

Still missing:

- Explicit 2D/3D mode lists and P1b’s fixed M,m.
- Frozen rules for each rank, especially unrun prefixes and the failing/deployed/finer P1e slice.
- State-source manifest, two time indices, case IDs and small-prefix policy.
- Kernel harness, split-query API, cross-mesh orchestration and timed/scored equivalence checks.
- Fresh-Jacobian stationarity certification and a specified tight/tighter schedule.
- Own-state quadrature indicators and numerical Q2 bars.
- Prefix-retaining high-mesh builder, pilot isolation and per-panel memory attribution.
- Aggregate resource accounting covering smoke, compilation, certification, failed attempts, repeats and contingency.

The existing speed3d certificate explicitly calls itself an estimate with headroom, S/resources.py:1–12. V3 also needs a rule saying **stop incomplete if the mandatory core cannot fit**; “never dropped” cannot override the hard cap.

**6. Statistical decision rules**

**Six-case bootstrap:** acceptable as exploratory pilot uncertainty, weak as population confirmation. Resample each case jointly across all meshes, repeats and configurations, then recompute the complete max/min statistic. Eighteen timings are not eighteen independent cases. Specify bootstrap method, repetitions, seed and treatment of drift failures. The nonsmooth max/min of sample medians and only six cases make coverage fragile.

**Injected 1.5× timing fault:** not guaranteed to fail. Multiplying the fastest mesh by 1.5 can reduce the original spread or leave max/min≤1.25. Freeze the injection target and require the resulting controlled contrast to exceed the bar. Do not tune the interval procedure until this one fixture rejects.

**Rank-cost slopes:** V3:60 is mathematically misstated.

- For \(t(r)=a+br\), the local log slope is \(br/(a+br)\), not necessarily zero or one.
- A log–log intercept estimates a multiplicative scale, **not additive fixed overhead**.
- Fixed-M,m residual work contains both \(mr\) and \(mM\).
- The fit has only four ranks ≥128 in 2D; GPU efficiency changes can invalidate a pure power fit without invalidating the operation count.

Use the declared FLOP/byte model plus an additive-overhead model, and distinguish operation-model failure from hardware scaling.

**Three-mesh sign test:** **degenerate as written**. If the sample consists of three independent signs, the smallest two-sided exact p-value is \(2/2^3=0.25\). Adjacent differences provide only two signs; all three pairwise differences are dependent. Consequently p>0.05 cannot establish “no trend.” If cases are intended as replicates, define one trend statistic per case and test across cases. Even then, non-rejection is not equivalence.

**The \(0.1e(1024^2)\) margin:** usable as a declared relative-stability margin, but gameable if interpreted as accuracy quality. A uniformly inaccurate method gets a larger allowed change. Freeze the denominator/configuration, label the claim relative stability, and report absolute errors alongside it. The 90% criterion on the common 16-case subset means at least **15/16**, not 90% of the 38-case lower-mesh cohort.

**“>50% improves and median change<0”:** descriptive majority improvement, not monotonicity or statistical evidence.

- On the same resolved, non-tied population, these conditions are largely redundant.
- One resolved improvement and many unresolved ties can pass unless minimum coverage is required.
- Publishing only passing adjacent steps selects favorable findings.
- A majority can improve while the worst case deteriorates severely.

Show every registered transition, tie/failure counts and effect sizes. A global monotonicity claim requires every registered transition to satisfy its predefined criterion.

**H-base matching:** overlapping intervals do **not** establish equivalence. Wider uncertainty makes “matches” easier. Use paired differences and explicit equivalence margins; otherwise say “unresolved.” Also reconcile the four declared objectives—online, offline, median, worst—with the dominance rule that currently uses only online time and worst error.

**7. Additional new problems**

Beyond the statistical failures above:

1. **Initialization contract is misstated.** V3:54 says full-field IC projection. Existing 2D `make_query` samples the supplied field at cold quadrature points and solves the reduced projection, C/t2core.py:259–263. Full-field **input** is not full-grid projection. Changing this would invalidate simple result reuse.
2. **Qualification is accidentally grandfathered.** Combined finalist labels do not establish the new pooled stationarity gate. The historical report’s role-qualified fallback labels also contradict V3’s intended separation of numerical qualification from usefulness.
3. **The 552 count includes reused results while being presented as rollout work.** Timing and qualification counts are separate and considerably larger.
4. **Q1 may require additional trajectory storage.** Six saved output states suffice for kernel fixtures and output-distance checks, but not fresh-Jacobian certification at every intervening step.
5. **The f1 bound label has the wrong direction.** V3:68 says “upper bound on FOM efficiency.” A finite ladder gives an upper bound on optimal FOM **cost**, hence an upper bound on the claimed ROM speedup—not FOM efficiency. S/DESIGN.md:373–376 states the correct interpretation.
6. **Dropping 4096² changes what can be concluded.** A two-mesh remainder cannot silently retain the registered three-mesh H-cost-N/H-acc-N verdict.
7. **“Fixed M=4r” conflates two panels.** At fixed M=1536 it equals 4r only at r=384. Keep fixed-mode and nominal-policy definitions separate.

**Final verdict: NEEDS-RESTATEMENT. Ranked remaining blockers**

1. **Priority 1:** freeze actual mode lists in both dimensions; correct the demonstrably false 2D invariance assertion.
2. **Priority 1:** complete Q1 inheritance and separate numerical qualification from historical role qualification; specify required certification/replays.
3. **Priority 1:** correct the run inventory, freeze rules and state fixtures, and enforce a cumulative 20-hour certificate with an explicit incomplete-core stop outcome.
4. **Priority 1:** correct the initialization contract and rank-scaling model; specify paired timing/bootstrap mechanics before collecting evidence.
5. **Before accuracy verdicts, fixable in lane designs:** replace the degenerate trend test; define equivalence, minimum resolved coverage and treatment of every rank transition.
6. **Before competition/final confirmation, fixable later:** repair interval-overlap “matching,” the FOM bound label, mesh-specific reuse restrictions and exact freeze artifacts.

The necessary banks and principal saved results are available. The remaining obstacles are specification and certification gaps, not a need to restart the campaign.