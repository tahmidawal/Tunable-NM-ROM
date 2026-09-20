# Adversarial review, round 2 — "Tunable Non-linear Manifold ROMs for Elliptic, Parabolic, and Hyperbolic PDEs via Matrix-Free Petrov–Galerkin Projection"

Paper at commit `d131ec11`, branch `exp/2026-09-16-paper-refresh`,
`worktrees/2026-09-16-paper-refresh/paper/`. `main.pdf` = 33 pages; main text ends
at the foot of p. 9; references p. 10–18; appendices p. 19–33.
Checked against: the round-1 review and `REVIEW-ROUND1-DISPOSITION.md`; the nine lane
`reports/summary.json` / `DESIGN.md` files at their pinned commits; `tables/numbers.tex`,
`tables-md/numbers.json`, `tables/provenance.json`, `gen_tables.py`; `old-neurips-main.tex`.

Line numbers are of `main.tex` at `d131ec11` unless another file is named. Tables are cited
by the label the paper uses; the printed number is given in brackets where it matters.

**Headline for the writer:** the numbers are in unusually good shape — I could not break a
single headline macro against its lane row, and every cost ratio I checked is same-job. What
is broken is the *argument*: the abstract, introduction and conclusion still carry the
rejected submission's framing sentence that §5.1 of this same paper explicitly withdraws;
the one positive result in the abstract (the L-shape cost win) is on a linear PDE that fails
the paper's own stated structural condition, is measured against the most expensive
full-order arm in the job, and is matched or beaten by plain POD in the same table; and the
entire tunability ladder moves the physical (vs-reference) error by 1.13× while being sold
as a 2.44× accuracy knob.

---

# 1. Number audit

## 1.1 Mechanical checks over the whole manuscript

| check | result |
|---|---|
| macros used in `main.tex` / `appendix.tex` / `extended-results.tex` but undefined | **0** (94 distinct macros used in main.tex, all defined) |
| `numbers.tex` vs `numbers.json` value mismatches | **0 numeric**; 48 entries differ in *formatting only* (LaTeX `--`/`\texttt{}`/`\ldots` vs Markdown `–`/backticks/`…`) — cosmetic, both sides agree on every digit |
| hard-typed result numerals in main-text prose | **0 in digit form** (all digits in the prose are mesh sizes, `q`, `k`, `M`, tolerances) |
| hard-typed result numbers in **word** form in main-text prose | **8** — see 1.3, this is where the discipline leaks |
| undefined `\ref`/`\cite` in `main.log` | 0 |
| uncited `\bibitem`s that still print | **9** — see finding N7 |

## 1.2 Traced numbers (35 sampled, spread across abstract, prose and every table)

Every row below was traced macro → `gen_tables.py` expression → lane `summary.json` row
(subject + job id). "matches" = exact to the printed digits after the generator's rounding.

### Abstract and contributions

| # | macro [value] | where in the paper | lane row (subject, job) | verdict |
|---|---|---|---|---|
| 1 | `nLshapeNeuralCheaper` [2.77] | abstract L92, Contrib. 3 L180, §5.6 L811 | `fom_splu` 8.3862 ms ÷ `neural_q64@head_sdf_R512_K16` 3.0283 ms, **both job 3784663** | matches, same-job |
| 2 | `nLshapeNeuralCheaperFiveTwelve` [7.60] | abstract L93, Contrib. 3, §5.6 | `fom_splu` 36.5046 ÷ head 4.8008 ms, **both job 3789568** | matches, same-job |
| 3 | `nQxmErrSpan` [2.44] | Contrib. 1 L157, §5.2 L699, conclusion L853 | `spans.worst_evolved_percent.fixed_M.1088.within_job.error_span`, **job 3780177** | matches |
| 4 | `nQxmCostSpan` [5.16] | same three places | 4377.884 ÷ 847.990 ms, **both job 3780177** | matches, same-job |
| 5 | `nAblBurgersNeural` [2.5629] | Contrib. 1 L161, §5.4 | T06a `(a) neural head, EQ`, job 3711424 | matches |
| 6 | `nAblBurgersLinear` [56.9296] | Contrib. 1, §5.4 | T06a `(b) linear map (fit to head outputs)`, 3711424 | matches |
| 7 | `nAblBurgersPodSixteen` [61.6503] | Contrib. 1, §5.4 | T06a `(e) POD-LSPG k'=16`, 3711424 | matches |

### §5.1 panel (Burgers 256² / 1024²)

| # | macro [value] | lane row (subject, job) | verdict |
|---|---|---|---|
| 8 | `nPanelSubjectCount` [48] | job 3789570, 48 non-`*` subjects | matches |
| 9 | `nPanelReducedCount` [39] / `nPanelReducedNonDomEvolved` [0] | `*` row, 3789570 | matches |
| 10 | `nPanelPodFiveTwelveErr` [0.2184] / `…Ms` [2790.8] | `pod512_M2048_dense`, 3789570 | matches |
| 11 | `nPanelFomBeatingMostAccurate` [4] | `dense_tight`, `fft_tight`, `nt1e-3_dt005`, `nt1e-4_dt005`, all 3789570 | matches |
| 12 | `nPanelFomNtThreeFineErr/Ms` [0.0489 / 31.8] | `nt1e-3_dt005`, 3789570 | matches |
| 13 | `nPanelFnoErr/Ms` [7.4164 / 7.2] | `fno-large`, 3789570 | matches |
| 14 | `nPanelCheapestRatio` [4.48] | `q0_M64_eqcert_g1em06_fastL4` 40.359 ÷ `nt1e-2_dt01` 9.009, **both 3789570** | matches, same-job |
| 15 | `nPanelTenTwentyFourCheapestRatio` [1.82] | `q0_M64_eqxfer_g0p001` 32.230 ÷ `nt1e-2_dt01` 17.724, **both 3789572** | matches, same-job |
| 16 | `nPanelTenTwentyFourReducedNonDomEvolved/Count` [5 / 20] | `*`, 3789572 (24 reduced, 4 uncertified `eqxfer` excluded) | matches |
| 17 | `nPanelEqSpeedupZero/OneTwoEight` [4.77 / 4.81] | dense ÷ `eqcert`, all four legs 3789570 | matches, same-job |
| 18 | `nPanelEqtopTwinSpeedupMin/Max` [4.2 / 5.3] | six dense÷`eqtop` ratios, all 3789570 (4.170…5.281) | matches, same-job |
| 19 | `nPanelRefDiscretisation` [4.0265] | `fft_tight.worst_reference_percent`, 3789570 | matches |

### §5.2 rank-vs-tests, §5.3 quadrature, §5.5 mesh/speed

| # | macro [value] | lane row | verdict |
|---|---|---|---|
| 20 | `nQxmFixedQzeroMs/QtopMs` [848.0 / 4377.9] | `fixed_M.1088.within_job`, job 3780177 both | matches, same-job |
| 21 | `nQxmTwoFiftySixErrSpan` [1.22], `…Passes` [no] | `fixed_M.256.within_job`, 3780177 | matches |
| 22 | `nQxmAnchorSpreadPct` [14] | `q0_M1088` 742.222 (3780175) vs 847.990 (3780177) | matches (this *is* a cross-job number, and the paper labels it as such) |
| 23 | `nQxmMstarTwoFiftySix` [2176] | `saturation.per_q.256.M_star`, job 3783899 | matches |
| 24 | `nMeshBurgersCachedFirst/Last` [44.16 / 43.58], `…UnknownGrowth` [264] | T10, Burgers mesh ladder | matches |
| 25 | `nMeshBurgersCompleteRatio` [1.033] | `flatness_complete.ratio_finest_over_coarsest` | **mislabelled — see N4** |
| 26 | `nSpeedGpuTwoFiftySix` [1.520x] | T15 `spd01`, 256, arm `L4` | matches |

### §5.6 linear cells, NS, L-shape

| # | macro [value] | lane row (subject, job) | verdict |
|---|---|---|---|
| 27 | `nPlinTenTwentyFourQzeroErr` [3.1495] | `q0_m4@new_K32`, job 3783813 | matches |
| 28 | `nPlinTenTwentyFourTopErr/Ms` [0.7421 / 4.42] | `d_linear_qr_m4@new_K32`, 3783813 | matches |
| 29 | `nPlinTenTwentyFourDstMs` [3.248] | `dst_direct`, 3783813 | matches |
| 30 | `nNsOracleRatio` [1.19] | `head_K16_R256`, gate `H-ORACLE_N256`, job 3787319 | matches |
| 31 | `nNsKthirtyTwoOracleRatio` [1.15] | `head_K32_R512`, 3787320 | matches |
| 32 | `nNsOracleBar` [2.0] | **hard-coded literal in `gen_tables.py`; reads no JSON** | **does not trace — see N1** |
| 33 | `nLshapeNeuralErr/Ms` [2.131 / 3.028] | `neural_q64@head_sdf_R512_K16`, 3784663 | matches |
| 34 | `nLshapeDevelopmentWorst/ValidationWorst` [6.8 / 16.7] | `head_arms`, mesh 256, job **3783786** | matches numerically, **wrong quantity for the sentence — see M6** |
| 35 | `nWaveBankOverHeadCostTwoFiftySix` [129] | `head_q0` 183.516 ÷ `linear_bank64` 1.428 `median_gpu_ms`, job 3783805 | matches, same-job, **but a different cost quantity from the L-shape ratios — see N2** |

## 1.3 What does NOT trace, or traces to the wrong thing

**N1 (MODERATE). Three constants are typed into the generator, not read from a lane.**
`gen_tables.py` sets `macro('nNsOracleBar','2.0')`, `macro('nEqtopBar','0.116')`,
`macro('nEqtopTightBar','0.06')` as string literals. `main.tex` L9–11 says "NO NUMBER IS
TYPED IN THIS TREE: every numeric value is a macro from `tables/numbers.tex` … written by
`gen_tables.py` from the lanes' machine-readable outputs", and the Reproducibility statement
(L869) repeats it. All three values are *substantively correct* against the lane DESIGNs
(NS H-ORACLE "≤ ½× POD-K held-out" ⇒ ratio ≥ 2.0, pre-registered in the ns2d DESIGN at
commit `1281ff73`, 2026-09-17 00:55, before every NS job), but they are hand-typed.
Round-1 finding 22 raised exactly this for `nEqtopBar`; the disposition marked it "fixed" by
*explaining* the origin in §3.4, which is not the same as generating it.
**Fix:** put the three bars in the lane JSONs (or a `bars.json` the generator reads) and
regenerate, or delete the "no number is typed" sentence.

**N2 (MODERATE). Two different cost quantities are quoted as if they were one.**
The wave macros (`nWave*Ms`, `nWaveBankOverHeadCost*` = 158/129/42×) use
`median_gpu_ms`; the L-shape (`nLshapeNeuralMs`, `nLshapeSpluMs*`) and p-linear
(`nPlinTenTwentyFour*Ms`) macros use `median_total_ms` / complete-query ms; the panel
macros use `median_gpu_ms`. At 256² on waves the complete-query costs are bank 16.659 ms,
POD-64 16.657 ms, DST 19.703 ms — the 129× bank-over-head separation is a GPU-kernel ratio
that essentially vanishes on the complete-query metric the L-shape section uses.
§4 "Metrics and timing" (L547–555) defines one notion of cost ("the median over repetitions
of a completed device computation") and never says that some tables charge the dense input
and output and others do not.
**Fix:** one sentence in §4 naming which tables report device ms and which report
complete-query ms, and a footnote on the wave ratios.

**N3 (MODERATE). Provenance table prints the wrong job for the L-shape bank/head rows.**
`gen_tables.py:1276` takes `job = s_rows[0]['job_id']`, the first row of the lane summary,
which is a *solve* row. So `\provLshapeJob` = `3784662` and `T18a`/`T18b` (Table
`tab:lshape`) are headed with that id, while every bank and head row in them carries
`job_id = 3783786` (the training job `lsh02`, which the lane report names in its first
line). `tab:provenance` therefore pairs job `3784662` with the *training* commit
`1086ccefdcb5…`. The numbers are right; the provenance is wrong, in the one table whose
whole purpose is provenance. Round-1 finding 21(a) was the same class of bug in the
w-ladder row.
**Fix:** key the job id on the rows actually printed.

**N4 (MODERATE). "the complete query grows 1.033×" measures the device time, not the
complete query.** §5.2 L669–672: "the cached reduced solve costs 44.16→43.58 ms … the
complete query grows 1.033× through dense input and output." `nMeshBurgersCompleteRatio`
= 1.033 is `flatness_complete.ratio_finest_over_coarsest`, which equals the **ROM device
ms** ratio 48.5→50.1 in Table `tab:mesh`. The actual complete (host) query in that same
table runs 50.4 → 76.8 ms, a ratio of **1.52×**. The sentence understates the cost of the
dense input/output by 1.5×, and it is the sentence that carries the mesh-independence
claim.
**Fix:** quote 1.52× and rename the macro, or say "device time" and quote 1.033×.

**N5 (MODERATE). Eight result numbers are hand-typed in word form in the main text**, in a
paper whose Reproducibility statement (L869) says "Every table and **every number in the
prose** is generated by one script". In order of severity:

| line | sentence fragment | status |
|---|---|---|
| L720 | "the learned bank is **three to four** times worse than a POD basis of the same rank" | true (Poisson 0.7421/0.1855 = 4.00; waves 5.121/1.502 = 3.41) but typed |
| L671 | "reaches ρ_max = 0.462 held-out, **four** times the primary bar" | true (0.462/0.116 = 3.98) but typed |
| L742 | "by 512² it is 7.60× cheaper … at **about two percent** error" | true (2.123 %) but typed |
| L635 | the pre-registered bar, spelled out: "monotone, at least **three** non-dominated points, at least **2×** on both axes, nothing early-stopped" | **and it is not the bar in the lane DESIGN — see M3** |
| L773 | "(six opened Burgers cases, **eight** wave and **twelve** Poisson)" | matches T01 |
| L571 | "the matched **eight**-case cohort" | matches T14 |
| L529 | "the **three** cheapest quadrature rungs are on the non-dominated set" | the macro says **5** reduced subjects are non-dominated (3 distinct rungs × tolerances); the sentence and the macro in the next paragraph disagree in form |
| L659 | "its **top two** rules single-draw" | matches T09d |

**Fix:** make each a macro, or accept the exception and delete the word "every" from the
Reproducibility statement.

**N6 (MINOR). `None` leaks into a printed table.** `T05_panel_all` prints the rule column
as `eqcert $m{=}2048$, None` for `q128_M576_eqcert_*` and `q256_M1088_eqcert_*` (four rows),
and `T03_tunability` prints a bare `—` in the matching status cells. A Python `None`
rendered into the paper's provenance-flagship table is the kind of detail a hostile reviewer
screenshots.

**N7 (MINOR). Nine stale bibliography entries from the rejected submission still print.**
`bib-inline.tex` is a `thebibliography` environment, so *uncited* `\bibitem`s appear in the
reference list. Nine are cited nowhere in the compiled document:
`Dosovitskiy2021ViT`, `Vaswani2017Attention`, `KoldaBader2009CP`, `Novikov2015TT`,
`Xiong2020PreNorm`, `Luo2016ERF`, `LoshchilovHutter2017SGDR`,
`LoshchilovHutter2019AdamW`, `Virtanen2020SciPy`. The first three are the ViT + attention +
canonical-polyadic references of the *previous* architecture, which this paper no longer
uses. A reference list containing "Attention Is All You Need" and a CP-decomposition paper
in a paper with no transformer and no CP decoder is both a loose end and a pointer to the
prior submission.

**N8 (MINOR). `b-qxm` is read unpinned.** `GIT_PINS` in `gen_tables.py` pins b-panel
(`13ddecac`) and b-seeds (`e533b48e`); `qxm_analysis` and `qxm_report` are read from the
working tree by absolute path. The bytes currently equal `b4e38103`
(sha256 `28819214…`, and `provenance.json` records it), so nothing is wrong today — but the
paper's headline tunability numbers are the ones read without a pin.

## 1.4 Cost-ratio audit — is every ratio same-job?

**Every cost ratio printed in the main text is formed inside one job.** Verified leg by leg:

| ratio | numerator job | denominator job | verdict |
|---|---|---|---|
| `nQxmCostSpan` 5.16× | 3780177 | 3780177 | same job ✓ (the paper's "inside one job" is correct; `nQxmAnchorJobs` "3780175 / 3780177" is an *anchor pair*, not the provenance of this ratio) |
| `nPanelCheapestRatio` 4.48× | 3789570 | 3789570 | ✓ |
| `nPanelTenTwentyFourCheapestRatio` 1.82× | 3789572 | 3789572 | ✓ |
| `nPanelEqSpeedupZero/OneTwoEight` | 3789570 | 3789570 | ✓ |
| `nPanelTolSavingZero/OneTwoEight` | 3789570 | 3789570 | ✓ |
| `nPanelEqtopTwinSpeedupMin/Max` | 3789570 | 3789570 | ✓ |
| `nEqtopLadderErrSpan/CostSpan` | 3780164 | 3780164 | ✓ |
| `nLshapeNeuralCheaper` 2.77× | 3784663 | 3784663 | ✓ |
| `nLshapeNeuralCheaperFiveTwelve` 7.60× | 3789568 | 3789568 | ✓ |
| `nLshapeNeuralCheaper{SixtyFour,OneTwentyEight}` 0.45 / 0.88× | 3784662 | 3784662 | ✓ |
| `nWaveBankOverHeadCost*` 158/129/42× | one job per mesh | same | ✓ (but see N2 on the cost quantity) |
| `nMeshBurgersFomOverRomRange` 0.265–0.495 | one job | same | ✓ |

**Three ratio-shaped statements that are not covered by that discipline and should be
flagged in the text:**

**M1 (MAJOR). "the cheapest reduced query falls from 4.48× to 1.82×" (L590–594) compares
two ratios across two meshes, two jobs, two GPU models and two different arm types.**
The 256² numerator is `q0_M64_eqcert_g1em06_fastL4` — the hand-ported **fast-kernel** arm at
tolerance 10⁻⁶ on an **A100 80GB**; the 1024² numerator is `q0_M64_eqxfer_g0p001` — a
**loose-tolerance** arm with a **transferred** EQ rule on an **H200**. The denominators are
the same-named FOM arm on different hardware. Every ingredient of the comparison changes at
once, and the sentence is the paper's only forward-looking "it gets better with mesh"
statement. The paper hedges with "we do not call that a crossover until the 512² panel now
queued brackets it", which covers the *conclusion* but not the *measurement*.
**Fix:** either quote 4.48× and 1.82× as two independent facts with their GPUs named in the
sentence, or wait for `bpn401` (3805065, same GPU model as 256²) and report the ladder on
one hardware family.

**M2 (MAJOR). The L-shape headline ratio is measured against the most expensive full-order
arm in the job, and the cheaper one is never mentioned in the main text.** At 512², the same
job 3789568 contains `fom_cg_gpu_r0.01` at **29.127 ms and 0.385 % error**, which is both
cheaper than sparse direct (36.505 ms) and **on the non-dominated set**. Against the
cheapest full-order comparator actually present, the head's margin is **6.07×, not 7.60×**,
and the full-order arm is **5.5× more accurate**. §4 (L566) says the L-shape comparators are
"sparse direct and IC(0)-PCG"; §5.6 (L803–806) says "the full-order comparator is a sparse
direct solve" and never names CG. The macros `\nLshapeFiveTwelveCgMs` / `…CgErr` exist in
`numbers.tex` and are **used nowhere in the compiled document** — the CG row is visible only
inside Table `tab:lshape-solve` in the appendix.
**Fix:** name CG in the §5.6 sentence and quote both margins.

**M3 (MAJOR, and it cuts both ways). The "pre-registered bar" as quoted is not the bar in
the lane's DESIGN, and `M=1088` *was* pre-registered.** Two separate errors in one
paragraph (§5.2, L688–708):
- L697–700: "$M=1088$ was chosen as the headline after the grid ran, by the lane's rule
  (the largest fixed $M$ holding every rung to $q=256$), **not pre-registered**."
  `b-qxm/DESIGN.md` (line 3: "Pre-registered before any job was submitted") pre-registers
  `M=1088` **twice**: §3's grid table names the column
  "`fixed1088` | 1088 = 4(K+256) | … the only fixed M in the scheduled ladder's range that
  holds every rung: the **pure-rank ladder**", and §6's headline rule is
  "**Headline the fixed-M ladder** (M = 1088, pure rank) if it is monotone in q on the
  evolved metric, every rung converged, and $S_q(1088) \ge 2\times$."
  The paper **understates its own pre-registration** and hands a reviewer a post-hoc-selection
  objection it does not have to answer.
- L633–635: the bar is quoted as "monotone, at least three non-dominated points, at least
  $2\times$ on both axes, nothing early-stopped". DESIGN.md §6 requires only *monotone,
  every rung converged, error span ≥ 2×*. The cost-span ≥ 2× and ≥ 3 non-dominated clauses
  come from `reports/generate_xm.py` and are inherited from an earlier campaign; the phrase
  "knob bar" does not appear in `b-qxm/DESIGN.md` at all. So the four-part bar the paper
  calls "pre-registered" is, in this lane, two parts pre-registered and two parts code.
  (The ladder passes either version: 2.437× error, 5.163× cost, 4 non-dominated.)
**Fix:** quote DESIGN §6's bar verbatim, say `M=1088` was pre-registered, and state
separately that the generator applies two extra clauses inherited from the earlier campaign.

---

# 2. Claim audit

Verdict key: **supported** = the tables say exactly this; **over-claimed** = the tables say
something weaker; **contradicted** = the paper's own tables or prose say the opposite.

## 2.1 Abstract, sentence by sentence

| # | claim (abstract, L68–99) | verdict |
|---|---|---|
| A1 | "Neural operators … **deliver one (accuracy, speed) point per trained model and offer no way to tune either at deployment**." | **CONTRADICTED by §5.1 of this paper.** L642–645: "The operators' own knob, evaluation resolution, is usable for `fno-large` only …; **''one accuracy–cost point per trained operator'' is withdrawn**." Table `tab:resolution` shows `fno-large` buying 2.12× its own speed at 1.14× its own error in one job. This sentence is verbatim from `old-neurips-main.tex:47–48`. See **B1**. |
| A2 | "exposes a family of accuracy/cost operating points from a single trained decoder, controlled at inference time by one primary knob … and three solver-side knobs" | supported (Tables `tab:tunability`, `tab:knobs`) |
| A3 | "the trained-once family is **monotone in $q$ on Burgers on each of three training seeds** and, with the test count held fixed so that $q$ is the only control, **meets a bar fixed before any run**" | **OVER-CLAIMED — the sentence splices two different experiments.** Monotone-on-3-seeds is Table `tab:seeds` (b-seeds), the **scheduled** ladder $M=4(K+q)$ — where $M$ moves with $q$, so $q$ is *not* the only control. Meets-the-bar is Table `tab:qxm` (b-qxm), the **fixed-$M{=}1088$** ladder, which is **single seed**. On the three seeds the knob bar is met on **2 of 3** (`nSeedsKnobBar` = 2; `seed3` fails: not every rung converged, Table `tab:seeds-verdicts`). Read plainly, A3 says the bar is met on three seeds. It is not. See **B2**. |
| A4 | "its quadrature rules are **validated on reachable states**, not by their fitting residual" | **OVER-CLAIMED.** True of the *procedure*; false of the *rules the paper ran above q=32*. Table `tab:replication`: confirmed at $q=0,16,32$ only; $q=64$ marginal (2/6), $q=128$ marginal (4/5), **$q=256$ marginal (1/5) with 0/4 of the independent re-draws passing either bar** (ρ = 0.1820, 0.1899, 0.1299, 0.2421 against a 0.116 bar). "Validated" with no qualifier in the abstract, against "never called certified" in §5.3, is not the same paper. See **B3**. |
| A5 | "its wins are cost-only and measured in the same job: 2.77× and 7.60× cheaper than a sparse direct solve on an L-shaped domain at 256² and 512²" | ratios **supported and same-job** (traced, #1–2). But see **M2** (a cheaper full-order arm, CG, sits in the same job and is not named) and **B4** (plain POD matches or beats the head's margin in the same table). |
| A6 | "and non-dominated on the evolved-times metric at 1024² on Burgers" | supported (5 of 20 admissible reduced subjects, job 3789572) — but the paper's own status block calls this **provisional** until `bpn401` lands, and the abstract carries no such marker. On the all-times metric it is **0 of 20**. |
| A7 | "Nothing reduced is on the frontier at 256², where a tuned full-order solver and a neural operator trained on the same data are both cheaper and more accurate" | supported (`nPanelReducedNonDomEvolved` = 0 of 39). Unusually honest; keep it. |
| A8 | "on Poisson, heat and waves the family collapses to a linear model, which says when the nonlinear manifold is worth having" | the collapse is **supported**; "which says when the nonlinear manifold is worth having" is **over-claimed and self-contradicted** — see **B5**. |

## 2.2 Contributions (L149–185)

| # | claim | verdict |
|---|---|---|
| C1 | "the Burgers 256² ladder is monotone with every rung converged, spanning 2.44× in error for 5.16× in cost inside one allocation and meeting a bar fixed before any run" | numbers supported and same-job. Omits that it is **one seed** and that the *cost* clause of the quoted bar is not in the lane DESIGN (**M3**). |
| C2 | "at matched $k=16$ … the neural head reaches 2.5629 % where the best linear map reaches 56.9296 % and POD-16 61.6503 %" | **supported, and it is the paper's strongest result** (T06a, job 3711424, one bank, one dimension). This is the only >20× positive effect in the manuscript. |
| C3 | "We identify the three constraints a decoder must satisfy … and show that a separable decoder … meets all three." | **ASSERTED, not shown.** No skip ablation (the paper says so, L462: "The skip is a design choice; we do not ablate it in this paper"). "Meets all three" is demonstrated for EQ compatibility and per-node evaluation; cold-start is supported only by Table `tab:coldstart`, which varies the *starting point* on Poisson, not the skip. Round-1 finding 9; disposition: "partly fixed", by rewording. Still the weakest contribution. |
| C4 | "a rule that a fitted quadrature is accepted only by its held-out error … and by independent re-draws, never by its fitting residual" | the **rule** is supported and is a genuine contribution (Fig. `fig:cert`, Table `tab:eqrules`, ~100 rules). But the paper's own headline EQ ladder **violates** it above $q=32$ and says so only in §5.3. Stating a rule and then reporting numbers that fail it is defensible only if the abstract says so; it does not (**A4**). |
| C5 | "Where the family is not worth having, reported as such" | supported, and a real strength. |

## 2.3 Conclusion (L845–865)

| # | claim | verdict |
|---|---|---|
| D1 | "This is the lever neural operators do not expose: **each trained model delivers one fixed operating point**." | **CONTRADICTED by §5.1**, same as A1, verbatim from `old-neurips-main.tex:194`. See **B1**. |
| D2 | "its wins are cost-only, on the L-shaped domain where no fast transform applies and, on the evolved-times metric, at 1024²" | supported modulo **M2**, **B4**, and the provisional status of the 1024² statement. |
| D3 | "**The trade is real where the residual is nonlinear in the coefficients and no fast transform makes the full-order solve cheap**, and it collapses where neither holds." | **CONTRADICTED by the paper's own cell.** The L-shape — the *only* cell where the paper claims the trade exists — is **Poisson**, a **linear** residual (`lshape/DESIGN.md` §1–2: "Poisson on the L-shaped domain", $A=-\Delta_h$, SPD). The authors know this: `WRITING-STATUS.md` records "the L-shape is Poisson (linear residual) and is written as the no-fast-transform case". §5.5 (L794–797) states the condition as requiring **both**; §5.6 then produces a win on a cell satisfying only the second; the conclusion re-states the conjunction as if it were satisfied. See **B5**. |
| D4 | "Natural next steps are a harder Burgers regime at lower viscosity, a Navier–Stokes head that beats POD held-out, and unstructured meshes." | fine. |

## 2.4 The six specific claims you asked about

**B1 (BLOCKER). The abstract, the introduction and the conclusion all assert the operator
premise that §5.1 explicitly withdraws.**
Three sites, all inherited verbatim or near-verbatim from `old-neurips-main.tex`:
- abstract L68–71 ≙ `old-neurips-main.tex:47–48`;
- intro L107–112 ("each trained model produces a single fixed (accuracy, wall-clock)
  operating point: there is **no** deployment-time knob a practitioner can turn") ≙
  `old-neurips-main.tex:83`;
- intro L177 ("deliver one fixed (accuracy, speed) point per trained model") ≙ `:194`;
- conclusion L854–855.
Against §5.1 L642–645, which says the claim is withdrawn, and Table `tab:resolution`, which
shows the FNO's own resolution knob working. `REVIEW-ROUND1-DISPOSITION.md` row 5 records
that this clause **fired** on 2026-09-17 (job 3787189) and that the abstract was rewritten
to "A learned PDE surrogate is fixed once trained…"; the 2026-09-17 rebuild from the old
source **reverted it**. This is the single most damaging thing in the paper: a reviewer who
reads the abstract and then §5.1 concludes the authors are overselling, which poisons every
other honest disclosure.
**Fix:** restore the disposition's replacement framing. The defensible version is narrow and
still interesting: *$q$ changes what the model can represent; a coarser evaluation grid does
not, and it works on one of four operator families we tested.*

**B2 (MAJOR). Tunability: "monotone on three seeds" and "meets a pre-registered bar" are
different ladders, and the bar is met on 2 of 3 seeds.**
Facts: `tab:seeds-verdicts` — seed1 knob bar **yes**, seed2 **yes**, seed3 **no**
(`seed3` at $q=256$ did not converge). `tab:seeds` is the **scheduled** $M=4(K+q)$ ladder.
The headline 2.44×/5.16× is the **fixed-$M=1088$** ladder, job 3780177, **one seed**
(the incumbent checkpoint). §5.2 L705–708 states this correctly. The abstract does not.
Second, undisclosed, problem: **the incumbent checkpoint is better than all three fresh
seeds at every rung $q=16\ldots128$** on the evolved metric (`tab:seeds`: 1.3985 vs
1.5988 ± 0.0995; 1.2336 vs 1.3727 ± 0.1314; 1.0843 vs 1.2130 ± 0.0580; 0.8930 vs
0.9562 ± 0.0264 — roughly 1–2 sd better each time). `WRITING-STATUS.md` records that the
sentence "the incumbent is inside the seed spread" was therefore *not written*. Nothing in
the paper tells the reader the hero checkpoint is a favourable draw.
Third, in the *other* direction — the paper again understates its pre-registration.
`b-seeds/DESIGN.md` §C4 reads: "**C4 — the knob bar, secondary.** qtd02's criterion —
converged non-dominated set of `dense_m4` spanning ≥2× in evolved error **and** ≥2× in cost
— holds on **at least 2 of 3 seeds**." So 2 of 3 *is* the pre-registered pass. And seed3
fails only on **convergence**, not on tunability: its spans are 3.93× error and **20.97×**
cost, the largest of the three; the single unconverged cell is `q=256 / M=1088` with 6
budget exits (job 3783778). The paper reports "meets the knob bar on 2 of 3" as if it were
a partial result.
**Fix:** split the abstract clause in two; say in §5.2 that 2 of 3 is the pre-registered
criterion and that seed3's failure is a convergence failure at one rung, with its spans
quoted; and add one sentence to §5.2 or Limitations saying the incumbent sits at or beyond
the top of the three-seed spread at $q=16$–128.

**B3 (MAJOR). Certification of quadrature: single-draw vs certified — and the status column
is misleading in both directions.**
There are **three** rule populations, and the paper's tables blur them.

| rung | rule the b-eqtop **ladder** ran (`tab:eqladder`, Fig. 1C, job 3780164) | rule the **panel** ran in the "replication-selected" column (`tab:tunability`, job 3789570) |
|---|---|---|
| q=128 | m=**2048**, 64 fit states, sha `e20f7de7…` — **re-drawn 4 times, 4/5 pass**, status "marginal at m=2048 (4/5)"; the lane marks this file `"superseded", "removed": true` | m=**2319**, sha `27ae705a…`, ρ_max 0.0277, **`draws: 1`, never re-drawn** |
| q=256 | m=**2048**, 64 fit states, sha `6e4e4de2…` — **re-drawn 4 times, all four FAIL both bars** (0.1820, 0.1899, 0.1299, 0.2421 vs a 0.116 bar), status "marginal at m=2048 (**1/5**)"; also `"removed": true` | m=**2560**, sha `3603d6e1…`, ρ_max 0.0647, **`draws: 1`, never re-drawn** |

Consequences a reviewer will find:
- **The status column says the wrong thing at both ends.** In `tab:tunability` the *b-eqtop
  ladder rules* column prints status "**—**" at q=128 and q=256 — blank — for precisely the
  two rules whose construction **is** known to be marginal (4/5 and 1/5). The
  *replication-selected* column prints "**certified in one draw**" for two rules that have
  **never been re-drawn at all**. The table is blank where a negative verdict exists and
  says "certified" where no replication was run. In `tab:panel-all` the same two eqcert rows
  print the literal Python `None`.
- **§5.2 L659 "its top two rules single-draw and never called certified" is contradicted by
  the table on the facing page**, which prints the word "certified" for exactly those two
  rules. The b-panel DESIGN §A8 instructs the opposite: "the report says 'single-draw at
  q ≥ 64' rather than 'certified' wherever those rungs appear." The prose complies; the
  generator does not.
- **"Marginal" is doing too much work.** q=128 (4/5 pass) and q=256 (**0 of 4 re-draws
  pass**) get the same adjective. A construction none of whose fresh draws passes is not
  marginal; it is not reproduced.
- The bars themselves are sound: `b-eqtop/DESIGN.md` §5 fixes primary 0.116 and tight 0.06,
  byte-identical from the file's first commit `33e8bced` (2026-09-17 01:11) through the
  pinned `8542c604`, and every b-eqtop job (3780164 at 10:48, 3780165, 3783811 at 12:01)
  post-dates it. The 0.116 value's own origin is `q-ridge/DESIGN.md` amendment A3, commit
  `a1822ae6`, 2026-09-16. Pre-registration is genuine; only the *transcription into the
  generator* is hand-typed (**N1**).
- The abstract's unqualified "its quadrature rules are validated on reachable states" covers
  none of this.
**Fix:** (i) change the `rule_status` string at source — "one passing draw, no re-draw" for
the m=2319/2560 rules, "0 of 4 re-draws pass" for q=256 at m=2048; (ii) fill the blank
status cells with the marginal verdicts that exist; (iii) add "at q ≤ 32" to the abstract's
validation clause.

**B4 (MAJOR). The L-shape cost wins: a linear-residual cell, an expensive comparator, and a
POD baseline that does at least as well.**
Three separate problems with the paper's only positive headline.
1. *Linear residual.* See **B5**/D3. By the paper's own structural condition the trade
   should not exist here.
2. *Comparator selection.* See **M2**: CG 10⁻² in the same job at 512² is 29.127 ms
   (vs splu 36.505), so the honest margin is 6.07×, and CG is 5.5× more accurate.
3. *The baseline matches it.* In the **same table** (`tab:lshape-solve`, same jobs):
   | mesh | head $q{=}64$ | POD $k'{=}128$ | splu | head margin | POD margin |
   |---|---|---|---|---|---|
   | 256² | 2.131 % @ 3.028 ms | 2.4575 % @ 2.850 ms | 8.386 ms | **2.77×** | **2.94×** |
   | 512² | 2.123 % @ 4.801 ms | 2.4511 % @ 4.031 ms | 36.505 ms | **7.60×** | **9.06×** |
   A plain linear POD ROM is **cheaper than the head at both meshes and beats the head's
   own cost-win ratio**, for 15 % more error. The nonlinear head buys 13 % lower error for 6 % (256²) to 19 % (512²)
   more cost. The abstract attributes a "win" to the method that its simplest
   baseline gets more cheaply; §5.6 says only "[the head] is the most accurate reduced
   subject on the set", which is true and not the point. The POD-128 figure appears in the
   appendix (`app:extended:linear`) only for 512², and the 256² POD comparison nowhere.
4. *A fourth, smaller one:* `tab:lshape-free` shows the **linear** free rung $q=R$ at
   **0.7791 %** — 2.7× more accurate than the head rung — in a separate job. Even inside the
   L-shape cell, the most accurate reduced subject is the linear one.
**Fix:** put the POD-128 row in the §5.6 sentence and state the margin as "2.77× (POD-128:
2.94×)". If the claim is that the *nonlinear* part earns its keep, the honest statement is
"13 % lower error than POD-128 at 6–19 % more cost", and that is a much weaker paper — which
is a reason to reframe, not to hide it.

**B5 (MAJOR). The structural condition is stated as a conjunction and then violated by the
paper's own positive cell.**
§5.5 L794–797: "the correction-rank trade needs **both** a residual nonlinear in the
coefficients **and** a manifold that beats POD held-out; Poisson, heat and waves lack the
first, this cell [NS] the second, and Burgers on the square had both yet still lost on
cost." Conclusion L862–864 repeats it. Then §5.6 L799: "**On the L-shaped domain the trade
does exist**" — on Poisson. Either the condition is wrong, or the L-shape is not an instance
of the trade (it is a cost win for *reduced models generally*, POD included — see B4).
Also note the condition is close to tautological as stated ("it works where a linear model
is not enough and a fast solver is not available"), which round-1 finding 10 already said.
**Fix:** drop "nonlinear in the coefficients" from the condition and state the real one the
data support: *the reduced solve wins on cost only where no fast transform exists; whether
the nonlinear head or a linear POD basis fills that slot is decided by the bank, and on our
cells POD usually wins.* That is a smaller claim the tables actually make.

**B6 (MODERATE). The 1024² non-dominated statement.**
Supported as written (evolved metric, 5 of 20, one job). Three qualifications the paper
carries in §5.1 but not in the abstract: (i) all-times = **0 of 20** at both meshes;
(ii) the transferred rules at $q\ge64$ miss the primary bar at 1024², so the non-dominated
rungs are exactly the three whose rules are confirmed; (iii) the paper's own status block
calls it provisional until the 512² panel lands. Also **M1**: the 4.48→1.82 trend that
motivates the paragraph changes mesh, job, GPU model and arm type simultaneously.

**B7 (MODERATE). Neural-operator comparison.**
Supported in substance: `tab:operators` shows unet-small 1.4712, unet-medium 1.5189,
tsol-refine 1.5224, unet-refine 1.7110 all below ROM 1.8671 (`nOpArmsBeatingRom` = 4 ✓), and
every FNO capacity above it (2.4829…3.2660). The matched eight-case cohort is genuinely
identical: `cohort_index_sha256 = 8b8a2ee1…` is the same value in the U-Net job (3780138),
the Transolver job (3780139), the control job (3783831) and the parent FNO job (3710846).
The budget is genuinely equal *wall*: measured `training_seconds` 3001.2–3007.2 for every
arm. Five things a reviewer will press:
- **"every operator number is a lower bound" (L639) is false, and the paper's own
  Limitations says so.** `tsol-refine` has `still_improving: false` (best epoch 1542 of
  1628 = 0.947 < the 0.95 threshold); limitation (v) correctly says "**7 of 8**". And
  `tsol-refine` is one of the four arms that beat the ROM. **Fix:** "all but one operator
  number is a lower bound", or drop the sentence and point at limitation (v).
- **The epoch confound runs in the operators' favour and is not stated.** Equal wall in
  float32 gave the new families 2.3–2.8× the float64 FNO's epochs (U-Net 908–2327,
  Transolver 586–1628, FNO 692–1741). The lane records this ("favourable to them"); the
  paper does not. This makes the *concession* (operators beat the ROM) safe, but it also
  means the one positive statement in the paragraph — "every FNO capacity is worse than it"
  — is the one the confound inflates. That sentence is not worth making next to
  `nOpFnoImproving` = 4 of 4.
- **The four arms that beat the ROM are untimed against it.** Only `fno-large` shares an
  allocation with the ROM (7.2 ms in job 3789570). Correctly handled in the text (L560–562);
  keep it that way.
- **Two FNO errors, unexplained, four paragraphs apart.** `tab:operators` gives `fno-large`
  = 2.4829 % (8-case cohort, job 3710846); `tab:panel-all` gives `fno-large` = 7.4164 %
  (6 development cases, worst evolved, job 3789570); the main text quotes both (L610, L635)
  with no word about why they differ by 3×. **Fix:** one clause naming the cohort in each.
- **Controls are real and they hold.** Job 3783831: a second-seed U-Net twin lands at
  1.5479 % (vs 1.5189 %) and a float64 twin at 1.6721 %, both still below the ROM. The
  float64 twins ran 0.25× and 0.42× of their parents' epochs at equal wall, so precision is
  confounded with epochs — `app:extended:operators` says this. The Transolver arm that beats
  the ROM has no twin at all. Accurate as written.

**B8 (MODERATE). The Navier–Stokes negative.**
Solid, and better than I expected. The 2.0× bar is pre-registered in the ns2d DESIGN at
commit `1281ff73` (2026-09-17 00:55), before jobs 3783796/3787319/3787320. Both arms fail
(1.19×, 1.15×). The budget exits are 12 (K=16) and 18 (K=32) of 384 held-out states, so the
oracle values are **upper bounds on the error**, which biases the ratio **downward** — i.e.
the negative verdict is conservative, not flattered. `tab:ns`'s caption carries this.
Two defects:
- The main text (L787–793) states the result as clean and never says the oracle is an upper
  bound; the qualifier lives only in the table caption and in the lane DESIGN §A8.
- **The caption of `tab:ns` is stale**: `appendix.tex:456–464` says "phase-2 gates on the
  **$K=16$** head over the full-rank **$R=256$** bank … The $K=32$ arm on the same bank
  (ns204) is **pending**", while the table it captions now contains six rows including three
  $K=32$/$R=512$ rows. The commit message says "ns2d closed"; the caption did not get the
  memo.
- Minor: `nNsKthirtyTwoPodOverOracleTzero` = 0.82, quoted at L791–793 as the "clean case of
  linear beating nonlinear", appears in **no printed table** (`tab:ns` has no $t{=}0$ column).

## 2.5 Does the title's "Elliptic, Parabolic, and Hyperbolic" hold?

| class | representative | evidence | verdict |
|---|---|---|---|
| Elliptic | Poisson (square + L-shape) | `tab:linear` (job 3783813), `tab:lshape-solve`, `tab:head-poisson` | **yes** |
| Hyperbolic | Burgers, reflective wave | the whole panel, `tab:waves` (job 3783805) | **yes** |
| Parabolic | heat 2D | **`tab:heat` only** — job **3511417**, commit `73fdaa88`, a **2026-09-10 report from a different campaign**, captioned "an earlier cell of the same decoder family", with $k=8$, $R=32$ (against $K=16$–32, $R=128$–512 everywhere else), twelve development cases, three meshes | **thin** |

The parabolic leg rests on one legacy table from a different checkpoint family, and that
table shows the method **losing badly**: the nonlinear head reaches 4.56 % at ~12 ms while
the linear bank reaches 1.68 % at 0.12–0.56 ms and the same-grid direct solve reaches
0.0004–0.09 % at 0.18–1.03 ms. The heat cell contributes one clause to §5.6 ("Heat …
behave[s] the same way") and one sentence in `app:extended:linear`. A reviewer who checks
will say the title's second adjective is carried by a table the paper itself brackets as
"earlier". **Fix:** either re-run heat on the current checkpoint family, or retitle to
"Elliptic and Hyperbolic" and keep heat as a supporting appendix cell.

---

# 3. Round-1 follow-up: all 30 findings at `d131ec11`

The disposition was written against commit `20c10e8b`. `d131ec11` **rebuilt `main.tex` from
`old-neurips-main.tex`**, so several round-1 fixes had to be re-applied by hand and one was
lost. Verdicts below are against what `d131ec11` actually compiles, not against the
disposition's claims.

| # | round-1 finding | at `d131ec11` |
|---|---|---|
| 1 | page budget (main text to p. 13) | **RESOLVED** — main text ends at the foot of p. 9; refs p. 10–18; appendices p. 19–33. But every table is now in the appendix (see R1 below). |
| 2 | unfilled `[pending: …]` placeholders in the text | **PARTIAL** — none in the main text; **Table 22 (`tab:sealed`, p. 31) is still a red `[ pending: b-seeds sealed cohort ]` box**, `T13_sealed.tex` being one `\gen{}` call. A placeholder table in a submitted appendix reads the same way as one in the body. |
| 3 | double-blind / rebuttal framing | **PARTIAL** — no reviewer IDs, no "resubmission", no Appendix D in any compiled file, and `PAPER.md` is clean. But (a) the **title is a two-word edit of the rejected NeurIPS title** ("…for Elliptic and Parabolic PDEs via Matrix-Free Galerkin Projection" → "…Elliptic, Parabolic, **and Hyperbolic** PDEs via Matrix-Free **Petrov–**Galerkin Projection"), which round-1 said was itself the search key; (b) `paper/old-submission/main.tex` and `paper/REVIEWER-RESPONSE-MAP.md` (11 reviewer-ID hits) are committed in the same `paper/` directory the Reproducibility statement promises to release anonymously; (c) the `main.tex` header comment (L1–26) says "Rebuilt 2026-09-17 from the previous submission's source" — harmless in the PDF, live if the `.tex` is supplied. See R3. |
| 4 | heat claimed but not tabulated | **RESOLVED as tabulated, THIN as evidence** — `tab:heat` (T11d) exists, job 3511417, but it is a legacy cell with $k=8$, $R=32$ from a different campaign, and it shows the head losing to its own linear bank by 2.7× and to the direct solve by ~50×. See §2.5. |
| 5 | the operator premise is one the authors pre-committed to withdraw | **REGRESSED — this is the worst item in the review.** The disposition records the clause firing and the abstract being rewritten. The rebuild restored the old sentence in **three** places (abstract L68–71, intro L107–112 and L177, conclusion L854–855) while §5.1 L642–645 still says the claim is withdrawn. See **B1**. |
| 6 | generator keyed on arm name, Table 7 corrupted | **RESOLVED** — `T14_operators` is keyed on (arm, job); FNO rows show job 3710846 with 692/1198/1741 epochs; `nOpFnoImproving` = 4 of 4, `nOpLaneImproving` = 7 of 8, both verified against the lane. |
| 7 | wave prose contradicted by its own table | **RESOLVED** — `app:extended:linear` now says the top rung "is not strictly the most accurate rung (strict reading: `nWaveTopStrictBestTwoFiftySix`) but matches the q=32 rung within the pre-registered integrator tie band", with the band as a macro. |
| 8 | cross-job cost inference for U-Net/Transolver | **RESOLVED** — L560–562: the FNO is timed in the same allocation, "the U-Net and Transolver are not, so no cost statement is made for them"; `tab:operators` has no timing column. |
| 9 | Contribution 2 asserted, not shown | **PARTIAL, as disposed** — the skip is now declared a design choice and explicitly not ablated (L462); the block-damped claim cites `tab:solver-variants` (job 3734098, `nBlockJointBudgetExits` = 6 vs 0); the Jacobian-rank claim is replaced by the `valid` column of `tab:linear`. No ablation was run. |
| 10 | structural condition confounded / near-tautological | **PARTIAL and now self-contradicted** — the confound is stated (§5.6 L779–782, "three to four times worse than a POD basis of the same rank"), but the condition is now asserted as a conjunction in §5.5 and the conclusion and violated by the L-shape cell in §5.6. See **B5**. The POD-bank ladder control was not run. |
| 11 | which PDEs, ICs, RHS | **RESOLVED** — `tab:spec` (T01b) gives $\nu \sim \log U(0.01,0.1)$, the Gaussian IC ranges, the Poisson source family, $\kappa = 0.02$, the wave bump family and $c \sim U(0.85,1.15)$, each with its generator source file. Note the **main text** still carries no $\nu$; a reader must reach p. 21. |
| 12 | harder problems not delivered | **PARTIAL** — L-shape now has a full solve layer at four meshes (`tab:lshape-solve`), NS has a closed pre-registered negative at both $K$ (`tab:ns`). Still 2D, still uniform Cartesian, still nothing harder than viscous Burgers solved successfully. Limitation (ii)/(iii) says so. |
| 13 | no anonymous code / checkpoints | **PARTIAL / deferred** — the Reproducibility statement now promises "An anonymised repository with the generator, the provenance registry, the lane summaries and the audit JSONs, and the checkpoints by hash, **accompanies the submission**" (present tense). Whether it exists is outside the paper; see R3 on what must not be in it. |
| 14 | selective fixed-$M$ reporting | **RESOLVED, then over-corrected** — `tab:qxm` shows both ladders and §5.2 states that $M=256$ fails and $M=1088$ passes. But the paper now says $M=1088$ was "not pre-registered" when the lane DESIGN pre-registers it twice. See **M3**. |
| 15 | metric-dependent head claim | **RESOLVED** — the matched-dimension result is the claim (L750–754); both metrics for POD-128 are in `app:extended:head` (all-times 10.1198 %, evolved 1.9464 % vs the q=0 rung's 1.8890 %). |
| 16 | prior art on hybrid linear/nonlinear reduction | **RESOLVED** — §2 ¶1 L223–231 positions against Barnett–Farhat–Maday, Peherstorfer–Willcox, Carlberg, Fresca–Manzoni, and states what is new narrowly ("nested prefix chosen offline, run-time selection without refit") and what is not ("we claim no novelty for either"). |
| 17 | offline cost never given | **RESOLVED** — `tab:offline` (T17) from the lane JSONs: head training, direction fit, NNLS per rule. Note the main text never quotes it; a reader has to notice that a "free" family costs ~1.7 h of offline NNLS. |
| 18 | no results figure; Fig. 1 illegible | **PARTIAL** — Fig. 1 is now the three-panel family figure in §5.1, which is the right call. The architecture diagram moved to Appendix B. But the paper now has **one figure and zero tables in nine pages of main text**; see R1. |
| 19 | overfull tables | **PARTIAL** — `main.log` still reports **7 overfull hboxes** (up to 160 pt) and one **"Float too large for page by 371.12 pt"** at `appendix.tex:321`, i.e. `tab:eqrules` (T09c, Table 24, the ~100-rule table). That table will not fit its page. |
| 20 | two "solved" numbers for one Poisson cell | **RESOLVED** — `app:extended:linear` gives the $M=129$ solve (3.1495 %) and `tab:layers` the $M=257$ solve (3.1146 %), with $M$ named in both. |
| 21 | provenance-table defects | **PARTIAL** — (a) the wave row is fixed; (b) the panel row still prints "incumbent (gate `checkpoint_unchanged`; hash in T2, tuning row)" instead of a hash, against the Reproducibility statement's "Each run's … checkpoint hash are in Table 8"; (c) declared in the table headers. **And a new one:** `\provLshapeJob` prints job `3784662` for `tab:lshape`, whose rows are all job `3783786` (**N3**). |
| 22 | origin of the 0.116 bar | **PARTIAL** — §3.4 now states the origin via `\nEqtopBarOrigin`, and it checks out against the b-eqtop DESIGN. The value is still a typed literal in the generator (**N1**). |
| 23 | cross-job spread calibration | **RESOLVED** — limitation (vi) prints "cross-job spread of an identical cell is 14 %" (`nQxmAnchorSpreadPct`), traced to 742.2 vs 848.0 ms on `q0_M1088` across jobs 3780175/3780177. |
| 24 | L-shape floor quoted on the cohort where it passes | **RESOLVED** — §5.6 L817–820 prints both (development 6.8 %, validation 16.7 %); `tab:lshape` prints the common-cohort floors. But the parenthetical is attached to the wrong quantity (**M6** below). |
| 25 | completion rule changed between jobs | **PARTIAL** — `tab:panel-all`'s caption defines "conv." (the pre-registered rule that accepts an attained initial fit) and "strict"; the main text no longer says that the rule post-dates the head-ablation job. That sentence was in the pre-rebuild `results.tex`, which is no longer compiled. |
| 26 | certification claim stronger in related work than in §5.3 | **RESOLVED** — §2 ¶2 L259–262 now says "a fitted rule is accepted by its held-out error and by re-draws, not by its fit residual", and §5.3 records the $q=64$ counter-example via `nEqtopSpreadMin/Max`. |
| 27 | "36 vs 35 timed subjects" | **RESOLVED** — the count is the macro, now 48 for job 3789570 (the re-run carries both rule sets). Worth noting `WRITING-STATUS.md` records "the bpn301 recheck JSON lists 48 arms, the message said 47" — the ledger is still one behind. |
| 28 | matched-dimension result buried | **RESOLVED** — it is Contribution 1's second clause and §5.4's headline sentence. |
| 29 | "0.18–0.21×" undersells the 5× | **RESOLVED** — §5.2 L711–714 quotes 4.77× and 4.81× at four-decimal-identical error. |
| 30 | certification scatter figure | **RESOLVED** — Fig. 3 (`fig:cert`, Appendix B) plots NNLS fit residual against held-out $\rho$ over every b-eqtop rule with both bars. It is the paper's most transferable result and it is on p. 22. |

**Score: 15 resolved, 12 partial, 2 regressed/thin (5 and 4), 1 resolved-then-over-corrected
(14).** The regression on finding 5 is worth more than the other 29 combined.

---

# 4. Reviewer pass

## 4.1 The five strongest objections to acceptance

**R1 (BLOCKER for the reader, not for the truth). Nine pages of main text contain one
figure and zero tables; the first supporting table is on page 24.**
`main.tex` `\input`s only `macros`, `tables/numbers`, `bib-inline` and `sections/appendix`.
Every `\begin{table}` in the document is inside `sections/appendix.tex`. §5 argues for four
pages against `tab:tunability` (Table 14, p. 24), `tab:operators` (Table 13, p. 24),
`tab:lshape-solve` (Table 38, p. 33) and twenty others, none of which the reader can see.
The round-1 page-budget blocker was fixed by evacuating the evidence. An ICLR reviewer reads
the main text once, linearly; this one asks for fifteen page-turns per paragraph.
**Fix:** bring back exactly three tables and cut prose to pay for them — (1) the 256² panel
non-dominated summary (six rows: cheapest FOM, best FOM, FNO, POD-512, best dense rung, best
EQ rung, with a `vs ref %` column — see R2); (2) the fixed-$M$ ladder, six rows; (3) the
L-shape solve at 256²/512² with POD-128, CG and splu in it. Everything else can stay in the
appendix. The three paragraphs of §5.1/§5.6 that those tables replace are longer than the
tables.

**R2 (MAJOR, substantive, and the objection I would lead a review with). On the physically
meaningful metric the knob does almost nothing, and a cheap full-order setting beats every
rung — and the paper has the column that shows it.**
`tab:panel-all` (Table 16, p. 25) prints `vs ref %` (error against the $4096^2$ reference) for every subject
in job 3789570:

| subject | evolved % (same-grid) | **vs ref %** | GPU ms |
|---|---|---|---|
| ROM `q0_M64_dense` | 1.8890 | **4.5575** | 283.9 |
| ROM `q256_M1088_dense` | 0.5194 | **4.0391** | 3939.8 |
| POD-512 | 0.2184 | 4.0266 | 2790.8 |
| FOM `nt1e-2_dt005` | 3.7127 | **2.4737** | **15.7** |
| FOM `fft_tight` (the reference discretisation) | 0.0000 | 4.0265 | 90.4 |

So the ladder that the abstract sells as **2.44× in error for 5.16× in cost** moves the
*physical* error from 4.56 % to 4.04 % — **1.13×** — because every rung is already below the
mesh's own 4.03 % discretisation error. And a full-order arm at **15.7 ms** reaches
**2.47 %** against the reference, better than every reduced subject in the job at 1/250th of
the top rung's cost. (The `nt1e-2_dt005` figure is partly error cancellation — a coarse time step
happening to offset the spatial error — and the paper is entitled to say so; it cannot say
that the ROM rungs are physically distinguishable from each other or from `fft_tight`, which
is the load-bearing point.) §4 L536–537 states the 4.0265 % figure once, in the setup, and
never returns to it. A reviewer who notices this concludes that the paper's headline knob is a
knob on an error that does not matter at this mesh.
**Fix:** this must be confronted in §5.1, not buried. Either (a) report the ladder at a mesh
where the reduction error is the binding one (the same-grid metric is the right one when the
ROM error is above the discretisation error — on the mesh ladder that is $64^2$–$128^2$,
where `tab:mesh` shows ROM-vs-ref 10.86 % and 6.54 %), or (b) state plainly that the
same-grid metric isolates the reduction and that at $256^2$ no rung is physically
distinguishable, and move the headline to the L-shape or to the matched-dimension result.
Not confronting it is the difference between "honest paper with a negative result" and
"paper that chose the metric that made the curve".

**R3 (MAJOR). Double-blind: the title is a two-word edit of the rejected submission's
title, and the release the paper promises would ship the reviewer map.**
Round-1 finding 3 named the title as the search key ("Anyone who searches 'Tunable
Non-linear Manifold ROMs' finds the authors"). The disposition marked the finding fixed by
removing reviewer IDs from the text; the title change that followed on 2026-09-17 was
**"Elliptic and Parabolic" → "Elliptic, Parabolic, and Hyperbolic"** and **"Galerkin" →
"Petrov–Galerkin"**, recorded in `ABSTRACT-2026-09-17.md` as "Two-word change, everything
else verbatim". The abstract's opening two sentences are also verbatim from that submission.
Separately: `paper/` contains `old-submission/main.tex` (the rejected paper),
`REVIEWER-RESPONSE-MAP.md` (11 reviewer-ID hits), `TABLE-PLAN.md` (9), `WRITING-STATUS.md`
(2) and `ABSTRACT-2026-09-17.md` (4, including "the rejected NeurIPS paper's
three-contribution skeleton"). If the promised anonymous repository is a mirror of this
tree, it de-anonymises the submission and shows a reviewer the response map.
**Fix:** (i) retitle — the title should describe the *finding*, not the old paper; something
like "When Does a Non-linear Manifold ROM Earn Its Keep? A Same-Job Study of a Run-Time
Accuracy–Cost Knob" both anonymises and improves it; (ii) rewrite the abstract's first two
sentences (which B1 requires anyway); (iii) build the anonymous repo from an allowlist of
files, never by mirroring `paper/`; (iv) strip the `main.tex` header comment before
submission (it says so itself, but say it in `build.sh`).

**R4 (MAJOR). The paper's only positive headline is a win for reduced models in general, on
a cell that fails the paper's own stated condition, against the most expensive comparator in
the job.**
This is **B4** + **B5** + **M2** together, and a reviewer will state it as one objection:
*the L-shape shows that at 256²–512² a reduced solve beats a sparse direct solve; POD-128
shows it more cheaply than the neural head does (2.94× and 9.06× vs 2.77× and 7.60×); CG in
the same job is cheaper still than sparse direct at 512²; and the cell is a linear Poisson
problem, which §5.5 says is exactly where the method should collapse.* Nothing in that
sentence requires the paper's decoder.
**Fix:** the L-shape result should be reported as what it is — "no fast transform ⇒ reduced
models win on cost; the head is the most accurate reduced subject and costs 6–19 % more
than POD-128 for 13 % less error" — and the abstract's win clause should be replaced by the
matched-dimension result (C2), which is the paper's only large, clean, uncontested effect.

**R5 (MAJOR). The central contribution is demonstrated on one checkpoint that the paper's
own seed study shows is a favourable draw, and that fact is nowhere in the paper.**
`tab:seeds`: at $q = 16, 32, 64, 128$ the incumbent reads 1.3985 / 1.2336 / 1.0843 /
0.8930 % against seed means 1.5988 ± 0.0995 / 1.3727 ± 0.1314 / 1.2130 ± 0.0580 /
0.9562 ± 0.0264 — the hero is 1–2 sd better than the population at every rung.
`WRITING-STATUS.md` records that the sentence "the incumbent is inside the seed spread" was
therefore **not written**. Every headline in the paper — the 2.44×/5.16× ladder, the panel,
the EQ ladder, the mesh ladder, the head ablation — is that one checkpoint.
**Fix:** one sentence in §5.2 and one in Limitations. It costs the paper almost nothing to
say and it is the single fact most likely to be found by a reviewer who reads Table 30, at
which point everything else in the paper is read differently.

## 4.2 Rejected-paper artefacts

Ranked by how much damage they do.

| # | artefact | evidence |
|---|---|---|
| 1 | **The operator premise, in three places.** | abstract L68–71, intro L107–112 and L177, conclusion L854–855, all ≙ `old-neurips-main.tex:47, 83, 194`; withdrawn by §5.1 L642–645 and `tab:resolution`. See **B1**. |
| 2 | **The title.** | two-word edit of the rejected title; see **R3**. |
| 3 | **Nine uncited `\bibitem`s that still print**, three of them the old architecture's (ViT, "Attention Is All You Need", Kolda–Bader CP). | see **N7**. A reference list advertising a transformer and a CP decomposition in a paper that uses neither. |
| 4 | **Seven orphaned source files** that no longer compile but are still committed and still describe the paper: `sections/{intro,setup,results,limitations,conclusion}.tex`, `methods.tex`, `related-work.tex`. `WRITING-STATUS.md`'s section table still points §5.1–§5.7 at `sections/results.tex`. The one substantive consequence: the sentence round-1 finding 25 asked for (the completion rule post-dates the head-ablation job) lives only in the orphaned `results.tex`. | `git ls-tree d131ec11` vs the four `\input`s in `main.tex` |
| 5 | **Stale caption**: `tab:ns` is captioned "phase-2 gates on the **$K=16$** head over the full-rank **$R=256$** bank … The $K=32$ arm on the same bank (ns204) is **pending**" while the table contains three $K=32$/$R=512$ rows and the commit message says ns2d is closed. | `appendix.tex:456–464` vs `T11e_ns.md` |
| 6 | **`tab:sealed` is a red `[ pending: … ]` box** (p. 31). | `T13_sealed.tex` |
| 7 | Old-submission style residue: `neurips_2026.sty`, `main-skeleton.tex`, `old-submission/` all still in `paper/`. Harmless to the PDF, live for the anonymous release (**R3**). | |

## 4.3 Double-blind

- **Compiled artefacts are clean.** No reviewer ID, no "resubmission", no "NeurIPS", no
  self-citation of the prior work in `main.tex`, `sections/*.tex`, `tables/*`, or `PAPER.md`.
  `\author{Anonymous authors\\Paper under double-blind review}`, `\iclrfinalcopy` not set.
- **Three live risks**, all in **R3**: the near-identical title; the verbatim abstract
  opening; the contents of `paper/` if the promised anonymous repository is a mirror.
- **One minor:** the `main.tex` header comment (L1–26) names `old-neurips-main.tex`, records
  "the title was changed by the user on 2026-09-17", and lists "Open decisions for the user".
  LaTeX comments do not render, but they travel with a `.tex` supplement.
- Check the PDF's `/Creator`, `/Producer` and `/Title` metadata before upload; `latexmk`
  will not have put a name there, but the file path can survive in some toolchains.

## 4.4 Readability for a reader new to the project

The user asked for plain, simple prose. This draft is not that. Concretely:

**The abstract is 314 words in five sentences.** Sentence lengths: 26, 64, 80, **94**, 50.
ICLR abstracts run 150–200 words with sentences under 30. Sentence 4 is:
> "Across 2D Burgers, Poisson, heat and waves, the trained-once family is monotone in $q$ on
> Burgers on each of three training seeds and, with the test count held fixed so that $q$ is
> the only control, meets a bar fixed before any run; its quadrature rules are validated on
> reachable states, not by their fitting residual; and its wins are cost-only and measured
> in the same job: 2.77× and 7.60× cheaper than a sparse direct solve on an L-shaped domain
> at 256² and 512², and non-dominated on the evolved-times metric at 1024² on Burgers."

Three unrelated claims, four qualifications and two numbers in one sentence. It is also the
sentence that commits the splice in **B2**. **Fix:** four sentences, one claim each, and
move the certification clause out of the abstract entirely.

**The abstract uses eight terms it never defines:** "rank $q$ of a linear correction",
"empirical-quadrature (EQ) sample count", "matrix-free least-squares Petrov–Galerkin
projection", "fixed smooth weak tests", "per-node spatial bank", "Fourier-feature coordinate
network", "reachable states", "evolved-times metric", "non-dominated". The glossary that
defines all of them is **Appendix G, page 33**. A reviewer reading the abstract cold
understands roughly the first sentence.

**Other specific offenders:**
- L143–147 (intro): a 60-word sentence listing four pieces of machinery with six citations.
- L472–479 (§3.4 "What is fixed, what is chosen"): one 55-word sentence enumerating seven
  frozen artefacts in parentheses — this is a list, set it as a list.
- L616–627 (§5.1, the 1024² paragraph): the reader must hold "evolved", "all-times",
  "$t=0$ compression", "admissible", "non-dominated" and two job ids simultaneously.
- §5.6's title, "Where the Family Collapses: Linear PDEs, Navier–Stokes, and the L-shaped
  Domain", promises three collapses and delivers two collapses and a win.
- The `\evolved` / `\alltimes` macros italicise two adjectives that appear ~20 times; the
  distinction is never stated in one plain sentence in the main text (it is in §4 L547–553,
  as a subordinate clause, and in the Appendix-G glossary).

**What already works and should not be touched:** "The losses come first." (L583); "Two
plain sentences first." (L588); "In plain terms: the reduced solve becomes the cheaper
option between 128² and 256²" (L800–801); "**The collapse is confounded with a weak bank,
and we say so**" (L779–780). Those four sentences are the voice the rest of the paper should
be rewritten into. The paper knows how to do it; it does it four times in nine pages.

**One concrete suggestion:** put a five-line "Reading this paper" box after the abstract
defining bank / head / rank $q$ / rung / evolved-vs-all-times / non-dominated. It costs eight
lines and it is the difference between a reviewer following §5 and skimming it.

## 4.5 Two smaller things worth fixing

**M6 (MODERATE). The L-shape "optimistic end of a range" caveat is attached to the wrong
quantity.** §5.6 L817–820: "every L-shape error quoted is the optimistic end of a range
(6.8 % worst best-found on development sources against 16.7 % on validation sources)."
Traced: both numbers are `head_arms`, mesh 256, job **3783786**, and they are the **max over
the seven head arms** of each arm's worst **best-found reconstruction** error — an untimed
multistart-LM fit, not a solve. The arm the headline uses, `head_sdf_R512_K16`, is 3.861 %
on development. The headline 2.131 % / 2.123 % is a different metric (`worst_same_grid` for
a solved `neural+linear` subject) on the 32-source development cohort. **No solve number on
the 461-source validation cohort exists anywhere in the lane** — the lane report says so:
"a 461-source solve sweep is the measurement that would replace them, and it has not been
run." The 2.5× factor is an analogy, not a measured bound on the quoted error.
**Fix:** say what it is — "the same banks' best-found reconstruction degrades 2.5× from the
32 development sources to the 461 validation sources; no solve sweep on the validation
cohort has been run."

**M7 (MINOR). `nLshapeNeuralMsTrend` is a three-job curve used as evidence of flatness.**
§5.6 L806–807: "while the head at $q=64$ stays nearly flat (2.85 → 2.81 → 3.03 → 4.80 ms)".
Those four numbers come from jobs 3784662, 3784662, 3784663 and 3789568. Each *ratio* in the
paper is same-job, but this *trend* is not, and the 2.85 → 2.81 dip is inside the 14 %
cross-job spread the paper itself reports in limitation (vi). The 64² and 128² head entries
also appear in no printed table (`tab:lshape-solve` prints non-dominated subjects only, and
the head is not on the set at those meshes), so a reader cannot check them.
**Fix:** print the four head rows in `tab:lshape-solve` even when dominated, and mark the
trend as cross-job.

---

# 5. Top 10 findings, by severity

| rank | id | severity | finding | one-line fix |
|---|---|---|---|---|
| 1 | **B1** | **BLOCKER** | The abstract, intro (×2) and conclusion assert "neural operators deliver one fixed (accuracy, speed) point per trained model" — verbatim from the rejected submission — while §5.1 of this paper explicitly withdraws it and `tab:resolution` shows the FNO's own resolution knob working. The round-1 disposition records this being fixed; the rebuild from the old source reverted it. | Restore the disposition's replacement framing; the defensible claim is that $q$ changes what the model can *represent*, which a coarser evaluation grid does not. |
| 2 | **R2** | **MAJOR** | On the `vs ref %` column of the paper's own `tab:panel-all`, the 2.44× knob moves physical error 4.56 % → 4.04 % (**1.13×**), every rung sits below the mesh's 4.03 % discretisation error, and a 15.7 ms full-order arm reaches 2.47 % — better than every reduced subject at 1/250th the cost of the top rung. Never confronted in the main text. | Confront it in §5.1: either move the ladder to $64^2$–$128^2$ where the reduction error binds, or state that the same-grid metric isolates reduction and no rung is physically distinguishable at $256^2$. |
| 3 | **R4** (= B4+B5+M2) | **MAJOR** | The only positive headline (L-shape, 2.77×/7.60× cheaper than sparse-direct) is on a **linear** Poisson residual that fails the paper's own structural condition; **POD-128 in the same table beats those margins** (2.94×/9.06×); and **CG in the same job at 512² is cheaper than the sparse-direct comparator** (29.13 vs 36.51 ms) and 5.5× more accurate but is never named in the main text. | Report it as "reduced models win on cost where no fast transform exists; the head costs 6–19 % more than POD-128 for 13 % less error"; name CG; drop "nonlinear residual" from the structural condition. |
| 4 | **B2 / R5** | **MAJOR** | The abstract splices two ladders ("monotone on three seeds" = the scheduled $M{=}4(K{+}q)$ ladder; "meets a bar with the test count held fixed" = the single-seed fixed-$M{=}1088$ ladder), and the knob bar is met on **2 of 3** seeds. Separately, `tab:seeds` shows the hero checkpoint is **1–2 sd better than all three fresh seeds at $q=16$–128**, a fact the paper never states and `WRITING-STATUS.md` records as deliberately unwritten. | Split the abstract clause; add one sentence in §5.2 and Limitations on the incumbent's position in the seed spread; note that 2 of 3 *is* the b-seeds pre-registered criterion and that seed3 fails on convergence, not on span. |
| 5 | **B3** | **MAJOR** | Quadrature status is misleading in both directions: `tab:tunability` prints a blank "—" for the two rules whose construction **is** known marginal (q=128 4/5, q=256 **0 of 4 re-draws pass**), and prints "**certified in one draw**" for two rules (m=2319, m=2560) that were **never re-drawn**, while §5.2 says those rules are "never called certified". `tab:panel-all` prints a literal `None`. | Change `rule_status` at source ("one passing draw, no re-draw"; "0 of 4 re-draws pass"); fill the blank cells; add "at $q \le 32$" to the abstract. |
| 6 | **R3** | **MAJOR** | Double-blind: the title is a **two-word edit** of the rejected NeurIPS title and the abstract's opening two sentences are verbatim from it; `paper/` also holds `old-submission/main.tex` and `REVIEWER-RESPONSE-MAP.md`, and the Reproducibility statement promises to release an anonymous repository from this tree. | Retitle around the finding; rewrite the abstract opening (B1 requires it anyway); build the anonymous repo from an allowlist, not a mirror. |
| 7 | **R1** | **MAJOR (presentation)** | Nine pages of main text contain **one figure and zero tables**; the first supporting table is on p. 24 and `tab:lshape-solve` is on p. 33. The round-1 page blocker was fixed by evacuating the evidence. | Restore three tables (panel summary with a `vs ref` column; the fixed-$M$ ladder; the L-shape solve with POD-128, CG and splu) and cut the prose they replace. |
| 8 | **M1 / M3** | **MODERATE** | Two ratio-provenance errors in §5.1–§5.2: "the cheapest reduced query falls from 4.48× to 1.82×" changes mesh, job, **GPU model (A100→H200)** and **arm type** (fast-kernel tight-tolerance → transferred-EQ loose-tolerance) simultaneously; and §5.2 says $M{=}1088$ was "not pre-registered" when `b-qxm/DESIGN.md` §3 and §6 pre-register it twice, while the four-part bar quoted as pre-registered is two parts DESIGN and two parts generator code. | Quote 4.48× and 1.82× as independent facts with GPUs named, or wait for `bpn401`; quote DESIGN §6's bar verbatim and claim the pre-registration the lane actually has. |
| 9 | **N1–N5** | **MODERATE** | Provenance-discipline leaks in a paper whose case is provenance discipline: three bars (`nEqtopBar` 0.116, `nEqtopTightBar` 0.06, `nNsOracleBar` 2.0) are **typed literals in `gen_tables.py`**; eight result numbers are hand-typed in word form in the prose against a Reproducibility statement that says "every number in the prose is generated"; `\provLshapeJob` prints job 3784662 for a table whose rows are job 3783786; "the complete query grows **1.033×**" is the *device*-time ratio, the complete query grows **1.52×**; wave cost ratios use device ms while L-shape/p-linear use complete-query ms, undeclared. | Read the bars from JSON; macro-ise or exempt the eight word-form numbers; key the job id on the printed rows; rename/requote the mesh ratio; declare which tables use which cost quantity. |
| 10 | **B8 / #2 / #19** | **MODERATE–MINOR** | Residue: `tab:ns`'s caption still says "$K{=}16$ … $R{=}256$ … the $K{=}32$ arm is pending" over a table containing three $K{=}32$/$R{=}512$ rows; `tab:sealed` (p. 31) is still a red `[ pending: b-seeds sealed cohort ]` box; `main.log` reports 7 overfull hboxes and one **"Float too large for page by 371 pt"** on `tab:eqrules` (Table 24); nine uncited ViT/attention/CP `\bibitem`s still print in the reference list; the NS "oracle values are upper bounds" qualifier lives only in a caption. | Regenerate the NS caption; drop `tab:sealed` or the sentence that needs it; split T09c; delete the nine `\bibitem`s; move the upper-bound qualifier into §5.5. |

---

# 6. Verdict

This is a scrupulously measured paper with a presentation that works against it and a
thesis its own tables do not support. The provenance machinery is real — I sampled
thirty-five numbers across the abstract, the prose and every table, traced each through
`gen_tables.py` to a named lane row and job id, and found **zero numeric mismatches** and
**zero cross-job cost ratios**; that is better discipline than most accepted ICLR papers and
the authors should be told so. But the argument does not survive contact with the evidence.
The abstract, introduction and conclusion still assert, in the rejected submission's own
words, that neural operators expose no deployment-time knob, while §5.1 of this same paper
withdraws that claim and the appendix tabulates the FNO's knob working — a reviewer who
reads both concludes the framing is being defended against the results. The headline knob
moves the same-grid error 2.44× but the *physical* error only 1.13×, because every rung
already sits below the mesh's own 4.03 % discretisation error, and a 15.7 ms full-order
setting in the same job is physically more accurate than the 3.9-second top rung; that
column is printed in the paper's own Table 16 and never discussed. The one positive headline
— the L-shape cost win — is on a linear Poisson residual that fails the paper's own stated
structural condition, is measured against the most expensive full-order arm in a job that
contains a cheaper one, and is matched or beaten by a plain POD basis in the same table. The
tunability claim is single-seed on a checkpoint the seed study shows is one to two standard
deviations better than every fresh seed, a fact the paper knows and does not print. And nine
pages of main text carry one figure and no tables, so none of this is checkable without a
fifteen-page turn. What remains after all that is genuinely worth publishing — the
matched-dimension head result (2.56 % against 56.93 % for the best linear map at $k=16$ in
one bank) and the quadrature-certification finding (the NNLS fit residual does not predict
held-out error; a single passing draw is not a certificate) — but neither is the paper's
current thesis, and the second is demonstrated on one PDE with three of six rungs
unreproduced. I would vote reject and say in the review that the honesty is exceptional and
that the paper it should be is a two-claim diagnostic paper built on those two results, with
the tunability ladder demoted to a worked example.

**Score: 3 / 10 (reject).** With B1 reverted to the disposition's framing, R2 confronted,
the L-shape reframed around POD and CG, the single-seed/favourable-checkpoint caveat stated,
and three tables back in the main text, this is a **5**. Reaching 6 or above needs the thing
round 1 already identified and this round confirms: **one cell in which the nonlinear head
plus the $q$-ladder is non-dominated against same-job full-order and POD controls on a
metric that is not below the discretisation error.** `b-lowvisc` (lvt01, job 3804337) is the
right idea; its mesh is under-resolved, which is fixable.

**Confidence: 4 / 5.** I read the compiled sources at `d131ec11`, `gen_tables.py`, all
generated tables, the nine lane `summary.json` and `DESIGN.md` files at their pinned commits,
`main.log`, `main.aux` and `old-neurips-main.tex`. I did not re-run any job, did not verify
the figures against their paired JSONs, and did not audit the ~800 macros I did not sample.
