# Adversarial review, round 1 — "Tunable Nonlinear-Manifold ROMs from One Trained Decoder"

Draft at commit `8dd88495`, `worktrees/2026-09-16-paper-refresh/paper/` (main.pdf, 25 pp, main text ends p. 13).
Reviewed against: the three NeurIPS 2026 reviews (`Older Paper /reviewer_comments `), the claim ledger
(`paper/ABSTRACT-2026-09-17.md`, `paper/TABLE-PLAN.md`), `tables/provenance.json`, and the eight lane
`reports/summary.json` files. Line numbers refer to the `.tex` sources; page numbers to `main.pdf`.

Number provenance was spot-checked for 30+ values (listed in finding 23). Everything that is a macro
traces to its JSON except the items in findings 6, 7, 20, 21, 22. The prose that is *not* a macro is
where the errors are.

---

## Findings

### BLOCKERS

**1. BLOCKER — page budget.** Main text (§1–§7) ends on p. 13 against a 9-page limit; references run to
p. 15. `WRITING-STATUS.md` already knows this ("~3.5 pages over"). There is no Fig. 2 yet, so the
real overrun after adding the one figure the paper needs is ≥ 4.5 pages. Nothing else matters until
this is fixed. Fix: Tables 3, 5, 6 and the full Table 7 to the appendix; §3.3–3.5 cut by a third; §5.1
and §5.6 halved; the intro's "reviewers' asks" paragraph deleted (see 3).

**2. BLOCKER — unfilled placeholders in the submitted text.** Red `[from generator: pending ...]` strings
appear in the main text at Table 1 (NS cohort cell, p. 8), §5.6 last sentence (p. 11: "whether any
reduced arm is non-dominated against a sparse direct solve is [pending: lshape solve jobs
3784662/3/4]"), §5.7 last sentence (p. 12: "the operators' own inference-time knob, evaluation
resolution, is [pending: no-second res01]"), and in the appendix at Tables 20, 21 and the T2 seeds row.
A reviewer opening this PDF stops reading at the first one. Either the jobs land before the 25th or
every sentence that depends on them is removed, not left as a hole.

**3. BLOCKER — the submission is not anonymous and reads as a rebuttal to another venue.** The draft
identifies its own rejected NeurIPS 2026 submission repeatedly and precisely:
- `intro.tex:26–28` "The paper has the same three-part skeleton as our earlier attempt at this claim".
- `intro.tex:77–83` "**The reviewers' asks.** This is a resubmission, and the three reviews of the earlier
  version asked for specific things: ... (fxe8) ... (GwrW) ... (5mgh)".
- `related-work.tex:28–29` "conflating tangent Galerkin with residual least squares was a defect of our
  own earlier presentation"; `:48–49` "the canonical-polyadic decoder of our previous submission".
- `limitations.tex:19` "Reviewer 5mgh asked for the shallow masked autoencoder ROM".
- Appendix D (`appendix.tex:250–278`) "The three NeurIPS 2026 reviews are answered as follows", a table
  keyed by reviewer ID.
- Appendix A.6 (`method-details.tex:192–209`) "Record of discrepancies with the previous manuscript ...
  Reading the code against the rejected manuscript", listing D1–D9 against "old-submission Eq. (1)",
  "Eq. (4)", "Sec. 4.3", "Sec. 5.1".

The NeurIPS submission is on OpenReview under a distinctive title with a ViT + LinearCPDecoder + "324×
over CG" abstract; the D1–D9 list quotes its equation numbers and architecture. Anyone who searches
"Tunable Non-linear Manifold ROMs" finds the authors. This is a double-blind violation, and quite apart
from policy, an ICLR reviewer does not want to grade a paper that is arguing with three other people.
Fix: delete every reference to the prior submission, the reviewer IDs, and Appendix D; keep A.6 only as
neutral statements of what the method is (e.g. "the projection is LSPG with a fixed test space, not
tangent Galerkin"), with no "previous manuscript" framing. Keep the reviewer map as a private
document.

**4. BLOCKER — the abstract and Contribution 3 claim a heat result the paper does not contain.**
Abstract: "On Poisson and heat the family collapses to a linear model". `intro.tex:52` "Where it is
linear (Poisson, heat, waves) the corrections are eliminated exactly, the top rung is the linear model".
Table 1's heat row says "method section only; no new run in this campaign"; no heat table exists
(`gen_tables.py:646` has a `build_heat()` reading a 2026-09-10 report, but no `T11d` is emitted or
`\input`). `WRITING-STATUS.md` item 1 flags exactly this. A claim in the abstract with no table behind
it is what reviewer fxe8 rejected the last version for. Fix: "Poisson and waves", or tabulate the heat
cell with its provenance.

**5. BLOCKER — the paper's opening premise is one the authors themselves say a pending job may
falsify.** Abstract sentence 1: "A learned PDE surrogate usually gives one accuracy–cost point; getting
another means retraining." `limitations.tex:39–44`: "The last one is binding: if a trained operator
exposes its own accuracy–cost family through evaluation resolution, the framing sentence ... is
withdrawn." An FNO's zero-shot evaluation at other resolutions is a headline property of Li et al.
(2020), which the paper cites; a reviewer will not need job 3783920 to doubt the premise. Reviewer 5mgh
already wrote that "the deployment-time tunable frontier is inherent to any solver-based ROM ... and is
novel only relative to neural operators" — and the draft has now conceded the ROM half and is betting
the operator half on a pending run. Fix: drop the "operators have one point" framing altogether. The
defensible statement is narrower: the rank $q$ is a run-time accuracy control with a *structural*
meaning (it moves the reachable set between the head's image and the bank's span), which resolution
scaling of an operator is not.

### MAJOR

**6. MAJOR — Table 7 (operators) and two prose numbers are corrupted by a generator bug; the paper's
"every number traces" claim fails on its own honesty table.** `gen_tables.py:439` sets
`meta[r['arm']] = r` keyed on arm name only; the Poisson-screen rows (job 3702464 / 3780625) share arm
names with the Burgers rows and overwrite them. Consequences in `tables/T14_operators.tex`:
- `fno-large / fno-medium / fno-small` show job `3702464` (the *Poisson* FNO job) with blank epochs and
  a blank "still improving" cell; their Burgers metadata (job 3710846; 692/1198/1741 epochs, best
  683/1190/1732) is lost.
- `unet-small / unet-medium / unet-large` show "500 epochs, job 3780625" (the Poisson epoch-cap run);
  their Burgers runs were 2327/1963/908 epochs in job 3780138.
- `\nOpFnoImproving` = "1 of 4" (`results.tex:355`, `limitations.tex:24–25`), contradicting the lane's
  own report ("4 of 4 FNO arms ... were still improving", `2026-09-17-no-second.md:31`) and the claim
  ledger ("four of four FNO arms were still improving"). The `7 of 8` figure is numerically right only
  because the Poisson U-Net rows happen to also be "improving".
Fix: key `meta` and `by` on `(arm, job_id)`; regenerate; re-read every sentence in §5.7 and §6 that
uses those macros.

**7. MAJOR — §5.6 states a wave result that its own Table 18 contradicts.** `results.tex:307–312`: "On
the reflective wave the top rung ... is at every mesh both the most accurate rung and 158–42× cheaper
... (error monotone in $q$, cost not ...)". Table 18 (T11a): `nested_q32` = 4.9161 / 5.1038 / 5.1164 %
against `linear_bank64` = 4.9323 / 5.1208 / 5.1335 % at 64² / 256² / 1024², and `pod_k40` = 4.8825 % at
64². The top rung is *not* the most accurate rung at any mesh, and the $q=32\to64$ step is
*non*-monotone. The lane report says "most accurate rung (within the integrator tie band of DESIGN §10
A2)"; the paper dropped the qualifier. This is a hand-typed prose claim, exactly the class the
reproducibility statement says cannot happen. Fix: "matches the $q=32$ rung to 0.02 pp and is 42–158×
cheaper", and make monotonicity a macro with a stated tie band.

**8. MAJOR — a cross-job cost claim the paper forbids itself.** `results.tex:351–354`: "Operators are
also far cheaper per query (the FNO at 7.2 ms in the same allocation as Table 2 ...), so on Burgers at
$256^2$ they dominate the ROM on both axes." The arms that beat the ROM on accuracy (U-Net-small,
U-Net-medium, Transolver-refine, U-Net-refine) were never timed in job 3780638; Table 7's caption says
"timing never is, and none is shown". Attributing the FNO's same-job cost to the U-Net/Transolver
accuracy is a cross-job cost inference. Fix: "the FNO dominates the ROM on both axes in the same job;
the U-Net and Transolver beat it on accuracy and were not timed against it".

**9. MAJOR — Contribution 2 is asserted, not shown.** (a) Abstract and `intro.tex:46–47`: "a linear skip
in the head, so the cold solve starts reliably". No with/without-skip ablation exists anywhere in the
paper; Table 15 (T08b) varies the *starting point* (nearest code vs zero) on Poisson, not the skip. The
old paper's evidence for the skip was the SMA cold-start failure, which this draft does not run
(`limitations.tex:19–22`). (b) `methods.tex:246–250`: block-damped variable projection "is what turned
the budget exits of a joint solve at $q\ge64$ into clean stationary exits at unchanged error" — no
table. (c) `methods.tex:199–201`: "we measure the rank of the projected Jacobian at every solved query
rather than asserting it" — never reported. Fix: run the skip ablation (one checkpoint, one job, cheap);
add a two-row table for (b); report the rank distribution or delete (c).

**10. MAJOR — the "structural condition" is confounded and near-tautological, and the paper never
exhibits the case it claims to characterise.** `intro.tex:50–57` promises "This says when the
nonlinear manifold is worth having." What the evidence shows: on Poisson the learned $R=512$ bank floor
is 0.7421 % where POD-512 reaches 0.1838 % (Table 6); on waves the $R=64$ bank is 5.12 % where POD-64 is
1.50 % (Table 18); on Poisson the head sits 4.2× above its own floor and the lane calls it "the fourth
Poisson cell to show the head is function-class limited". So the collapse on linear PDEs is at least as
much "this bank and this head are poor here" as "the residual is linear". The exact-elimination
statement itself is just: a linear least squares on the whole bank is cheaper than a nonlinear iteration
on a subset of it — a restatement of Kolmogorov-width intuition, not a finding. And on the one nonlinear
cell, the manifold is *also* not worth having at matched cost (0 of 27 reduced arms non-dominated;
POD-512 beats the best rung on the evolved metric). The paper therefore says only when the nonlinear
manifold is *not* worth having. Fix: (i) reword Contribution 3 honestly ("where it is not"); (ii)
de-confound with the missing control — run the same correction ladder on a POD bank of the same $R$ on
Burgers, so "learned bank vs POD bank" and "linear vs nonlinear residual" are separated; (iii) if a cell
exists where the head + ladder is non-dominated (e.g. lower-$\nu$ Burgers where POD-512 and Newton both
degrade), that cell is the paper.

**11. MAJOR — reviewer GwrW's first question is still unanswered.** GwrW: "Please specify what PDEs you
are actually solving. What is the equation? ... What are the initial conditions ... the boundary
conditions and RHS." Table 1 gives the operator and the domain but no $\nu$, no initial-condition family
for Burgers or waves, no source family for Poisson, no $\kappa$, no $c$. `grep` over the sources finds
$\nu$ and $\kappa$ only as symbols. Appendix D claims this ask is answered by Table 1 and §3.1; it is
not. Fix: a column or an Appendix-A paragraph with the sampling distributions, verbatim from the
generator configs.

**12. MAJOR — "harder problems" was promised in the rebuttal and is not delivered.** L-shape: bank and
head layers only, no solve, no timing, no non-dominance verdict ("the row is the whole point of the
cell and is empty"). Navier–Stokes: full-order solver only, "no reduced model yet". Heat: not run. 3D:
none. The hardest problem the method is *solved* on is 2D viscous Burgers on a square at an unspecified
$\nu$ — the same cell as the rebuttal. GwrW's "the paper needs a much bigger rethink than a rebuttal
will allow" will recur verbatim.

**13. MAJOR — no anonymous code or checkpoints.** fxe8: "Since neither anonymous code nor checkpoints
are provided, these ... claims are difficult to verify independently." Reproducibility statement
(`main.tex:74–75`): "Code, checkpoints and run records will be released with the camera-ready." A paper
whose entire case is provenance discipline and that ships neither the generator nor the JSONs it hashes
undermines itself. Fix: anonymous repo with `gen_tables.py`, `tables/provenance.json`, the eight
`summary.json` files and the audit JSONs; the checkpoint by hash.

**14. MAJOR — selective reporting of the fixed-$M$ ladder.** §5.2 says the crossed grid runs "fixed
$M\in\{256,1088\}$" and headlines $M=1088$ (2.44× error span, passes the bar). The lane JSON also
records the fixed-$M=256$ ladder: `fixed_M_within_job.error_span.evolved = 1.22` (to $q=128$), which
*fails* the pre-registered $\ge 2\times$ bar. The paper never says so. The reader is entitled to know
that of two fixed-$M$ ladders, one passes and one fails, and how $M=1088$ was chosen as the headline
(was it pre-registered, or picked after the fact?). Fix: report both ladders in Table 3; state the
selection rule and when it was fixed.

**15. MAJOR — the head-ablation claim is metric-dependent and only the favourable metric is quoted.**
`results.tex:233–235`: "no POD rank up to $k'=128$ matches it (POD-128: 10.1198 % ...)" — that is the
all-times (t=0-dominated) metric. On the evolved metric in the same panel job, POD-128 = 1.9464 %
against the head's 1.8890 % (Table 12), and POD-256 = 0.7109 % beats every rung below $q=256$. The
matched-*dimension* result at $k=16$ (POD-16 61.65 %, best linear map 56.93 %, head 2.56 %) is the clean
and robust one and should be the claim; "no POD rank up to 128" should carry both metrics or go.

**16. MAJOR — the augmentation of a nonlinear manifold with linear directions is not positioned against
its prior art.** $C_q$ is a POD of the head's closure residual added to a learned decoder. The related
work has no paragraph on hybrid linear/nonlinear reduction: NN-augmented projection-based ROMs
(Barnett, Farhat & Maday 2023), the linear-plus-quadratic split already inside quadratic manifolds
(Geelen–Wright–Willcox; Barnett–Farhat), POD-DL-ROM and other "linear trunk + nonlinear head" decoders,
nested/hierarchical bases and rank-adaptive ROMs. 5mgh's "no component is fundamentally new" stands
until the paper says precisely what is new (the run-time *prefix* selection and the
constant-projector elimination at $q=R$), and what is not.

**17. MAJOR — offline cost is promised and never given.** `related-work.tex:88–91`: "holding tests, row
scaling, ... fixed, including the offline cost of each"; `setup.tex:31–33`: offline setup "reported
separately". No training time, no PCA time, no NNLS fit time appears. The lane JSON has `fit_seconds`:
the 64-state $q=128$ rule that certifies took 2920 s (49 min) of NNLS; the $q=256$ one is of the same
order. "One trained decoder → a family" is then "one trained decoder plus roughly an hour of offline
fitting per certified rung", which changes what "free" means. Fix: one offline-cost table.

**18. MAJOR — no results figure; Fig. 1 is illegible.** The paper's thesis is a curve and it shows no
curve (Fig. 2 is "not yet re-pointed ... not included in main.tex"). Fig. 1 (p. 4) is a full-page
TikZ diagram shrunk to `\linewidth` with ~4 pt labels; nothing in it can be read in print. Fix: Fig. 2
(error vs same-job ms, exit reason as marker style, FOM controls and POD on the same axes) as Figure 1;
the architecture diagram to the appendix at a legible size or redrawn with a quarter of the boxes.

### MINOR

**19. MINOR — tables overflow the margin.** Table 1 (146 pt overfull; the NS row and "cohorts" column
are cut off on p. 8) and Table 7 (134 pt; the "job" column is cut on p. 12). `main.log` lists 11
overfull boxes and a "Float too large for page by 77 pt" (T9c).

**20. MINOR — the same cell carries two "solved" numbers.** Table 5 gives Poisson $1024^2$, $R=512$,
$q=0$ solved = 3.1146 % (this is the $M=257$ solve, `q0_m256@new_K32`); §5.6 (`results.tex:304–306`)
gives "solved 3.1495 %" (the $M=129$ ladder rung, `q0_m4@new_K32`) beside the *same* floor and
best-found. Both are job 3783813, but the reader sees one three-layer decomposition with two different
third layers and no mention of $M$.

**21. MINOR — provenance table (Table 8) defects.** (a) w-ladder row prints the job id as the commit:
"$64^2$: job 3780447 (0bb3cc86, commit 3780447)". (b) The hero panel (T3, T5) lists checkpoint
"asserted unchanged by gate" — no hash — while the reproducibility statement promises "Each run's ...
checkpoint hash are in Table 8". (c) `WRITING-STATUS.md` admits b-speed job ids and the T1 sizes are
typed, and T15 is parsed from a Markdown table rather than JSON. Fix all three; (b) especially.

**22. MINOR — typed constants with no stated origin.** `gen_tables.py:905`: `macro('nEqtopBar','0.116');
macro('nEqtopTightBar','0.06')`. Eq. (6) presents 0.116 as "the primary bar" with no derivation. A
reader will ask why 0.116 and not 0.1; say where it came from (pre-registered design, and from what).

**23. MINOR — provenance spot-check record (for the record; these trace).** Verified against
`summary.json` / `analysis.json`: panel POD-512 0.2184 % / 2928.9 ms; Newton $10^{-3}$ 0.0489 % / 31.4 ms;
q0 EQ 57.9 vs dense 284.8 ms; ref discretisation 4.0265 %; 0 of 27 non-dominated (recomputed from
T05 by hand — correct); dense ladder spans 3.64× / 14.62×; tolerance saving 26 % / 23 %; fixed-$M$
1.2657→0.5194 % and 848.0→4377.9 ms (G2), span 2.44× / 5.16×; ANOVA 85.1 / 6.1 / 8.8 %; corner shares
69.0 / 31.0 %; eqtop parent rules 14 and 8 fit states, $\rho_{\max}$ 0.1908 / 0.1678, 64-state 0.0669 /
0.1074, ladder 1.8891→0.5389 % at 59.1→722.2 ms, cost ratios 0.184–0.207; static $q=256$ fit 1.49e-4 /
$\rho$ 0.4621; U-Net/Transolver/ROM/FNO 1.4712 / 1.5189 / 1.5224 / 1.7110 / 1.8671 / 2.4829 %; wave
bank/head cost ratios 158 / 129 / 42; L-shape floor 0.6650 %, enrichment 1.067×, head ratios
2.24–5.77; mesh ladder 44.2→43.6 ms, 264×, FOM/ROM 0.265–0.495; speed 1.520× / 1.469×. One
calibration the paper should print: the *identical* cell `q0_M1088_dense` costs 742.2 ms in G1
(3780175, A100-80GB) and 848.0 ms in G2 (3780177, A100-40GB), 14 % apart; `q256_M1088_dense` is 3915 /
4165 / 4378 ms across three jobs, 12 % apart. The no-cross-job rule is respected, but the spread is
the same size as the tolerance saving (23–26 %) and the kernel-port gain (1.52×), and the reader
should be told so.

**24. MINOR — L-shape floor "target" is quoted on the cohort where it passes.** `results.tex:321–323`:
"the bank clears its floor target (0.6650 %)". The same bank is 1.7028 % on the pre-registered common
selection cohort (sdf: 1.6155 %), above 1.0 %; the lane notes the two cohorts disagree on the ranking.
Print both.

**25. MINOR — the completion rule was changed between jobs, in the baseline's favour, and the paper does
not say when.** T06a (job 3711424) shows POD-LSPG $k'\ge64$ "stationary: no"; the panel (job 3780638)
introduces "conv." which accepts an attained initial fit. JSON confirms the explanation (pod512
`max_step_stationarity` 7.6e-7, `max_joint_stationarity` 0.045). The rule was pre-registered before the
panel ran, which is fine, but say explicitly that it post-dates the head-ablation job.

**26. MINOR — the related-work paragraph on certification is stronger than §5.3.** `related-work.tex:85–88`:
"we show that neither predicts the rule's error ... and that the number of fit states binds before the
node count does." §5.3 and Table 4/16 show the 64-state rules *fail* at $q=64$ where a 25-state rule
passes, and that "the draw is first-order". State it as "at $q\ge128$" in both places.

**27. MINOR — "36 timed subjects" (`results.tex:69`) vs the ledger's "35 timed subjects".** The JSON has
37 distinct subject keys including the `*` aggregate; 36 is right, the ledger is stale. Trivial, but
it shows the ledger and the paper are not being reconciled.

### Underclaiming (also defects)

**28.** The matched-dimension head result — at $k=16$ inside one bank, the neural head reaches 2.56 %
where the best linear map reaches 56.9 % and POD-16 61.7 % — is the paper's only clean, large, positive
effect (>20×), and it is buried in §5.4 under a title that leads with the loss. It should be
Contribution 1's second clause and in the abstract.

**29.** "0.18–0.21× the dense cost at matched error" (`intro.tex:38`) undersells a same-job 5× cost
reduction at four-decimal-identical error. Say "5×".

**30.** The certification result (NNLS fit residual anti-correlates with held-out $\rho$: the static
$q=256$ rule fits to 1.5e-4 and reaches $\rho=0.46$; the std $q=256$, $m=3410$ rule fits to 1.4e-5 and
still fails at $\rho=0.18$; Table 17 has 100 such rows) is the most transferable thing in the paper and
the one an ECSW/EQ practitioner would cite. It is Contribution 4b, has no figure, and is labelled
provisional. A scatter of NNLS fit vs $\rho_{\max}$ over all 100 rules, with the bar drawn, would carry
the paper's diagnostic framing on its own.

---

## Answer to the three prior reviews — does this draft answer, dodge, or worsen each ask?

| reviewer | ask | this draft |
|---|---|---|
| fxe8 | one checkpoint per curve; protocol | **answered** (hash-pinned checkpoint, same-job panel, exit reasons). |
| fxe8 | internal inconsistencies | **partly**: the generator removes the typed-number class, but findings 6, 7, 20 are new inconsistencies of the same kind, one of them in the honesty table. |
| fxe8 | heat step invalid; CG on weighted operator | **dodged**: derivation fixed in A.2, but heat is never run and is still claimed (finding 4). |
| fxe8 | EQ counts under-specified | **answered** (A.4 separates $m$, support, $n_{\rm fit}$). |
| fxe8 | only unpreconditioned CG; the 324× | **answered by conceding**: DST and tuned Newton in every job; the method loses. |
| fxe8 | asymmetric tuning vs operators | **answered by conceding**: three families, equal budget; the ROM loses to two of them. |
| fxe8 | timing methodology | **answered** (§4). |
| fxe8 | no code / checkpoints | **not addressed** (finding 13). |
| GwrW | which PDEs, ICs, RHS | **not answered** (finding 11). |
| GwrW | harder problems (NS, corner) | **not delivered** (finding 12): L-shape without a solve, NS without a ROM. |
| GwrW | direct solver outperforms | **conceded**, which sharpens GwrW's real objection: why does this method exist? |
| 5mgh | broken/untuned baselines | **answered** (U-Net/Transolver; POD-LSPG through the same solver) — and the answer is that the method loses. |
| 5mgh | when is the nonlinear manifold needed | **answered negatively and confounded** (findings 10, 15): never, at matched cost, on any cell run; the linear collapse is confounded with a weak bank. |
| 5mgh | seeds / variance | **not yet** (b-seeds pending; every number single-seed). |
| 5mgh | SMA cold start | **not run**, conceded. |
| 5mgh | originality | **worsened**: the q-ladder is not positioned against manifold-augmentation prior art (finding 16), and the operator premise is now a pending falsification (finding 5). |
| 5mgh | Galerkin vs LSPG | **answered** (§3.3, D2). |
| 5mgh | is the family free from one model | **answered honestly**: rules re-solved offline; the offline cost of that is not given (finding 17). |

Net: the draft answers the *mechanical* complaints well and the *scientific* ones by concession. Two of
the three reviewers' central objections (GwrW: simple problems, direct solver wins; 5mgh: nonlinear
manifold not shown to be needed) are now confirmed by the paper's own tables rather than answered.

---

## Meta-review verdict

The one-sentence contribution a reviewer would write down is: "A frozen bank + neural-head decoder
admits a run-time correction rank $q$ that gives a monotone four-point accuracy–cost ladder on one
2D Burgers checkpoint, at one seed, on six development cases; the ladder is dominated on both axes by
a tuned Newton solve and by an FNO, beaten on accuracy by a U-Net and a Transolver, beaten on one of
the two metrics by POD-512, and collapses to a linear ROM on Poisson and waves, where the learned bank
is itself 3–4× worse than POD." That is not enough for ICLR as a method paper, and the draft knows
it — hence the diagnostic framing. But the diagnostic framing is not yet earned: a diagnostic paper
needs a transferable lesson shown to generalise, and the two candidates (the three-layer decomposition;
certify hyper-reduction on reachable states) are (a) close to existing practice in the ROM community
and (b) demonstrated on one PDE, one checkpoint, one draw, and labelled provisional. The "structural
condition" reads as a restatement of $n$-width intuition confounded by a weak bank, and the framing
premise about operators is one the authors have pre-committed to withdraw. Add the double-blind
violation, the page overrun, the placeholders, the corrupted Table 7, and the wave prose contradicted
by its own table, and this draft would be desk-rejected or scored 3/10 by all three reviewers. The
honesty is real and unusual and reviewers will say so; it does not substitute for a result.

**The single change that would most raise the score:** produce one cell in which the nonlinear head
plus the $q$-ladder is non-dominated against the same-job full-order and POD controls — most plausibly
a lower-viscosity Burgers (or the L-shape solve, if it lands) where POD's $n$-width decays slowly enough
that POD-512 fails and Newton needs enough iterations to lose. One such cell turns "when the nonlinear
manifold is worth having" from an unbacked promise into the paper's thesis, and every other table
becomes its characterisation. If that cannot be run by the 25th, the fallback is to re-scope hard:
lead with the certification finding (finding 30) and the matched-dimension head result (finding 28) as
the two claims, demote the tunability ladder to a worked example, cut to 9 pages, remove every pending
cell and every reference to the prior submission, and fix findings 6 and 7 — and accept that it is then
a workshop-strength diagnostic paper submitted to a main track.
