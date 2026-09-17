- **My own first reading of the local smoke was wrong, and it nearly became a code bug.** `smoke_xm.py`'s
  gate 1 — the retained $q=0$ path at 64 intervals against the consolidated audited fixture — returned
  $2.14\times10^{-7}$ on two consecutive runs, five orders above its $10^{-12}$ bar, while the *identical*
  construction in `smoke_gate1.py` returned $1.56\times10^{-14}$ three times in the same minutes. GPU
  contention was ruled out (the passing runs were at 96 % utilisation). The difference is the initial-condition
  Levenberg–Marquardt fit landing at a different point *inside* its own $10^{-6}$ stationarity tolerance —
  111 iterations in the failing process against 86 in the passing one — which an ulp-level GEMM difference is
  enough to cause when XLA's GPU autotuner picks a different algorithm on a contended shared box. **Nothing
  in the lane's code was wrong**; what was wrong was the gate's premise that a $10^{-12}$ field bar is
  reachable across processes on the shared GB10. Consequences, all applied before submission: the $t=0$
  invariance gate is blocking at $10^{-6}$ with $10^{-12}$ and bitwise as probes (DESIGN §9 A0); gate 1 now
  records its IC iteration count and `XLA_FLAGS` and is asserted at the end so the other gates still produce
  data. On the cluster — one process, one exclusively allocated GPU — the question is settled empirically by
  the in-job fidelity gates, not assumed.
- **"The arrays and peak workspace this lane needs ($\le$ ~10 GB, measured by `probe_memory.py`)" was an
  extrapolation stated as a measurement**, and the pre-submission audit caught it. At the time it was written
  the probe had measured 6.6 GB at $(256, 2176)$ and had not run its largest cell. It finished later at
  11.8 GB for $(64, 4096)$ — *above* the claimed bound. The sentence was corrected to say what was measured
  before any job was staged.
- **The memory probe itself was a process error.** It held one of three shared local GPU slots for 27 minutes
  against a sub-minute rule for local runs, with eight other lanes queued behind it, and GB10 numbers under a
  36 GB unified-memory cgroup do not predict an A100 anyway. It was not restarted; memory was handled by
  design instead (the grid is split into three jobs by $q$, each with its own attempt directory and its own
  FOM controls). Recorded as a deviation in DESIGN §9 A0.
- **Codex was unavailable** (account over its usage limit until 2026-09-19), so the protocol's independent
  audit of DESIGN.md was run by a fresh Claude Opus agent with no lane context, read-only. That is a weaker
  control than a different model family: it is the same family as the author. Its 14 findings and their
  disposition are in `reports/design-audit.md`; two were blocking and both were fixed before the first
  submission (S1 was missing the $(64, 320)$ cell its own cost ratios needed; the memory claim above). The
  final report has **not** had a cross-family audit, and this lane does not claim one.
- **Four design defects the audit found would each have produced a wrong or unfalsifiable number**, and are
  recorded because they were in the pre-registered design, not in a draft: the fixed-$M$ span silently
  dropped a non-converged rung and quoted the span over the shortened ladder (now reported *unavailable*);
  a cross-job fidelity pair whose comparator carried a null metric would have passed while comparing nothing,
  and counted toward the "two unconditional reproductions" gate (now requires two compared metrics); the
  H(rank) falsification lever could have been flipped by the single near-square $(64, 128)$ cell (now reads
  only cells with $M \ge 2(K+q)$); and the inner linear solve switches from Gauss–Jordan to LU at exactly the
  rung where the fixed-$M$ ladder crosses between jobs, confounding solver with rank (now controlled by a
  duplicate $(32, 1088)$ arm run with LU in the same job as its Gauss–Jordan twin).
- **The report generator's first within-job cost ladder was wrong, and it under-reported the lane's own
  headline.** It searched only each cell's *primary* job, so for the fixed-$M=1088$ column it found G1's
  three rungs ($q = 0, 16, 32$: error span $1.088\times$, cost span $1.291\times$, **fails** the tunability
  bar) and never saw that G2 holds four rungs of the same column ($q = 0, 64, 128, 256$) because G2 runs the
  $q = 0$ cell as an in-job anchor — which is exactly why the anchors were pre-registered. Corrected to
  search every appearance: the ladder is error span $2.437\times$, cost span $5.163\times$, four
  non-dominated points, and it **passes**. The wrong version existed for one generator run and never left
  the worktree, but it would have inverted the lane's recommendation.
- **No cross-family audit of the final report exists.** Codex remained over quota, so the numbers were
  checked by `checks/selfaudit.py` — an independent recomputation sharing no code with the generator, which
  re-derives three cells' errors from the archived field tarballs and verifies the decomposition identities
  — and all 27 checks agree. That is a self-check by the same author, not the protocol's independent
  auditor, and the lane says so rather than implying otherwise.
- **I hand-typed a number into DESIGN §A2 and it was wrong.** The solver control's
  evolved-metric agreement was written as $4.7\times10^{-14}$ from memory; read from
  `summary.json` it is exactly 0.0 — the Gauss–Jordan and LU solves of the $(32, 1088)$
  cell agree to the last bit. Corrected in place, and the paragraph now quotes the generated
  value. No other prose number in this lane was typed by hand; every table is generated.
- **Round 2 (2026-09-17, `bqx401`/`bqx501`): the report generator let a failed extension of a
  column erase an already-certified shorter result inside the same named object.** Round 1's
  fixed-$M=1088$ column (`bqx201`/G2, $q = 0, 64, 128, 256$, monotone, every rung converged,
  span $2.437\times$) was reported correctly in the committed `analysis.json`
  (`4b9723e8`). Round 2 (`bqx401`/E1) legitimately extended the same $M=1088$ column with a
  $q = 512$ rung that does **not** converge (worst joint gradient $1.77\times10^{-1}$ against
  the $10^{-6}$ bar). `spans()` computed `all_converged` over the union of both rounds' cells
  at that $M$, so the one non-converged addition flipped the flag for the whole object and, per
  DESIGN.md §5's "not patched" rule, nulled `span` entirely — silently erasing round 1's own,
  already-certified $2.437\times$ result and, downstream, flipping `rank_claim_false` to `true`
  and the headline to "scheduled ladder". Neither round-1's four converged rungs nor round-2's
  results support either of those. **Fixed:** `span` (the full declared range, correctly
  `null`/unavailable when a rung fails) is now reported separately from `certified_span` (the
  converged prefix, unaffected by a later failed extension); `verdict()` reads
  `certified_span` for `rank_claim_false`/`headline_fixed_M`. Re-derived `span_q_at_M1088 =
  2.4368429602045354`, bit-identical to the value that was briefly overwritten. This defect
  reached only the *working tree* between the previous session being killed and this one
  finishing the fix — the committed `analysis.json` at `4b9723e8` was correct throughout and
  the paper's pinned commit never carried the wrong value. Full mechanism: DESIGN.md §A5.
