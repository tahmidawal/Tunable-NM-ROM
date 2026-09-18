# Writing status — ICLR 2027 draft

## 2026-09-17 night — round-2 review applied (current state; everything below is history)

Review round 2 (3/10 at d131ec11) applied item by item, one commit each; disposition in
`private/REVIEW-ROUND2-DISPOSITION.md`. Claims changed where the review showed them false or
unsupported (each recorded in `ABSTRACT-2026-09-17.md`): the operator premise; the vs-reference
error (the knob moves the reduction error, 1.13x physically at 256²); the L-shape reframed (linear
residual, CG named, POD-128 cheaper than the head); the two ladders separated and the incumbent
disclosed as a favourable draw; one status vocabulary for quadrature rules; M=1088 pre-registered.
Main text now carries four tables (ladder with vs-ref, panel summary, L-shape, seeds) and Figure 1;
Figure 2 stays in Appendix B; main text ends on page 9. Open: title (user decision), no skip
ablation (C3), thin parabolic leg, tab:sealed until job 3804465 lands, b-qxm unpinned (clean tree).

## 2026-09-17 late — REBUILT FROM THE OLD SOURCE (current state; everything below is history)

**Release plan (double-blind).** The released/anonymised tree is `paper/` as committed, minus
nothing; `private/` at the worktree root (the previous submission's source, its style file, the
reviewer map, the review dispositions) is excluded and is never mirrored. `main.tex`'s header
comment is stripped at release.

User direction: keep the wording, style and title of the previous paper; update with the current
architecture and the new results. `main.tex` is now the old NeurIPS `main.tex`
(`private/old-neurips-main.tex`, archived outside the release tree) with the ICLR 2027 kit, the user-approved
two-word title change (see `ABSTRACT-2026-09-17.md`), the same section order and voice, and only
the overturned sentences rewritten. Details, derivations and every full table live in the
appendices (`sections/appendix.tex`, `sections/method-details.tex`); the inline bibliography is
`bib-inline.tex` (the old 74 entries plus 13 new ones). `main.tex` is canonical; `PAPER.md` is
regenerated from it by `gen_paper_md.py` and no longer carries the bibliography.

Status for the reader is a comment block at the top of `main.tex` (removed before submission):
pending T13 / seeds-sealed (job 3804465) only (ns2d closed at 50bf36da: K=32 fails the same bar); in flight lvt01
(3804337) and bpn401 (3805065); Figure 2 sits in Appendix B for the page budget; provisional T12 until T13, the 1024² frontier statement until
bpn401, the two top EQ rungs single-draw; open user decisions: headline Burgers metric (decisive
at 1024²), and the abstract keeps the old opening two sentences.

Coordinator notes carried over: the L-shape is Poisson (linear residual) and is written as the
no-fast-transform case; the committed seeds summary shows the incumbent better than every seed at
q=16–128 on the evolved metric, so "incumbent inside the seed spread" was not written; the bpn301
recheck lists 48 arms (message said 47).

## 2026-09-17 evening — reader-ready pass (this is the current state; sections below are history)

Lanes read in this build, all from **committed** lane state (`GIT_PINS` in `gen_tables.py` pins
b-panel 13ddecac, b-seeds e533b48e; b-qxm b4e38103 and lshape dc762ed3 are clean trees):

| lane | state | what the paper reads |
|---|---|---|
| b-panel | closed | bpn301 (3789570) = 256² with both rule sets, replaces bpn101; bpn203 (3789572, H200) = 1024². T3, T3b, T5, T5b. Claim mesh-qualified in §1, §5.1, abstract ("at 256²"). bpn401 (3805065, 512²) in flight |
| b-qxm | closed | pin dropped; §5.2 numbers unchanged; q=512 extension unconverged (one sentence, not plotted); M-saturation at q=256 (one sentence); five jobs in T2 |
| lshape | closed | 64²–512² solves (T18c); head q=64 2.77× at 256² and 7.60× at 512² cheaper than SuperLU; in abstract, Contribution 3, §5.6, conclusion |
| b-seeds | development cohort landed | T12 + T12b from e533b48e; sealed cohort (3804465) pending → T13 placeholder |
| b-eqtop, no-second, w-ladder, p-linear, ns2d | unchanged | same summaries as the previous provenance |
| b-lowvisc | register only | lvt01 (3804337) in T2c; no prose |

Figures: Fig. 2 (architecture) now sits in §3.2 beside eq. (ladder); Appendix B keeps the block
table. Figs. 1–3 have standalone captions (what is plotted, lane/job, one takeaway).
`PAPER.md` opens with an italic "Status for the reader" block (generated; removed before submission).

Remaining PENDING markers: T13 / seeds-sealed / T2 sealed row (b-seeds sealed cohort, job 3804465);
NS K=32 arm (ns204, job 3787320).

Open decisions (user's): headline Burgers metric — now decisive for §5.1 at 1024² (reduced rungs
non-dominated on evolved only); sign-off on the abstract's opening two sentences (unchanged here).

Coordinator notes: (1) the L-shape is Poisson, residual linear in the coefficients — written as the
no-fast-transform case, not "nonlinear residual + manifold"; (2) the committed seeds summary shows
the incumbent *better* than every seed at q=16–128 (evolved), so "incumbent inside the seed spread
at every rung" was not written; (3) bpn301 recheck JSON lists 48 arms, the message said 47.

Updated 2026-09-17 (third checkpoint: round-1 review addressed; see `REVIEW-ROUND1-DISPOSITION.md`). Build: `./build.sh` in this directory (regenerates `tables/` and
`tables-md/` from the lanes' JSON, renders `PAPER.md` from the LaTeX sources, then
`latexmk -pdf main.tex`); PDF at `paper/main.pdf` (git-ignored).

**Canonical source from now on: `paper/PAPER.md`** (coordinator instruction 2026-09-17). It was
rendered once from the LaTeX by `gen_paper_md.py`; user edits come back as diffs and are
reconciled into `PAPER.md`; the LaTeX is a build target that may lag. Section headings in
`PAPER.md` are numbered exactly as the LaTeX numbers them (2–7 main, A–E appendices) and must
stay stable so the mirrored document can be matched section by section. Generated tables sit
behind `<!-- table: Tnn_... -->` comments and are regenerated in place from `tables-md/`;
prose numbers are in `tables-md/numbers.json`. Style: official ICLR 2027 kit
(`iclr2027_conference.sty/.bst`, fetched from `media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip`),
anonymous, `\iclrfinalcopy` commented out.

## Page count (superseded: after the 2026-09-17 evening pass the main text runs to page 10; the rebuild from the old source addresses the budget)

26 pages total. Main text (intro through conclusion) ends on **page 9** (reproducibility statement opens page 10), within the
ICLR 2027 submission limit of 9 pages (10 at camera-ready). References p10–11, appendices
p12–24. Trim done by moving T1, T4, T7, T11b, T14 to the appendix (T3 and the family figure
stay in the main text), compressing method §3.3–3.5 into three subsections, and shortening
the intro, related work, setup, limitations and conclusion.

## Sections

| section | file | state |
|---|---|---|
| Abstract | `main.tex` | revision 2 of `ABSTRACT-2026-09-17.md`, verbatim |
| 1 Introduction | `sections/intro.tex` | drafted; three contributions + methodological findings; non-claims; reviewer pointer |
| 2 Related work | `related-work.tex` | extended (U-Net, PDEBench, Transolver; GNAT, accelerated ECSW; certification paragraph); five new bib entries verified 2026-09-17 against dblp/Springer (Ronneberger et al. 2015, MICCAI pp. 234–241), NeurIPS 2022 D&B proceedings (Takamoto et al.), PMLR v235 (Wu et al. 2024), JCP 242:623–647 (Carlberg et al. 2013) and IJNME 109(12):1623–1654 (Chapman et al. 2017) |
| 3 Method | `methods.tex` | tightened; correction ladder §3.2 first-class; instance derivations + DISCREPANCY record moved to Appendix A (`sections/method-details.tex`) |
| Fig. 1 architecture | `figures/architecture.tex` | TikZ translation of `architecture.mmd`; comparator row narrowed; `figures/architecture.png` rendered by a standalone compile for the Markdown |
| 4 Setup | `sections/setup.tex` | drafted; T1 in |
| 5.1 panel | `sections/results.tex` | drafted from b-panel; losses first |
| 5.2 rank vs tests | `sections/results.tex` | drafted from b-qxm |
| 5.3 EQ certification | `sections/results.tex` | drafted from b-eqtop, marked provisional (job 3783811 pending) |
| 5.4 head ablation + layers + training | `sections/results.tex` | drafted |
| 5.5 mesh ladder + speed | `sections/results.tex` | drafted |
| 5.6 linear PDEs (Poisson both meshes, heat, waves, L-shape bank/head) | `sections/results.tex` | drafted; p-linear closed; heat rows (T11d) generated from the 2026-09-10 heat linear-bank report, job 3511417 |
| 5.7 operators | `sections/results.tex` | drafted; claim withdrawn as instructed |
| 6 Limitations | `sections/limitations.tex` | written first; includes single seed + 1e-9→1e-3 gate amendment, 2D only, dev cohort, no SMA cold start, operators as lower bounds, one-draw certification, fixed-M dearer baseline, pending cells |
| 7 Conclusion | `sections/conclusion.tex` | drafted |
| Reproducibility / AI-use statements | `main.tex` | present (AI-use is required by ICLR 2027) |
| App. A method details | `sections/method-details.tex` | complete |
| App. B provenance (T2) | generated | complete for landed lanes |
| App. C full tables | `sections/appendix.tex` | T4b, T5, T6a/b, T8, T8b, T9b, T9c, T10, T11a, T11c, T12*, T13*, T14b, T15, T16, T18a/b |
| App. D reviewer map | removed for double-blind; the private map stays in `REVIEWER-RESPONSE-MAP.md` |
| App. E glossary | `sections/appendix.tex` | complete |
| Fig. 2 tunability family | `figures/gen_fig_tunability_family.py` | re-pointed: A rank vs error (b-qxm, no cost axis because its cells span three jobs), B the 256² same-allocation panel (b-panel), C the primary-rule EQ ladder vs dense twins (b-eqtop, provisional); paired JSON with SHA256s; included in §5.1 |

## Placeholders, from `tables/PENDING.md` (regenerated each build)

| placeholder | waits on |
|---|---|
| T5 at 1024² | b-panel **bpn203, job 3789572** (H200, running). Earlier attempts retracted: bpn201/3783817 config-parsing bug, bpn202/3787247 OOM in an untimed diagnostic — both listed in the retracted-attempts table, not deleted |
| T12 seeds, T13 sealed cohort, seeds status, T2 seeds row | b-seeds |
| NS K=32 arm | ns2d ns204, job 3787320 (the K=16 arm CLOSED as a pre-registered negative: H-ORACLE ratio 1.19 vs bar 2.0, job 3787319; phase 3 never submitted; T11e generated; ns202/3783797 retracted and listed in T02b) |

Also running: b-panel **bpn301, job 3789570**, the 256² re-run carrying both quadrature rule
sets (it will put the certified-EQ cheap arm beside the dense ladder in one allocation), and the
L-shape solve at 512² (job 3789568). When they land, the provenance rows must carry the live job
ids, and the retracted attempts stay in Table `T02b`.

Landed since the first checkpoint and no longer pending: the operator resolution knob (job
3787189, Table T14c, fired its falsification clause), the operator seed/precision controls
(job 3783831, Table T14d), the L-shape solve layer (jobs 3784662/3/4/3784910, Tables T18c/T18d;
$512^2$ still to come), and the EQ draw replication (job 3783811, Table T9d).

## Numbers with a caveat in the generator

- T15 (speed) is parsed from the lane's generated Markdown table (`2026-09-16-b-speed.md`),
  because its `result.json` lives only inside chunked Git archives; the audit JSONs are
  hashed in `tables/provenance.json`.
- b-speed job ids in T2 are typed from the lab-log entry, not read from JSON.
- Sizes in T1 (problem spec) come from the tuning config for Burgers; the Poisson/wave/
  L-shape rows are typed from the lanes' reports (they carry no numbers that appear in prose).
- `\nEqtopBar`/`\nEqtopTightBar` (0.116 / 0.06) are typed constants of the pre-registered design.

## Contradictions between sources and the ABSTRACT ledger (reported, not resolved)

1. (Resolved by the coordinator.) The heat collapse is tabulated from the 2026-09-10 heat
   linear-bank cell (job 3511417, A100-PCIE-40GB) as Table T11d, labelled an earlier cell of
   the same decoder family; the abstract's "Poisson and heat" stands and waves are added.
2. The ledger's "EQ rules certify on the primary bar only for q ≤ 64" is superseded by
   b-eqtop's interim report (every rung certifies in one draw, provisional); the paper
   carries the provisional version.
3. The b-panel EQ ladder is not monotone at q = 256 (secondary rules), while the abstract
   says "the family is monotone". The dense fixed-M ladder is monotone (b-qxm) and the
   primary-rule EQ ladder is monotone in the b-eqtop draw; the paper says exactly that.
4. The lab log's b-speed entry calls 1024² "the crossover reached (0.98–1.01×)" against
   the cheapest fair Newton control, while the mesh ladder reports no crossover at any
   rung against the cheapest target-meeting FOM. Different comparators; the paper cites
   only the mesh ladder and the parity speedup.
5. The p-linear q>0 best-found oracle is retracted by the lane (mis-scaled); the
   generator skips every row flagged `retracted` and the three-layer table uses only
   the q = 0 oracle.

## Open decisions (user's)

- Burgers headline metric (evolved vs all-times): both printed everywhere; none chosen.
- Whether to keep "heat" in the abstract (item 1 above).

## Round-1 review (commit 8dd88495) — status

Disposition per finding in `REVIEW-ROUND1-DISPOSITION.md`. New generated artefacts: T01b (sampling
families), T14c (resolution ladder), T17 (offline cost), T19 (solver variants), Fig. 3
(NNLS fit vs held-out rho). Abstract sentence 1 was changed by the pre-registered falsification
clause of the resolution job; the user should confirm the new wording.

## RESOLVED 2026-09-17 evening: the b-qxm pin (lane committed b4e38103; pin dropped; history kept below)

`gen_tables.py` now reads b-qxm's `analysis.json` from the lane's **committed** state
(`GIT_PINS`, commit `4b9723e8`, 09:40), not its working tree. Reason: the lane's uncommitted
working tree (modified 13:43, alongside `generate_xm.py`, `summary.json` and both figures —
it is mid-regeneration for "round 2") reverses the paper's headline:

| field | lane commit 4b9723e8 (what the paper uses) | lane working tree (uncommitted) |
|---|---|---|
| `span_q_at_M1088` | 2.4368 | `null` |
| `fixed1088_all_converged` | true | false |
| `fixed1088_passes_tunability_bar` | true | `null` |
| `fixed1088_within_job` | job 3780177, 4 rungs | `null` |
| `headline` | fixed-M ladder (M = 1088, pure rank) | scheduled ladder |
| `rank_claim_false` | false | **true** |
| `unavailable_reason` | — | "a rung is not converged; the span is not patched" |

The paper's Contribution 1, §5.2, the abstract's "meets a bar fixed before any run" and the
conclusion all rest on the committed reading. The working-tree reading would withdraw them.
A plausible benign explanation is that the fixed-$M$ column was extended to $q=512$, where
$(512, 1088)$ is only 2.06 tests per unknown and does not converge, so the lane's own rule
refuses to patch the span — i.e. a longer ladder, not a refutation of the published four-rung
result. **I have not assumed either way.** Nothing in the paper was changed on the strength of
an uncommitted file; the pin keeps the build reproducible and traceable. The coordinator should
say which state is authoritative, and the pin comes out as soon as the lane commits.

**Coordinator's ruling on the pin (2026-09-17).** Keep it. That working tree belongs to a lane
agent mid-regeneration, collecting two round-2 extension jobs: the pure-rank ladder extended to
$q=512$, and a saturation sweep in $M$ at $q=256$. The extended column adds $(512, 1088)$ at
2.06 tests per unknown — the under-tested case that lane pre-registered as possibly
uninformative — which does not converge, and the lane's rule then refuses to compute a span
across a ladder containing a non-converged rung. That is a longer ladder failing at its new top
rung, not a refutation of the four-rung result. The lane has been asked to confirm or correct
this, to report the four-rung and extended ladders as separately named objects so a flag cannot
flip merely because a longer ladder was appended, and to commit once its regeneration is
coherent. **Until that is relayed: do not soften Contribution 1, §5.2, the abstract or the
conclusion.** If the four-rung ladder is genuinely overturned, all four are rewritten together.

**Page budget (approved 2026-09-17).** The correction-ladder table stays in the appendix because
Figure 1 carries the family in the main text; do not trade it back.
