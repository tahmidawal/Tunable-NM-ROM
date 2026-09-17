# Writing status — ICLR 2027 draft

Updated 2026-09-17 (second checkpoint: Markdown canonical, main text trimmed to 9 pages,
family figure re-pointed). Build: `./build.sh` in this directory (regenerates `tables/` and
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

## Page count

24 pages total. Main text (intro through conclusion) now ends on **page 9**, within the
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
| App. D reviewer map | `sections/appendix.tex` | complete |
| App. E glossary | `sections/appendix.tex` | complete |
| Fig. 2 tunability family | `figures/gen_fig_tunability_family.py` | re-pointed: A rank vs error (b-qxm, no cost axis because its cells span three jobs), B the 256² same-allocation panel (b-panel), C the primary-rule EQ ladder vs dense twins (b-eqtop, provisional); paired JSON with SHA256s; included in §5.1 |

## Placeholders (`\gen{pending: ...}`), from `tables/PENDING.md`

| placeholder | waits on |
|---|---|
| T5 at 1024² (`\nPanelTenTwentyFour`) | b-panel job 3783817 |
| T12 seeds, T13 sealed cohort, `\nSeedsStatus`, T2 seeds row | b-seeds |
| T18 solve layer (`\nLshapeSolve`) | lshape jobs 3784662/3/4 |
| NS reduced model (`\nNsRom`), T1 NS row | ns2d phases 2–3 |
| operator resolution knob (`\nOpResolutionKnob`) | no-second res01, job 3783920 (binding for the framing sentence) |
| operator seed/precision control (`\nOpSeedControl`) | no-second ctrl01, job 3783831 (not cited in prose yet) |
| T9 top-rung certification status | b-eqtop draw replication, job 3783811 (numbers present, flagged provisional) |

`q = 512` extension: not in any lane's output; not referenced.

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
