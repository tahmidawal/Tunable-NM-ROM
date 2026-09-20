# Handoff: rethink the NM-ROM paper from the evidence

This is a fresh-context ideation handoff, not a chosen paper outline or a claim that the current draft is ready. The numerical excerpts below are generated from retained JSON; acceptance and provisional status are stated beside each result. The canonical project history remains `LAB-LOG.md`.

## What the user wants now

The user says the results do not resonate and wants to start a fresh discussion about **how to write the paper**. Do not simply continue polishing the current framing or launch another tuning campaign. Help identify the scientific question, the strongest defensible result, and the story those support. Treat the current draft as revisable, not as an agreed final narrative.

Suggested opening task for the fresh session:

> Read the canonical LAB-LOG first, then this handoff, the current manuscript and the review summary. Help me rethink the paper from the evidence. Explain what is actually new, what result should lead, and what claims we can defend. Propose a few coherent paper stories with their strongest evidence and gaps before rewriting. Keep FOM-focused, readable tables and avoid a long archival appendix. Do not change the manuscript or launch experiments merely because the prior session listed possible next steps.

## Authoritative files and read order

Repository: `/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude`.

1. Read `/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/LAB-LOG.md` first. Latest dated entries supersede earlier status statements.
2. Current paper: `paper/main.pdf`, `paper/main.tex`, `paper/sections/appendix.tex`. **All manuscript edits/builds must occur only in this root `paper/` directory.** Do not edit the historical paper-refresh worktree.
3. Independent review synthesis: `paper/reviews/2026-09-20/SUMMARY.md`; detailed `mathematics.md`, `experiments.md`, `reproducibility.md`, `issues.json` and frozen hashes in `manifest.json`.
4. Older paper style/sample criticism: `Older Paper /neurips26__Copy_ (3).pdf`, the sibling source ZIP, and `Older Paper /reviewer_comments `. **The directory and reviewer-comments filename contain trailing spaces.** Read with properly quoted paths. These are historical sources, not current result evidence.
5. Numerical evidence: current table manifests and the latest lane paths listed below. Do not use conversational numbers as a source.

The manuscript last changed at `e6f07fe5`. Reviews are retained at `3584e848`. Three independent subagent reviews were completed; the session thread cap prevented a fourth subagent. A fourth coordinator check is explicitly **not independent**, since the coordinator edited the draft. Do not describe this as four independent reviews.

## User preferences to preserve

- Keep the older paper's clean style and core organization, using the ICLR 2027 anonymous format and main-text page limit.
- Put meaningful experiments in the main section, with 2D and 3D coverage clearly identified. Do not imply all dimensions/problems have matching completed panels.
- External comparison tables should be **NM-ROM versus named FOMs only**. Other method comparisons remain in the repository, not dumped into the PDF.
- For appropriate linear PDEs, use CG as the requested main comparator; no sparse-direct comparison in the current main presentation. Burgers is Newton–BiCGStab and Navier–Stokes is CNAB2, **not plain CG**.
- The user wants to retain the earlier Burgers tight-FOM speedup. It is already in the paper, together with the relaxed comparator and the accuracy/stall qualifications below. Do not present it as equal-error or fastest-FOM acceleration.
- Latest preference: show **relative error and speedup as a multiple of FOM**, not milliseconds in the main tables. Keep measured times/repetitions in supporting data. This was discussed and supported, **but has NOT been implemented**. The PDF still displays milliseconds.
- Suggested main-table columns: problem/mesh, NM-ROM setting, NM-ROM error, FOM error, speedup versus the named FOM. Any internal correction/EQ cost ratio needs its own explicit denominator; it is not automatically FOM speedup.
- Avoid junk controls, repeated tables and a long appendix. The compact appendix must still contain necessary equations, configurations and validation.
- The architecture diagram was redrawn; it still needs a direct source/parameter dependency into the solve, identified by review.

## What the method actually is

The implemented central representation uses a frozen learned spatial bank $G$, a nonlinear coefficient head $h_\theta$, and nested correction directions $C_q$:

$$u(z,y)=G[h_\theta(z)+C_qy].$$

The online algorithm solves reduced coordinates; it does not retrain the network or directly substitute a neural-operator forward prediction. In the main scalar weak-residual formulation, fixed smooth tests project the PDE residual, with row scaling and a PDE-specific reduced nonlinear solve. Linear corrections can be eliminated analytically for suitable linear PDE formulations; Burgers keeps nonlinear coupling. The wave arm is materially different and must be described separately rather than forced into the generic formula.

Three distinct ideas must not be conflated:

- **Correction rank $q$** changes the reachable representation inside the same fixed bank. It is the main demonstrated accuracy control.
- **Solver tolerance/budget** changes numerical work and stopping. It does not necessarily improve field accuracy once representation error dominates.
- **Empirical quadrature (EQ)** approximates residual evaluation at stored nodes. A new rule is fitted offline; selecting a stored rule is not retraining the decoder. Current 3D/wave panels do not establish EQ acceleration.

The unrestricted linear-bank solver bypasses the nonlinear head. It is an internal baseline using the learned bank, **not the nonlinear NM-ROM**, and its speedups must not be attributed to the nonlinear method. Nesting does not prove monotone errors from a local solver or a universal speed advantage.

## Why the current story is unsatisfying

The strongest demonstrated mechanism is a correction-rank accuracy/cost tradeoff, especially in Burgers. The main acceleration examples are largely comparator-specific linear-PDE results and an older Burgers configuration. Current dense nonlinear 3D results do not establish broad accuracy/runtime superiority. Those observations should shape the story rather than be hidden by selecting a convenient baseline.

Potential directions to discuss, not decisions already made:

1. **An empirical study of post-training accuracy control.** Lead with nested correction directions and distinguish fixed-test-space causal evidence from the scheduled multi-seed study. Needs careful novelty positioning and stronger fixed-test replication if claiming generality.
2. **A weak-form reduced-solver architecture with explicit deployment tradeoffs.** Lead with bank/head separation, residual evaluation and named-FOM comparisons; explain where it helps and where it fails. Needs a precise account of what is new over established reduced models.
3. **A narrower nonlinear benchmark paper.** Center Burgers and use linear/geometry cases as validation, rather than an encyclopedic PDE survey. This requires checking whether the retained nonlinear evidence is strong enough; it is not authorization to suppress relevant failures or tune on final data.

FOM comparisons show the value of reduction against a particular full solver. They do **not** by themselves isolate the benefit of the nonlinear head. The review suggested a small same-bank internal ablation as one option; alternatively, narrow that claim. The user has not approved restoring external POD/operator comparison tables or a new experiment plan.

## Evidence map: printed snapshots versus newest results

The PDF includes accepted Burgers3D final evidence, but still uses older Poisson3D replay07, Heat3D extra03 and NS3D confirmation06b **development** snapshots. Newly accepted final Poisson/NS and the numerically audited Heat final are separate; the review findings refer to the printed snapshots. Do not silently mix final accuracy with development timing.

The next table is a generated orientation aid, not a proposed paper table or a complete sweep. It lists representative corrected rows. Error is worst same-grid relative $L^2$ in percent: Burgers/NS use initial-normalized evolved error, heat current-normalized evolved error, and Poisson steady-state error. Ratios use the row's named same-job comparator; CG selection policies differ from a fixed denominator. These norms and policies must not be treated as interchangeable.

| Problem | Setting | NM-ROM error (%) | FOM error (%) | FOM/NM-ROM | FOM setting | Evidence status |
| --- | --- | --- | --- | --- | --- | --- |
| Burgers3D | native; q192 | 4.399 | 2.312 | 0.0229× | Newton–BiCGStab; fixed displayed control | Accepted final; physical refinement fails; already in PDF |
| Poisson3D | N=32; q96 | 0.265 | 0.157 | 0.941× | cg_identity_plain_rtol1e-02 | Accepted final; NOT yet in PDF |
| Poisson3D | N=64; q96 | 0.263 | 0.114 | 1.33× | cg_identity_plain_rtol1e-02 | Accepted final; NOT yet in PDF |
| Heat3D | N=32; q96 | 0.762 | 0.325 | 0.0664× | fom_cn_cg_dt0.05_rtol1e-04 | PROVISIONAL: numerical audits pass; complete Git retention/remote cleanup pending; NOT in PDF |
| Heat3D | N=64; q96 | 0.754 | 0.337 | 0.127× | fom_cn_cg_dt0.05_rtol1e-04 | PROVISIONAL: numerical audits pass; complete Git retention/remote cleanup pending; NOT in PDF |
| NS3D | native; q256 | 18.921 | 2.472 | 0.000683× | CNAB2; frozen dt0.01 | Accepted final; accuracy target fails; NOT yet in PDF |

### The earlier Burgers result the user remembers

This is the separate fine-grid development experiment, not the newer corrected stationary ladder. Its source is `paper/evidence/burgers-iterative-2026-09-11/results.json`, extracted from the original iterative06 run. Both named timings are paired with the same old NM-ROM in one allocation; the review independently verified the arithmetic.

| Comparator | NM-ROM error (%) | FOM error (%) | FOM/NM-ROM |
| --- | --- | --- | --- |
| Tight Newton–BiCGStab | 3.908 | 2.142 | 14.53× |
| Relaxed passing Newton–BiCGStab | 3.908 | 2.390 | 1.63× |

The common accuracy target is not identical measured accuracy. The NM-ROM stopping contract permits stalls and does not establish stationarity. Tight and relaxed FOM tolerances, older checkpoint identity and the reduced/test/quadrature dimensions need to remain visible. Detailed protocol: `reports/2026-09-11-iterative-fom-multiresolution.md`, particularly the relaxed-FOM section and Burgers protocol. Neither CG nor the full correction mechanism should be retroactively attributed to this result.

### Other useful evidence already retained

- Current 2D CG rows: `paper/evidence/paired-cg-2026-09-20/results.json` and `paper/tables/rewrite-provenance.json`. Poisson and L-shaped Poisson contain faster but less accurate NM-ROM results against their named CG controls. Heat fine-grid acceleration uses an earlier audited checkpoint. Wave displacement speed alone does not meet the full-state target.
- Main fixed-test Burgers ladder: `paper/tables/TR_correction_main.tex`, generated by `paper/gen_tables.py` from pinned evidence. One development checkpoint; do not present the changing-test-count sealed study as independent replication of this exact intervention.
- Scheduled sealed Burgers results and failures: `paper/tables/T13_sealed.tex`, `T13b_sealed_verdicts.tex`; source design `worktrees/2026-09-17-b-seeds/experiments/b-seeds/DESIGN.md`.
- Dense/EQ comparison and construction status: `paper/tables/TR_figure1_table.tex`, `T09_eq_ladder.tex`; generated from one paired allocation for the runtime comparison. This measures internal residual-evaluation cost, not FOM speedup.
- Broader POD/operator experiments exist, including cases where they dominate NM-ROM. Removing their tables from the PDF did not invalidate or erase those results. Avoid making a contradictory broad superiority claim.

## Independent review: essential unresolved points

The reviewers judged the frozen draft not yet submission-ready. They did not establish false timing arithmetic or fabricated fields. They recognized repairs to the old heat derivation, projection terminology, EQ-count explanation, timing protocol and overstated superiority claims.

1. **Verified overstatement:** “full pre-registered criterion” in the sealed-study prose actually refers to the secondary knob bar. Stricter sealed-ratio and universal-convergence checks fail. Fix the wording and define the criterion without changing measured numbers.
2. **Verified source mismatch:** Heat2D setup tables point to a different experiment from the printed CG measurements. Separate checkpoint lineage from measurement provenance and state the refined-reference convention.
3. **Missing formulations:** actual wave acceleration/curvature/RK4 route, corrected heat recurrence, heat endpoint operator, and Burgers block damping. The generic fully discrete overdetermined formulation does not describe every displayed arm.
4. **Missing experimental definitions:** 3D families, domains, coefficients, horizons, training/evaluation counts/seeds, bank/test dimensions and solver settings. Restore a compact configuration table, not a historical archive.
5. **Missing construction/assumptions:** training loss and correction metric/order; rank requirements and fallbacks; linear skip is motivation, not a conditioning guarantee; EQ certification is empirical and finite-state.
6. **Evidence gap:** FOM-only comparisons do not establish nonlinear necessity. Fixed-test-space rank causality is demonstrated on one checkpoint; the multi-seed study changes test count too.
7. **Scope/uncertainty:** setup names NS2D and lower-viscosity results not shown; near-break-even timing lacks dispersion; CG-specific speedups are not strongest-FOM guarantees. Retain all negative physical/full-state findings.
8. **Reproducibility/presentation:** absolute-path worktree dependencies need a portable evidence bundle; artifact access is not established by listing local manifests. L-shaped boundary/test construction and the diagram's direct input dependency need concise explanations.

Some fixes are writing/source integration. New experiments were only proposed for stronger claims, not launched or approved as part of the review. First choose the scientific story; then distinguish what must be fixed from what is optional evidence.

## Operational loose end: Heat final retention

The latest Heat owner handoff says its final numerical audits pass, but complete Git retention and exact remote cleanup remain blocked by local disk space. It is therefore not yet accepted final evidence under the repository's retention rules. Raw local and remote scientific data remain intact.

Resume from `worktrees/2026-09-20-paper-h3d/experiments/paper-h3d/HANDOFF.md` and `runs/final08/`. The pending rows are explicitly named `paper-tables-pending-retention.json`. Do not rename them accepted or delete the remote directory before actual-Git restoration verification.

A verified plan to free old archived Burgers duplicate fields exists at branch commit `37f92a5d`, path `worktrees/2026-09-20-paper-b3d/experiments/paper-b3d/checks/older-field-storage-20260920/PLAN.json`. **No deletion under that older-attempt plan was authorized/performed in this session.** Its proof and restoration helper are retained. This is an operational item, not a reason to stall the ideation discussion. Verify current disk/queue state before acting; do not treat an old estimate as current.

Poisson final08 and devicecheck09 are accepted, retained and remotely cleaned according to their owner handoff (`…/paper-p3d/HANDOFF.md`). The additional device check confirms the physical GPU stayed unchanged; it is development reproducibility evidence, not a second final test. NS final07 is accepted and retained (`…/ns3d/runs/final07/closure.json`); preserve its documented cross-runtime scalar-exp audit qualification.

## Workflow constraints and unresolved choices

- Read and append the absolute canonical `LAB-LOG.md`; this handoff does not replace it.
- User explicitly authorized root `paper/` as the only manuscript location, overriding the old paper-refresh worktree rule. Experiment lanes remain separate. Ask before creating new experiment worktrees or merging them.
- Do not change `best-results/` or retune on final outcomes. Same-job accuracy/time pairing, retained repetitions, f64/highest precision, GPU preflight, checksum collection and actual-Git restoration remain mandatory.
- Local Python: `/home/tahmid/Dev/.venv/bin/python`; cluster Python: `/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python`. Real experiments run on the approved cluster GPU lane, not unrestricted local jobs. No bare `scancel`.
- Preserve unrelated edits under `understand/2026-09-02-poisson-rom-training-to-prediction.*`. The canonical log also has entries from a separate Claude-integration session; preserve them. Do not stage unrelated changes into a manuscript commit.
- No merge, push, publication, new review-driven GPU campaign, or speedup-only table rewrite has occurred since the review. A prior positioning note in `reports/` is untracked and is not an agreed story.
- Start the new context by identifying the user’s desired takeaway and offering evidence-backed alternatives. Do not promise a universal win or guarantee paper acceptance.

## Frozen sources for this handoff

The handoff generator is `reports/generate_paper_ideation_handoff.py`. Re-run it with the absolute local Python path. These hashes identify the source bytes read for its numerical excerpts; re-running after source changes is an explicit refresh, not proof that the old handoff had those results.

| Source JSON | SHA256 |
| --- | --- |
| `paper/tables/main-experiments-provenance.json` | `bb9681ab0454f4058501e5a6f069243b3c2e9e340032a04a07e8e7226bbbad8e` |
| `worktrees/2026-09-20-paper-p3d/experiments/paper-p3d/runs/final08/paper-comparisons.json` | `48fad9067babede53918a07ca1168ad9dfbcf44196de41e0c332062a16479a97` |
| `worktrees/2026-09-20-paper-ns3d/experiments/ns3d/runs/final07/paper_summary.json` | `5c01c13c9e326a63821dec434fdc651c540ac3a19424bdd604744df88b408e7e` |
| `worktrees/2026-09-20-paper-h3d/experiments/paper-h3d/runs/final08/paper-tables-pending-retention.json` | `a0e9aca275f6f267ccbc49194ce18cbdabca41d223e7d84283c8d470629a7e89` |
| `paper/evidence/burgers-iterative-2026-09-11/results.json` | `8e5fe13d13d0289681004e4f76d219c2d3493f5f3a7d7ebf3ab1577a8f522c10` |
| `paper/rewrite-verification.json` | `4998aa1310027d9aecb07a68cfed642d61484e594dcd2eaabc3999b17aec9e34` |

Current PDF SHA256: `89f55938c0993c1bd5cf0d4c929551c30b48c818e48b0d9a60f51023f4d4fb35`. Recorded PDF length: 18 pages; main text ends on page 9.

## Glossary

**FOM:** full-order numerical solver. **NM-ROM:** nonlinear-manifold reduced-order model. **CG:** conjugate gradient; **BiCGStab:** a different Krylov method used inside the Burgers Newton solver. **CNAB2:** Crank–Nicolson/Adams–Bashforth time stepping used in the NS comparison. **POD:** proper orthogonal decomposition, a linear reduced-space construction. **Bank:** frozen spatial basis-like matrix $G$. **Head:** nonlinear map from reduced coordinates to bank coefficients. **$k,R,q,M,m$:** nonlinear coordinate count, bank width, correction count, weak-test count, quadrature-node count. **EQ/NNLS:** empirical quadrature/nonnegative least squares used to fit stored rules. **Weak residual:** PDE residual projected onto smooth test functions. **Stationarity:** numerical first-order stopping; it does not prove global optimality. **Stall:** stopping because a step/improvement is too small, without necessarily meeting stationarity. **Development/final:** cases available for selection versus cases opened after choices are frozen. **Same-grid/refined-reference error:** reduction error on the same discretization versus an error that also reveals discretization effects. **FOM/NM-ROM:** FOM runtime divided by NM-ROM runtime; above one means faster NM-ROM. **Matched CG:** a qualifying tested CG setting under the source's stated accuracy/stopping selection rule, not an unlimited global optimum. **Native/transfer:** training/evaluation mesh versus evaluation on a different mesh. **Accepted:** numerical and retention checks completed; **provisional:** the row's stated checks remain outstanding. **Actual-Git retention:** reading and restoring committed scientific bytes, not merely trusting a local file or commit message. **SHA256:** content checksum. **Internal ablation:** changing a component of the same method to isolate its effect.
