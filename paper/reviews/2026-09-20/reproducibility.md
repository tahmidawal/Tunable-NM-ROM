# Independent review: reproducibility

Structured summary of the subagent review, not a verbatim transcript. The reviewer had prior Burgers experiment ownership; it did not consult the other reviews and assessed the full frozen manuscript.

**Assessment:** Not yet submission-ready on reproducibility. **Confidence:** 4/5.

## 1. Public reproduction and portable build are missing

**P1; Verified portability gap; artifact availability not established.** PDF pp.8,17; main.tex:638–643; appendix.tex:93–111; build.sh:8; gen_tables.py:34,151–172.

The PDF lists local manifests without access instructions. Build requires an absolute Python path and sibling worktrees. Provide an anonymous versioned artifact, a complete relative-path evidence bundle and reproduction commands; do not infer that unseen uploads do not exist.

## 2. 3D experiments lack essential definitions

**P1; Verified omission.** PDF pp.6–7,17; main.tex:500–535; appendix.tex:75–91.

Specify source/initial-condition families, ranges, domain, NS equation/discretization, horizons/outputs, training/evaluation counts/seeds and k/R/M. A compact four-row table and NS paragraph suffice.

## 3. Training and correction recipes are not reconstructible

**P1; Verified omission.** PDF pp.3–5,14–17; main.tex:179–209,327–337,363–371.

State common loss/normalization, coordinate features, architecture, initialization, optimizer/schedule and checkpoint selection. Give correction equations/metrics; keep full schedules in artifact.

## 4. Full preregistered criterion is confused with secondary knob bar

**P1; Verified overstatement.** PDF p.8 lines398–401; p.18 Table D.1; main.tex:597–604.

The three passing entries concern secondary C4. Source design reports incumbent sealed-ratio C2 failure and universal-convergence C3 failure. Replace full criterion with secondary knob criterion, define it and retain strict failures; numbers need not change.

## 5. Numerical protocol is incomplete

**P2; Verified omission.** PDF pp.3,15–17; main.tex:276–289; method-details.tex:175–190; appendix.tex:57–73.

Supply panel thresholds, caps, initial guesses, block damping, preconditioner, GPU and repetitions. Clarify aggregation and the displayed coarser wave setting; Poisson3D uses identity preconditioning.

## 6. L-shaped implementation is not specified by square formulas

**P2; Verified omission.** PDF pp.3,7,16–17; main.tex:185–195,560–573.

State the signed-distance boundary factor and masked-domain eigenmode tests, selected dimensions and homogeneous conditions on every segment. The square factor does not enforce interior cut boundaries.

## 7. Heat2D setup pointers do not identify the displayed measurement

**P2; Verified documentation mismatch.** PDF pp.6,16–17; T01_problems.tex:10; T01b_spec.tex:11.

Setup names job3511417; current CG table comes from3529772 and a continuum-spectral refined reference. Distinguish checkpoint lineage from timing job and specify reference/cohort/family. Not evidence of false heat numbers.

## 8. Diagram omits direct PDE-input dependency of residual

**P3; Verified graphical omission.** PDF p.16 Figure B.1; architecture.tex:23–36.

Inputs currently feed only initialization. Add source/parameters dependency into the reduced solve while retaining the compact layout.

## Resolved historical concerns

- All four displayed 3D snapshots match their committed Git sources.
- Nonlinear heat recurrence, weak fixed-test residual, EQ counts/refitting and named timing conventions are clearer.
- Failures and development/final scope remain visible; universal superiority withdrawn.

Internal retention is not sufficient publication-facing reproducibility. These findings largely need concise documentation and artifact packaging, not restoration of all comparison tables.
