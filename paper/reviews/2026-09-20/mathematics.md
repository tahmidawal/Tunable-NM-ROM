# Independent review: mathematics

Structured summary of the subagent review of the frozen manuscript; not a verbatim transcript.

**Assessment:** Weak reject / major revision. **Confidence:** 4/5.

## 1. Nonlinear-head benefit is not isolated

**High; Missing evidence.** PDF pp.1–2,6–8; main.tex:112–145,415–422,540–557.

FOM comparisons show reduction benefit, not benefit over nested linear enrichment. Add a compact same-bank internal ablation or narrow the contribution.

## 2. Wave and corrected heat formulations are incomplete

**High; Verified omission/scope mismatch.** PDF pp.3,14,16–17; main.tex:225–267; method-details.tex:57–88.

The retained wave arm uses continuous acceleration with a decoder-curvature term and RK4, rather than the common fully discrete overdetermined objective; its q32 setting is square. State the actual wave formulation and corrected heat recurrence. This does not establish a code error.

## 3. Heat exact modal propagation is ambiguous

**High/medium; Missing assumptions.** PDF pp.3,14; main.tex:257–265; method-details.tex:83–88.

Distinguish projected Crank–Nicolson powers from the exponential of a reduced continuous generator. Exact physical modal propagation requires appropriate invariance; specify which operator is propagated.

## 4. Correction construction and rank assumptions are missing

**Medium; Verified omission.** PDF pp.3,14, equations2,6.

Specify per-PDE coefficient metric, snapshots, ordering, full-span completion and rank fallback. Nesting guarantees only monotonic globally minimized objective, not local-solver/full-state errors.

## 5. EQ certification scope is finite-state empirical validation

**Medium; Missing qualification.** PDF pp.4,15,18; equations10–11, Table D.2.

Dense-solver reachable-state checks do not establish all EQ-optimizer states or derivative fidelity. Specify what confirmed means and retain actual-returned-state checks or limitations.

## 6. Block-damped Burgers update is not reconstructible

**Medium; Verified omission.** PDF pp.3,15,18; main.tex:267–292.

The displayed scalar damping equation does not define separate block damping or elimination. Give the actual block update and distinguish it from Poisson exact elimination.

## 7. Linear skip does not guarantee a usable descent direction

**Medium/low; Mathematical overstatement.** PDF p.4; main.tex:317–331.

The nonlinear derivative can cancel the skip and projection can annihilate it. Describe conditioning as motivation and state actual rank checks.

## Resolved historical concerns

- Nonlinear heat recurrence replaces the old linear-latent derivation.
- Fixed-test weak least squares is distinguished from tangent Galerkin.
- Poisson constant-projector elimination is correct under full-rank assumptions.
- Online resizing of NNLS, projected CG and universal superiority are no longer claimed.

No false reported numerical values established; missing assumptions and publication-facing evidence are distinguished from code errors.
