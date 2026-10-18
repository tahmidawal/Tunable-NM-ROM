**Verdict: SUBMIT all three staged jobs through the guarded submission script. No blocking defect found.** Live queue availability remains a submission-time check; I did not access the cluster.

| Job | Verdict | ROMs / FOM candidates / groups | Projection | Required 1.5× | Limit |
|---|---|---:|---:|---:|---:|
| c2256 | **SUBMIT** | 32 / 134 / 884 | 2.864 h | 4.295 h | 4.5 h |
| c21024 | **SUBMIT** | 33 / 134 / 846 | 2.891 h | 4.337 h | 4.5 h |
| c24096 | **SUBMIT** | 33 / 134 / 846 | 2.891 h | 4.337 h | 4.5 h |

Verified:

- **Provenance:** FRONT is byte-identical to `2374fafe1`, matching SHA-256 `a51ac1a250f00d4b0006a84f2434ab3c96407b1f575f4ffff7a3eb25b7a8b77d`. All 122 manifest entries per staged job pass; staged files match local sources. The 118 provenance records checked against git also pass.
- **ROM subjects:** Regenerated targets and K2 exactly match the saved files. Subjects include precisely the entries carrying both qualification labels. The 256² R384 fallback correctly uses `gauss160`. Solver cores match qualify byte-for-byte; forms, schedules through factor 5, budget 600, production tolerances and `nstep=800` agree with `qacc.run`.
- **Errors and matching:** Every dev6 ROM target error comes from FRONT. Regenerated configs and plan match exactly, including FOM `ladder_e`. Recomputed admissibility from persisted evidence and all 196 target candidate sets agree. No family triggers extension. H/X populations coincide on all 38 cases.
- **K2 and output scope:** All 33 qualified 4096² identities are included; no hidden qualified entry invalidates completeness. Analysis covers H-comp at 1024²/4096², K2, crossover (a)/(b), family/union cells and median-/worst-only comparisons. Deferred same-resolution PS cells remain explicit.
- **Reference A:** R384, `gauss128`, frozen modes, LSPG-CNR, `dt=0.005`, production tolerances.
- **Resources:** Certificates reproduce exactly. Heavy candidates are included: UB L4096/0.000625 and CC L4096/0.0025 have recorded lattice screening costs of 13.041 and 4.915 seconds. The projection charges selection/inference work to every candidate; groups share subject executions.
- **Memory:** The 33 identities share **four Settings and one bank**, via [the cache](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-fom2d/experiments/jcp-fom2d/c2time.py:113). Production quadrature tables total approximately 1.435 GB before small auxiliary tables. Smoke peak was 96.592 GB against a 138.102 GB device allocation, with heavy FOM pilots passing. Device memory and 240G host memory are plausible; the complete production peak is not yet measured.
- **Budget and concurrency:** Ledger spending is 1.7845 GPU-h. Adding all three limits gives **15.2845/30 GPU-h**, leaving 14.7155. Submission code enforces lane cap 3, account count ≤6 before submission, and the post-submit account cap.

**NOTED:** Resource estimates are not hard bounds: retry costs and compilation variability remain. Identity fingerprints are accidental-corruption guards, not cryptographic protection against engineered substitutions; that forgery-only limitation does not block submission.

Audit scope: matching was recomputed from persisted ladder scores/evidence; I did not repeat every lattice-field rescore. No cluster commands, submissions, repository changes or LAB-LOG writes occurred. Scratch contains approximately 23 MB.