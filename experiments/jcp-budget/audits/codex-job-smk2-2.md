**No remaining blockers found.** Audited staged `smk2` against HEAD `0f7fb2d10778985ba2d5a2cfb8f4255d9ee60ec4`. Paths below are relative to `experiments/jcp-budget`.

| Item | Assessment and evidence |
|---|---|
| 1. Script and provenance | **CORRECT.** All **149 provenance entries** match committed HEAD; all **153 manifest entries** match staged bytes. H200, `gpu`, two hours, namespace-local outputs, x64/highest and GPU preflight are enforced: `runs/smk2/run.sbatch:3–34`. Sequential-smoke and reservation checks remain in `cluster/submit.sh:18–37`. |
| 2. Predicate/configuration coverage | **CORRECT.** All requested predicates have evaluable paths. Reference predicates use case 0; B2/archive comparisons use cases 0–1. All three members have matching qualify archives containing both cases (`q3d2`/`q3d3`). Every configured rule exists, including gl56/gl64, successors and lat256. Evidence: `configs/smk2_b3.json:11–89`, `ana3d.py:86–151,233–242`, `smoke_accept.py:115–140`. |
| 3. Horizons and output contract | **CORRECT.** B2 runs at 0.25; B7 switches to 0.5 and clears both horizon-dependent caches. GLAT and `W_b3.npz` outputs match the expected inventory: **91 coefficient arrays, 179 case-rollouts**. Adaptive LSPG/GAL use horizon-aware `budcore`: `b3run.py:169–198`, `s3ctx.py:142–169`, `s3common.py:208–219`. |
| 4. Runtime and memory | **CORRECT for a bounded smoke.** Two hours remains plausible, not a demonstrated completion guarantee. Adaptive/star solves and possible L768 reference refinement remain the principal runtime/memory uncertainties. The telemetry blocker is fixed: cumulative allocator peaks are recorded after solve phases and persisted; acceptance rejects absent/nonpositive peaks and peaks above 120 GiB. Evidence: `b3run.py:33–42,173–197`, synchronization at `s3ctx.py:148–157`, memory gate at `smoke_accept.py:136–138`. |
| 5. Acceptance reliability | **CORRECT.** G-B3 now requires all **six** comparisons, zero missing comparisons, finite values and the registered tolerance. Executing the actual gate expression in memory accepted complete evidence and rejected short, missing, NaN, infinity and over-tolerance evidence: `ana3d.py:329–334,570–574`. |

The absent case-1 long reference only omits its long-reference error/growth metrics; it does not prevent G-L1 or B2 evaluation (`ana3d.py:470–480`).

No files changed, SSH used, jobs submitted, or scratch created. Live queue and remote disk state were not inspected.

Blocking fixes:

1. None.

**SUBMIT**