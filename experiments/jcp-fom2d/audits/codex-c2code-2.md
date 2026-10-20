**NOT READY to submit `runs/smkc2` as staged.** The job appears feasible on one H200, but executable release remains incomplete and the new release/recompile path loses compilation timing. These are instrumentation/lifecycle defects, not an observed OOM.

Audited HEAD `e7fe4b14e5fbd50cdc42a76ccb96640b4f8fefad`. All paths below are relative to `experiments/jcp-fom2d/`.

| Round-1 fix | Status | Evidence |
|---|---|---|
| 1. T-id exercises identity enforcement | **FIXED** | `c2time.py:366–373` corrupts the alias reference and calls the ordinary triplet; `:215–219,250` perform comparison and veto. CPU analysis fixtures also confirmed production-subject veto exclusion. |
| 2. K2 producer-schema completeness | **FIXED** | `targets.py:140–150` recognizes `acc_label`/`timing_label` and missing-accuracy records. Replaying the round-1 producer-shaped fixture now yields `complete: false`. |
| 3. Input sharing and executable release | **PARTIAL** | Inputs share `(discretisation, case)` at `c2time.py:129–145`. PS dictionary entries are removed at `:308–314`, but surviving closures retain executables. FD/maps and prepared inputs remain cached; pilots release arrays but not programs (`:374–383`). |
| 4. Pilot validity affects acceptance | **FIXED** | `c2time.py:379–385` checks full-output finiteness and solver convergence, records shape/evidence/memory, and requires every requested pilot. `c2analyze.py:98–102` gates the panel. A failed-pilot fixture invalidated it. |
| 5. Missing/deferred/invalid cells | **PARTIAL** | Empty groups now become cells (`c2analyze.py:126–130`), but too-small targets still disappear from H-comp; crossover summaries still say “no qualified mesh.” Early identity vetoes can still become “no candidate.” |
| 6. Selected-comparator annotations | **FIXED** | `make_c2configs.py:70–76,99,108` annotates every family/union candidate, including PS anchors; `c2analyze.py:124` reports the frozen selection’s annotations. |

Ranked fixes:

1. **P2 — Finish executable release before submission.**  
   The main-scope `fn` assigned by `c2time.py:225` survives stage 0. In this staged configuration, its final value closes over **PS N256**, the last FOM. Removing that program from `foms.ps` at `:311` therefore does **not** release it. If it loses selection, the pilot subsequently builds another wrapper/program while the old closure remains alive.

   Clear obsolete callable/output references and implement a lifecycle that accounts for controls and pilots. For production, also release unused input, FD and map caches. A6’s claim that unselected PS programs are released is currently stronger than the implementation.

2. **P2 — Record recompilation events before submission.**  
   `fomrun2.py:109–112` records only the first compilation for each name. If PS N256 is dropped after selection and rebuilt for its requested pilot (`c2time.py:374–377`), that compilation disappears from the final log.

   A CPU fixture calling `_log` with successive compilation durations of 2 and 3 seconds retained **2 seconds**, not both events. Persist each compilation event or accumulate durations/counts. This affects the smoke’s requested compilation accounting and subsequent resource estimates.

3. **P2 — Complete missing-cell reporting before production conclusions.**
   - `make_c2configs.py:89–111` creates no family groups for a too-small target; `c2analyze.py:131–148` emits only a target-status cell and no H-comp entries.
   - `c2analyze.py:189–195` retains failure reasons in sequences but discards them when constructing summaries. My too-small fixture produced an empty H-comp and **“no qualified mesh”** despite an explicit population-too-small sequence.
   - `c2time.py:281,306` loses the distinction between originally empty candidates and candidates all vetoed before shortlisting.

The staged job itself checks out:

- **Integrity:** `MANIFEST.sha256` verification and `bash -n run.sbatch` pass. Staged sources match the working tree, including `mr-burgers2d`, `separable-decoder`, and the ROM checkpoint. CPU imports resolve those staged dependencies.
- **Configuration:** the requested bank edges, A, R128 target, four accepted-smoke FOMs, six dev6 cases and three combined-workload pilots are present. All copied ladder and ROM target errors match their source JSONs exactly.
- **Batch setup:** one H200, GPU partition, 240G host memory, 90 minutes, f64/highest precision, GPU preflight, private paralab directory, source verification and output checksums are correctly configured (`runs/smkc2/run.sbatch:3–35`). Reservation and staged commit agree.
- **Memory feasibility:** bank storage is **68.686 GB**; one six-field output is **0.806 GB**. Archived P1’s substantially broader ROM pilot peaked at **113.469 GB**. This supports trying the combined workload on an H200; it does not establish its actual combined peak. **240G host memory appears adequate.**
- **Time:** **90 minutes appears ample for this bounded smoke.** Archived P1’s broader pilot completed in 278.6 seconds; accepted ladder pilots measured approximately 5.06 seconds for UB4096 and 4.00 seconds for CC4096 at Δt=0.005. The smaller-step pilots increase work, but these measurements do not suggest a 90-minute requirement. This is a feasibility assessment, not a measured C2 certificate.

The smoke exercises G-RR, G-FR, A-valid, stages 0/1/2, frozen selection before fresh inference, T-dbl, T-fault, T-id and combined-workload memory/validity. Its candidate set exercises union pooling, but cannot test truncating a family with more than three candidates. GPU reproducibility and control firing remain empirical smoke outcomes.

**`c2analyze.py` processes the supplied smoke plan successfully.** Synthetic passing data produced both expected cells; failed pilots invalidated the panel; a production identity veto invalidated both affected cells. K2 remains explicitly inconclusive. H-comp is empty because the smoke target key is `smk`, not an `|H` key—appropriate only for the plan’s stated mechanics-only scope.

**NOTED — forgery-only:** full-field fingerprints remain reductions plus lattice samples rather than full-byte hashes; analysis still trusts the recorded frozen-selection hash rather than independently validating its artifact. Neither is a demonstrated honest-run failure.

No repository or lab-log changes, cluster operations, or bank builds were performed. Scratch remains approximately 6.2 MB.