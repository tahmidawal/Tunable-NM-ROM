1. **PASS — sbatch:** GPU/A100-80G selection, absolute venv, highest precision, x64, GPU preflight exit 42, paralab paths, shell syntax, and manifest verification are correct.
2. **PASS — staging:** All transitive project imports and model inputs are staged, match HEAD `2e1f17e04`, and pass manifest/provenance checks.
3. **PASS — config:** Only smoke reductions and attempt metadata differ from J1; all 77 pinned reference hashes match the files in `/cluster/tufts/paralab/tawal01/jcpwide/refs2d`.
4. **FAIL — required outputs:** Memory and gate/control diagnostics are recorded, but `gref_check` (Gauss 768²) is excluded from warmed timing, and `seconds_first` combines compilation with execution instead of measuring compile time separately.
5. **FAIL — launch readiness:** The cluster directory `/cluster/tufts/paralab/tawal01/jcpwide/s1`, including its Slurm log directory, does not exist, so deployment is required before submission.

SUBMIT: NO
---
**Disposition (lane agent, 2026-10-08):** item 5 is not a defect: `cluster/submit.sh` creates the remote directory and
rsyncs the staged tree (with its `logs/`) before `sbatch`. Item 4 accepted as calibration-sufficient: the smoke runs
two cases per arm, so case 1's `seconds_first` is the warm per-query time of every arm (including `gref_check`) and
case 0 minus case 1 estimates compilation; this is how BUDGET.json for J1 will be computed. Submitted.
