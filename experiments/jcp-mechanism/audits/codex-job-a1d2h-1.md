| Item | Verdict | Finding |
|---|---|---|
| `run.sbatch` | **CORRECT** | Only GPU selection, job name and paths changed: A100-80G → H200, `jm_a1d2` → `jm_a1d2h`, directory `a1d2` → `a1d2h`. |
| Code/config/input tree | **CORRECT** | All 16 files are byte-identical, including checkpoints and both configs. |
| `MANIFEST.sha256` | **CORRECT** | Both manifests pass. Only hashes for `COMMIT.txt`, `PROVENANCE.json` and `run.sbatch` differ. |
| “Only GPU, name and paths changed” | **WRONG**, literally | Commit metadata also changed: `025b11f99…` → `30539d990…` in `COMMIT.txt` and every provenance entry. Both versions match their recorded committed blobs; no code drift. |
| `REFS.json` / `REFS.sha256` | **CORRECT** | Byte-identical. The previous audit’s separate reference-copy requirement and cohort-hash qualification remain applicable. |
| Config dependence | **CORRECT** | No numerical configuration depends on the scheduler job name or GPU model. GPU/job identifiers are recorded as metadata. |
| “No timing is measured” | **WRONG**, literally | Both configs disable benchmark timing, but diagnostic durations remain: setup, truth solves, `seconds_first`, rho and total elapsed time. |
| H200 suitability | **CORRECT**, static assessment | The accuracy/mechanism experiment remains valid on H200 with unchanged f64, `highest`, GPU preflight and numerical gates. Acceptance still requires passing runtime checks; numerical outputs need not be bit-identical across GPUs. Diagnostic durations cannot support cross-GPU speed comparisons. |

No files modified, jobs submitted, SSH connections or GPU computations performed.

Remaining WRONG: “only GPU/name/paths changed” omits commit metadata; “no timing is measured” should read “benchmark timing is disabled.”