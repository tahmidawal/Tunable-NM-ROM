Read-only audit of `014b64e02`; no files changed, jobs submitted, or arrays copied. Paths below are relative to `experiments/imp-geometry/`.

- **CORRECT — inherited failure propagation:** `cluster/job.sbatch.tmpl:24–26,31–32` records inherited ERR events in a shared file and checks it in the parent. Subshell and command-substitution failures survive subsequent successful commands.
- **CORRECT — display pipelines:** `cluster/job.sbatch.tmpl:23` disables `pipefail` before the bodies. The unmatched `grep | tail` at `cluster/bodies/smoke6.sh:2` produces no false ERR event. In-memory Bash checks also confirmed that successful-command accounting and an absent `.failures` file do not trigger false failures.
- **CORRECT — collection transport failure:** `cluster/collect.sh:13–15` aborts on SSH failure/timeout and requires confirmed cell presence for `L_wide_S1`. Archive verification and local checksum verification precede deletion (`:17–24`).

No new blocking issues found for these two submissions. Their bodies and allocations match DESIGN A8’s first release step; combined allocation is 11.65 GPU-h. Shell syntax checks passed; live cluster availability was not checked.

CODE: READY-TO-RUN  
JOBS(step 1): READY