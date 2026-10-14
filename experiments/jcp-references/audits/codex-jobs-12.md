**SUBMIT both jobs for provisional generation.**

| Job | Walltime | Verdict |
|---|---:|---|
| `fd2dval32p1` | 12 h | **SUBMIT** |
| `fd2dtest64p0` | 12 h | **SUBMIT** |

Verified:

- Sbatch changes only A100 → H200 and removes the A100-80G constraint. Precision, GPU preflight, memory, isolation and walltimes remain unchanged.
- Both manifests pass; all 18 staged source entries per job match provenance and audits 10/11.
- Configs remain byte-identical to audit 9: val32 cases 16–31; test64 cases 0–15; both S/ST.
- Slurm confirms cancelled jobs **5217960 / 5219092** never started: `Start=None`, zero elapsed time, no node assigned. No local outputs; both remote directories are absent.

Qualification: relative to audit 9, staging also incorporates the already-audited `floor_stab` metadata addition and A11.5 documentation; numerical methods are unchanged.

Prior release conditions and walltime limitations remain. No files modified or jobs submitted.