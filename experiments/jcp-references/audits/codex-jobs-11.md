**SUBMIT all eight jobs for provisional generation.** The H200 change introduces no method blocker.

| Job | Walltime | Verdict |
|---|---:|---|
| `fd2dtest64p1` | 12 h | **SUBMIT** |
| `fd2dtest64p2` | 12 h | **SUBMIT** |
| `fd2dtest64p3` | 12 h | **SUBMIT** |
| `fd2dsealed128p0` | 24 h | **SUBMIT** |
| `fd2dsealed128p1` | 24 h | **SUBMIT** |
| `fd2dsealed128p2` | 24 h | **SUBMIT** |
| `fd2dsealed128p3` | 24 h | **SUBMIT** |
| `fd2dood32p0` | 24 h | **SUBMIT** |

Verified against the local staging:

- **Sbatch:** only `--gres=gpu:a100:1` → `--gres=gpu:h200:1` and removal of `--constraint=a100-80G`. Walltimes, 120G host memory, isolation, absolute venv, f64/highest precision and GPU preflight remain unchanged.
- **Integrity:** all manifests and all 18 provenance entries per job pass against staged bytes and recorded commit `5a0c3ddc0`. Every staged source also matches audit 10’s commit `4df6a834a`.
- **Qualification to “only GPU changed”:** test64 p1–p3 additionally acquire audit 10’s one-line `floor_stab` result-metadata addition and A11.5 documentation. They are therefore **not byte-identical to audit 9**, but their numerical method is unchanged. Sealed/OOD sources exactly match audit 10.
- **Configs/cases:** byte-identical to their respective audited versions, including custody paths/hashes and both S/ST kinds.
- **Hardware independence:** no GPU-model-dependent solver or acceptance logic; elapsed times are telemetry, with no timing claim. The recorded 35 GB estimate at \(L=12288\) fits comfortably within H200’s [141 GB memory](https://www.nvidia.com/en-au/data-center/h200/). This is capacity clearance, not a measured H200 peak.

Prior release conditions and tight-refinement walltime limitations remain; submission approval does not certify reference release.

No files modified, lab-log entry appended, or jobs submitted.