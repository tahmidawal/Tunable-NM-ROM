Audited committed HEAD `553301061`, read-only. **No submission blocker found.**

1. **CORRECT — bodies and execution.** All four body hashes match `RELEASE.json`; each runs `ref → data → pod → bank → ladder → time → controls` for its intended cell, without smoke-size overrides. Full-size defaults apply; W28 intentionally omits bank/translation arms. GPU preflight, f64, highest precision, and mandatory-control failure handling are present.

2. **CORRECT — A12 certificate.** Read-only recomputation reproduces:
   `5.7917 + 24 + 1.15 × 24.91 + 6.2275 = 64.6657 GPU-h < 87`.
   The 1.5 factor applies once despite both policy and source changes. This certifies this wave, not the remaining campaign or test jobs; the factor is a registered estimate, not a measured production bound.

3. **CORRECT — wall time.** Smoke elapsed was **4.1517 h**; multiplied by 1.5 gives **6.2275 h**. The 10-hour limit leaves **3.7725 h** beyond that estimate.

4. **CORRECT — smoke supports release, with a memory correction.** All seven stages completed; all registered controls fired. Selection contains **41 finalist entries and one arm-censored entry** (`trans|0.01|worst`), with no construction/screen failures. MF-ECSW’s failed positive control is explicitly permitted by A6.6.  
   **24/27 GB are not whole-job memory peaks:** external sampling reached **62.59 GB host and 67.84 GiB device**. These remain comfortably within 240G/H200 capacity. The ladder logged an overflow warning; invalid configurations are excluded. No OOM or truncated completion was found. Production additionally requires the new G-XFORM control.

5. **CORRECT — question coverage.** Production computes paired \(S\) per arm, tolerance, and median/worst statistic against qualified fixed/adaptive FOM finalists, including dense transforms and the A10 additions. Qualifications, censoring, and timing validity remain explicit. These are validation results; frozen test confirmation and the full width/steepness question require later jobs. Smoke speedups are not production evidence.

The live queue showed **one imp-box job and three account jobs**, leaving room for the first two submissions. Submit the second pair only when capacity permits; retain a detached collector per job.

**SUBMIT**