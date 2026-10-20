**DO NOT SUBMIT any of the 12 jobs yet.** A11.3 still has a reproducible release bypass. Staging passes, and the reported memory analysis provides a reasonable 80 GB budget. Walltime has improved but remains vulnerable to tight refinement.

No files were modified, no jobs submitted, and no lab-log entry appended.

**1. Release binding — BLOCKER**

I executed the actual `gather()` function in memory with simulated files and intercepted links. In [collect.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/collect.py:70), evidence candidates come from raw result files, **not successfully collected calibration jobs**.

The counterexample:

- Calibration result: `complete: true`, matching dimension/hash, passing proxy, but `gates_all_pass: false`.
- Final status: `provisional: false`, passing proxy, correct dimension and factor.
- FD result: complete, gate-passing, `calib_status: provisional`.

**Observed:** the collector refused the calibration job for gate failure, then accepted the FD job and linked its provisional case metadata into the simulated store. Calibration evidence bytes were absent. This violates the “collected, complete calibration job” requirement.

Additional findings:

| Check | Result |
|---|---|
| Final remains provisional | Correctly held |
| Final has wrong dimension | Correctly held |
| Final factor differs from current collector constant | Correctly held |
| Final evidence hash has no matching candidate | Correctly held |
| Final has `provisional: null` | Released; explicit Boolean `false` is not enforced |
| Collector constant and final factor both change to 100; case remains judged at 10 | Released; original judgment factor is not checked |

The last test confirms that comparing against the collector’s current `V.FLOOR_STAB` does not bind release to the factor used by the staged `refjob`. Case metadata already records `floor_stab`, but collection ignores it.

The earlier resume bypass also remains: resumed provisional cases can sit under a new job-level `calib_status: ok`, skipping this gate entirely. The calibration writer still omits `provisional: false`, so copying its status verbatim continues to hold production.

Required correction: validate calibration jobs and their evidence first; authorize release only from that verified set; bind final status to calibration and case judgment factors; enforce explicit finality and case-level protection, including resumed outputs and sealed QC.

**2. Walltime — ordinary refinement covered; tight contingency limited**

Using the previous audit’s production measurements:

| Cases | Limit | Base, both kinds | Every case refines both kinds | Remaining before overhead |
|---:|---:|---:|---:|---:|
| 6 | 12 h | 0.76 h | 2.72 h | 9.28 h |
| 16 | 12 h | 2.02 h | 7.25 h | 4.75 h |
| 32 | 24 h | 4.04 h | 14.50 h | 9.50 h |

These exclude compilation, initialization, gates, serialization, retries and tight attempts.

A11 now reports **3.5 h**, rather than approximately 3 h, for one tight 8192 solve:

- A 16-case all-refinement job plus one such solve reaches **10.75 h**; two reach **14.25 h**.
- A 32-case all-refinement job plus two reaches **21.50 h**; three reach **25 h**.
- Area × timestep scaling predicts roughly **11.8 h for a tight ST 12288 solve alone**, before the other members of its tight ladder. This is an extrapolation.

Thus 24 h resolves the earlier ordinary-refinement shortfall. It does **not** cover every mandatory tight path. Calibration proxy margins still do not establish that production floor failures are rare. Timeout is particularly costly for sealed/OOD jobs because ciphertext prevents resume.

**3. Memory — adequate planning budget, not a measured peak**

Taking the reported compiled analysis as accurate, **35.0 GB is a credible budget on an 80 GB A100**:

- The 8192 observation exceeds its compiled estimate by only about 3.2%.
- Applying that ratio gives approximately **36.1 GB** at 12288.
- The configured 0.92 allocation fraction nominally leaves about **73.6 GB**, providing substantial allowance for cached arrays, padding, runtime workspace and fragmentation.

Inspection shows full-resolution outputs are released between attempts; retained case evidence is restricted to the scoring lattice. I would retire memory as an independent submission blocker. However, I did not independently reproduce the compilation analysis or measure a 12288 runtime peak.

**4. Restaging — PASS**

All 12 manifests verified. Every provenance entry matched staged bytes and git content at `68d2afaec5da65acba1aec614edece4064021de5`.

Verified: A100-80G, 120G host RAM, GPU partition, absolute venv, f64/highest precision, GPU refusal preflight, isolated paths, and the revised limits. Custody paths/hashes match the committed seed record. Coverage is exact and duplicate-free: **262 cases, 524 case-kind outputs**, both ST and S.

| Job | Cases | Limit | Verdict |
|---|---:|---:|---|
| `fd2ddev6p0` | 6 | 12 h | **DO NOT SUBMIT** |
| `fd2dval32p0` | 16 | 12 h | **DO NOT SUBMIT** |
| `fd2dval32p1` | 16 | 12 h | **DO NOT SUBMIT** |
| `fd2dtest64p0` | 16 | 12 h | **DO NOT SUBMIT** |
| `fd2dtest64p1` | 16 | 12 h | **DO NOT SUBMIT** |
| `fd2dtest64p2` | 16 | 12 h | **DO NOT SUBMIT** |
| `fd2dtest64p3` | 16 | 12 h | **DO NOT SUBMIT** |
| `fd2dsealed128p0` | 32 | 24 h | **DO NOT SUBMIT** |
| `fd2dsealed128p1` | 32 | 24 h | **DO NOT SUBMIT** |
| `fd2dsealed128p2` | 32 | 24 h | **DO NOT SUBMIT** |
| `fd2dsealed128p3` | 32 | 24 h | **DO NOT SUBMIT** |
| `fd2dood32p0` | 32 | 24 h | **DO NOT SUBMIT** |

The common decisive blocker is the demonstrated release bypass.