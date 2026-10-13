**SUBMIT all seven jobs.** No blocking defect found in the staged inputs or execution paths. The 16-hour budget and L=768 device fit are credible estimates, not demonstrated worst-case guarantees.

No files were modified, no jobs submitted, and no lab-log entry written, as requested.

| Audit item | Verdict | Evidence / qualification |
|---|---|---|
| Calibration refusal input | **PASS** | Every staged job contains committed `checks/calib_3d.json`, with `proxy_ok_all=true` and `floor_stab=10.0`, matching `verdict.FLOOR_STAB`. No smoke override. `mode_fd` checks it before generating production references. |
| Calibration evidence | **PASS** | Verified calibration input/output hashes. Independently recomputed all **24 tolerance differences** and checked residual/proxy/Newton evidence for **36 solves** across 12 rows. All pass. Worst difference/proxy ratios: **0.024578 production**, **0.015848 tight**. Status matches `calib3d2/output/result.json`. |
| A9 consistency | **PASS for these FD jobs** | Calibration used tighter budget 60; maximum observed Newton count was only 3. Staged code changes that budget to 8, but `mode_fd` never uses `tighter`. Production, retry and tight settings remain unchanged. |
| Cohort partitions | **PASS** | dev4: 0–3; val64: 0–31 and 32–63; test32: 0–31; sealed64: 0–31 and 32–63; ood16: 0–15. Each cohort is complete, with no duplicate indices or overlapping partitions. |
| Levels and time steps | **PASS** | ST: L=128/256/512, Δt=.0025/.00125/.000625, CN-R; refinement L=768, Δt=.05/120. S: same spatial levels/refinement, BE at Δt=.005. ST has 100/200/400 nominal steps, **102/202/402 actual stages**; refinement has 600 steps, **602 stages**. S has 50. |
| Tolerances and failure procedure | **PASS** | Production `(1e−11, 1e−9, 20)`; failed production/refinement gets one retry `(1e−11, 1e−10, 40)`. Floor failure reruns all three levels tight `(1e−13, 1e−12, 40)`. Order/Péclet failure adds refinement and judges the finest three. Persistent solver/floor failures terminate without inappropriate spatial refinement. |
| Sealed configuration | **PASS** | Secret-related entries contain only cluster paths and hashes matching committed `checks/sealed_seeds.json`. No secret seeds are staged. Ciphertext hash is checked before seed decryption; plaintext hash afterward. |
| Sealed logging and outputs | **PASS** | `QUIET` is set before secret cohort construction. Per-attempt telemetry is suppressed; progress contains only item counts. Fields and descriptors enter encrypted `.npz.enc`/`.json.enc` files. Public result contains cohort/case indices, aggregate QC and hashes—not descriptors or fields. Case filenames use `cNNN`, never seeds. |
| Encryption, resume, QC | **PASS** | Memory-piped AES-256-CBC/PBKDF2, decrypt-roundtrip verification, atomic read-only writes. Existing ciphertext triggers refusal to resume. QC is per kind, includes every outcome, refinement/tight/attempt counts and explicit finite/undefined denominators. |
| Pull and publication | **PASS for completed jobs** | `pull.sh` excludes `*.enc` and verifies returned outputs. `publish_sealed.sh` verifies every listed ciphertext and encrypted-manifest hash before copying into the sealed store and deleting the job directory. It does **not** rehash the destination after copying; this is a robustness limitation. |
| Resources | **CREDIBLE** | All request H200, 240G host RAM and 16 h, satisfying A5.6. Detailed estimates below. L=768 remains unmeasured and its proxy uncalibrated, as explicitly allowed by A5.4. |
| Sbatch, provenance, manifests | **PASS** | All seven pin **`698f3d810715fd2da9b9f2d428385dde236f973b`**. Verified every staged manifest, all 17 provenance entries per job against committed bytes, and absence of extra files. Scripts pass `bash -n`; GPU partition, absolute venv, GPU refusal preflight, f64/highest, unique paralab paths and output hashing are present. Submission wrapper enforces the four-job lane cap. |

The implementation references are [solver procedure]( /home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/refjob.py:102), [sealed output handling](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/refjob.py:192), and [publication verification](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/cluster/publish_sealed.sh:15).

**Runtime:** summing measured production attempt times gives **139–251 seconds per case for both ST and S**, including base levels. Scaling L=512 to L=768 by interior volume, and ST additionally by 602/402 stages, gives:

| Per-case scenario | Estimated time |
|---|---:|
| Base ST + S | **2.3–4.2 min** |
| Base plus refinement of both kinds | **11.6–20.0 min** |
| Base, refinement, then tight reruns of the refined triples | **approximately 27–46 min** |

These are extrapolations from two development cases, before additional cohort setup, new compilation and output overhead. Both calibrated cases pass ST and S on the base triple without refinement. OOD costs and retry-heavy cases are not measured.

**Memory:** L=768 has \(767^3\) interior unknowns: **3.610 GB per f64 state**, with six retained output states alone taking **21.66 GB**. Scaling the measured **30.35 GB process-lifetime device peak** by \((767/511)^3\) gives approximately **102.6 GB**. The calibration H200 reported **143,771 MiB**, approximately **150.8 GB**, with roughly **138.7 GB** available under the staged 0.92 allocator fraction. **Likely fits**, with meaningful headroom; FFT workspace, cached solvers and allocation behavior prevent treating this as a measured guarantee. The 240G request is host RAM.

| Job | Cases | Base estimate | If every case refines both kinds | Verdict |
|---|---:|---:|---:|---|
| `fd3ddev4p0` | 4 | 0.15–0.28 h | 0.78–1.33 h | **SUBMIT** |
| `fd3dval64p0` | 32 | 1.23–2.23 h | 6.21–10.67 h | **SUBMIT** |
| `fd3dval64p1` | 32 | 1.23–2.23 h | 6.21–10.67 h | **SUBMIT** |
| `fd3dtest32p0` | 32 | 1.23–2.23 h | 6.21–10.67 h | **SUBMIT** |
| `fd3dsealed64p0` | 32 | 1.23–2.23 h | 6.21–10.67 h | **SUBMIT** |
| `fd3dsealed64p1` | 32 | 1.23–2.23 h | 6.21–10.67 h | **SUBMIT** |
| `fd3dood16p0` | 16 | 0.62–1.12 h | 3.10–5.34 h | **SUBMIT** |

A 32-case job with pervasive refinement **and** tight reruns could require roughly **14.5–24.7 hours**, so 16 h is credible for expected production, not every permitted execution path. This matters especially for sealed jobs, which deliberately cannot resume. Verdicts cover the local staging audit; live queue, custody-file availability and cluster disk state were not inspected.