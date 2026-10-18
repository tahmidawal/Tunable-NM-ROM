**Part A: ACCEPTED. Part B: SUBMIT for all four jobs**, through `cluster/submit.sh`, with at most three concurrent lane jobs. No blocking finding.

Audited staged HEAD `24a01e155eeb6d4b372364fd20bce068e4bb981b`. No files changed, no jobs submitted, and no cluster commands run.

Smoke evidence:

| Check | Recomputed finding |
|---|---|
| G-FD1 | Copied modules match the actual pinned E0 git blobs byte-for-byte, as amended by A2.6. |
| G-FD2 | Fresh CPU check: relative difference **0**, Newton counts identical. |
| G-E0 | Both recorded differences pass: **8.52e-15**, **9.08e-15**, against 1e-10. |
| G-PS1 | Fresh CPU comparison passes; half-step differences **3.59e-6**, **1.17e-5** both exceed 1e-9. |
| G-PS2 | Recorded difference **5.20e-15** passes 1e-10; registered stored run and step match. |
| G-INT | All 15 numerical checks pass their respective thresholds and cover the registered sizes. |
| G-CT | All five registered configurations pass; maximum difference **1.63e-15**. |
| K-loose | Fresh CPU replay confirms agreement rejection: differences **1.43e-4–1.29e-3**, above 1e-6. Convergence itself passes, as intended. |
| K-swap | Rescored original/swapped/shifted errors: **0.00147147 / 1.12294695 / 0.18071960**. Both mutations fire. |

Calibration recomputed from `C_*.npz` and stage residual/Newton evidence selects **t2 for both groups**. Maximum t2 differences are **7.99e-8** (CC) and **4.73e-9** (UB); t3 fails with **2.08e-4** and **3.41e-4**.

All four production field archives match their recorded hashes; offline scores reproduce exactly. Each configuration has the required six warm timing rows plus two warmups, all recording bitwise identity and zero difference. All 81 pilots are present; per-unit timing arithmetic and required compile-log coverage check out. The log contains `jax_backend=gpu`, f64/highest precision, H200 identification and `ALL-DONE`; stderr is empty.

**NOTED — evidence granularity:** gate-comparison fields and warm-repeat fields are not individually archived. G-E0/G-PS2/G-INT/G-CT thresholds were independently evaluated from their numerical diagnostics; warm identity was checked from individual comparison records, not independently retained repeat arrays. This limits forgery detection; it is not evidence of a failed gate.

Production verdicts:

| Job | Configurations | Projection | Required ×1.5 | Limit | Verdict |
|---|---:|---:|---:|---:|---|
| `fcc` | 48 | 1.639 h | 2.459 h | 2.500 h | **SUBMIT** |
| `fub` | 42 | 1.405 h | 2.108 h | 2.250 h | **SUBMIT** |
| `fub4k` | 7 | 2.443 h | 3.665 h | 3.750 h | **SUBMIT** |
| `fps` | 88 | 0.753 h | 1.130 h | 1.250 h | **SUBMIT** |

All certificates reproduce exactly. Independent arithmetic spot-checks give:

- CC, L=4096, dt=0.0025: **837.907 seconds/group**.
- UB, L=4096, dt=0.000625: **3967.527 seconds/group**.
- PS, N=256, dt=0.0005: **42.727 seconds/configuration**. Projected query time is **0.16665 seconds**, versus the smoke’s **0.13965-second** warm median.

These remain one-case pilot extrapolations with headroom, not runtime guarantees.

The configs exactly implement A1.1’s **185 configurations**, all38, calibration dev6 0–3, warm dev6 0–1 ×3, all four tolerance levels and burn 0.1. Each staged tree’s 108 provenance-listed files matches HEAD; all staged manifests verify. The C1 execution code is unchanged since the smoke.

Saved `result.json`, `F_*.npz` and `C_*.npz` provide the fields, calibration evidence, convergence records, scores and screening timings consumed by `match2.py`, supporting C1. Estimated array output is **25.34 GiB total**, excluding JSON and transfer packaging.

The ledger records **0.0889 GPU-h spent**, zero active reservations. Adding all four staged limits gives **9.8389 GPU-h**, below 30. Reservations are acquired at submission, not already active. `submit.sh` enforces the lane cap, account margin ≤6 before submission, and post-submit account-cap check. Live queue/storage availability was not queried under the read-only constraint.

Keep **`fub4k` first to defer** if budget binds; submit `fcc`, `fub`, and `fps` first.