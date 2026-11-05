# ns3d-offmesh-quadrature: job and GPU-hour ledger

Every allocation, failures included. GPU-hours = elapsed wall × GPUs from `sacct` once final;
running jobs count at their wall limit. Hard cap 6 GPU-h (DESIGN §7).

| job | id | purpose | GPU | submitted (commit) | wall limit | elapsed (sacct) | GPU-h | outcome |
|---|---|---|---|---|---|---|---:|---|
| q96 | 6896984 | 96³ primary (DESIGN §2–6, A1, A2) | A100-80GB pax049 | 2026-11-05, 79520f9d4 | 1:30:00 | 0:07:29 | 0.125 | COMPLETED; status final; flag: net continuum p128 vs p96 1.36e-9 > 1e-10 |
| q64 | 6897000 | 64³ primary | A100-80GB pax105 | 2026-11-05, 79520f9d4 | 1:00:00 | 0:05:52 | 0.098 | COMPLETED; status final; same continuum flag (1.36e-9) |
| q32 | 6897410 | 32³ secondary | A100-80GB pax105 | 2026-11-05, 7a0efa618 | 0:40:00 | 0:05:07 | 0.085 | COMPLETED; status final; continuum flag (1.44e-9) |

Local smoke (GB10, not counted): 32³, 2 cases, 7 rules. Run 1 stopped at G2 on the NUDFT-vs-FFT
sub-criterion (1.78e-12 > 1e-12) → amendment §A1. Run 2: G1/G2/G4 pass; timing gate failed on the
GB10 (solo/in-block ratio 0.52 before and after, end ratio 1.01; not a drift); continuum target
p=48 vs 40 unconverged (smoke-sized, by design). Record in `smoke-record/`.

Running total: **0.308 GPU-h** final (cap 6); nothing running; namespace `ns3d-offmesh-q-20261105` emptied and removed. q32 was submitted from 7a0efa618, which differs from 79520f9d4 only by the ledger and the pulled q64 results (no code change).
