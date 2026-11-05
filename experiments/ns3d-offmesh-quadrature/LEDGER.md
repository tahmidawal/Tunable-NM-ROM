# ns3d-offmesh-quadrature: job and GPU-hour ledger

Every allocation, failures included. GPU-hours = elapsed wall × GPUs from `sacct` once final;
running jobs count at their wall limit. Hard cap 6 GPU-h (DESIGN §7).

| job | id | purpose | GPU | submitted (commit) | wall limit | elapsed (sacct) | GPU-h | outcome |
|---|---|---|---|---|---|---|---:|---|
| q96 | 6896984 | 96³ primary (DESIGN §2–6, A1, A2) | A100-80GB pax049 | 2026-11-05, 79520f9d4 | 1:30:00 | running | (1.50) | running |
| q64 | 6897000 | 64³ primary | A100-80GB pax105 | 2026-11-05, 79520f9d4 | 1:00:00 | running | (1.00) | running |

Local smoke (GB10, not counted): 32³, 2 cases, 7 rules. Run 1 stopped at G2 on the NUDFT-vs-FFT
sub-criterion (1.78e-12 > 1e-12) → amendment §A1. Run 2: G1/G2/G4 pass; timing gate failed on the
GB10 (solo/in-block ratio 0.52 before and after, end ratio 1.01; not a drift); continuum target
p=48 vs 40 unconverged (smoke-sized, by design). Record in `smoke-record/`.

Running total: ≤ 2.50 GPU-h committed at wall limits (two jobs running).
