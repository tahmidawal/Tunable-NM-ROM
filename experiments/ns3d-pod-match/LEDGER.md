# ns3d-pod-match: job and GPU-hour ledger

Every allocation of this lane, failures included. GPU-hours = elapsed wall × GPUs, from `sacct`
once final; running jobs count at their wall limit (DESIGN §A1 item 12). Budget: 15 GPU-h at
pre-registration; lifted by user authorization on 2026-11-04 (coordinator message). The gates
stay exactly as pre-registered. Every attempt is kept here.

| job | id | attempt / amendment | GPU | submitted (commit) | wall limit | elapsed (sacct) | GPU-h | outcome |
|---|---|---|---|---|---|---|---:|---|
| bank_a | 6846165 | arm (a), DESIGN §3 + §A1 | A100-80GB pax049 | 2026-11-03, 37f9d0072 | 6:40:00 | 4:56:12 | 4.937 | COMPLETED; G1 pass (1.35/1.44/1.44×), G2 pass (1.105), G3 FAIL (6.7e-3/6.8e-3) → bank fails |
| bank_b | 6846174 | arm (b), DESIGN §3 + §A1 | A100-80GB pax050 | 2026-11-03, 37f9d0072 | 6:40:00 | 4:56:08 | 4.936 | COMPLETED; G1 FAIL (1.75/1.87/1.87×), G2 pass (1.106), G3 FAIL (9.6e-3/9.7e-3) → bank fails |

Running total: **9.873 GPU-h** final (after bank_a, bank_b); nothing running.
