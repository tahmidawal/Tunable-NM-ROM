# ns3d-pod-match: job and GPU-hour ledger

Every allocation of this lane, failures included. GPU-hours = elapsed wall × GPUs, from `sacct`
once final; running jobs count at their wall limit (DESIGN §A1 item 12). Budget: 15 GPU-h at
pre-registration; lifted by user authorization on 2026-11-04 (coordinator message). The gates
stay exactly as pre-registered. Every attempt is kept here.

| job | id | attempt / amendment | GPU | submitted (commit) | wall limit | elapsed (sacct) | GPU-h | outcome |
|---|---|---|---|---|---|---|---:|---|
| bank_a | 6846165 | arm (a), DESIGN §3 + §A1 | A100-80GB pax049 | 2026-11-03, 37f9d0072 | 6:40:00 | 4:56:12 | 4.937 | COMPLETED; G1 pass (1.35/1.44/1.44×), G2 pass (1.105), G3 FAIL (6.7e-3/6.8e-3) → bank fails |
| bank_b | 6846174 | arm (b), DESIGN §3 + §A1 | A100-80GB pax050 | 2026-11-03, 37f9d0072 | 6:40:00 | 4:56:08 | 4.936 | COMPLETED; G1 FAIL (1.75/1.87/1.87×), G2 pass (1.106), G3 FAIL (9.6e-3/9.7e-3) → bank fails |
| bank_a2_mu1 | 6850334 | §A2 warm start from bank_a + μ=1 G3 penalty | A100-80GB pax049 | 2026-11-04, 78ac5c10f | 8:00:00 | 6:16:46 | 6.279 | COMPLETED; **all gates pass**: G1 1.325/1.426/1.426×, G2 1.112, G3 4.5e-6/2.6e-4 |
| bank_a2_mu001 | 6850341 | §A2 warm start from bank_a + μ=1e-2 G3 penalty | A100-80GB | 2026-11-04, 78ac5c10f | 8:00:00 | 6:16:47 | 6.280 | COMPLETED; **all gates pass**: G1 1.325/1.417/1.418×, G2 1.106, G3 7.0e-5/3.1e-4 → **selected** (lower max ratio, §A2 rule); frozen as banks/a2_mu001/bank_selected.npz sha256 a1af98e2… |

Running total: **22.432 GPU-h** final; next: dev96/dev64/dev32 ROM jobs (wall 0.75/0.33/0.25 h).
