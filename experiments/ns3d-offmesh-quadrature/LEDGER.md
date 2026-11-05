# ns3d-offmesh-quadrature: job and GPU-hour ledger

Every allocation, failures included. GPU-hours = elapsed wall × GPUs from `sacct` once final;
running jobs count at their wall limit. Hard cap 6 GPU-h (DESIGN §7).

| job | id | purpose | GPU | submitted (commit) | wall limit | elapsed (sacct) | GPU-h | outcome |
|---|---|---|---|---|---|---|---:|---|

Local smoke (GB10, not counted): 32³, 2 cases, 7 rules. Run 1 stopped at G2 on the NUDFT-vs-FFT
sub-criterion (1.78e-12 > 1e-12) → amendment §A1. Run 2: G1/G2/G4 pass; timing gate failed on the
GB10 (solo/in-block ratio 0.52 before and after, end ratio 1.01; not a drift); continuum target
p=48 vs 40 unconverged (smoke-sized, by design). Record in `smoke-record/`.

Running total: 0.000 GPU-h.
