# pois01 — job 3780224 FAILED in the preamble (2026-09-17)

`sacct`: FAILED after 00:00:51 on pax049. Preflight passed (`jax_backend=gpu`, 196 data files
verified, torch CUDA), the precision and training smokes passed, then the first training arm
died immediately: `FileNotFoundError: code/configs/unet-poisson/small.json`. Cause: the stager's
explicit `CODE` list named only the Burgers config files, so the Poisson configs were never
staged. No training ran, so no number exists to retract. Fixed by staging every file under
`configs/`; resubmitted as attempt `pois02` from the fixed commit. Per the lane protocol a
preamble death with zero training GPU time does not count toward the job cap, but it is
recorded here and in the lab log.
