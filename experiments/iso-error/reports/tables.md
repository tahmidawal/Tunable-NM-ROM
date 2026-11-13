## Reproduction of the existing row

Stored: job 4319328 (pax105, NVIDIA A100 80GB PCIe). Reproduction: job 7174297 (pax106, NVIDIA A100 80GB PCIe), same driver/config/checkpoint.

| width | median % stored | median % repro | worst % stored | worst % repro | errors bit-identical | ms stored | ms repro | ratio |
|---|---|---|---|---|---|---|---|---|
| 384 | 0.026419 | 0.026419 | 0.874860 | 0.874860 | no | 45.237 | 45.281 | 1.0010 |
| 96 | 0.529332 | 0.529332 | 3.845644 | 3.845644 | yes | 14.244 | 14.374 | 1.0091 |
| 128 | 0.267509 | 0.267509 | 2.546901 | 2.546901 | no | 18.279 | 18.152 | 0.9931 |

Full-order solver of Table 1 (`lean_nt3e-3_l3e-3_dt005__graphs`): stored 17.416 ms, reproduction 17.127 ms.
Gates of the reproduction job: {'evaluation_cohort_hash': True, 'phi_free_operator_parity': True, 'sep_lattice_projection': True, 'drift': True, 'neighbour': True, 'repetition_output_identical': True, 'retained_repetitions': True}. Check passed: True.

## Per-width results (test64, 256², primary compile mode)

Sweep job 7175375 (pax050, NVIDIA A100 80GB PCIe); gates {'evaluation_cohort_hash': True, 'phi_free_operator_parity': True, 'sep_lattice_projection': True, 'drift': True, 'neighbour': True, 'repetition_output_identical': True, 'retained_repetitions': True}; drift ratio A2/A1 in [0.9955, 1.0077]; neighbour ratio max 1.0066.

| width R' | M | median % | mean % | p90 % | worst % | time ms (median) | min–max ms | recon ms | solve ms (derived) | LM iters/query (median, max) | stalled exits | mode |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 128 | 4.0859 | 5.6266 | 12.9445 | 18.3423 | 9.28 | 7.90–11.90 | 0.14 | 9.14 | 25, 44 | 0 | graphs |
| 48 | 192 | 2.0673 | 3.0235 | 7.0245 | 11.4996 | 10.32 | 8.95–13.56 | 0.16 | 10.16 | 29, 50 | 0 | graphs |
| 64 | 256 | 1.3214 | 1.9270 | 5.0385 | 7.4278 | 11.55 | 9.64–14.68 | 0.17 | 11.37 | 31, 51 | 0 | graphs |
| 96 | 384 | 0.5293 | 0.9337 | 2.5412 | 3.8456 | 15.27 | 11.91–19.11 | 0.24 | 15.03 | 36, 53 | 0 | graphs |
| 128 | 512 | 0.2675 | 0.5596 | 1.5823 | 2.5469 | 18.45 | 13.71–21.27 | 0.23 | 18.22 | 43, 53 | 0 | graphs |
| 192 | 768 | 0.0973 | 0.2436 | 0.7096 | 1.5699 | 25.58 | 18.76–27.29 | 0.27 | 25.31 | 51, 55 | 0 | default |
| 256 | 1024 | 0.0485 | 0.1411 | 0.4445 | 1.1586 | 33.81 | 25.60–37.94 | 0.31 | 33.51 | 51, 61 | 0 | default |
| 384 | 1536 | 0.0264 | 0.0875 | 0.2403 | 0.8749 | 45.55 | 33.42–52.31 | 0.37 | 45.18 | 52, 63 | 0 | graphs |
| 512 | 2048 | 0.0407 | 0.1345 | 0.3668 | 1.3814 | 59.99 | 43.02–68.32 | 0.40 | 59.59 | 52, 61 | 0 | graphs |

### Other compile mode (secondary; not used in any table)

| width R' | primary mode | primary ms | other-mode ms | other/primary | errors equal to 4 sig. digits |
|---|---|---|---|---|---|
| 32 | graphs | 9.28 | 9.53 | 1.027 | yes |
| 48 | graphs | 10.32 | 10.54 | 1.021 | yes |
| 64 | graphs | 11.55 | 11.76 | 1.019 | yes |
| 96 | graphs | 15.27 | 15.08 | 0.987 | yes |
| 128 | graphs | 18.45 | 18.30 | 0.992 | yes |
| 192 | default | 25.58 | 25.71 | 1.005 | yes |
| 256 | default | 33.81 | 33.92 | 1.003 | yes |
| 384 | graphs | 45.55 | 45.55 | 1.000 | yes |
| 512 | graphs | 59.99 | 60.59 | 1.010 | yes |

### Consistency of widths 384 / 96 / 128 across jobs

| width | errors equal to stored (4 sig.) | bit-identical | ms stored (4319328) | ms reproduction | ms sweep |
|---|---|---|---|---|---|
| 384 | yes | no | 45.237 | 45.281 | 45.547 |
| 96 | yes | no | 14.244 | 14.374 | 15.270 |
| 128 | yes | yes | 18.279 | 18.152 | 18.452 |

Full-order solver of Table 1 in the sweep job: 17.454 ms (stored 17.416 ms).

## Iso-error table (median relative error on test64)

| level (median ≤) | ours: narrowest passing width | our median % | our worst % | our time ms | fastest operator meeting the level | operator median % | operator worst % | operator time ms |
|---|---|---|---|---|---|---|---|---|
| 0.2 % | 192 | 0.0973 | 1.5699 | 25.58 | FNO F0 seed 1 (paper architecture; seeds 0 and 2 give 0.220 % and 0.233 %) [T1 line 16; S1] | 0.195 | 0.435 | 4.06 |
| 0.1 % | 192 | 0.0973 | 1.5699 | 25.58 | not reached yet (best so far: 0.0730 % validation mean, about 0.065 % validation median, FNO Fw128m64L4 on 36,864 trajectories, 400k steps; no test evaluation [S2 line 68]) | — | — | — |
| 0.05 % | 256 | 0.0485 | 1.1586 | 33.81 | not reached yet (best so far: 0.0730 % validation mean, about 0.065 % validation median, FNO Fw128m64L4 on 36,864 trajectories, 400k steps; no test evaluation [S2 line 68]) | — | — | — |
| 0.026 % | 384 | 0.0264 | 0.8749 | 45.55 | not reached yet (best so far: 0.0730 % validation mean, about 0.065 % validation median, FNO Fw128m64L4 on 36,864 trajectories, 400k steps; no test evaluation [S2 line 68]) | — | — | — |

