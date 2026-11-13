## Reproduction of the existing row

Stored: job 4319328 (pax105, NVIDIA A100 80GB PCIe). Reproduction: job 7174297 (pax106, NVIDIA A100 80GB PCIe), same driver/config/checkpoint.

| width | median % stored | median % repro | worst % stored | worst % repro | errors bit-identical | ms stored | ms repro | ratio |
|---|---|---|---|---|---|---|---|---|
| 384 | 0.026419 | 0.026419 | 0.874860 | 0.874860 | no | 45.237 | 45.281 | 1.0010 |
| 96 | 0.529332 | 0.529332 | 3.845644 | 3.845644 | yes | 14.244 | 14.374 | 1.0091 |
| 128 | 0.267509 | 0.267509 | 2.546901 | 2.546901 | no | 18.279 | 18.152 | 0.9931 |

Full-order solver of Table 1 (`lean_nt3e-3_l3e-3_dt005__graphs`): stored 17.416 ms, reproduction 17.127 ms.
Gates of the reproduction job: {'evaluation_cohort_hash': True, 'phi_free_operator_parity': True, 'sep_lattice_projection': True, 'drift': True, 'neighbour': True, 'repetition_output_identical': True, 'retained_repetitions': True}. Check passed: True.

