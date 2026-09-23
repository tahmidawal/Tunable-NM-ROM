summary.json sha256 f19d01721a0ba999518eeb3f05ee49ac4f0c10f52ca43313d35ce4c08940f67b

**Step 0 — stage split (median ms, separately jitted stages; fused GPU-query and host copies alongside):**

| mesh | arm | project+start | LM solve | y elim | reconstruction u=Gc | fused GPU | host copies | dominant |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 256² | `R512_q256` | 0.48 | 2.05 | 0.21 | 0.27 | 2.25 | 0.98 | lm_solve |
| 256² | `R512_q0` | 0.21 | 2.23 | 0.14 | 0.27 | 1.98 | 0.99 | lm_solve |
| 256² | `R512_linear` | 0.75 | 0.00 | 0.00 | 0.46 | 0.77 | 0.85 | project_and_start |
| 256² | `R128_linear` | 0.19 | 0.00 | 0.00 | 0.15 | 0.59 | 0.89 | project_and_start |
| 1024² | `R512_q256` | 0.97 | 1.76 | 0.20 | 2.85 | 4.83 | 3.53 | reconstruction |
| 1024² | `R512_q0` | 0.23 | 2.13 | 0.14 | 2.85 | 4.60 | 3.51 | reconstruction |
| 1024² | `R512_linear` | 0.24 | 0.00 | 0.00 | 3.86 | 3.54 | 3.39 | reconstruction |
| 1024² | `R128_linear` | 0.19 | 0.00 | 0.00 | 1.87 | 1.44 | 3.39 | reconstruction |
| 2048² | `R512_q256` | 1.24 | 1.56 | 0.20 | 11.01 | 13.23 | 13.69 | reconstruction |
| 2048² | `R512_q0` | 0.29 | 2.38 | 0.15 | 11.03 | 12.98 | 13.66 | reconstruction |
| 2048² | `R512_linear` | 0.34 | 0.00 | 0.00 | 12.18 | 12.01 | 13.56 | reconstruction |
| 2048² | `R128_linear` | 0.56 | 0.00 | 0.00 | 3.69 | 3.74 | 13.50 | reconstruction |
| 4096² | `R512_q256` | 1.97 | 1.66 | 0.26 | 40.96 | 43.73 | 56.26 | reconstruction |
| 4096² | `R512_q0` | 0.75 | 2.33 | 0.16 | 40.93 | 43.42 | 56.34 | reconstruction |
| 4096² | `R512_linear` | 0.99 | 0.00 | 0.00 | 41.79 | 42.39 | 55.82 | reconstruction |
| 4096² | `R128_linear` | 0.74 | 0.00 | 0.00 | 12.52 | 12.54 | 56.11 | reconstruction |

**Table-1 settings (pre-registered rule) per mesh:**

| mesh | GPU | job | accurate arm | err % | GPU ms | fast arm | err % | GPU ms | FOM (CG) | FOM err % | FOM ms | × accurate | × fast | gates |
|---|---|---|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---|
| 256² | NVIDIA A100 80GB PCIe | 4199318 | `R512_linear` | 0.746 | 0.77 | `R128_linear` | 2.314 | 0.59 | `cg_0.03` | 0.548 | 10.3 | 13.4 | 17.5 | FAIL: neighbour, drift |
| 1024² | NVIDIA A100 80GB PCIe | 4199318 | `R512_linear` | 0.742 | 3.54 | `R128_linear` | 2.306 | 1.44 | `cg_0.03` | 0.220 | 98.7 | 27.9 | 68.5 | FAIL: neighbour, drift |
| 2048² | NVIDIA A100 80GB PCIe | 4199318 | `R512_linear` | 0.742 | 12.01 | `R128_linear` | 2.306 | 3.74 | `cg_0.1` | 0.609 | 650.2 | 54.1 | 174.0 | FAIL: neighbour |
| 4096² | NVIDIA A100 80GB PCIe | 4199321 | `R512_linear` | 0.742 | 42.39 | `R128_linear` | 2.306 | 12.54 | `cg_0.1` | 0.451 | 4490.0 | 105.9 | 358.0 | FAIL: neighbour |

**R' ladder — bank-span linear rung q=R' (worst err % / floor % / GPU ms / × vs own matched CG):**

| R' | 256² | 1024² | 2048² | 4096² |
|---:|---|---|---|---|
| 512 | 0.746 / 0.746 / 0.77 / 13× (`cg_0.03`) | 0.742 / 0.742 / 3.54 / 28× (`cg_0.03`) | 0.742 / 0.742 / 12.01 / 54× (`cg_0.1`) | 0.742 / 0.742 / 42.39 / 106× (`cg_0.1`) |
| 384 | 0.773 / 0.773 / 0.71 / 14× (`cg_0.03`) | 0.769 / 0.769 / 2.78 / 35× (`cg_0.03`) | 0.768 / 0.768 / 8.92 / 73× (`cg_0.1`) | 0.768 / 0.768 / 32.48 / 138× (`cg_0.1`) |
| 256 | 0.945 / 0.945 / 0.67 / 15× (`cg_0.03`) | 0.941 / 0.941 / 2.04 / 42× (`cg_0.1`) | 0.940 / 0.940 / 6.31 / 103× (`cg_0.1`) | 0.940 / 0.940 / 22.56 / 182× (`cg_0.2`) |
| 128 | 2.314 / 2.314 / 0.59 / 15× (`cg_0.1`) | 2.306 / 2.306 / 1.44 / 54× (`cg_0.2`) | 2.306 / 2.306 / 3.74 / 160× (`cg_0.2`) | 2.306 / 2.306 / 12.54 / 299× (`cg_0.4`) |
| 64 | 5.345 / 5.345 / 0.56 / 16× (`cg_0.1`) | 5.336 / 5.336 / 1.07 / 73× (`cg_0.2`) | 5.335 / 5.335 / 2.36 / 228× (`cg_0.4`) | 5.335 / 5.335 / 7.12 / 508× (`cg_0.5`) |
| 32 | 9.176 / 9.176 / 0.20 / 39× (`cg_0.2`) | 9.166 / 9.166 / 0.91 / 73× (`cg_0.5`) | 9.165 / 9.165 / 1.61 / 302× (`cg_0.7`) | 9.165 / 9.165 / 4.39 / 769× (`cg_0.7`) |

**R' ladder — head-only q=0 (worst err % / floor % / GPU ms / × vs own matched CG):**

| R' | 256² | 1024² | 2048² | 4096² |
|---:|---|---|---|---|
| 512 | 3.157 / 0.746 / 1.98 / 4× (`cg_0.1`) | 3.150 / 0.742 / 4.60 / 17× (`cg_0.2`) | 3.149 / 0.742 / 12.98 / 43× (`cg_0.3`) | 3.149 / 0.742 / 43.42 / 86× (`cg_0.4`) |
| 384 | 3.156 / 0.773 / 1.97 / 4× (`cg_0.1`) | 3.148 / 0.769 / 3.94 / 20× (`cg_0.2`) | 3.148 / 0.768 / 10.37 / 54× (`cg_0.3`) | 3.148 / 0.768 / 33.66 / 111× (`cg_0.4`) |
| 256 | 3.142 / 0.945 / 1.88 / 5× (`cg_0.1`) | 3.135 / 0.941 / 3.27 / 24× (`cg_0.2`) | 3.135 / 0.940 / 7.47 / 76× (`cg_0.3`) | 3.135 / 0.940 / 23.79 / 157× (`cg_0.4`) |
| 128 | 3.044 / 2.314 / 1.85 / 5× (`cg_0.1`) | 3.036 / 2.306 / 2.61 / 30× (`cg_0.2`) | 3.036 / 2.306 / 4.95 / 114× (`cg_0.3`) | 3.036 / 2.306 / 13.90 / 269× (`cg_0.4`) |
| 64 | 5.430 / 5.345 / 1.77 / 5× (`cg_0.1`) | 5.420 / 5.336 / 2.23 / 35× (`cg_0.2`) | 5.420 / 5.335 / 3.55 / 152× (`cg_0.4`) | 5.420 / 5.335 / 8.35 / 433× (`cg_0.5`) |
| 32 | 9.176 / 9.176 / 5.13 / 2× (`cg_0.2`) | 9.166 / 9.166 / 5.25 / 13× (`cg_0.5`) | 9.165 / 9.165 / 6.25 / 78× (`cg_0.7`) | 9.165 / 9.165 / 9.17 / 368× (`cg_0.7`) |

**Verdict inputs:**
- 256²: C_q reference q=max: monotone err False, monotone cost False, R'=512/R'=32 cost 0.44×; nm-rom q=0: monotone err False, monotone cost False, R'=512/R'=32 cost 0.39×; linear rung q=R': monotone err True, monotone cost True, R'=512/R'=32 cost 3.78×
- 1024²: C_q reference q=max: monotone err False, monotone cost False, R'=512/R'=32 cost 0.92×; nm-rom q=0: monotone err False, monotone cost False, R'=512/R'=32 cost 0.88×; linear rung q=R': monotone err True, monotone cost True, R'=512/R'=32 cost 3.90×
- 2048²: C_q reference q=max: monotone err False, monotone cost False, R'=512/R'=32 cost 2.12×; nm-rom q=0: monotone err False, monotone cost False, R'=512/R'=32 cost 2.08×; linear rung q=R': monotone err True, monotone cost True, R'=512/R'=32 cost 7.47×
- 4096²: C_q reference q=max: monotone err False, monotone cost True, R'=512/R'=32 cost 4.77×; nm-rom q=0: monotone err False, monotone cost False, R'=512/R'=32 cost 4.73×; linear rung q=R': monotone err True, monotone cost True, R'=512/R'=32 cost 9.65×

**Gates:**
- 256² (ABA): {'parity': True, 'deterministic': True, 'cg_converged': True, 'neighbour': False, 'drift': False, 'profile_matches_fused': True, 'device_guard': True}; audit PASS; parity 1.4e-13, 5.2e-13; primary-arm neighbour worst 1.458, failing primary ['R32_q0@romA1 1.425', 'R64_linear@romA2 1.458'], failing reference ['R64_q32@romA1 1.101', 'R256_q224@romA2 1.307', 'R128_q96@romA2 1.180', 'R64_q32@romA2 1.335'], drift failing ['R256_q224 0.879']
- 1024² (ABA): {'parity': True, 'deterministic': True, 'cg_converged': True, 'neighbour': False, 'drift': False, 'profile_matches_fused': True, 'device_guard': True}; audit PASS; parity 1.4e-13, 5.4e-13; primary-arm neighbour worst 1.663, failing primary ['R32_q0@romA2 1.663'], failing reference ['R256_q224@romA2 1.394', 'R128_q96@romA2 1.164'], drift failing ['R128_linear 0.898']
- 2048² (ABA): {'parity': True, 'deterministic': True, 'cg_converged': True, 'neighbour': False, 'drift': True, 'profile_matches_fused': True, 'device_guard': True}; audit PASS; parity 1.4e-13, 5.2e-13; primary-arm neighbour worst 1.079, failing primary none, failing reference ['R128_q96@romA1 1.204', 'R128_q96@romA2 1.129'], drift failing none
- 4096² (ABA): {'parity': True, 'deterministic': True, 'cg_converged': True, 'neighbour': False, 'drift': True, 'profile_matches_fused': True, 'device_guard': True}; audit PASS; parity n/a; primary-arm neighbour worst 1.165, failing primary ['R32_q0@romA1 1.165'], failing reference ['R64_q32@romA2 1.185'], drift failing none

- 256²: pbkH/output result.json sha256 a4eacc7b35d2a929a4df7cb261b330d61ef4f81f4de14ecb6a828610b7fd153f, commit 50f2df489a465100dd5a5dd7e947c7758e109acf, GPU NVIDIA A100 80GB PCIe GPU-15e19d78-1dff-7ce0-3e6d-2ffe9e4ba214
- 1024²: pbkH/output2 result.json sha256 d5b9cadedc2be56ad8f6b9f80a8b3d9670d5af011e632bfedcf2ae74a7163724, commit 50f2df489a465100dd5a5dd7e947c7758e109acf, GPU NVIDIA A100 80GB PCIe GPU-15e19d78-1dff-7ce0-3e6d-2ffe9e4ba214
- 2048²: pbkH/output3 result.json sha256 5b5bd43ee622db959f00376a9c7f72b2706eff67be9009850ad87aee880615c0, commit 50f2df489a465100dd5a5dd7e947c7758e109acf, GPU NVIDIA A100 80GB PCIe GPU-15e19d78-1dff-7ce0-3e6d-2ffe9e4ba214
- 4096²: pbkI/output result.json sha256 780ee322774f9c3479b67e059c5e8c306980a01e59de984215ed5104fe08fd9b, commit 50f2df489a465100dd5a5dd7e947c7758e109acf, GPU NVIDIA A100 80GB PCIe GPU-c58462eb-c0bf-1c85-741e-f8dc31e64312
