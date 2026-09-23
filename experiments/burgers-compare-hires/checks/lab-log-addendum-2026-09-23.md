
### burgers-compare-hires — addendum (15:30 EDT): 2048^2 is FINAL from p2048e; namespace empty, nothing running

The H200 panel `p2048e` (job 4218390) completed 15:06 EDT, 0 failed gates, remote/local audits agree to 5.9e-15, so by
the pre-registered A7 rule it replaces the A100 fallback `p2048f` as the 2048^2 panel (p2048f: run, not used; identical
errors, other GPU's timings, never mixed). The 2048^2 table in the entry above is SUPERSEDED by this one. Quadratic
manifold r=64 is dropped at 2048^2 (XLA autotuning failure on the 72 GB bank); POD-512 is present. Summary
`checks/p2048e-summary.json` SHA256 3b10737153515abe7de715d507c46a481ab3f8f4f440d01e6154e31aa8f8a735; report + `reports/summary.json`
(45a5946a…) regenerated, status FINAL. `bcmp_20260923/` is empty.

## $2048^2$ — attempt `p2048`, job 4218390, NVIDIA H200, commit `f57b2effce`
| method | unknowns | worst % | median % | GPU ms | FOM chosen (its GPU ms, worst %) | speedup | status / notes |
|---|---|---|---|---|---|---|---|
| NM-ROM fast (q=0, M=64) | 16 | 2.3714 | 1.0499 | 26.92 | `lean_nt1e-2_l1e-2_dt01` (63.74, 1.6773) | 2.37× | confirmed at 2048^2 (hires-burgers hb2k02: held-out rho 0.0404, deployed 0.0424, bar 0.116) |
| reference only: NM-ROM with corrections (q=256, M=1088) | 272 | 0.5979 | 0.1857 | 112.1 | `lean_nt3e-3_l3e-3_dt005` (122.5, 0.0494) | 1.09× | MARGINAL at 2048^2: passes burgers-eqcert bc2048b draws (rho_max 0.0794, confirmation 0.1156 vs bar 0.116 -- thin) but reaches 0.183 on other trajectories (unaudited exploration); labelled marginal |
| reference only: NM-ROM with corrections (q=256), robust rule (1 exact step) | 272 | 0.5979 | 0.1857 | 1,836.9 | `lean_nt3e-3_l3e-3_dt005` (122.5, 0.0494) | 0.067× | confirmed at 2048^2 (burgers-eqcert bc2048b: j=1 confirmation 0.051) |
| NM-ROM bank-span R'=128 (M=512, `lat128` m=16129) | 128 | 1.8719 | 0.6287 | 125.6 | `lean_nt1e-2_l1e-2_dt01` (63.74, 1.6773) | 0.51× | held-out rho_max 0.1624 over all states (k>=0, the house convention) -- EXCEEDS the 0.116 bar; 0.0329 over time-stepped states (k>=1); 408 states |
| NM-ROM bank-span R'=128 (M=512, `lat64` m=3969) | 128 | 1.8716 | 0.6286 | 73.54 | `lean_nt1e-2_l1e-2_dt01` (63.74, 1.6773) | 0.87× | held-out rho_max 0.0479 over all states (k>=0, the house convention) -- within the 0.116 bar; 0.0182 over time-stepped states (k>=1); 408 states |
| NM-ROM bank-span R'=256 (M=1024, `lat128` m=16129) | 256 | 0.7282 | 0.1213 | 224.2 | `lean_nt3e-3_l3e-3_dt005` (122.5, 0.0494) | 0.55× | held-out rho_max 0.3521 over all states (k>=0, the house convention) -- EXCEEDS the 0.116 bar; 0.0385 over time-stepped states (k>=1); 408 states |
| NM-ROM bank-span R'=256 (M=1024, `lat64` m=3969) | 256 | 0.7293 | 0.1213 | 126.1 | `lean_nt3e-3_l3e-3_dt005` (122.5, 0.0494) | 0.97× | held-out rho_max 0.0780 over all states (k>=0, the house convention) -- within the 0.116 bar; 0.0227 over time-stepped states (k>=1); 408 states |
| NM-ROM bank-span R'=384 (M=1536, `lat128` m=16129) | 384 | 0.2194 | 0.0652 | 346.6 | `lean_nt3e-3_l3e-3_dt005` (122.5, 0.0494) | 0.35× | held-out rho_max 0.4145 over all states (k>=0, the house convention) -- EXCEEDS the 0.116 bar; 0.0345 over time-stepped states (k>=1); 408 states |
| NM-ROM bank-span R'=384 (M=1536, `lat64` m=3969) | 384 | 0.2195 | 0.0652 | 186.4 | `lean_nt3e-3_l3e-3_dt005` (122.5, 0.0494) | 0.66× | held-out rho_max 0.0777 over all states (k>=0, the house convention) -- within the 0.116 bar; 0.0099 over time-stepped states (k>=1); 408 states |
| NM-ROM bank-span R'=512 (M=2048, `lat128` m=16129) | 512 | 0.2909 | 0.1090 | 464.1 | `lean_nt3e-3_l3e-3_dt005` (122.5, 0.0494) | 0.26× | held-out rho_max 0.5058 over all states (k>=0, the house convention) -- EXCEEDS the 0.116 bar; 0.0446 over time-stepped states (k>=1); 408 states |
| NM-ROM bank-span R'=512 (M=2048, `lat64` m=3969) | 512 | 0.2907 | 0.1090 | 242.8 | `lean_nt3e-3_l3e-3_dt005` (122.5, 0.0494) | 0.50× | held-out rho_max 0.0573 over all states (k>=0, the house convention) -- within the 0.116 bar; 0.0139 over time-stepped states (k>=1); 408 states |
| POD-LSPG k=16 (M=64) | 16 | 29.3744 | 22.2014 | 290.8 | `nt1e-2_dt01` (56.68, 3.5129) | 0.19× |  |
| POD-LSPG k=64 (M=256) | 64 | 7.4711 | 3.6922 | 1,491.8 | `nt1e-2_dt01` (56.68, 3.5129) | 0.038× |  |
| POD-LSPG k=256 (M=1024) | 256 | 1.1681 | 0.4611 | 19,106.2 | `lean_nt3e-3_l3e-3_dt005` (122.5, 0.0494) | 0.006× |  |
| POD-LSPG k=512 (M=2048) | 512 | 0.4876 | 0.0554 | 69,310.3 | `lean_nt3e-3_l3e-3_dt005` (122.5, 0.0494) | 0.002× |  |
| quadratic manifold r=16 (M=64) | 16 | 22.6973 | 15.0651 | 1,306.1 | `nt1e-2_dt01` (56.68, 3.5129) | 0.043× |  |
| quadratic manifold r=32 (M=128) | 32 | 13.0600 | 6.7252 | 3,816.5 | `nt1e-2_dt01` (56.68, 3.5129) | 0.015× |  |
| FNO (`fno-large`) | — (no online solve) | 8.0222 | 5.8823 | 458.7 | `nt1e-2_dt01` (56.68, 3.5129) | 0.12× | trained at 2048^2, 19 epochs (killed (host memory) after 19 completed epochs, 2897 s of the 3000 s budget) |
| FNO (`fno-large-256`) | — (no online solve) | 6.1086 | 2.8851 | 459.5 | `nt1e-2_dt01` (56.68, 3.5129) | 0.12× | trained at 256^2 |
| U-Net (`unet-refine`) | — (no online solve) | 9.4989 | 7.7430 | 151.4 | `nt1e-2_dt01` (56.68, 3.5129) | 0.37× | trained at 2048^2, 62 epochs (wall_budget) |
| Transolver (`tsol-refine`) | — (no online solve) | 18.7123 | 13.9462 | 143.0 | `nt1e-2_dt01` (56.68, 3.5129) | 0.40× | trained at 2048^2, 24 epochs (wall_budget) |
| DeepONet (`don-small`) | — (no online solve) | 38.2515 | 21.2456 | 83.63 | `nt1e-2_dt01` (56.68, 3.5129) | 0.68× | trained at 2048^2, 69 epochs (wall_budget) |
