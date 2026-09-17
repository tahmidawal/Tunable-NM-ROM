# Self-audit — Phase 2 attempt `ns201` (job 3783796) and the §A4 amendment

**This is a written self-audit by the lane agent, not an independent-model audit.** Codex is
quota-blocked until 2026-09-19 11:33 (DESIGN §A5). Each claim below names the JSON field it
rests on and the check that was run. The mechanical check is `audit_phase2.py`
(`artifacts/ns201/audit.json`, 67 checks); its six MISMATCH rows are a finding, not a bug in
the audit, and are explained under claim 4.

| # | claim | rests on | check run | status |
|---|---|---|---|---|
| 1 | The `ns201` bank has numerical rank 128 of R=256 at every evaluation mesh | `gates.B-ORTH_N{64,128,256}.rank` | `audit_phase2.py` recomputes the bank in NumPy from the pickled weights (integer Fourier features → SiLU MLP → mean-zero columns) and its SVD rank; equals 128 at all three meshes | verified |
| 2 | The rank cap is structural: rank(G) ≤ g_hidden because the g-MLP's last layer is linear | `config.ARCH.g_hidden` = 128; `sep_common.apply_mlp` (last layer `x @ w + b`) | read the code; the parent lane's `sep_burgers_r3.py` header records the same landmine and the `G_HIDDEN >= 2R` fix; smoke on the patched driver gives rank 16/16 for R=16, g_hidden=32 | verified by reading + smoke; the cluster rerun is the confirmation at scale |
| 3 | H-ORACLE fails: oracle median 0.203 vs POD-16 median 0.240 (ratio 1.19, bar 2.0) at every mesh | `oracle.{N}.oracle_median`, `.podK_median`; `gates.H-ORACLE_N{N}` | audit recomputes both medians from the saved per-state arrays (`oracle_N{N}.npz`) and the pass rule; matches to 1e-9; the JSON per-state lists equal the npz arrays exactly | verified |
| 4 | The reported oracle numbers are contaminated by ≤ 1.1e-2 relative (64²), 7.6e-3 (128²), 3.0e-3 (256²) because they were computed through R_b⁻¹ of a singular R_b | `oracle_N{N}.npz: Z, e_oracle`; ns101 `dev8_N{N}.npz` | audit recomputes ‖G h(z) − u‖/n₀ exactly at the saved codes on the 48 archived states; a debug run reproduced the driver's formula in NumPy (|c|max ≈ 1e12, cancellation) and showed the exact norm equals the rank-128 projection formula to all digits; the patched driver reports `formula_vs_field_worst_rel` = 1.3e-16 on a full-rank smoke | verified; contamination does not change any verdict |
| 5 | The bank floor gate B-FLOOR passes (ratio 1.82–1.85 ≤ 2) but is itself computed with a 256-column QR whose 128 extra columns are roundoff directions | `floors.{N}.bank.256.per_case_worst_fixed`, `gates.B-FLOOR_N{N}` | audit recomputes the per-case floor for the 8 archived cases with a rank-128 projection: differs by 2.0e-2 (64²), 7.4e-3 (128²), 2.6e-3 (256²), recorded as value-level checks; the gate arithmetic from the stored numbers matches exactly | verified with the caveat recorded |
| 6 | The 256² cohort hash mismatch on `pax049` is node-dependent roundoff, not a code change | `gates.B-DATA_{train,dev}_N256` (ns201, `pax049`) vs the same gates in the `ns202` log (`pax106`, PASS, same commit `dddb41fb`) | compared the two job logs; 64² and 128² dev hashes match on both nodes | plausible, not proven by value: `ns201` saved no fields. The value gate (§A4, ≤1e-8 on the archived 8 trajectories) is asserted in every later job; the audit's claim-4 agreement on 256² states (3e-3, dominated by the cancellation effect) does not bound it tightly |
| 7 | `dev8_eval_ref.npz` is Phase 1's data | cut from `runs/ns101/archive/.../dev8_N{64,128,256}.npz` (job 3780151, archived and checksum-verified) at output indices [0,5,10,15,20,25] | audit checks the archived `physical` rows equal a column-by-column NumPy redraw of seed 20260918 (max abs diff 0) | verified |
| 8 | H-SOLVED passes: single-start median within 1.5× of the oracle (0.213 vs 0.203) | `oracle.{N}.single_start_median` | audit recomputes from the npz arrays and the pass rule | verified (but the number carries the claim-4 contamination) |
| 9 | Training reconstruction: rel-L2 mean 0.209, median 0.206, max 0.621, 30 000 steps, 3202 s | `training.*` | reported by the driver from the fitted codes; not independently recomputable (training fields not saved). Consistent with the oracle level (0.20) | not independently verified |
| 10 | The reruns `ns203`/`ns204` change only `G_HIDDEN` (2R), the head width/depth (512×3) and the step count (100 000) | `configs/phase2-k{16,32}r{256,512}-v2.env` vs `phase2-k{16,32}r{256,512}.env`; `runs/ns20{3,4}/PROVENANCE.json` | `diff` of the env files; staged bytes equal the committed blobs at `beffbb1b` | verified |

## What this self-audit cannot establish

- Whether the H-ORACLE bar (oracle ≤ ½ POD-K) is reachable on this family by this decoder:
  the parent lane's evidence on Burgers (longer training ≈ 2×) is the only calibration.
- Whether the rank-capped bank materially biased the *head's* training (the head was trained
  against a 128-dimensional effective span while nominally 256): a full-rank rerun answers it.
- Claim 6 by value, until a job on a mismatching node reports `value_worst_rel`.
