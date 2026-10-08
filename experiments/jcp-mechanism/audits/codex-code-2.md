**Remaining defects prevent a CORRECT verdict.** Read-only review completed using source inspection, NPZ/hash checks, and isolated NumPy reproductions. No files modified; no GPU jobs or JAX execution.

All paths below are relative to `experiments/jcp-mechanism/`.

| File | Verdict |
|---|---|
| `a1_3d.py` | **WRONG** — G2a can silently ignore NaNs |
| `q2d/qstudy.py` | **WRONG** — same G2a defect |
| `make_configs.py`, production 3D configs | **WRONG** — incompatible cohort hashes |
| Other configs, including smoke configs | **CORRECT** for the reviewed fixes |
| `make_inputs.py`, `inputs/a2_states.json` | **CORRECT** |
| `a2_gap.py` | **CORRECT** as the data collector; adjudication belongs to `make_report.py` |
| `cluster/stage.py` | **CORRECT** |
| `cluster/submit.sh` | **WRONG** — locked queue check fails open |
| `DESIGN.md`, Amendment 4 | **NEEDS-RESTATEMENT** — dimensional scope of gate coverage |
| `make_report.py` | **WRONG** |

**Disposition of the previous seven WRONG items**

1. **Empty reference manifest:** fixed. `stage.py:113` and `submit.sh:26` skip an empty `REFS.sha256`.
2. **SSH consuming reference-loop input:** fixed. `submit.sh:18` uses `ssh -n` and redirects rsync stdin.
3. **Wrong A2 populations:** fixed. Actual NPZ shapes match the metadata: fixed 3D `(200,512)/(200,256)`, own-mesh `(100,R′)`; fixed 2D `(300,384)/(300,128)`, own-mesh `(96,R′)`. The NPZ hash matches both metadata and A2 configs.
4. **Weakened/unconsumed gates:** substantially fixed, but not fully. Full 3D G3, four-state G2c, coordinatewise G1, direct 2D bank comparison, target flags, and `gref` nodes-reached rho are implemented. G2a’s NaN handling remains defective.
5. **Non-atomic cap:** the namespace lock fixes concurrent check/submit races. A failed queue query inside that lock still permits submission.
6. **Hardcoded reference paths:** fixed. Config paths resolve through exported `TASK_ROOT`.
7. **Smoke cohort-hash mismatch:** fixed by setting the smoke expected hash to `null`, disabling that assertion. Production hashes have a separate defect below.

**Concrete remaining bugs**

1. **Production 3D cohort hashes contradict each other.**  
   `configs/a1d3_n65.json:38` expects `de2d05a4…`; `configs/a1d3_n129.json:38` expects `b7993291…`. Both specify seed `923801`, count `64`, and execute the identical mesh-independent `C.table(seed,count)` at `a1_3d.py:219–221`, sequentially on the same allocation.

   `make_configs.py:48` copies each historical job’s hash independently. The hash includes GPU-computed `s_star` (`vendor/.../b3d_common.py:150–159`), so historical floating-point differences can contaminate it. These two expected hashes cannot both match identical regenerated table bytes. Use a coherent cohort contract; do not fingerprint GPU-derived floating-point values as portable exact input identity.

2. **Both G2a accumulators can accept non-finite bank blocks.**  
   `a1_3d.py:254–261`; `q2d/qstudy.py:140–147`.

   They accumulate with Python `max(previous, block_max)`. Reproduced: `max(0., NaN)` returns `0.0`. A NaN-containing block can therefore disappear from the error maximum; the subsequent finiteness check tests the accumulator, not the arrays. Check every block’s finiteness explicitly before reduction.

3. **Submission’s authoritative queue check fails open.**  
   `cluster/submit.sh:24–26` runs:
   ```bash
   squeue ... | grep -c '^jm_' || true
   ```
   If `squeue` fails, `grep` can print `0`; `|| true` masks the failure and submission proceeds. The earlier successful queue check does not make this later check reliable. Capture and require successful `squeue` execution inside the lock before counting jobs.

4. **Report labels silently select available cases.**  
   `make_report.py:261–265` intersects the three arms’ available keys, without validating the registered cohort or listing missing cases. The loaders also ignore `complete`.

   CPU reproduction: an incomplete result containing only one dev6 case returned **R**. This violates Amendment 2’s missing-case accounting and prohibition on selecting subsets after outcomes. Plotting also assumes complete arms/settings (`285–316`), so other partial outputs crash instead of producing an incomplete-result report.

5. **Solver-provisional threshold is pooled across arms.**  
   `make_report.py:218–238` and `268–278` aggregate non-stationary exits across all three arms before applying 1%.

   Amendment 3 requires provisional status when **any arm** exceeds 1%. Reproduced: nodes at 2/100 non-stationary steps, the other arms at zero, produces **R without provisional status** because the implementation tests 2/300.

6. **Invalid-result processing can crash before returning X.**  
   `make_report.py:174–178` computes the median and exclusions before checking `invalid`. The collectors serialize non-finite values as `null`; a distance array containing `None` raises `TypeError` before the invalid branch. Likewise, `227` and `232` compare possibly null target/sensitivity statistics numerically.

   Invalid/non-finite evidence must be handled before numerical summaries.

7. **Intermediate X retains a bootstrap bound.**  
   `make_report.py:184–192` stores the bound before choosing R/N/X and never clears it for intermediate X. Reproduced: median recovered fraction `0.7` returns `label='X', boot_low=0.7`. Amendment 3 explicitly requires **N/A for X and X0**.

8. **Unresolved A2 slopes crash report generation.**  
   The analysis correctly returns `None` when survival/window requirements fail, but `make_report.py:428–429` formats upwind and central slopes with `:.2f`. That raises `TypeError`. “Unresolved slope” must remain a reportable outcome.

9. **Independent-family validation misses own-mesh state sets.**  
   `make_report.py:72–75` checks only fixed-state independent-family rho. Lines `96–103` report own-mesh gaps without consuming their saved `*_own*_lat32768_rho` / `*_own*_fib121393_rho`.

   Amendment 3 declares target validity **per state set**. Own-mesh results can therefore be presented without flagging an independent-family failure.

10. **Report text and diagnostics do not faithfully expose the decisions.**
    - `564–565` still hardcodes **104/150 states**, contradicting restored **200/300** populations.
    - `175`, `261–264`, and `549–552`: exclusions are positional indices after lexicographic key sorting; the rendered report prints only their count, not the required case identities.
    - `177` records invalidity reasons, but the rendered label summaries/table never display `why`.
    - `344–363` reports historical **coefficient-space** differences, not the amended 3D field distance. `368–380` omits the required 2D field comparison on saved audit cases 0 and 2.

**What the new report implements correctly**

For finite, complete inputs, it correctly implements:

- A2 screening on **every window mesh**, strict `>100×check` and `>1e-12`, at least 25% survival and three meshes.
- All three slope statistics and the upwind/central acceptance intervals.
- C-pos slope tolerances, leading-coefficient minimum, and finest-mesh vector agreement; C-neg-a’s strict `|s|<0.05`.
- Fixed-population independent-family `1e-2` rejection.
- A1 X → X0 → R/N/X ordering, separation exclusion, seed-0 2,000-resample bootstrap and 2.5th percentile, and sensitivity threshold `0.1×median(s)`.
- False 3D `continuum_target_valid` or 2D `targets_valid` forcing X when the surrounding data are numerically usable.

**Amendment 4:** A4-1 and A4-2 are **CORRECT**. A4-3 needs an explicit **3D** qualifier for its chunked G3/four-state Jacobian G2c prescription. The 2D implementation uses full `jacfwd` at one state and value-only G2c; the amendment currently states the stronger coverage without a dimensional qualifier.

**Remaining WRONG: `a1_3d.py`, `q2d/qstudy.py`, `make_configs.py` and production 3D configs, `cluster/submit.sh`, `make_report.py`.**