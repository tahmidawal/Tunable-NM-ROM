You are an adversarial reviewer. DO NOT modify, create or delete any file in this repository; read only. Do not run GPU jobs, do not use ssh, do not run git commands that write.

Scope: experiments/poisson-bank-knob-3d/ (DESIGN.md with amendment A1, pbk3_core.py, pbk3_prep.py, pbk3_lshape.py, pbk3_cube.py, pbk3_audit_np.py, make_tables.py, make_report.py, config-*.json, frozen-N32.json, frozen-N64.json, reports/tables.generated.md, reports/summary.json, runs/*/archive/output/result.json + audit.json). Parents for comparison: experiments/hires-poisson/hpl_solve.py, experiments/hires-poisson/hp3d_solve.py, experiments/hires-poisson/lshape/lsh_core.py, experiments/paper-p3d/{poisson.py,common.py,iterative_cg.py}, experiments/poisson-bank-knob/pbk_core.py (sibling lane).

Check each item and give a verdict CORRECT / WRONG / NEEDS-RESTATEMENT with a one-paragraph reason and file:line evidence:
1. Rotation math (make_rotation): T = R_G^{-1} V_s, L = V_s^T R_G, training data only (no development/final case enters pbk3_prep.py).
2. Truncated NM-ROM arm: residual through B T[:, :R'] L[:R'] (B at R'=R exactly), corrections eliminated with Q,R of (B P) C_q, decode (G T)[:, :R'] (L[:R'] c). Is this the same as the sibling lane's construction? Any off-by-one/ordering bug in the nested column blocks (split_blocks/nblocks/decode)?
3. Linear rung: least squares of B T[:, :R'] against the weak source, decode from the same blocks; L-shape uses M=257 so R' in {384,512} not constructible — correctly excluded?
4. Cube lean query: DST route with weak moments dst3(f)[triples]/(n^1.5 lambda) equals poisson.engine's projection (parity to engine is gated at n<=64). Anything that would make the parity gate vacuous?
5. Timing: fused_device_seconds and total_seconds contract identical for ROM and CG subjects; randomized order; burn-in; slow CG in its own phase. Is the device-guard issue in amendment A1 correctly diagnosed from runs/c32diag*/archive/output/result.json (neighbour rows with variant field)? Is the amended order gate (pbk3_core.order_gate) a meaningful test, and would it detect a real order effect (check its negative control claim)?
6. make_tables.py rule: accurate = min worst error (tie->cheaper), fast = cheapest with worst <= orig_q0 worst, FOM = fastest CG with worst <= accurate worst, speedup = FOM ms / arm ms in the Table-1 scope (L-shape total_seconds, cube fused_device_seconds). Any bug (e.g. CG summarised from wrong phase, medians mixing phases, wrong scope)?
7. Final-cohort discipline: frozen-N32/N64.json were created from development runs (c32b, c64b) and committed before c32final/c64final were staged (check git log order of commits touching frozen-N*.json vs runs/c32final, runs/c64final). The final job cannot change the selected arms.
8. The audit (pbk3_audit_np.py) is independent of the driver's truth path, and its controls would fail on real data.
9. Claims in reports/tables.generated.md: are the accurate/fast rows correctly derived from the result JSONs? Recompute at least two rows by hand from result.json.
10. Anything else that would make a Table-1 number wrong or misleading (e.g. the 32^3/64^3 speed gain coming from the linear rung / DST route rather than from R' truncation — is the report explicit?).

Write your answer as a numbered list with verdicts, then a short list of concrete defects (if any).
