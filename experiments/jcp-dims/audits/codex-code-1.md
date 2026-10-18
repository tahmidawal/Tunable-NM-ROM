**Overall: NEEDS-FIX. Do not launch the current smokes.** The design is ready, but the implementation has reproducible integration defects and acceptance gaps.

This audit was read-only: no files changed, jobs submitted, GPU use, or scratch files created. I reran the 72 CPU checks without invoking their file-writing entry point; all passed. I also regenerated all 17 config files in memory: they match the committed files exactly. Those checks do **not** cover the defects below.

All file references below are relative to `experiments/jcp-dims/`.

**Ranked findings**

1. **Wrong square-system anchors in `dtime`.**  
   `dtime.py:36` imports **`qacc`**, and lines 175–176 run tight/tighter anchors through `QA.run`. `qacc.py:74` uses the inherited `P.tol_of`, whose LSPG tight/tighter residual tolerances are zero. Production uses the amended square tolerances through `dcert`, but its anchors do not. This violates A2.1 and can invalidate square timing qualification. Use the square-aware runner consistently.

2. **A 3D timing panel can pass without in-job certification.**  
   At `d3run.py:375–378`, `scored_equals_certified` returns `None` or `False`, but this result never affects invocation validity or `valid_all` at line 414. Reference A’s expected hash is also generated without checking certification (`:346`).  
   **Reproduced with the actual `run_panel` function and tiny mocked queries:** empty certification stores produced `scored_equals_certified=None` and **`valid_all=True`**. Require complete certification and matching identity for A and every ordinary B before timing acceptance.

3. **Adaptive 3D timings omit full-field decoding.**  
   `d3run.py:83–86` returns coefficients, scale and statistics; it never decodes fields. `subject` times that query at `:324–328`, whereas FS/QN queries include decoding through `s3common.make_query`. These are different cost contracts, despite the registered full-field mesh-path contract. A tiny CPU adaptive solve confirmed the returned keys are only `W`, `scale`, `stats`, `w0`.

4. **Missing lower candidates can produce a false \(m^*\).**  
   `dstats.py:135–141` returns the first common qualified candidate before checking missing lower candidates. CPU counterexample:
   ```
   BE: [missing, qualified]
   CN: [qualified, qualified]
   ```
   returns `m* = 2`, instead of “not reached (incomplete ladder).”

   `family_mstar`, lines 116–127, has the analogous intersection problem: a draw with an earlier individual \(m^*\) can hide a missing candidate below the eventual all-draw \(m^*\). A five-draw counterexample returned `m* = 3` despite missing evidence at candidate 2. Apply missing-lower-candidate precedence to the **intersection**, not just individual ladders.

5. **The 3D timing identity record is incomplete as an audit trail.**  
   `payload_hash` itself hashes the entire returned tree correctly. However:
   - `d3run.py:390` records keys once per setting/rule block, using the **last subject’s** `names`; FS/QN/adaptive subjects can have different payloads. A3.1 requires the list per subject.
   - `scored_equals_certified` compares only six output vectors for FS/QN (`:435`), not their internal execution payload.
   - `run_ladder` calls `Ctx.rollouts` without retaining internal states (`:225`). `s3ctx.py:163–174` retains `W_all` only when `keep_wall` requests it. The resulting archive cannot independently reconstruct every step certificate for new ladder configurations.

6. **Current `smk2`, `b2f` and `b2m` crash on fixgate blocks.**  
   `dacc.py:274–275` unconditionally evaluates `blk['setting']`. Fixgate blocks generated at `make_configs.py:139` and `:197–199` have `r`/`M`, but no `setting`. Their intended second dispatch loop at `dacc.py:289` is never reached.  
   **Reproduced from all three generated configs:** `KeyError('setting')`.

7. **Timing-block cleanup retains the supposedly released tables.**
   - **2D:** `dtime.py:251` leaves `d` and `q` bound; `cold`, `F`, `fo`, `W`, and `s_` also survive the loop. Deleting `st` at `:289` does not release the table dictionary referenced by `d`. The next block is built while these references remain alive.
   - **3D:** after `for x in Bs`, `x` remains bound to the last subject and its closure over `d` (`d3run.py:391–396`). `del Bs` does not release it. `cx.queries` is never cleared, and adaptive cache cleanup is literally `pass`.

   The 1 GB check can therefore abort a healthy block because cleanup is incomplete. On smaller blocks it can pass while carrying stale allocations into the next block.

8. **The smokes do not exercise the registered maximum live, compiled workload.**
   - `make_configs.py:179–182`: `smk2` builds gauss384 as a successor but never runs the heavy gauss384 production query.
   - `make_configs.py:204–208`: the largest `smk2t` block contains LSPG only, without the heavy GAL subject.
   - `make_configs3d.py:57–58`: `smk3` pilot builds tables/form data only. It omits gl48 from the registered tail and does not compile/run the heavy ADT/GAL queries.
   - Its timing panel (`:71–75`) uses W512 subjects, not the W1024 gl64/lat262144 block sequence required by A2.4.
   - Peak memory is logged, but the drivers do not enforce the **120 GB** acceptance threshold.

9. **Smoke coverage and evidence are silently narrower than registered.**
   - `make_configs.py:187–190` supplies one non-default draw and one production rung per random family; its successor table is used for rho only. This is not the specified two draws/two production rungs.
   - `make_configs3d.py:61` first saves W512/lat16384/CN on cases 0–3; line 63 reruns the same key on cases 0–1. `Ctx.rollouts` overwrites `Wstore`/`Cstore` under that key, losing the four-case reproduction archive.
   - `smk3` does not certify its reference A before the panel.
   - The square conditioning requirement is missing: `dacc.py:171` initializes `jcond`, but the square LSPG branch never fills it; its optional Q3 samples are only selected outputs. `dtime` and the 3D adaptive branch likewise do not record square step-Jacobian conditioning for every certified step as A2.1 requires. The requested unweighted residual diagnostic is also absent.

10. **Manifests do not contain the registered numerical identities.**  
    `make_configs.py:158–164` writes only setting, r, M, policy, rule, qp, form, scheme and factor. It omits bank/rotation pins, mode-list hash, point count, qpp links, tolerances, budget, initialization, weighting, output times and metric identity.  
    `make_configs3d.py:95–96` records `M='default'`; its manifest is a ladder list, not a complete configuration-identity manifest. Neither satisfies A1.7. The production `t2`, `t3s`, `t3c` generators/templates and rounds 5–8 selection/configuration path are also absent.

11. **The production resource-certificate path is still qualify’s implementation.**  
    `resources.py:128` recognizes only `qacc`, `qtime`, `q3run`; staging accepts only `dacc`, `dtime`, `d3run`. Its smoke paths remain `smkA`/`smkB`, and its projections assume old schemas, including unconditional `qp`/`qpp` access. Production staging therefore cannot obtain the prescribed certificate through the supplied toolchain. Relabeling driver names alone would not account for M, new families, adaptive forms and certification work.

12. **Reservation validation can bypass the 55-hour cap.**  
    `ledger.py:70–79` validates positive finite hours only for mandatory entries; `pend` sums **all** entries at `:85`. A pending inventory containing the mandatory jobs plus `b2m=-100` passes validation.  
    **Reproduced:** no validation errors and a pending total of **−90 hours**. Validate every entry before storing/summing it. Also reserve an explicit allowance for mandatory round continuation: `MANDATORY` includes only `q3r4`, not rounds 5–8.

13. **Identity checks and decision guards remain incomplete.**
    - `dacc.py:307` and `d3run.py:301` load fixture arrays without checking their pinned identity; recording a descriptive `source` string is not validation. The current extracted fixtures **do** match the pinned source archives, which I verified, but the production readers do not enforce that fact.
    - `dstats.cross_family_gate([], True, 0)` returns **pass** (`:160–169`), accepting absent component-(b) evidence.
    - `dstats.h2q` uses the comparator’s uncertainties without checking agreement with the candidate (`:252–255`), unlike speed3d’s `common` function.
    - `PROVENANCE.json` still labels the five edited original-path files as unchanged and contains only their old hashes; the promised edited hashes are not appended.

14. **The passing tests leave the important failure paths untested.**  
    `test_dims.py` lacks the integration paths above, missing-lower intersection cases, missing fixture evidence and certification-less panels. Its generator checks at `:175–186` omit the Gaussian bump and 2D MC integration tests required by §7. The Smolyak comparison at `:174` sorts each coordinate column independently and ignores weights, so it does not establish equality of point/weight rules.

**Per-file verdicts**

“READY” below means no blocking defect found in that file’s inspected implementation; it is not GPU-result acceptance.

| File | Verdict | Assessment |
|---|---|---|
| `p1common.py` | **READY** | Explicit M, 4096 ceiling and LMF ordering match the design. Independently checked the frozen 2048-mode prefix against its stored list. |
| `rules2d.py` | **READY** | Registered parser names, draws, MC and bad-vector construction are consistent. |
| `rules3d_names.py` | **READY** | Registered rules and lattice-vector recovery are consistent; seed-zero regeneration test passes. |
| `rules3d/make_rules_dims.py` | **READY** | Uses the registered CBC generator and records runtime, merit and hashes. |
| `rules3d/rules_dims.json` | **READY** | NPZ hash matches both metadata and loader pin; recorded merit/check agree. |
| `d3common.py` | **READY** | M slicing, mode hashing and signed-weight Gram omission are consistent with A1.3/A1.4. Real-size reproduction remains required. |
| `dcert.py` | **READY** | Square tolerances, weighted predicate and A3.3 recovery are implemented; CPU checks pass. |
| `dacc.py` | **NEEDS-FIX** | Fixgate crash; incomplete square diagnostics and fixture validation. |
| `dtime.py` | **NEEDS-FIX** | Wrong square anchors; retained block references; incomplete square diagnostics. |
| `d3run.py` | **NEEDS-FIX** | Certification acceptance, decode contract, lifetime, payload records and smoke defects above. |
| `dstats.py` | **NEEDS-FIX** | Missing-evidence precedence and identity guards. It also lacks the complete qualification/analysis orchestration. |
| `test_dims.py` | **NEEDS-FIX** | Existing 72 checks pass, but coverage is insufficient as described. |
| `checks/test_dims.json` | **READY** | Accurately records the existing tests’ outcome; not evidence of integration readiness. |
| `establish_fixture.py` | **READY** | Independent NumPy evaluation and finite-evidence admission implement A4.1 for the fixed nonempty fixtures. |
| `checks/fixture-badlat.json` | **READY** | All three panels contain six finite qualifying states. Minimum recorded ratios are approximately 3.137, 2.460 and 1.746. I verified source pins/extracted states, but did not rerun the large evaluations. |
| `make_configs.py` | **NEEDS-FIX** | Main cell inventory correct; smoke coverage, manifests and timing-generation gaps remain. |
| `make_configs3d.py` | **NEEDS-FIX** | Main ladders correct; smoke workload, overwritten reproduction evidence and manifests remain defective. |
| `configs/f2a.json`, `f2b.json`, `f2k.json` | **READY** | Inventory: 208 CNR, 160 GAL-CN, 160 LSPG-CN, 60 BE arms; all38, tighter dev6. Driver fixes still required. |
| `configs/f2l.json` | **READY** | Correct 15-rung LMF inventory; separate-job split is explicit. |
| `configs/b2f.json`, `b2m.json` | **READY** | Registered family ladders/draws and all38 population match; blocked by `dacc` dispatch defect. |
| `configs/m3a.json`, `m3b.json` | **READY** | Five settings, both ladders, all three amended specs; 64 cases and tighter cases 0–7. |
| `configs/k3.json` | **READY** | Correct shell counts, adaptive forms and no fixed-sweep κ=1; fixed-M panel also includes an extra lat8192 rung beyond §2.3’s stated ladder. |
| `configs/b3f.json`, `b3m.json` | **READY** | Registered family/draw inventories and populations match. |
| `configs/q3r4.json` | **READY** | Generator takes the 16 exposed candidates and supplies extended successors. Later-round machinery remains outstanding. |
| `configs/smk2.json`, `smk2t.json`, `smk3.json` | **NEEDS-FIX** | Findings 6–9. |
| `configs/MANIFEST-cells.json`, `MANIFEST-cells3d.json` | **NEEDS-FIX** | Incomplete identity manifests. |
| `ledger.py` | **NEEDS-FIX** | Negative/nonfinite optional reservation values; round-continuation allowance missing. |
| `cluster/stage.py` | **NEEDS-FIX** | Certificate integration is unusable with current `resources.py`; also permits non-H200 GPU selections despite the lane’s H200-only contract. |
| `cluster/submit.sh` | **READY** | Correct namespace/prefix, actual lane cap 3 and account check 8, reservation and before/after queue checks. The comment saying 4 is stale. Its lock is lane-local, so it does not provide atomic cross-lane campaign-cap enforcement. |
| `cluster/collect.py` | **READY** | Correct namespace/prefix and checksum collection; cleanup requires completed evidence and refuses partial archives. |
| `PROVENANCE.json` | **NEEDS-FIX** | Edited-file hashes/status missing. |
| `resources.py` | **NEEDS-FIX** | Required dependency, still wired to qualify. |

I compared all provenance-listed files with their sources and pins. The only original-path differences were `p1common.py`, `ledger.py`, and the three cluster scripts. I also reviewed the derived `dacc`/`qacc` and `dtime`/`qtime` changes.

**What the corrected real-size smokes must demonstrate**

- Explicit-M reproduction at the registered cases and ≤1e−10 bar; independent 3D M slicing and G-modes checks.
- Correct square production/tight/tighter tolerances, weighted residual certificates, GAL G-cert and required conditioning diagnostics.
- Actual heavy query compilation/execution, including GAL and anchors, with peak ≤120 GB and released memory within 1 GB after successive blocks.
- Certification/identity faults invalidate timing; full-field decode is inside every timed contract; payload keys are recorded per subject.
- Complete registered populations and non-overwritten reproduction archives.
- Admitted bad-vector fixtures trigger gate component (b), with pinned fixture identity and independent recomputation.
- Under-resolved, budget-zero, wrong-table and timing controls behave as registered. Smolyak failure and GAL=LSPG agreement are **not gates** under the amendments.
- Smoke-derived resource certificates work through the actual staging path.

Fix in order: **(1)** certificate/timing correctness, **(2)** crash and retained references, **(3)** missing-evidence decision rules, **(4)** complete smoke workloads/evidence, **(5)** manifests, resource certificates and reservation validation.

**Final verdict: NEEDS-FIX.**