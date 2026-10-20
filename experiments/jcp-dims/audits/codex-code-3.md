**NEEDS-FIX.** The fixes improve coverage, but certification acceptance and memory handling still have gaps.

Read-only audit completed: no files changed, jobs submitted, GPU used, or scratch files created. Python ran with bytecode writes disabled and `JAX_PLATFORMS=cpu`.

Evidence:

- **96/96 existing CPU checks passed.**
- Tiny adaptive LSPG and GAL queries compiled and executed, including both certifiers.
- Both returned `W.shape == (6, 2)` and `fields.shape == (1, 6, 2)`. `S.decode_lattice(W[:, None], d)` is correct.
- No interface or traced-Python-control-flow crash surfaced in these bounded checks. Real-size GPU memory behavior remains untested.

All references below are relative to `experiments/jcp-dims/`.

| Round-2 outstanding finding | Verdict | Evidence |
|---|---|---|
| **1. Certification identity and success** | **NEEDS-FIX** | `d3run.py:471` now compares execution records and `:473` requires run validity. However, it ignores `gam_cert`/`nonstationary`, recorded separately at `s3ctx.py:157–159`. Adaptive links similarly use `ad_valid`, without the independent certificate outcome (`d3run.py:183`, `:458`). |
| **2. Heavy smoke panel and cases** | **CORRECT** | W1024/gl64 is now a panel subject with preceding runs on cases 0–1 (`make_configs3d.py:71`, `:85`). The two-rung W512 lattice block uses cases 0–3 (`:69`). Verified the generated config too. |
| **3. Memory handling** | **NEEDS-FIX** | Adaptive panel cache deletion is added at `d3run.py:424–425`; rejection flags and resource-reader refusals exist. But the 2D post-setting fixgate phase escapes rejection checks, and 3D ladder tables survive beyond their last use. Details below. |
| **4. Adaptive output validity** | **CORRECT** | `d3run.py:109–112` checks decoded fields, initialization and scale for finiteness. The nonfinite-fields regression passes. |
| **5. Missing certification keys** | **CORRECT** | `dstats.py:279–283` rejects absent required keys; `d3run.py:435–436` supplies A and subject/case keys. A separate invocation-membership hole remains. |
| **6. Manifest, provenance and predecessor evidence** | **NEEDS-FIX** | Round entries now exist, but bypass the numerical fields added to ordinary entries (`manifest_cells.py:115–123` versus `:142–145`). All **17 q3r4 entries lack `tol`, `weighting`, and `mode_hash`**. The resources provenance hash matches; predecessor-member checks are present at `make_round.py:34–37`. |
| **7. Regression coverage** | **NEEDS-FIX** | `test_dims.py:230–261` covers mutated execution records, invalid runs and nonfinite adaptive fields. It does not test failed numerical certificates, FS/QN decoded-field validity, table lifetimes, or the required smoke inventory. |

**Ranked remaining findings**

1. **Certification links still accept failed numerical certification.**  
   I reproduced `scored_equals_certified(...) == True` with `valid=True`, matching coefficients and execution records, but `nonstationary=25` and all 25 certificate values equal to `1.0`.

   `s3common.run_valid` checks solver execution validity, not the independent stationarity certificate. The link at `d3run.py:473` therefore cannot establish numerical certification success. Adaptive links have the same separation: `noncert` and `gcert` are computed at `d3run.py:158–172`, but only `R['valid']` reaches the link.

   Preserve identity, execution validity, and numerical certification as separate evidence, then enforce the registered pooled certificate predicate before declaring numerical acceptance.

2. **FS/QN decoded-field failures remain accepted.**  
   With matching finite coefficients and execution records, replacing `fields` with infinity still produced both:

   ```text
   S.run_valid(...)                 True
   scored_equals_certified(...)     True
   ```

   See `s3common.py:221` onward and `d3run.py:459–473`. FS/QN linkage does not compare decoded fields or `w0`; its `payload` argument is unused. Subsequent timed-versus-scored hashing can faithfully reproduce an invalid scoring output. Validate the complete output contract and retain the certification invocation’s full payload identity.

3. **Memory rejection misses the final 2D fixgate phase.**  
   `dacc.py:289–292` checks memory inside the settings loop. Fixgate blocks execute afterward at `:297–301`; `:303` records final memory without applying the threshold. Consequently, a fixgate-only threshold crossing can leave `memory_rejected` absent, and `resources.ref_dacc` accepts it at `resources.py:59`.

   Apply the threshold after those blocks and at finalization; resource readers should also validate the recorded peak.

4. **3D ladders retain an obsolete rule while building the next successor.**  
   A tiny orchestration reproduction of `run_ladder` observed:

   ```text
   r0
   r0, r1
   r0, r1, r2
   ```

   `d3run.py:248` builds the successor before `:270–271` drops the previous rung. Thus the W1024 gl48→gl56→gl64 pilot temporarily retains all three tables despite gl48 having no remaining use. The adaptive ladder has the analogous ordering at `:185` and `:206–207`.

   Drop obsolete tables before allocating the next successor. This is a demonstrated lifetime defect; I have not claimed it necessarily exceeds 120 GB.

5. **Round manifest identities remain incomplete.**  
   Populate round-member numerical identities through the same complete schema as ordinary ladder entries. Adding round membership alone did not finish finding 6.

6. **`panel_accept` permits unexpected invocations.**  
   I reproduced acceptance of an invocation for `B='rogue'` with `expected={}` and only A certified. At `dstats.py:287`, counts are checked only for supplied expected subjects; invocation subjects are not required to belong to that set.

   The current driver builds a consistent inventory, so this is a decision-function boundary defect, not a demonstrated normal-driver failure. Require exact subject membership and case/repetition coverage.

The smoke and existing accuracy collection paths **do not depend on the unwritten analysis/scoring code or `t2`/`t3s`/`t3c` configs**. Their deferral is not a smoke blocker. Scientific qualification, selection, and later-round generation still require that subsequent audited work.

**Overall: NEEDS-FIX.**