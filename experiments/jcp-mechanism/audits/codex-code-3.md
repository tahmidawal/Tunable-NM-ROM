**`make_report.py` remains WRONG.** Reviewed commit `dc3e38f06` using source inspection, saved-array schema checks, synthetic NumPy inputs, and mocked shell commands. No files modified or GPU jobs run.

Paths below are relative to `experiments/jcp-mechanism/`.

| File | Verdict |
|---|---|
| `a1_3d.py` | **CORRECT** for the reviewed fixes |
| `q2d/qstudy.py` | **CORRECT** for the reviewed fixes |
| `make_configs.py`, production 3D configs | **CORRECT** under Amendment 5 |
| `cluster/submit.sh` | **CORRECT** |
| `DESIGN.md`, Amendment 5 | **CORRECT**; implementation remains incomplete |
| `make_report.py` | **WRONG** |

**Disposition of the previous ten bugs**

| # | Status | Evidence |
|---|---|---|
| 1. Incompatible production cohort hashes | **FIXED** | `a1_3d.py:219–222` records the comparison without asserting equality. The differing hashes at both production configs’ line 38 are now historical metadata, consistent with `DESIGN.md:370–371`. |
| 2. G2a silently ignores NaNs | **FIXED** | Explicit block-finiteness assertions precede accumulation: `a1_3d.py:257`; `q2d/qstudy.py:143`. |
| 3. Locked queue check fails open | **FIXED** | `submit.sh:24–26` captures `squeue` separately under `set -e`. Mocked execution: queue failure exits before submission; `jm_busy` exits 4; empty queue submits. The quotes inside the SSH double-quoted string reach the remote shell correctly. |
| 4. Labels select available cases; incomplete outputs crash | **NOT FIXED** | 2D now checks the expected intersection (`make_report.py:278–306`), but 3D checks only `nodes` case identities (`232–234`), ignores `complete`, and plots still require complete arms/settings (`331–347`). |
| 5. Solver threshold pooled across arms | **FIXED** | Per-arm numerators and denominators at `make_report.py:236–244` and `292–301`. A synthetic 2% `nodes` failure rate is now marked provisional. |
| 6. Null values crash before invalid classification | **FIXED in the identified label paths** | `make_report.py:185–190` checks invalidity/null distances first; nullable target/sensitivity checks are guarded at `248` and `255`. Null handling elsewhere still prevents report completion; see below. |
| 7. Intermediate X retains a bootstrap bound | **FIXED** | `make_report.py:206` sets `boot_low=None`; rendering uses N/A at `378` and `619`. Synthetic median recovery 0.7 produces X with N/A. |
| 8. Unresolved A2 slopes crash formatting | **FIXED** | Nullable formatting at `make_report.py:496–498` and `648–650`. |
| 9. Own-mesh independent-family validation omitted | **FIXED for finite inputs** | `make_report.py:104–108` consumes own-mesh checks; `132–133` propagates failure. A non-finite regression remains below. |
| 10. Incorrect populations, exclusions, reasons and historical metrics | **FIXED for complete, finite inputs** | Counts come from arrays (`633–635`); exclusions retain case identities (`193`, `620`); reasons are printed (`383–386`, `621`); historical field distances replace/add the required diagnostics (`428`, `448–456`). |

**Remaining defects and regressions**

1. **3D completeness enforcement is insufficient.**  
   `make_report.py:222–261` never checks `r['complete']` and derives all case identities from `nodes`. It neither validates incumbent/converged coverage nor aligns their distance arrays by identity.

   Reproductions:
   - `complete=False` with otherwise valid evidence returns **R**.
   - Removing an incumbent case and its distance raises **IndexError**, rather than listing that case as INCOMPLETE.
   - Missing an entire arm produces the placeholder “arm output missing” (`229`), not the required case list.

   `label():180–198` also assumes equal lengths and alignment of `keys`, `s`, and `dn`; it does not enforce them.

2. **The report still crashes on incomplete or invalid evidence.**  
   `main():471` calls plotting before rendering the labels. Reproduced:
   - Missing 3D arms: **KeyError `tensor_R512`** at `331`.
   - Missing 2D rows: **ValueError: max() iterable argument is empty** at `341`.
   - Null serialized error in the 2D table: **TypeError** from `np.median(st)` at `598`.

   Thus the repaired `label()` can return X/INCOMPLETE without the report ever displaying it. Partial 2D setup also remains vulnerable to unconditional `r['setup'][s]` access at `581`.

3. **New early returns bypass provisional-status cleanup.**  
   `label():181` stores provisional status, but X, X0 and INCOMPLETE return at `184–196`, before cleanup at `207–208`. The table prints that retained status at `620`.

   An in-memory rendering produced **X** with “no sensitivity rerun (2D)” in the solver-provisional column. Amendment 3 limits that qualification to R/N. Uncomputed exclusions are also displayed as **“none”**, rather than unavailable.

4. **The new own-mesh validity check fails open on NaN and misreports finite failures.**  
   At `107`, `NaN > 1e-2` is false, so a non-finite own-mesh check leaves validity unchanged.

   Separately, `639–641` prints the **fixed-state** worst value alongside a pass/fail flag that now includes **own-mesh** failures. Fixed-state worst `0.001` and own-mesh worst `0.02` therefore render as:

   > independent family worst 1.0e-03 (FAILS the empirical 1e-2 check)

   The failing state set and its value are absent from the report.

5. **New historical field comparisons silently omit missing audit evidence.**  
   `g4_2d():448–456` accepts whichever of cases 0 and 2 exists, while `679–680` describes the result as covering **“the two saved audit cases.”** One missing file can therefore yield an apparently complete comparison. `g4_3d():427–430` likewise silently intersects available coefficient keys without reporting coverage.

   For complete inputs, the calculations themselves check out: `g65_rows()` has the correct lattice ordering and transposed bank orientation; saved 3D coefficients are `(6, R′)` and saved 2D fields are `(6,257,257)`. No transpose or coefficient-key regression found.

`expected_2d()` correctly reconstructs the production 38-case population and honors configured subsets. It does, however, trust saved cohort metadata: missing metadata raises `KeyError` at `279` rather than producing INCOMPLETE.

Amendment 5 correctly resolves the dimensional gate wording, makes the historical hash report-only, and specifies complete cohorts plus per-arm thresholds. No restatement is needed; the report needs to implement that contract consistently.

A concurrent, uncommitted change to `a1_3d.py`’s target-retry logic appeared during inspection. It was not made by this audit; the verdict above covers the reviewed `dc3e38f06` fixes.

Remaining WRONG: `experiments/jcp-mechanism/make_report.py`.