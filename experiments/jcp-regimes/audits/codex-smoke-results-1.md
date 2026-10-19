**At `3f7ee1957`: (A) SMOKE-ACCEPTED; (B) NEEDS-FIX; (C) NEEDS-FIX.**

Read-only audit completed: no files changed, jobs submitted, arrays copied, or LAB-LOG writes. Checks ran CPU-only with bytecode writes disabled.

Paths below are relative to [experiments/jcp-regimes](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes).

**A — Smoke results: SMOKE-ACCEPTED**

| Item | Verdict | Independently checked result and evidence |
|---|---|---|
| R-repro | **CORRECT** | Finest field equals the E0 store exactly. Maximum order difference **8.28705e-10**, below **1e-9**; both outcomes accepted. `smoke_accept.py:106–115`; `checks/smoke-accept.json:83`. |
| R-upwind | **CORRECT** | Recomputed **order-fail**, with orders **0.936422–0.978027**. Solver, shared initial field, and floor checks pass. `smoke_accept.py:118–136`; `checks/smoke-accept.json:91`. |
| Calibration, including A3.1/A6 identity checks | **CORRECT** | Production members match the last recorded production/retry attempt. Saved residual histories and fields pass validation. Deviations below. `smoke_accept.py:139–160`. |
| Q-repro | **CORRECT** | Against FRONT-2D’s six dev6 cases, maximum relative error discrepancies are **7.26e-14** for R384 CN-R and **3.80e-15** for R128 BE, comfortably below **1e-9**. `smoke_accept.py:189–206`. |
| Under-resolved control | **CORRECT** | Recomputed all six distances; **every case** exceeds **2.5e-4**. `smoke_accept.py:208–216`; `checks/smoke-accept.json:320`. |
| Budget-zero LSPG | **CORRECT** | Independently recomputed all **300** gamma values. Range **0.0490544–0.0546933**; **100% non-certified**. Maximum relative discrepancy from saved certificates **4.88e-15**. `smoke_accept.py:217–244`. |
| Budget-zero GAL | **CORRECT** | Complete saved evidence and **100% non-certified** in-job; remains diagnostic/non-blocking, as A6 requires. No independent eta recomputation claimed. `smoke_accept.py:245–246`. |
| Generator and synthetic controls | **CORRECT** | CPU rerun returned **ALL-PASS**, including seed/bounds/disjointness, paired geometry, IC, spectral IC, separation, order and Péclet fixtures. Also independently checked exact 1024-grid restriction to the 256 lattice for B1/B2 cases 0–1. `test_regimes.py:86–121`; `DESIGN.md:411–430`. |
| Wrong mesh/substituted table | **CORRECT** | CPU re-execution accepts the own table and refuses both wrong-mesh and perturbed tables. `ctrl_wrongmesh.py:19–45`; real-size in-job evidence at `checks/smoke-accept.json:378`. |
| Full-field verification and custody | **CORRECT, bounded verification** | All nine R-file/config bindings match fverify. All **16 locally present output checksums** match; nine archived F-file hashes match the original output manifest. Archived full fields were not downloaded or decoded again. `fverify.py:34–64`; `runs/smk2/OUTPUTS.sha256:1`; `runs/smk2/ARCHIVE.sha256.txt:1`. |
| A7 split smoke and logs | **CORRECT** | smk1’s sole traceback differs only in fingerprint. Reference source hashes agree across smk1, smk2 and audited checkout. GPU/f64/highest precision recorded; smk2 ends with successful fverify and `ALL-DONE`. No additional hidden failure found. `runs/smk1/logs/6068450.err:1`; `runs/smk2/logs/6074276.out:228–241`. |

Calibration recomputation:

| Pilot | Deviation | Floor proxy | Verdict |
|---|---:|---:|---|
| C2 c9 | 4.69626e-12 | 7.92150e-10 | agree |
| B2 c15 | 2.45823e-12 | 6.56977e-10 | agree |
| A+ c13 | 1.55114e-13 | 9.96714e-10 | agree |

Under-resolved distances, in registered order: **0.00407894, 0.00105302, 0.125210, 0.377176, 0.0283103, 0.0552518**.

**The C1/C2 diagnostic failures are real.** Decoding the raw R384 LSPG-CN-R production and successor coefficients gives:

- C1 c0: **0.000533979**, versus Q2 limit **0.00025**.
- C2 c9: **0.01425546**, versus the same limit.

Thus these failures do not depend on the stored qualification label. Evidence: `rscore.py:298–307`, raw `runs/smk2/output/qacc_smk1-rom/R_R384_LSPG_CNR_1.npz`. A3.3 correctly keeps these smoke cases diagnostic.

**B — Production code: NEEDS-FIX**

| Round-3 finding | Verdict | Verification |
|---|---|---|
| 1. Calibration identity/completeness and missing PS configs | **NEEDS-FIX — partially fixed** | Calibration now compares identities and rejects duplicates/missing cohorts (`analyze.py:212–227`). Missing PS configs are recorded (`analyze.py:299–307`), but disappear from the rendered report; see below. |
| 2. False val32 Q2 failure | **CORRECT** | Requirement now follows the expected check population. Executing the actual Q2-construction block gives **pass** with no scheduled check, and **unverified** when a required check is absent. `rscore.py:329–334`. |
| 3. Allowance/selection publication context | **CORRECT for the principal fixes** | Baseline exclusions and allowance counts accompany ratio cells; absolute errors carry coverage and unusability; the selection-bias disclosure is explicit. `analyze.py:245–271`; `make_report.py:94–104,123–125,143–149`. |
| 4. PS deferral lost in export | **CORRECT** | `ps_status` now survives export, including deliberate deferral. `make_report.py:250–255`. |

**Remaining defect — missing PS configuration is still silently omitted from the report.**

`analyze.py:303` stores the missing run in `ps_status.missing_configs`, then skips its expected case. At `make_report.py:166–173`, the report prints `ps_status` only when it is a string; dictionary status is ignored.

Executing that exact rendering block with:

```text
ps_status = {expected: [], missing_configs: ["refjob:configs/ps-B2.json"]}
ps = {}
```

produced only a blank line. The report therefore loses the scheduled diagnostic’s absence, although EXPORT retains the metadata.

**Required fix:** render missing configurations explicitly and expose scheduled diagnostic completeness. Keep those missing runs in the expected inventory even when their configuration cannot be read.

**C — Resource plan: NEEDS-FIX**

| Item | Verdict | Evidence |
|---|---|---|
| Reference populations and allowances | **CORRECT** | Seven cohorts ×16 plus N0s ×8 = **120** new references. Ascending full populations; only A+ has allowance **1**. Slim sources match their reference jobs. `make_configs.py:94–103`. |
| ROM and verification inventory | **CORRECT** | Nine arms; **158 cases/arm**, **38 tighter cases/arm**; **4,950 rollouts**, **1,422 verification fields**. `resources.py:28–42`; `make_configs.py:107–109`. |
| PS cases | **CORRECT** | Independently regenerated maxima: N0 **2**, A+ **7**, A− **5**, C1 **11**, C2 **9**, B1 **0**, B2 **15**. `make_configs.py:104–111`. |
| Certificate hashes and arithmetic | **CORRECT** | Every configuration hash and cost component matches. Projection **18.598012 GPU-h**; rounded reservations **29 GPU-h**. `resources.py:62–84`; `checks/resources-plan.json:96–106`. |
| Budget and concurrency | **CORRECT** | **29 + 0.5494 = 29.5494 GPU-h**, below 60. Lane cap three enforced under the lane submission lock; H200-only staging. `runs/GPU-LEDGER.json:26`; `cluster/submit.sh:12–24`; `cluster/stage.py:24,47–58`. |
| Low-viscosity reference time limits | **NEEDS-FIX** | Refinement workload is omitted; the three-hour limits are not adequately supported. Details below. |

**The low-viscosity cost model drops the refinement reserve.** `resources.py:57` takes **350.367 seconds** from the unrefined C2 pilot; `resources.py:68–70` multiplies this by population and 1.2, without any refinement term. Yet production can execute the 12288-level refinement (`refjob.py:173–179`), and the design reserves for low-viscosity refinement (`DESIGN.md:434–435,501–503,556`).

The four pilots all avoiding refinement does not establish that all production cases will avoid it. The current **three-hour** limits for A−, C1 and C2 therefore cannot be certified safe from this evidence.

For a concrete conservative alternative, retaining the design’s **850-second low-viscosity allowance** within the existing formula gives approximately:

- **4.6167 GPU-h projection per low-viscosity job**;
- **7-hour rounded limit per job**;
- **26.5921 GPU-h total projection**;
- **41-hour total reservations**, or **41.5494 GPU-h including spent time**.

That still fits the cap without reducing populations or dropping PS. Regenerate the certificates with an explicit refinement allowance—or document and substantiate an amended resource model—before production.