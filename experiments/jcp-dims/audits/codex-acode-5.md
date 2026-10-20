**Overall: NEEDS-FIX.** Two round-4 findings remain partially unresolved. Both were reproduced using in-memory CPU fixtures.

Read-only audit: no files changed, production results opened, jobs submitted, LAB-LOG writes, or scratch files created.

| Round-4 finding | Disposition | Assessment |
|---|---|---|
| 1. Adaptive certification inventories | **NEEDS-FIX** | Case counts, distance counts and `noncert` lengths are checked, but the certificate’s step dimension is not. |
| 2. Evidence-bound numerical caches | **CORRECT** | Added bindings cover the identified NumPy checker, bank/rotation, rule and fixture dependencies. |
| 3. Incomplete round evidence | **NEEDS-FIX** | Missing/blocked labels remain pending downstream, but incomplete evidence can still receive a terminal failure label. |
| 4. Cost Shapley and elasticity statistic | **CORRECT** | Both now use pooled valid-triplet medians and resample cases with their triplets. |

Ranked remaining defects:

1. **Adaptive Q1 accepts an empty step-certificate inventory.**  
   [analyze3d.py:550](experiments/jcp-dims/analyze3d.py#L550) checks `cert.shape[0] == n`, then uses the unchecked `cert.shape[1]` as its denominator. With all eight cases present, valid anchors, zero distances, zero `noncert` counts and `cert.shape == (8, 0)`, **`cert_of_job` returns `True`**.

   The accuracy path has the same defect at [analyze3d.py:151](experiments/jcp-dims/analyze3d.py#L151): a `(64, 0)` certificate makes **every Q1 predicate true** in the corresponding fixture. Missing step evidence can therefore qualify accuracy, screening and confirmation.

   Require the full certificate shape against the registered step count derived independently from the configuration—not from the certificate array itself.

2. **Incomplete rounds become completed failures when any available run is invalid.**  
   [analyze3d.py:335](experiments/jcp-dims/analyze3d.py#L335) tests own-run invalidity **before** population completeness, contrary to A1.5’s missing-first precedence.

   **Reproduced:** 64 registered cases with only 63 validity entries, one false, produce `fail (own runs invalid)`. Making those 63 entries true produces `missing (incomplete population)`.

   Consequently, [make_round.py:25](experiments/jcp-dims/make_round.py#L25) accepts the first label as completed evidence, and the open-candidate filter does likewise. The candidate can disappear despite missing evidence. Check population completeness first, before numerical verdicts and computations requiring complete arrays.

No separate new blocker was established; these are residual defects in round-4 findings.

| File | Verdict |
|---|---|
| `dscore.py` | **READY** |
| `analyze2d.py` | **READY** |
| `analyze3d.py` | **NEEDS-FIX** — both findings above |
| `score3.py` | **READY** for the setting-name mapping edit |
| `make_configs_late.py` | **READY** |
| `make_round.py` | **READY** for its revised missing/blocked-label handling; depends on corrected producer labels |
| `waitjobs.sh` | **READY** |
| `d3run.py` | **READY** for the adaptive `Wstore`/`Cstore` change |

Verification included the restricted diff against speed3d’s scorer, adaptive/explicit-M mapping checks, shell syntax, and the adversarial certificate and round-state reproductions above.