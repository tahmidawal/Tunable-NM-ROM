**The four X0 labels are correct. No numerical errors found in the audited 2D report.** There are minor wording and reporting omissions below.

I independently aggregated the JSON rows with NumPy, checked the saved ρ arrays, and recomputed historical comparisons. No files were modified; no GPU work was run.

**1. CORRECT — case pairing, decision order, and X0 labels.**

Each cell contains all 38 cases for every arm. I paired cases by `(cohort, case)` and used `vs_gref_restricted_evolved` for both \(s_j\) (`dense`) and \(d_j\) (`nodes`). All required arms are finite, with no failure exits; continuum checks pass. Thus the invalid-X condition does not fire, and the median-separation test gives X0 before any R/N statistics.

All distances below are percentages of the initial-field norm.

| Cell | Median \(s_j\) | Worst \(s_j\) | Median \(d_j\) | Worst \(d_j\) | Label |
|---|---:|---:|---:|---:|---|
| 256² acc | 0.882898446 | 4.25773556 | 0.000317676 | 0.033884995 | X0 |
| 256² fast | 0.882939943 | 4.05342555 | 0.000732916 | 0.030407677 | X0 |
| 1024² acc | 0.224657748 | 1.15446441 | 0.000000993 | 0.000455534 | X0 |
| 1024² fast | 0.224675145 | 1.08726064 | 0.000001161 | 0.000072666 | X0 |

Every median is below **1%**. Cases with \(s_j<0.1\%\): none at 256²; `val32:1`, `val32:21`, `val32:27` at 1024², for both settings. These exclusions correctly do **not** alter the median used to decide X0. Bootstrap bounds are appropriately N/A.

Saved field checks on dev6 cases 0 and 2 reproduce the corresponding row distances within \(1.4\times10^{-17}\).

**2. CORRECT — ST/S error values; NEEDS-RESTATEMENT — include S medians.**

The printed ST worst/median and S worst values match. Independently recomputed values follow; each entry is **worst / median, in percent**. Refined-reference errors remain provisional.

| Cell | Arm | ST | S |
|---|---|---:|---:|
| 256² acc | dense | 6.205203 / 1.254917 | 4.157536 / 0.856781 |
| | lat64 | 6.205244 / 1.254816 | 4.157410 / 0.856860 |
| | gauss96 | 2.747376 / 0.909195 | 1.078286 / 0.047666 |
| | gref | 2.747418 / 0.909197 | 1.059786 / 0.047710 |
| | nodes | 2.747253 / 0.909195 | 1.051389 / 0.047681 |
| 256² fast | dense | 6.923358 / 1.292488 | 4.973523 / 0.901882 |
| | lat64 | 6.924389 / 1.292249 | 4.974324 / 0.901171 |
| | fib1597 | 4.640922 / 1.036055 | 3.455885 / 0.414035 |
| | gref | 4.640565 / 1.035941 | 3.464220 / 0.414041 |
| | nodes | 4.640577 / 1.035938 | 3.463072 / 0.414141 |
| 1024² acc | dense | 3.382982 / 0.981533 | 1.344996 / 0.196494 |
| | lat64 | 3.383906 / 0.981528 | 1.346874 / 0.196634 |
| | gauss96 | 2.746514 / 0.906333 | 1.063627 / 0.050754 |
| | gref | 2.746552 / 0.906333 | 1.045759 / 0.050795 |
| | nodes | 2.746552 / 0.906333 | 1.045608 / 0.050795 |
| 1024² fast | dense | 5.031554 / 1.040971 | 3.479700 / 0.427639 |
| | lat64 | 5.032508 / 1.040982 | 3.480463 / 0.427266 |
| | fib1597 | 4.637343 / 1.034805 | 3.445118 / 0.412150 |
| | gref | 4.636994 / 1.035475 | 3.451109 / 0.412131 |
| | nodes | 4.636994 / 1.035475 | 3.451108 / 0.412131 |

**3. CORRECT — ρ statistics.**

For all rules, NumPy maxima and medians from `rho_per_state_*.npz` match the JSON **exactly**. Below, each pair is **continuum worst / mesh worst**, dimensionless. “Selected” means gauss96 for acc and fib1597 for fast.

| Cell | Rule | 1,900 lat64-reached states | 400 nodes-reached states |
|---|---|---:|---:|
| 256² acc | dense | 0.122667 / 0 | 0.0830109 / 0 |
| | lat64 | 0.138685 / 0.0859111 | 0.0598179 / 0.0840973 |
| | selected | 0.00486316 / 0.136782 | 0.00464865 / 0.0773180 |
| | fib121393 | 8.23523e-5 / 0.136715 | 1.01437e-5 / 0.0797105 |
| | gref | 0 / 0.136711 | 0 / 0.0797105 |
| | nodes | 0.0122031 / 0.137922 | 0.0117875 / 0.0852230 |
| 256² fast | dense | 0.121901 / 0 | 0.0824672 / 0 |
| | lat64 | 0.131152 / 0.0895072 | 0.0590591 / 0.0893169 |
| | selected | 0.0564823 / 0.136998 | 0.0563512 / 0.0983528 |
| | fib121393 | 2.66706e-5 / 0.134963 | 7.30397e-6 / 0.0799827 |
| | gref | 0 / 0.134965 | 0 / 0.0799827 |
| | nodes | 0.0219659 / 0.134901 | 0.0146613 / 0.0899243 |
| 1024² acc | dense | 0.0353255 / 0 | 0.0177484 / 0 |
| | lat64 | 0.0709173 / 0.0540738 | 0.0403567 / 0.0527335 |
| | selected | 0.00462996 / 0.0366646 | 0.00436392 / 0.0172077 |
| | fib121393 | 0.000130754 / 0.0364366 | 2.23369e-5 / 0.0176061 |
| | gref | 0 / 0.0364291 | 0 / 0.0176061 |
| | nodes | 6.44446e-5 / 0.0364319 | 3.68849e-5 / 0.0176364 |
| 1024² fast | dense | 0.0344253 / 0 | 0.0171277 / 0 |
| | lat64 | 0.0494662 / 0.0459246 | 0.0305879 / 0.0455983 |
| | selected | 0.0647105 / 0.0671486 | 0.0641318 / 0.0665651 |
| | fib121393 | 3.19887e-5 / 0.0353981 | 8.08708e-6 / 0.0170317 |
| | gref | 0 / 0.0354023 | 0 / 0.0170318 |
| | nodes | 0.000173804 / 0.0354059 | 4.34667e-5 / 0.0170453 |

The nodes-reached entries agree with their saved JSON summaries; those summaries are not an independent reconstruction of advection vectors.

**4. CORRECT — gates and `targets_valid`.**

Amendment 6 supersedes the original fixed-step G1 bar. All four cells satisfy error at \(10^{-6}\le10^{-5}\) and error ratio within \([30,300]\).

| Cell | G1 error, \(h=10^{-5}\) | G1 error, \(h=10^{-6}\) | Recomputed ratio | G2b | G2c |
|---|---:|---:|---:|---:|---:|
| 256² acc | 2.732745e-6 | 2.734991e-8 | 99.917882 | 2.838701e-14 | 1.196538e-14 |
| 256² fast | 8.157807e-7 | 8.176148e-9 | 99.775672 | 1.420912e-14 | 4.526475e-15 |
| 1024² acc | 9.496110e-5 | 9.495949e-7 | 100.001700 | 2.841824e-14 | 1.191622e-14 |
| 1024² fast | 1.468289e-5 | 1.468278e-7 | 100.000723 | 1.420999e-14 | 2.673837e-15 |

G2a and G3 are zero in every cell. Bars: G2a/G2b/G3 \(10^{-12}\), G2c \(10^{-11}\). The staged code also enforces the reference-array magnitude floor \(10^{-8}\).

Continuum Gauss-check maxima:

| Cell | lat64 population | gref population | nodes population |
|---|---:|---:|---:|
| 256² acc | 3.347205e-8 | 3.174130e-8 | 3.262696e-8 |
| 256² fast | 2.915264e-8 | 2.893962e-8 | 2.915283e-8 |
| 1024² acc | 4.320803e-8 | 4.132735e-8 | 4.132636e-8 |
| 1024² fast | 3.240186e-8 | 3.200689e-8 | 3.200650e-8 |

All are below \(10^{-5}\); flux checks also pass, with overall maximum \(3.931163\times10^{-8}\). Independent-family worst ρ never exceeds \(1.307545\times10^{-4}<10^{-2}\). Both meshes’ `targets_valid=True` are supported.

**5. CORRECT — exits, completion, and logs.**

Every arm in every cell has **1,900 stationary exits**, zero budget, tolerance, tiny-step or damping-limit exits, and all finite flags true. Printed iteration medians match: acc 54–55, fast 51.

Job 5015576 records `jax_backend=gpu`, `x64=True`, `precision=highest`, NVIDIA H200, both `QSTUDY COMPLETE` markers and `ALL-DONE`. Stderr is empty; no error, OOM, fallback or warning markers were found. All `OUTPUTS.sha256` checks pass.

**6. CORRECT — historical reproducibility.**

All 38 historical cases match per arm. Independent recomputation reproduces every printed historical value at its displayed precision:

| Cell | Range of worst per-case relative ST differences across arms | Range of worst audit-field distances across arms |
|---|---:|---:|
| 256² acc | 7.149860e-14–1.043301e-13 | 8.779046e-16–1.078663e-15 |
| 256² fast | 2.894032e-14–5.913171e-14 | 1.009660e-15–1.450067e-15 |
| 1024² acc | 9.994590e-14–5.249248e-13 | 8.970232e-16–1.531681e-15 |
| 1024² fast | 3.610155e-14–2.316973e-13 | 1.189264e-15–1.408997e-15 |

The field comparisons correctly cover **only dev6 cases 0 and 2**, as disclosed. Historical reproducibility is correctly report-only.

**7. CORRECT in substance; NEEDS-RESTATEMENT in presentation — Answers and label wording.**

The descriptive statements beside X0 report actual worst distances and explicitly avoid assigning a mechanism label. They do not over-claim recovery or causality.

Three small changes would make the limits clearer:

- Replace `0.000%` nodes distances with **0.000456%** and **0.0000727%** at 1024²; they are nonzero.
- Rename “invalid because” to “label reason”: X0 means **below the declared median-separation threshold**, not invalid evidence or zero separation.
- Explicitly state that 2D solver sensitivity was **not tested**, as amendment 2 requests. The X0 rows’ “provisional (solver): no” is technically consistent with amendment 3, which applies that designation to R/N, but can obscure this omission.

Remaining WRONG: none