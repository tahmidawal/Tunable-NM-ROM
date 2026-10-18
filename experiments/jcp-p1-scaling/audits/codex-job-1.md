**Verdict: FIX-FIRST for the staged pair.** The batch scripts and staged dependencies pass inspection. The blockers are incomplete resource accounting and an overstated smoke-acceptance contract—not a demonstrated GPU-memory failure.

No files changed and no jobs submitted. Verification was static: hashes, Git blobs, Python syntax, dependencies, configs, and producer/consumer contracts.

| Item | Assessment | Finding |
|---|---|---|
| `smk/run.sbatch` | **CORRECT** | `gpu` partition; A100 with `a100-80G` constraint; 200G host memory; 02:30:00; four requested commands in order, with matching output directories. |
| `pilot/run.sbatch` | **CORRECT** | `gpu` partition; H200; 240G host memory; 02:30:00; correct pilot command and output directory. |
| Environment and safeguards | **CORRECT** | Both use the absolute cluster venv, enable f64 and highest precision, check the manifest before execution, and exit 42 for a non-GPU backend. Logs, outputs, cache, and temporary files use the requested paralab namespace. |
| Manifest/provenance integrity | **CORRECT** | Every listed manifest hash and provenance size/hash verified in both staged trees. No missing provenance files or differences between staged source bytes and their corresponding current-HEAD blobs. All 58 staged Python files parsed successfully. |
| Claimed source revision | **NEEDS-RESTATEMENT** | Both reservations, provenance records, and `COMMIT.txt` consistently identify `4d0b5731…`, whereas current HEAD is `ee14bb539…`. The intervening change only adds a `.gitignore` entry; staged source content matches HEAD. Describe these as staged from `4d0b5731`, not literally current HEAD. |
| Driver dependency closure | **CORRECT** | Required quad2d modules and vendor files, quad3d modules and nested vendor subtrees, fixtures, rotations, W1024 bank, rules3d, separable-decoder checkpoint, `engines.py`, `iterative_paths.py`, and `sep_common.py` are present. |
| Output filenames and keys | **CORRECT** | The four smoke directory names match `smoke_accept.py`; pilot writes `pilot_pilot/result.json`. Required coefficient archives, control archives, timing records, kernel-equivalence arrays, and pilot timing fields have matching producers. |
| Offline acceptance dependencies | **CORRECT**, locally scoped | Acceptance additionally depends on the local E0/combined worktrees and `checks/test_p1.json`; it is not self-contained in the cluster staging directory. This matches the proposed post-collection evaluation. |

The resource-accounting defects are concrete:

- **WRONG — compilation is excluded from accuracy projections.** In [resources.py](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling/resources.py:32), `per_case_times()` only subtracts consecutive `RUN` timestamps within the same configuration/scheme. Each smoke population has two cases, so its measurement is the **second case only**. First-case compilation is discarded. `setup_seconds` measures setting construction before those queries, so it does not restore that cost.
- **WRONG — block costs are asserted rather than measured.** The same script assigns **900 seconds per extra block**, with a comment claiming smoke measurement. The drivers do not provide separate block elapsed measurements supporting that bound. Order checks, controls, and vendor bridges have substantially different workloads.
- **WRONG — incomplete pilot data can produce a certificate.** `resources.py` does not require pilot completion or valid, finite timing measurements. Missing matched pilot timings silently become a factor of **3**. An interrupted pilot must not become a seemingly measured resource certificate.
- **NEEDS-RESTATEMENT — “upper proxies.”** Scaling by rank and quadrature-point count does not bound nonlinear iteration counts, compilation, or bank construction. These are estimates requiring explicit measured allowances, not established upper bounds.

Fix these before relying on the pair to authorize production: retain first-call/compile costs, measure each extra block, and reject incomplete or unusable pilot measurements.

The memory assessment is favorable for the smoke, but these are **array-size calculations, not measured peaks**:

| Resident allocation | Decimal GB |
|---|---:|
| Full 1024² R512 bank | 4.287 |
| R512 gauss192 / gauss256 / gauss320 tables | 0.906 / 1.611 / 2.517 |
| R384 gauss640 tables | 7.550 |
| 3D 65-node W1024 mesh bank | 2.048 |
| 3D fixed gl24 tables, r=1024, M=2052 | 0.453 |

**CORRECT — no obvious A100-80G capacity violation.** The four drivers run as separate processes. The 3D builder explicitly uses `tensor=False`, avoiding the large \(M r^2\) tensor. However, quad2d table construction retains chunks while concatenating them; gauss640 therefore needs materially more than its final 7.55 GB during construction. Compiler workspaces, executable caches, and Jacobian temporaries also count.

**NEEDS-RESTATEMENT — pilot peak guarantees.** Its three banks, twelve distinct timing settings, all-38 input fields, gauss640 tables, and executed solver paths provide a substantial memory exercise. The 4096² bank alone is 68.686 GB; resident timing-rule tables total 15.610 GB before the historical-rule addition. But it exercises one physical case per subject, and only writes the final threshold decision after all work completes. OOM or timeout must explicitly mean **full-bank feasibility unestablished**, with a defined prefix-pilot follow-up—not an absent decision or a fallback timing factor.

Control coverage:

| Required control group | Assessment |
|---|---|
| A1.1 under-resolved BE rule: six cases, rollout failure, own-state ρ and independent NumPy agreement | **CORRECT** |
| A1.6 budget-zero fixture: all 300 steps and independent γ agreement | **CORRECT** |
| Zero-update fixture, wrong-mesh rejection, sign-flipped probe, vendor bridges, amended temporal-order check | **CORRECT** |
| Doubled work, split/fused equivalence, recording overhead, synthetic controls | **CORRECT** |
| A3.3 numerical kernel equivalence and dispatch-sync at 2D r=512 / 3D r=1024 | **CORRECT** |
| A4.1 injected identity failure, invalid triplets, subject-wide veto | **CORRECT** |
| “Every DESIGN 6 control fully demonstrated” | **NEEDS-RESTATEMENT** |

That last claim needs two qualifications:

1. The reproduction check covers **dev2**, whereas DESIGN 6 specifies **all 38 cases**. It is a preliminary reproduction check; the full gate remains outstanding.
2. The production timing-fault probe supplies only two meshes to a procedure that refuses a registered three-mesh verdict. Its unmodified and faulted outcomes cannot demonstrate sensitivity to the injected timing fault. The separate synthetic three-mesh control does provide that demonstration.

Finally, **NEEDS-RESTATEMENT — 2.5-hour completion confidence** for both jobs. Neither has a measured certificate establishing that allocation suffices. The smoke includes tight/tighter solves, six-rule vendor bridges and order anchors reaching 1,600 steps; the pilot builds and executes many configurations. A timeout during the early accuracy driver prevents the later smoke controls from running.

**FIX-FIRST:** repair resource measurements and incomplete-pilot handling, explicitly distinguish preliminary smoke gates from full-population gates, and document the bounded-attempt/timeout disposition before submission.