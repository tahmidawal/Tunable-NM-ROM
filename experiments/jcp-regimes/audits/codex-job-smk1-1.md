**SUBMIT** — the staged job is consistent with the registered smoke. Two hours is plausible, though completion remains unverified.

1. **Staging integrity: PASS.** All **119 manifest entries** verify; all **115 staged source/data files** match committed HEAD blobs at `828826f61305ad66b1b425f4a91276370608ce6f`. Changes since audit 6 are audit/heartbeat records, not executable code.

2. **Batch configuration: PASS.** [run.sbatch](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/runs/smk1/run.sbatch) requests one H200, `gpu`, 200 GB host RAM and `02:00:00`. It uses the prescribed cluster venv, x64, highest matmul precision and GPU preflight exit 42. Logs, outputs, temporary files and caches resolve under the `smk1` namespace. Bash syntax passes. `set -euo pipefail` stops on command failures, including `fverify` exit 3; `ALL-DONE` follows successful output checksumming. Scientific control acceptance remains a separate offline gate.

3. **Configurations and identities: PASS.** References are dev6 c0, C2 c9, B2 c15 and A+ c13; probes and zero tight-path allowances match A2.5/A3.2. Both ROM controls use the exact ordered six-case population. All nine arms have the same registered 22 smoke cases, with tighter and Q2 populations set to `all`. Checkpoint, rotation and all four main-rule manifests match the hash-verified FRONT-2D artifact. Smoke cases remain diagnostic, separate from production populations.

4. **Acceptance evidence and resource inputs: PASS for smoke.** The batch produces the reference records, probe arrays, slim references, nine R/F pairs, control arrays, unit-gate result and bound `fverify.json` at the paths expected by `smoke_accept.py`. Acceptance additionally requires the existing local E0 store and FRONT-2D file. Pilot seconds and aggregate `qacc_elapsed` supply current `resources.py`; probe seconds and level timing/memory records are also retained. **A6’s complete production resource plan and measured verification-cost work remain deferred production gates**, not approval granted by this verdict.

5. **Runtime and memory: plausible within two hours.**
   - Public E0 H200 logs independently confirm **80 ST cases, median 145.5 s**, with the refined case **693 s**, including **526 s at L=12288**.
   - Four reference pilots suggest roughly **600–1,700 s** before probes and overhead, depending on refinement. These new regimes can be harder.
   - ROM work is **990 rollouts**—198 case/arm pairs × five variants—plus controls, certification and **594 Q3 SVDs**. For context only, archived qualify `acc1024` completed 1,064 case/arm rows in 1,425 s on A100; its populations and step sizes differ, so this is not a direct runtime prediction.
   - A reasonable planning envelope is **about 45–100 minutes**, with uncertainty from stiff solves and tight calibration probes. I do **not** find evidence that exceeding two hours is likely. Drop no mandatory step. Moving `qacc → fverify` before references would preserve ROM evidence earlier if references stall, but would not reduce total work.
   - E0’s refined solve recorded **40.56 GB peak device allocation**. The six L=12288 f64 output fields alone occupy **7.247 GB**; added stiffness temporaries require more. H200 capacity appears adequate; no inevitable OOM identified.

6. **Cohort and write isolation: PASS.** No staged invocation opens sealed/OOD cohorts or writes scientific outputs outside `jcp-regimes/smk1`. Submission’s shared campaign lock and local reservation bookkeeping are intentional administrative writes.

Read-only audit completed: no files changed, jobs submitted, arrays copied, scratch created, or LAB-LOG writes.