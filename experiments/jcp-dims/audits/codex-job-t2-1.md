FIX-FIRST

- **Cross-job identity comparison is missing.** `dtime` records fresh in-job outputs and hashes but never reads accuracy archives ([dtime.py:168](experiments/jcp-dims/dtime.py:168)); `timing_costs()` never compares them offline ([analyze2d.py:321](experiments/jcp-dims/analyze2d.py:321)). A1.1 still requires reporting the ≤1e-10 comparison. All 298 subjects have locally available accuracy `W` and dev6 IDs: slim copies suffice; `f2kA` remains under `archive`. No large-array retrieval is needed.

- **Identity failures and controls do not veto acceptance.** `usable` checks inventory and valid fused counts, allowing isolated identity failures; it ignores split identity failures, doubled-work results and split/fused equivalence ([analyze2d.py:362](experiments/jcp-dims/analyze2d.py:362)). Apply the subject-wide identity veto and enforce controls before accepting costs.

- **No injected identity fault in t2.** The staged configuration has an empty list ([t2.json:3619](experiments/jcp-dims/configs/t2.json:3619)), generated explicitly that way ([make_configs_late.py:50](experiments/jcp-dims/make_configs_late.py:50)). The smoke exercised this control, but the requested production panel does not.

- **Exact memory-envelope coverage is unproven.** Production includes R512/M4096 main gauss320 plus gauss384 ([t2.json:2500](experiments/jcp-dims/configs/t2.json:2500)). The timing smoke used main gauss256 with successors 320/384, peaking at **22.15 GB**. Thus it does not demonstrate the exact heaviest simultaneous table footprint required by [DESIGN.md:489](experiments/jcp-dims/DESIGN.md:489). This is a coverage gap, not evidence of an impending OOM.

Checks that passed:

- Independently reconstructed **298 unique subjects**, zero missing/extra; correct reference A, dev6 ×2, burn, randomized blocks/invocations, synchronized triplets, drift band and doubled-work execution.
- Reproduced **5.270219632 h → 8 h** from smk2t measurements; config hash matches [resources-t2.json:6](experiments/jcp-dims/checks/resources-t2.json:6).
- Entire manifest verifies; **176 provenance files match HEAD blobs** at `6008ffe3e`. H200, `gpu`, exit-42 preflight, f64/highest and paralab paths are correct ([run.sbatch:3](experiments/jcp-dims/runs/t2/run.sbatch:3)).
- Budget: **28.6212 + 12 + 8 + 2 = 50.6212 ≤ 55 GPU-h**, leaving **4.3788 h** ([GPU-LEDGER.json:247](experiments/jcp-dims/runs/GPU-LEDGER.json:247)). Both SSH queue checks showed only pending `6056422 / jd_t3s`: lane **1 < 3**, account **1 < 8**.

No files changed, scratch arrays copied, or jobs submitted.