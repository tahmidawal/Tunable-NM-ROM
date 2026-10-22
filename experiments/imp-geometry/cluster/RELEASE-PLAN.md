# imp-geometry release plan (proposed; nothing submitted)

Status: proposed after the design-mode work of 2026-10-21; production is NOT releasable on the current cost certificate
(`runs/cost/projection.json`, conservative bounds: wide panel above the 62 GPU-h cap). Every job goes through
`cluster/submit.sh` (committed tree, campaign lock, lane cap 3, account cap 8).

| order | job | body | GPU / mem / time | purpose |
|---|---|---|---|---|
| 1 | `meas6` | `bodies/smoke6.sh` | a100-80G / 120G / 00:09:00 (cap 0.15 GPU-h; two probes, 220 s timeout each) | bank steady-state step time, ECSW NNLS fit time (replace the two largest bounds) |
| 2 (timed out, A9) | `L_wide_S1` | `bodies/release1_L_wide_S1.sh` | a100-80G / 160G / 11:30:00 (step cap 12 GPU-h incl. meas6; stages checkpointed, a timeout leaves a resumable archive) | first full cell (stages ref..freeze); its measured GPU-h re-certify the panel (A6) |
| 3 | recertify | `resources.py` + amendment | — | apply A5.2 reductions (a)-(d) in order if spent + projection > 62 |
| 4.. | remaining wide cells, box cells | `make_body.sh <dom> wide <lev> ...` | a100 | only after 3 |
| last | test-once per cell | stages `test` with the committed `runs/frozen/<cell>.json` | a100 | after the freeze of every cell is committed |

Collection after each job (within 1 h): pull JSON / logs / files < 100 MB, archive larger arrays under
`/cluster/tufts/paralab/tawal01/burgers-archive/imp-geometry/<job>/` with a committed sha256 listing, update the ledger,
delete the job directory.

Re-certification after step 2: `python resources.py --measured runs/cells/L_wide_S1/out/cell runs/cells/meas6/out/probe/probe_H_S2_bankstep.json`
→ `runs/cost/projection_measured.json` (measured stage hours, per-stage device/host peaks, remaining-panel projection
×1.5 vs remaining cap, `fits`, `memory_ok`). Test jobs: `make_body.sh test <dom> wide <lev>` restores the archived cell
(`collect.sh` archive + SHA256SUMS) and runs the test stage against the committed `runs/frozen/<cell>.json`.

**A9 (2026-10-22):** job 6175736 timed out in the ladder. Next job: `L_wide_S1_r` = `bodies/resume_L_wide_S1.sh`,
a100-80G / 160G / 08:00:00 (projection 5.1 h, 7.7 h with headroom; `runs/cost/projection_A9.json`), restoring the archived cell. Collect
with `collect.sh L_wide_S1_r L_wide_S1`. Further cells only after the coordinator's decision on the box controls.

**Step 2 (A10, authorized; hard stop 30 GPU-h in the step):** `L_wide_S1_r` (6192623) and `H_wide_S1` =
`bodies/H_wide_S1.sh`, a100-80G / 160G / 16:30:00 (projection 10.9 h, 16.4 h with headroom; total step requests 24.5 h
≤ 30). Then test-once jobs per cell after its `frozen.json` is committed (`make_body.sh test <dom> wide S1`,
a100-80G / 120G / 02:00:00).
