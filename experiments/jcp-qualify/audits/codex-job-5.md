**Overall: SUBMIT-fb1024 / FIX-FIRST timing.** Read-only audit; no files changed, scripts executed, GPU use, or cluster jobs.

Paths below are relative to `experiments/jcp-qualify/`.

| Item | Verdict | Evidence |
|---|---|---|
| Fallback trigger | **CORRECT** | Archived 38-case run confirms worst successor distance **0.0002502734547105819**, at `val32\|5`; Q2 remains **fail**. `configs/fb1024.json:30`; `DESIGN.md:106`. |
| Fallback ladder/population | **CORRECT** | `fib6765 → fib10946 → fib17711`; all38 production/tight, sub16 Q2, dev6 tighter/check/Q3. `configs/fb1024.json:13`; `qacc.py:150`. |
| fbstate/reservation | **CORRECT** | Recorded parent, 0.5 h fallback reservation, enlarged time1024 certificate/reservation, no duplicate 1024 allowance. Total recorded commitment **22.3991 / 35 GPU-h**. Atomic implementation matches A4.1/A5.1. `runs/GPU-LEDGER.json:98`; `ledger.py:205`. |
| Fallback certificate/staging/sbatch | **CORRECT** | Independently reproduced **0.292950 h → 0.5 h**; config hash matches. All **142 manifest entries** verify; staged source matches `8a8ec1d1b1c4621e57c6212895df0f58092a9697`. GPU partition, A100-80G, f64/highest, GPU preflight, isolated paralab output, 30-minute limit. `checks/resources-fb1024.json:1`; `runs/fb1024/run.sbatch:2`. |
| Timing subjects/protocol | **CORRECT** | **36 / 37 / 36** subjects; time1024 includes the fallback. Correct A, dev6 × 3, seed 20261018; split and fused triplets, burns, synchronization, identity checks. `configs/time256.json:24`; `configs/time1024.json:29`; `configs/time4096.json:24`; `qtime.py:237`. |
| Memory/certificates | **CORRECT** | Pilot peak **96.592 GB < 120 GB**; successor tables dropped before timing. Recomputed projections **2.072851 / 2.134049 / 3.645703 h**, limits **3.25 / 3.25 / 5.5 h**; all config hashes match. Estimates, not bounds. `qtime.py:201`; `resources.py:85`; pilot `result.json:564`. |
| B5 numerical eligibility | **CORRECT** | Requires both accuracy and timing qualification, usable split/fused timings, no identity failure, complete38; identity hashes retained. `analyze.py:345`, `:371`, `:388`. |

Ranked must-fix before timing production:

1. **WRONG — timing controls do not gate B5.** Split/fused and doubled-work verdicts are computed, but front construction ignores them. Doubled-work acceptance also ignores `drift_ok`. Block dependent fronts when controls fail or are unavailable; use drift-valid doubled measurements. `analyze.py:379–385`, `:403–429`.

2. **NEEDS-FIX — missing doubled-work checks in two panels.** `configs/time256.json:400` and `configs/time4096.json:400` contain `doubled: []`; their control necessarily reports failure. Add the registered A fixture and refresh certificates/reservations. time1024 already includes it.

3. **NEEDS-FIX — time1024 staging must wait.** fb1024 is pending, so A2.2 is not satisfied. Finish fallback accuracy and compute its labels before staging time1024. Inclusion of the fallback subject alone does not satisfy sequencing. `DESIGN.md:484`; `runs/GPU-LEDGER.json:105`. Other meshes need not wait.

Minor: reference A appears as a spurious unqualified entry in other meshes’ analysis; zero subject triplets prevent frontier contamination (`analyze.py:339`, `:411`).

**SUBMIT-fb1024** is a static audit verdict; live queue/cap checks remain with submission. **Timing: FIX-FIRST**, then respect the mesh-local fallback dependency.