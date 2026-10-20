**FIX-FIRST.** The staged job is internally consistent and appears feasible on H200, but its acceptance program does not fully demonstrate every required gate/control.

Read-only audit completed: no files changed, scratch created, GPU computation run, cluster access, or submissions.

Paths below are relative to [experiments/jcp-budget](/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-budget/experiments/jcp-budget).

1. **Batch script and provenance — CORRECT.**

   - `gpu` partition, one H200, 240G host RAM, two-hour limit: `runs/smk1/run.sbatch:3–9`.
   - x64, highest precision, memory fraction `0.92`, task-local cache/tmp/output, GPU preflight exiting 42 on fallback: `run.sbatch:13–27`.
   - All three reference steps precede B1024; pilot runs last: `run.sbatch:29–38`.
   - `set -euo pipefail` stops subsequent steps on failure. `OUTPUTS.sha256` and `ALL-DONE` are produced only after successful completion: `run.sbatch:12,39–41`. A timeout/OOM therefore leaves partial outputs without the final manifest; this is fail-closed.
   - **Verified:** all **148** provenance-listed files equal committed HEAD `ff24894b453b067b69392084b4ef43a7237a7f7c`; all **152** manifest entries match; no unlisted staged files. Bash syntax passes.
   - Sequential-smoke enforcement exists in `cluster/submit.sh:13–26`. Actual queue occupancy remains unverified under the no-SSH constraint.

2. **Smoke coverage — mostly CORRECT; acceptance has two NEEDS-FIX gaps.**

   | Predicate | Assessment and evidence |
   |---|---|
   | G-R1 | **CORRECT:** separate E0 reproduction, three base levels at \(T=0.25\); `ref/refrun.py:280–294`. |
   | G-R2 | **CORRECT, conditional:** long-horizon final levels are matched against local E0 attempts by mesh, timestep and purpose; `anaref.py:59–79`. If refinement/tightening produces an unmatched finest level, this gate fails rather than becoming vacuous. |
   | G-R3 | **CORRECT coverage:** in-job shared-node checks for all final levels, plus local finest-file recomputation; `ref/refrun.py:218–231`, `smoke_accept.py:56–57`. |
   | C-R1 | **CORRECT:** first-order ladder on dev6 case 0; offline acceptance checks prerequisites and order failure; `ref/refrun.py:297–315`, `smoke_accept.py:59`. |
   | G-C1 | **CORRECT as the registered CPU fixture:** acceptance reads the existing passing fixture, rather than rerunning it on H200; `test_budcore.py:1–3,23`, `smoke_accept.py:62–63`. |
   | G-B0–G-B5 | **CORRECT coverage:** full factorial, sham runs, archive reproduction, every Q-containing corner check, star checks and LS initialization; `b2run.py:180–238,482–505`, `smoke_accept.py:69–80`. |
   | G-B6 | **NEEDS-FIX:** all **114** configured logical tuples pass my independent static check, and execution asserts them. However, neither offline analysis nor smoke acceptance verifies the saved tuples or reports G-B6; `b2run.py:194–197`, `ana2d.py:506–515`, `smoke_accept.py:69–88`. |
   | C-Q, C-S, C-T | **CORRECT coverage:** required R384 BE control, R384 CNR budget-zero control, and R128 BE member are present; `b2run.py:473–490`, `ana2d.py:517–522`. Firing remains an empirical smoke question. |
   | C-I, C-F, C-score / C-R2 identity check | **CORRECT coverage:** perturbation, manufactured floor and identity-error checks are evaluated; `ana2d.py:523–525,550–597`. |
   | G-Q, G-L1 | **CORRECT coverage:** both budget-zero certifiers, wrong-table guard and every member’s archive prefix; `b2run.py:506–536`, `smoke_accept.py:88–94`. |
   | C-A | **NEEDS-FIX:** raw detection is exercised, but classifier evidence can be empty and pass through `all([])`. Also, the recorded “classifier” boolean repeats the threshold inequality without exercising the eligibility-dependent classification branch; `ana7.py:294–307,348`, `smoke_accept.py:97–100`. |
   | C-E, G-D1, fine recomputation | **CORRECT coverage:** evaluated on available case-0 reference data for all three members; `ana7.py:308–350,461–495,552–558`. |
   | Memory pilot | **CORRECT coverage:** both M settings, both large rules, fine bank, solves and fine QR; `b2run.py:543–571`. |

   **Population:** dev6 cases 0–1 support B2 and B7 certification/reproduction. Only case 0 supports new-reference B7 scores and B8. Missing case 1 is explicitly skipped, and fine recomputation expects only cases with reference files: `b2run.py:369–405`, `ana7.py:240–248,461–464`. That is sufficient for the registered “at least one smoke case” requirement; it is not two-case B8 evidence.

3. **Paths and isolation — CORRECT.**

   `configs/smk1_b1024.json:60–61` matches the reference output directory and `result.json` written by `run.sbatch:32`. The resulting filenames are exactly `2d/dev6/T0.5/c000.{json,npz}` and `c000_fine.npz`: `ref/refrun.py:257–270`.

   B8 verifies these files against the producing step’s manifest: `b2run.py:327–350`. The scheduled drivers regenerate cohorts and do not require the E0 store. E0 reads occur in the later local analyses (`anaref.py:33–45`, `ana2d.py:77–87`). I found no scheduled write path to E0, `sealed/`, `custody/`, or outside the job directory.

4. **Runtime and memory — CORRECT sizing choice; two-hour completion remains unproven.**

   - **References:** 8192² uses 1,602 stages at \(T=0.5\); refinement uses 2,402 at 12288². Eleven full-grid output buffers alone occupy **5.50 GiB** and **12.37 GiB**, respectively, before solver/FFT workspace (`ref/refrun.py:41–43`; `ref/refcore.py:165–168,216–219`). Levels execute sequentially.
   - Historical E0 case 0 needed approximately **326 seconds total** at \(T=0.25\) on A100, with the 8192 level peaking at **15.7 GB**. This is planning context, not an H200 timing comparison or bound: E0 `store/2d/dev6/ST/c000.json:95–99,203–210`. Nominal reference work is plausibly a modest fraction of two hours; retries and refinement increase it substantially.
   - **B1024:** **228 primary corners**, **102 check-rule corners**, additional star families, controls and B7 solves. Gauss640 Jacobian work is the principal runtime uncertainty (`b2run.py:180–234`). The primary corner count alone substantially understates this workload.
   - **4096 pilot:** persistent rotated bank is **47.98 GiB**. R384 Gauss320/640 tables across M=1536 and M=2048 total approximately **19.53 GiB**, plus main-rule tables. Fine bank adds **2.99 GiB**. Thus roughly **71 GiB** is already resident before construction temporaries, QR workspace, solver allocations and other state. Chunk concatenation temporarily duplicates table storage (`vendor/quad2d/qcore.py:213–227`).
   - I found **no deterministic OOM**, but cannot certify the peak below 120 GiB analytically. Bank construction and blocked LS QR are significant peaks; fine QR adds further workspace. The pilot meaningfully measures these risks.

   **Two hours is plausible for the nominal path, not established for escalation or slow corner convergence.** There is insufficient measured H200 evidence to give a defensible tighter runtime prediction.

5. **Will the outputs answer the smoke question? — NEEDS-FIX for complete acceptance; otherwise informative.**

   The outputs provide reference timings/peaks, corner timings, B7 timings, setup costs, pilot elapsed time and peak memory—the inputs consumed by `resources.py:43–73`.

   Two limitations matter when subsequently issuing production certificates:

   - B8 measures one reference-bearing case while the overhead extrapolation divides by two configured cases; B8’s contribution is therefore diluted (`resources.py:55–66`).
   - The certificate asserts H200 provenance without checking recorded hardware (`resources.py:113–115`).

   These require attention before production certification; they do not prevent this smoke from collecting useful measurements. Missing G-B6 verification and potentially vacuous C-A classifier acceptance do prevent the stronger claim that **every required gate/control was demonstrated**.

**FIX-FIRST — blocking fixes only:**

1. Add independent offline verification of the saved G-B6 tuples against every configured member/corner, with an explicit fail-closed smoke acceptance row.
2. Exercise the actual C-A classification path, report resolved/unresolved counts, and require at least one resolved smoke example to be classified “inconsistent”; empty classifier evidence must not count as a demonstrated control.