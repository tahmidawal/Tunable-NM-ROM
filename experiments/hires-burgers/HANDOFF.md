# HANDOFF — hires-burgers (kept current; read this first after an interruption)

**Lane:** `experiments/hires-burgers/` in `worktrees/2026-09-20-hires-burgers`
(`exp/2026-09-20-hires-burgers`, fork `exp/2026-09-17-b-panel` @ `25434a27`). Cluster namespace
`/cluster/tufts/paralab/tawal01/hires_b_20260920/`. Contract:
`reports/2026-09-20-speed-accuracy-campaign-protocol.md` (on main). Design: `DESIGN.md`.

## State (update at every milestone)

- Jobs used: **5 / 8** (hb2k01 done; hb4k01, hb4k02 failed before any ROM number). RUNNING: `hb2k02` = 4071616 (2048², speed loop) and `hb4k03` = 4071625 (4096²), both H200 pax008, source = HEAD at submission.
- **hb2k01 (2048², job 4054951, H200) collected, NumPy-audited (no failed gate), remote dir deleted.**
  `checks/hb2k01-summary.json`, report `reports/2026-09-20-hires-burgers.md`, `reports/summary.json`.
  Accuracy SURVIVES transfer: q=256/M=1088 0.598 % worst evolved (dense truth 0.5985 %), q=128 1.075 %, q=0 2.37 %,
  zero stalled exits. Bar at 2048²: NOT met — cheapest certified ≤1 % arm `q256_M1088_lat64_g0p001_fast_chol`
  195 ms vs lean_tight 751 ms = 3.85×; vs relaxed passing FOM (164 ms) 0.84×. Only the 63×63 lattice rule certifies
  at q=256/M=1088 (ρ_max 0.100 / 0.098); scaled and refit b-eqtop rules FAIL the certificate (0.25–0.29, 0.21–0.22)
  although their error is the same; control bad0 fails as required (0.484, error 1.04 %).
- Speed loop so far (SPEED-LOG.md): hfast 1.11–1.16× at parity; Cholesky 1.56–1.80× at parity; M=544 reverted;
  first time step costs 41–88 of 170–494 LM iterations → `clip` and `lamcarry` arms written, pending measurement.
- hb4k01: 64 GiB staged upload OOM. hb4k02: Triton gemm cannot autotune > 2^31 elements → bank is now a tuple of
  row blocks (`hops.build_bank/bank_apply`). Records in `artifacts/hb4k0{1,2}-failed/`.
- Next: smoke (blocked bank + variants) → commit → submit `hb2k02` (`config-2048-speed.json`) and `hb4k03`
  (`config-4096-speed.json`, `--mem 320G`), both H200.

## How to run a job

```bash
cd worktrees/2026-09-20-hires-burgers        # commit first: stage.py refuses uncommitted files
PY=/home/tahmid/Dev/.venv/bin/python
$PY experiments/hires-burgers/cluster/stage.py <attempt> config-2048.json --gpu h200 --mem 240G
ssh tufts-login 'squeue -u $USER'                                    # before
rsync -a experiments/hires-burgers/runs/<attempt>/ tufts-login:/cluster/tufts/paralab/tawal01/hires_b_20260920/<attempt>/
ssh tufts-login 'cd /cluster/tufts/paralab/tawal01/hires_b_20260920/<attempt> && sbatch run.sbatch'
ssh tufts-login 'squeue -u $USER'                                    # after: exactly one job for this dir
# when done:
$PY experiments/hires-burgers/cluster/collect.py <attempt>
$PY experiments/hires-burgers/audit_hires.py experiments/hires-burgers/runs/<attempt>/archive --out experiments/hires-burgers/checks/<attempt>-summary.json
ssh tufts-login 'rm -rf /cluster/tufts/paralab/tawal01/hires_b_20260920/<attempt>'   # only this attempt dir
```

## Next step

See the bottom of this file's job table.

| attempt | job id | mesh | GPU | state | summary |
|---|---|---|---|---|---|
| hb2k01 | 4054951 | 2048² | H200 (pax008) | DONE, audited, remote deleted | `checks/hb2k01-summary.json` |
| hb4k01 | 4055954 | 4096² | H200 (pax010) | **FAILED, no number** (64 GiB staged bank upload OOM; `artifacts/hb4k01-failed/`), remote dir deleted | — |

Next: watch `logs/4054951.out` for the `QUICK` lines (early answer), then stage `hb4k01` with `config-4096.json --mem 400G` once the early phases are seen to work and the account has < 6 running.
| hb4k02 | 4059827 | 4096² | H200 (pax010) | **FAILED, no ROM number** (Triton gemm > 2^31 elements; `artifacts/hb4k02-failed/`), remote deleted | — |
| hb2k02 | 4071616 | 2048² | H200 (pax008) | RUNNING (config-2048-speed.json) | — |
| hb4k03 | 4071625 | 4096² | H200 (pax008) | RUNNING (config-4096-speed.json, blocked bank) | — |
