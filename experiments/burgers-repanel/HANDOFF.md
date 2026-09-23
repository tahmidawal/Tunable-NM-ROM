# HANDOFF — burgers-repanel (kept current; read this first after an interruption)

**Lane:** `experiments/burgers-repanel/` in `worktrees/2026-09-22-burgers-repanel`
(`exp/2026-09-22-burgers-repanel`, fork of `exp/2026-09-20-hires-burgers` @ `0ab60014`).
Cluster namespace `/cluster/tufts/paralab/tawal01/brepanel_20260922/`.
Budget: **≤ 1 running GPU job, ≤ 4 total** (account cap 6, shared with four other lanes).
Paper needs the rows by **2026-09-24**. Design: `DESIGN.md`.

**Job:** re-time the paper's Burgers 2D rows at $1024^2$, $512^2$, $256^2$ (that order) so they
measure the optimised solver path, with the pre-optimisation arm the paper prints as a same-job
control. No retraining, frozen checkpoint, frozen rules, no re-certification.

## State (2026-09-22)

- Design written and independently audited (see `reports/`). Codex could not run (bubblewrap
  sandbox failure, a recorded landmine on this box); an independent subagent audited instead.
- Local $64^2$ smoke run: base arms register, parity passes at ~2e-14 with identical integers.
- **br1024 submitted: job 4186871** (H200, 240G, 8 h), 2026-09-22 21:45 EDT. Jobs used 1 / 4.
- **The coordinator's 4096² addendum needs no job on the dev6 cohort.** `reports/unify_4096_ratio.py`
  (run, output in `reports/2026-09-22-4096-single-job-ladder.md`) shows job 4079320 already timed every
  tunability-ladder rung AND both comparators in one allocation: applying the paper's rule inside that one
  job gives 9.96× for the fast arm (comparator `lean_nt1e-3_l1e-3_dt01`, 405.52 ms) and 6.77 / 4.67 / 4.12 /
  4.87 / 3.96× for the other rungs (comparator `lean_nt3e-3_l3e-3_dt005`). Both tables can print 9.96×.
  The hold64 job 4079321 timed only four full-order settings (no dt = 0.01 arms), so its 13.24× is
  rule-correct for the grid it tested but not comparable; that, not the dev6 ladder, is what a fourth job
  would fix.

## Files

- `repanel.py` — driver, from `burgers-eqcert/eqcert.py` @ `176b2a9a` plus `base` (audited
  pre-optimisation arm, parity twin) and `skip_certificates`.
- `make_configs.py` → `config-{256,512,1024}.json`, `config-smoke64.json`. **Edit the generator,
  not the JSON.**
- `xfast.py` (exact-first-steps query), `audit_repanel.py` (NumPy-only audit).
- `cluster/{stage.py,submit.sh,collect.py}`.

## How to run a job

```bash
cd worktrees/2026-09-22-burgers-repanel        # commit first: stage.py refuses uncommitted files
PY=/home/tahmid/Dev/.venv/bin/python
$PY experiments/burgers-repanel/cluster/stage.py br1024 config-1024.json --gpu h200 --mem 240G --hours 8
experiments/burgers-repanel/cluster/submit.sh br1024        # waits for slots; squeue before/after
# when done:
$PY experiments/burgers-repanel/cluster/collect.py br1024
$PY experiments/burgers-repanel/audit_repanel.py experiments/burgers-repanel/runs/br1024/archive \
    --checkpoint experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl \
    --out experiments/burgers-repanel/checks/br1024-summary.json
ssh tufts-login 'rm -rf /cluster/tufts/paralab/tawal01/brepanel_20260922/br1024'   # that attempt only
```

Local smoke (GB10, not a result):

```bash
source /etc/profile.d/jax-mem.sh
PYTHONPATH=experiments/mr-burgers2d:experiments/separable-decoder:experiments/head-ablation:\
experiments/cheap-corrections:experiments/b-ladder-top:experiments/b-panel/speed:\
experiments/hires-burgers:experiments/burgers-repanel \
JAX_DEFAULT_MATMUL_PRECISION=highest JAX_ENABLE_X64=true \
jaxrun /home/tahmid/Dev/.venv/bin/python experiments/burgers-repanel/repanel.py \
  --config experiments/burgers-repanel/config-smoke64.json \
  --checkpoint experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl \
  --inputs experiments/b-panel/inputs --out <scratch>/smoke64
```

## Jobs

| attempt | job id | mesh | GPU | state | summary |
|---|---|---|---|---|---|
| br1024 | 4186871 | $1024^2$ | H200, 240G, 8 h | submitted 2026-09-22 21:45 EDT | |
| br512 | — | $512^2$ | — | not submitted | |
| br256 | — | $256^2$ | — | not submitted | |

Fourth job (4096² tunability ladder re-timing, coordinator's addendum) only if the three above
are on track. See `DESIGN.md` §6.

## Fallback if time runs out

The paper keeps its existing audited "confirmed rule" rows from `burgers-eqcert`
(0.091× at $512^2$, 0.32× at $1024^2$). Say early which mesh will not land.
