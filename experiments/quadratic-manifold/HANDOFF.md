# quadratic-manifold — running state

Branch `exp/2026-09-22-quadratic-manifold`, forked from `exp/2026-09-17-b-panel` @ `25434a27`.
Namespace `/cluster/tufts/paralab/tawal01/qman_20260922/`. Budget: ≤ 1 running job, ≤ 4 total.

Read `DESIGN.md` first — it is the pre-registered contract and nothing above its `§A` amendments
is edited.

## Where things stand

| | |
|---|---|
| status | `qmn101` **submitted**: job **4185793**, A100-80G, 180G, 12 h, `qman_20260922/qmn101` |
| jobs used | 1 of 4 |
| running | 1 (the cap for this lane) |
| audit | done before the submit — `checks/design-audit.md`, dispositions in `DESIGN.md` §A1; one BLOCKING finding fixed |
| smoke | green after every amendment: 14 subjects, no drops, no failed gate, baseline case to 1.6e-14 |

**The blocking finding, for whoever picks this up.** The ridge holdout used to split snapshot
*columns*; the snapshot matrix concatenates trajectories at 26 states each, so it was holding out
near-duplicates and could not see overfitting — it handed the *baseline* its weakest $W$. Measured,
not argued: `checks/probe64.json`. It now splits by trajectory. If a future lane reuses
`qman.fit`, keep that.

## The lane in one paragraph

b-panel's same-allocation $256^2$ Burgers harness with one new subject family, `qman`:
$u = u_{\rm ref} + V_r a + W\,\mathrm{vech}(a a^\top)$, $W$ from one ridge-regularised linear
solve on the panel's own truth snapshots (Geelen–Wright–Willcox 2022; Barnett–Farhat 2022). No
network, no training run. The trial map is the only thing that changes: the columns
$[u_{\rm ref}\mid V_r\mid W]$ become an `arms.GridBank` and `qman.head` is the coefficient map, so
the residual, test projection, initializer, LM driver and output contract are the ones POD-LSPG
and the NM-ROM arms already run. `qman{r}_quad` and `qman{r}_lin` at $r\in\{8,16,32,64\}$ isolate
the quadratic term; the NM-ROM fast and accurate settings, POD-LSPG 8/16/32/64/256/512 and the
eight Newton–BiCGStab full-order settings are in the **same job**, so every speedup is a
single-job ratio.

## Files

| path | what |
|---|---|
| `DESIGN.md` | the pre-registered design; §7 pass/fail, §8 stop rules and job accounting |
| `qman.py` | the trial map: fit, head, bank columns, and `demo()` — the runnable self-check |
| `panel.py` | b-panel's driver + the `qman` family (diff listed in `COPIED-FROM.json`) |
| `audit_panel.py` | the independent NumPy audit, `qman`-aware |
| `smoke_panel.py` | local 64-interval smoke over every family, then the audit and report generator |
| `config-256-qman.json` | the cluster job (`qmn101`) — 29 subjects: 21 reduced, 8 full-order |
| `checks/design-audit.md` | the independent pre-job audit, verbatim |
| `checks/probe64.json` | the measurement that confirmed the blocking finding |
| `reports/make_table.py` | the table, `summary.json` and DESIGN §1's three answers, from the audit JSON |
| `cluster/stage.py` | stage one attempt into `qman_20260922/<attempt>`; `collect.py` pulls it back |
| `COPIED-FROM.json` | every copied file, its b-panel commit and hash, and what was changed |

## Commands

```bash
LANE=/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-22-quadratic-manifold
cd "$LANE/experiments"
source /etc/profile.d/jax-mem.sh
PY=/home/tahmid/Dev/.venv/bin/python
P=quadratic-manifold:quadratic-manifold/speed:head-ablation:mr-burgers2d:separable-decoder:cheap-corrections:b-ladder-top:q-ridge

# the trial map's own self-check (seconds)
PYTHONPATH=$P JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest jaxrun $PY quadratic-manifold/qman.py

# the local smoke (one driver run at 64 intervals; exceeds the sub-minute rule, DESIGN A2)
PYTHONPATH=$P JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest \
  jaxrun $PY quadratic-manifold/smoke_panel.py quadratic-manifold/checks/smoke-panel.json

# stage and submit ONE job (squeue before AND after)
ssh tufts-login "squeue -u tawal01"
$PY quadratic-manifold/cluster/stage.py qmn101 config-256-qman.json --gpu a100-80G --mem 180G --hours 12
# rsync the staged dir to the printed remote path, then sbatch run.sbatch from there
ssh tufts-login "squeue -u tawal01"

# afterwards
$PY quadratic-manifold/cluster/collect.py qmn101
```

## Next

1. Watch 4185793; on completion run the audit, `reports/make_table.py`, the checksum-verified pull
   (`cluster/collect.py qmn101`), then delete the remote job dir.
2. Write `reports/2026-09-22-quadratic-manifold.md` from `reports/summary.json` — no hand-typed
   numbers — answering DESIGN §1's three questions, and append the lab-log entry.
3. Read the ladder with `DESIGN.md` §A1.1 in hand: check `ridge_at_grid_endpoint` and
   $\lVert W\rVert_F$ per rung, and compare `qman32_quad_M128` with its two pre-declared
   sensitivity arms (`qman32_quad_M512`, `qman32_quadg0p0001_M128`) before reading any rung as a
   property of the method rather than of the selection rule or the test space.
