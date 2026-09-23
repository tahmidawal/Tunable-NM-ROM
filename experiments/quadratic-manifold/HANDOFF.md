# quadratic-manifold — running state

Branch `exp/2026-09-22-quadratic-manifold`, forked from `exp/2026-09-17-b-panel` @ `25434a27`.
Namespace `/cluster/tufts/paralab/tawal01/qman_20260922/`. Budget: ≤ 1 running job, ≤ 4 total.

Read `DESIGN.md` first — it is the pre-registered contract and nothing above its `§A` amendments
is edited.

## Where things stand

| | |
|---|---|
| status | **DONE.** `qmn102` (job 4186220) landed clean, audited, reported, pulled, remote deleted |
| jobs used | 2 of 4 (`qmn101` died with no numbers — DESIGN §A2; `qmn102` is the result) |
| running | none; namespace `qman_20260922/` empty |
| report | `reports/2026-09-22-quadratic-manifold.md`, generated from `reports/summary.json` |
| numbers | `reports/summary.json`, `checks/qmn102-audit.json`, `artifacts/qmn102/result.json.gz` |

**The result.** The quadratic manifold beats POD-LSPG at matched solved dimension at every rung it
is fitted at (1.29× / 1.28× / 1.48× lower worst evolved error at $r=8,16,32$), and stops paying at
$r=64$ where the ridge pins at the grid top. It does **not** reach the nonlinear head: 22.21 %
against 1.89 % at 16 solved unknowns, a factor of 11.8. The head needs 4× fewer unknowns than the
best quadratic arm, is 17.6× cheaper and 3.7× more accurate. The binding constraint is
representational — every arm's error equals its own representation floor — not the solve.

**Two things a reader must not miss.** (1) The ridge rule leaves accuracy on the table: the
pre-declared fixed $\gamma=10^{-4}$ beats the rule-selected one by 1.42× at $r=32$, so the ladder
is a **lower bound** on a tuned quadratic manifold. (2) This lane gives the baseline no
hyper-reduction, which GWW and BF both supply; the dense-against-dense comparison is in the report's
§5 and the head still wins it 2.44× on cost and 3.7× on error.

**The blocking finding, for whoever reuses `qman.fit`.** The ridge holdout used to split snapshot
*columns*; the snapshot matrix concatenates trajectories at 26 states each, so it held out
near-duplicates, could not see overfitting, and handed the *baseline* its weakest $W$. Measured, not
argued: `checks/probe64.json`. It now splits by trajectory. Keep that.

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

Nothing is required. 2 of 4 jobs remain unspent and the lane's question is answered. If it is
reopened, the two open threads are named in the report's §5: a ridge chosen better than the
held-out-snapshot rule (§2 shows it is worth ~1.4× at $r=32$), and hyper-reduction for the
baseline (an EQ rule for the quadratic manifold, which is a lane's worth of work and would have to
buy more than an order of magnitude to change the verdict).

**Do not merge or push without asking.** The branch is the archive.
