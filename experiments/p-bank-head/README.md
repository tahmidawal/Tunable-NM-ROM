# p-bank-head

How far the Poisson 2D **bank floor** and the **head** can be pushed inside the same
separable architecture, and whether the solved answer follows. Pre-registration:
[`DESIGN.md`](DESIGN.md).

## Layout

| file | what it is |
|---|---|
| `DESIGN.md` | pre-registered question, losses, seeds, gates, success and honesty clauses |
| `pbh_core.py` | shared numerics: cohorts, fields, banks, floors, head oracles, streaming POD |
| `pbh_fit.py` | training phases — joint/bank (minibatched, the accepted recipe) and the frozen-bank head objective |
| `pbh_train.py` | job 1 driver: diagnose the incumbent, sweep the bank, sweep the head, freeze checkpoints |
| `pbh_solve.py` | job 2 driver: the frozen solve through the **unchanged** `poisson_ablation` machinery |
| `config-train.json`, `config-solve.json` | the two configurations |
| `pabl01-reference.json` | per-case `pabl01` arm errors, the cross-job fidelity reference |
| `pbh_audit_np.py` | independent pure-NumPy/SciPy decoder, DST solve and reference |
| `audit_train.py`, `audit_solve.py` | audits that import neither the driver nor JAX |
| `cluster/stage.py` | flat, Git-verified staging into the paralab namespace |
| `cluster/collect.py`, `cluster/preserve_archive.py` | checksum collection and chunked Git-tracked archives |
| `checks/` | local smoke records and their audits |
| `reports/` | the generated report and its generator |

## Running

```bash
# local smokes (sub-minute class, GB10)
source /etc/profile.d/jax-mem.sh
JAX_DEFAULT_MATMUL_PRECISION=highest jaxrun $PY pbh_train.py --smoke ...
JAX_DEFAULT_MATMUL_PRECISION=highest jaxrun $PY pbh_solve.py --smoke ...

# cluster
python cluster/stage.py train pbh01 --hours 6
python cluster/stage.py solve psol01 --hours 6 --extra <checkpoints> <bases> models.json
```

Cluster namespace `/cluster/tufts/paralab/tawal01/p_bank_head_20260916/`, one attempt
directory per job, `squeue` checked before and after every submit.
