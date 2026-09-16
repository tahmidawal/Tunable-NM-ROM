# Per-query head refinement as an inference-time knob

Is refining the frozen head's own weights at query time — anchored to the trained weights —
a usable accuracy/cost knob? Two pre-registered variants, on Burgers 2D and Poisson 2D, at one
frozen checkpoint each. The predeclared protocol is [DESIGN.md](DESIGN.md); project state stays
in the canonical
[lab log](/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/LAB-LOG.md).

Everything except the refinement is the head-ablation arm (a) contract: the frozen bank $G$, the
latent dimension, the weak objective, the test-mode family, the time discretization, the
initializer policy, the Levenberg–Marquardt stopping rule and the output contract. The only
online knobs are the variant, the number $n$ of Adam steps taken on $\theta$ **inside the timed
query**, and the anchor weight $\mu$. $n=0$ is arm (a) with $\theta$ carried as a traced runtime
operand instead of a compile-time constant.

## Layout

| file | what it is |
| --- | --- |
| `refine_core.py` | $\theta$ handling, the inline Adam step, the V1/V2 Burgers queries, the Poisson query, and the refined-manifold reconstruction |
| `refine.py` | the Burgers 2D driver |
| `poisson_refine.py` | the Poisson 2D driver; V2 collapses onto V1 there and only V1 is run |
| `make_gate_reference.py` | extracts the small in-job gate references from the retained `abl01` / `pabl01` archives |
| `smoke_refine.py` | the local fidelity gate, under a minute |
| `config-refine*.json` | the sweep configuration and its tiny local-dry-run twin |
| `audit_refine.py`, `audit_poisson_refine.py`, `audit_common.py` | independent NumPy audits; no JAX, no GPU |
| `reports/generate_head_refine.py` | the one report for both PDEs, generated from the raw JSON |
| `cluster/` | staging, checksum collection and the Git-trackable chunked archive |
| `checks/` | retained smoke and audit evidence |
| `artifacts/<attempt>/` | the checksum-verified raw job archive, in bounded chunks |

`runs/` is Git-ignored scratch for staging and collection; `artifacts/` is the retained record.

## Gates

Nothing is believed before all five pass; they are stated in full in `DESIGN.md`.

```bash
source /etc/profile.d/jax-mem.sh
PY=/home/tahmid/Dev/.venv/bin/python
JAX_DEFAULT_MATMUL_PRECISION=highest jaxrun "$PY" experiments/head-refine/smoke_refine.py \
  experiments/head-refine/checks/smoke-refine.json
```

## Running a cluster attempt

One job per directory, `gpu` partition only, and the batch script asserts `jax_backend=gpu`
before doing any work. Check `squeue` before and after every submission.

```bash
PY=/home/tahmid/Dev/.venv/bin/python
"$PY" experiments/head-refine/cluster/stage.py <attempt>          # Burgers, nested layout
"$PY" experiments/head-refine/cluster/stage_poisson.py <attempt>  # Poisson, flat layout
scp -r experiments/head-refine/runs/<attempt> \
  tufts-login:/cluster/tufts/paralab/tawal01/head_refine_20260915/
ssh tufts-login "cd /cluster/tufts/paralab/tawal01/head_refine_20260915/<attempt> && \
  sha256sum -c MANIFEST.sha256 --quiet && sbatch run.sbatch"
```

Collect, audit, archive and then delete only that exact remote attempt directory:

```bash
"$PY" experiments/head-refine/cluster/collect.py <attempt>
"$PY" experiments/head-refine/audit_refine.py \
  experiments/head-refine/runs/<attempt>/archive/output/result.json \
  --out experiments/head-refine/checks/<attempt>-audit.json \
  --fields experiments/head-refine/runs/<attempt>/archive/output
"$PY" experiments/head-refine/cluster/preserve_archive.py <attempt>
ssh tufts-login "rm -rf /cluster/tufts/paralab/tawal01/head_refine_20260915/<attempt>"
```

The report's plain-language glossary defines every term and column it uses.
