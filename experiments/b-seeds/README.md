# b-seeds — three training seeds and a sealed cohort for the Burgers correction ladder

Is the Burgers $256^2$ dense correction ladder a property of the method or of one lucky
checkpoint (reviewer 5mgh, M2)? Three decoders are retrained from seeds 1, 2, 3 with the
incumbent's own three-script recipe (bank and head, `SEED0` the only variable), each is run
through the exact qtd02 ladder configuration on the six development cases with the incumbent
as a same-job control, and a sealed cohort — drawn once, opened only in the last job — grades
the incumbent and all three seeds at every rung. The pre-registered protocol, criteria and
every deviation are in [`DESIGN.md`](DESIGN.md); the report is generated from the audited JSONs.

## Layout

| path | what it is |
|---|---|
| `DESIGN.md` | pre-registered design, criteria, falsification, deviations D1–D6, amendments |
| `seeds_run.py`, `trajdirs.py` | the ladder driver: `q-trajdirs/qtd_run.py` with four non-numerical edits (`checks/seeds_run.diff`), and its helper verbatim |
| `audit_seeds.py` | independent NumPy audit of one ladder invocation (cohort modes, qtd02 fidelity, three layers, training gates); imports neither driver nor JAX |
| `make_configs.py` | writes every `config-*.json` from the two audited parents and draws the sealed cohort (`checks/sealed-cohort.json`) |
| `config-dev-*.json`, `config-sealed-*.json`, `config-eqcert-*.json` | the ladder / sealed / EQ-certification configurations |
| `deps/` | the incumbent's exact generator dependencies (hashes equal its staged manifests; `PROVENANCE-COPIES.json`) |
| `comparators/` | qtd02's config, result and audit (the incumbent fidelity comparator) |
| `cluster/stage.py`, `collect.py`, `preserve_archive.py` | staging (byte-checked against Git), checksum collection, Git-chunked archiving |
| `smoke_seeds.py`, `smoke_chain.py` | local smokes: the parent lane's baseline reproduction; the whole pipeline at 64 intervals from a staged tree |
| `reports/generate_b_seeds.py` | the one report and `summary.json`, generated from the audit JSONs |
| `checks/` | smoke, audit and diff evidence |
| `artifacts/<attempt>/` | checksum-verified raw job archives in bounded Git chunks |
| `checkpoints/` | the three seed checkpoints (produced artifacts, committed for the sealed job) |

## Running

```bash
PY=/home/tahmid/Dev/.venv/bin/python
NS=/cluster/tufts/paralab/tawal01/b_seeds_20260917

$PY experiments/b-seeds/make_configs.py                      # configs + the sealed draw
$PY experiments/b-seeds/cluster/stage.py s1                  # one seed job (s1, s2, s3)
rsync -a experiments/b-seeds/runs/s1/ tufts-login:$NS/s1/
ssh tufts-login "cd $NS/s1 && sha256sum -c MANIFEST.sha256 --quiet && sbatch run.sbatch"
$PY experiments/b-seeds/cluster/collect.py s1
$PY experiments/b-seeds/audit_seeds.py experiments/b-seeds/runs/s1/archive/output/ladder_seed/result.json \
   --fields experiments/b-seeds/runs/s1/archive/output/ladder_seed --cohort dev \
   --train experiments/b-seeds/runs/s1/archive/output/train --out experiments/b-seeds/checks/s1-ladder_seed-audit.json
$PY experiments/b-seeds/audit_seeds.py experiments/b-seeds/runs/s1/archive/output/ladder_incumbent/result.json \
   --fields experiments/b-seeds/runs/s1/archive/output/ladder_incumbent --cohort dev \
   --out experiments/b-seeds/checks/s1-ladder_incumbent-audit.json
$PY experiments/b-seeds/cluster/preserve_archive.py s1
ssh tufts-login "rm -rf $NS/s1"
# after all three seeds: copy the checkpoints into checkpoints/, commit, then
$PY experiments/b-seeds/cluster/stage.py final               # the sealed-cohort job
$PY experiments/b-seeds/reports/generate_b_seeds.py --audits experiments/b-seeds/checks/*-audit.json \
   --out experiments/b-seeds/reports/2026-09-18-b-seeds
```

## Glossary

- **Seed:** one complete retraining of bank and head from a different random initialisation and state pick; the data are the same for every seed.
- **Incumbent:** the one frozen checkpoint every earlier Burgers number used; re-run in every job as the same-job control.
- **Correction ladder, rung $q$:** letting the solver use $q$ extra fixed coefficient directions on top of the head; a rung is one $q$.
- **Development / sealed cohort:** the six cases every earlier lane measured on / six cases drawn once and opened only in the final job.
- **Three layers:** bank floor (best any coefficients could do), best-found (best the checkpoint's manifold reaches on the reference), solved (what the online solve delivers).
