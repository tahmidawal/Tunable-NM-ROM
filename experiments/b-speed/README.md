# b-speed — how much faster the frozen Burgers query gets at parity, and where its time goes

Read `DESIGN.md` first: it is the pre-registered protocol, including the algebra of every
optimisation, the parity gates, the timing protocol, the falsification condition and the five
recorded deviations (D1–D5). The result is
`reports/2026-09-16-b-speed.md`.

**Headline.** The fastest arm that passes every parity gate is **1.52x** the incumbent's
single-query GPU latency at 256 intervals and **1.47–1.50x** at 1024, reproduced in two
independent jobs. The pre-registered 2x target **fails**, and the profile says why: the query
is kernel-count bound, the 1D study's largest win (killing a cuSOLVER call) was already taken
in this path, and what remains is the residual-and-Jacobian evaluation itself.

## Layout

| file | what it is |
|---|---|
| `DESIGN.md` | the pre-registered protocol and the deviation register |
| `fast.py` | the optimisations; every path computes the same residual, the same LM iteration and the same outputs as `arms.make_query` |
| `ladders.py` | the arm list with each arm's declared parity intent |
| `speed.py` | the one-job driver: profile, parity gates, interleaved timed panel, throughput |
| `audit_speed.py` | the independent audit; imports neither JAX nor any driver |
| `smoke_speed.py` | the local parity smokes, outputs in `checks/` |
| `cluster/` | staging, checksum collection and archive chunking |
| `artifacts/<attempt>/` | `result.json`, `audit.json`, manifests, and the raw archive as bounded chunks |
| `reference/` | the retained `abl01` `a_neural_eq` fields at 256, for the in-job provenance gate |

## Regenerating the report

Nothing is hand-typed; the generator reads only the tracked result JSONs and audits:

```bash
cd reports
python generate_speed.py \
  --jobs spd01:../artifacts/spd01/result.json:../artifacts/spd01/audit.json \
         fine01:../artifacts/fine01/result.json:../artifacts/fine01/audit.json \
         comp01:../artifacts/comp01/result.json:../artifacts/comp01/audit.json \
  --out 2026-09-16-b-speed.md
```

It prints the file's sha256, which must be
`fc8fa7edbfcb12844ed7b679297b8ec3428d73e3a1f4cb110edb08b3021be456`.

## Restoring a raw archive

```bash
cd artifacts/<attempt>
sha256sum -c SHA256SUMS
cat collection.tar.gz.part* > collection.tar.gz
mkdir restored && tar -xzf collection.tar.gz -C restored
cd restored && sha256sum -c OUTPUTS.sha256 && sha256sum -c MANIFEST.sha256
```

Then the audit can be rerun against the restored fields:

```bash
python ../../audit_speed.py --result restored/output/result.json \
  --fields restored/output --reference ../../reference --out /tmp/audit.json
```

## Rerunning on the cluster

```bash
python cluster/stage.py <attempt> --config config-256.json --hours 05:30:00 --mem 200 --reference
rsync -az runs/<attempt> tufts-login:/cluster/tufts/paralab/tawal01/<namespace>/
ssh tufts-login "cd /cluster/tufts/paralab/tawal01/<namespace>/<attempt> && sbatch run.sbatch"
```

Change `NAMESPACE` in `cluster/stage.py` and `cluster/collect.py` first: the namespace this
cell used, `b_speed_20260916`, was deleted after collection and must not be reused blindly.
One job per attempt directory; check `squeue` before and after every submission.
