"""jcp-robust: production job specs (DESIGN 7, A3.10, A6.1, A7.1). Deterministic; writes jobs/<attempt>.json.

  python make_jobs.py e1        E1: e1m256 (A100-80G), e1m1024 (A100-80G), e1m4096 (H200: the full 4096^2 bank peaks
                                ~97-113 GB), e13d (A100-80G)
  python make_jobs.py e5        E5: e5a1, e5a2 (A100-80G), e5h1, e5h2 (H200, + 4096^2)
  python make_jobs.py e2train   E2 training: e2t2d (A100-80G x 3 GPUs, seeds 1-3 in parallel, Stage A + B + rotation);
                                e2t3d1..3 (H200, one seed each, 420G host: the incumbent's host RSS was ~287 GB)
The time limit of a driver job is set by certify.py's projection (x 1.5, rounded up to 0.25 h) via --hours after
certification; training jobs carry a stated projection from the incumbent logs (smoke-measured per-step rates).
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def w(spec):
    (HERE / 'jobs' / f"{spec['attempt']}.json").write_text(json.dumps(spec, indent=1) + '\n')
    print(spec['attempt'])


def e1():
    for m, gpu in ((256, 'a100-80G'), (1024, 'a100-80G'), (4096, 'h200')):
        w(dict(attempt=f'e1m{m}', gpu=gpu, gpus=1, hours=1.0, mem='200G' if m < 4096 else '240G', cpus=8, production=True,
               mem_fraction='0.95', groups=[[dict(driver='r2acc', config=f'configs/e1m{m}.json')]]))
    w(dict(attempt='e13d', gpu='a100-80G', gpus=1, hours=1.0, mem='200G', cpus=8, production=True,
           groups=[[dict(driver='e1run3d', config='configs/e13d.json')]]))


def e5():
    for att, gpu, cfg in (('e5a1', 'a100-80G', 'e5prod_a100'), ('e5a2', 'a100-80G', 'e5prod_a100'),
                          ('e5h1', 'h200', 'e5prod_h200'), ('e5h2', 'h200', 'e5prod_h200')):
        w(dict(attempt=att, gpu=gpu, gpus=1, hours=1.0, mem='240G' if gpu == 'h200' else '200G', cpus=8, production=True,
               mem_fraction='0.95', groups=[[dict(driver='e5time', config=f'configs/{cfg}.json')]]))


def e2train():
    w(dict(attempt='e2t2d', gpu='a100-80G', gpus=3, hours=6.0, mem='480G', cpus=24, production=True,
           groups=[[dict(sh='train2d/run_seed.sh', args=[str(s), f'OUT/train2d_s{s}'], complete=f'OUT/train2d_s{s}/SEED-COMPLETE',
                         projection_hours=4.0, projection_basis='incumbent dn256b: Stage A 10,685 s (A100), Stage B 2,676 s hfit '
                         '+ extraction (job 2837431); smk2 measured extraction ~900 s on A100-80G')] for s in (1, 2, 3)]))
    for s in (1, 2, 3):
        w(dict(attempt=f'e2t3d{s}', gpu='h200', gpus=1, hours=4.0, mem='420G', cpus=8, production=True, mem_fraction='0.90',  # = j3b (audit 3D-1 #8)
               groups=[[dict(sh='train3d/run_seed.sh', args=[str(s), f'OUT/train3d_s{s}'], complete=f'OUT/train3d_s{s}/SEED-COMPLETE',
                             projection_hours=2.6, projection_basis='incumbent W1024 job j3b 5012847: 6,660 s on H200 x 1.4 (smk3: data generation 1.4x slower) '
                             '(bank fit 4,104 s); smk3 pilot data generation measured')]]))


def e2sel(b):
    w(dict(attempt=f'e2s{b}', gpu='a100-80G', gpus=1, hours=1.0, mem='200G', cpus=8, production=True, mem_fraction='0.95',
           groups=[[dict(driver='r2acc', config=f'configs/e2s{b}.json')]]))


def e2ver(b):
    for tag, gpu in (('a', 'a100-80G'), ('b', 'h200'), ('c', 'a100-80G')):
        w(dict(attempt=f'e2v{b}{tag}', gpu=gpu, gpus=1, hours=1.0, mem='240G' if gpu == 'h200' else '200G', cpus=8,
               production=True, mem_fraction='0.95', groups=[[dict(driver='r2acc', config=f'configs/e2v{b}{tag}.json')]]))
    w(dict(attempt=f'e2t{b}', gpu='a100-80G', gpus=1, hours=2.0, mem='200G', cpus=8, production=True, mem_fraction='0.95',
           groups=[[dict(driver='dtime', config=f'configs/e2t{b}.json')]]))


def e2run3d(name):
    """E2 3D evaluation job (e2q<b>a/b, e2f<b>) from configs/<name>.json. A retrained bank (> 100 MB) stays in the cluster
    archive: the config's cluster_inputs (archive path + sha256 from inputs/banks3d/s<b>/BANK.json) are propagated into the
    JOB SPEC, which is what stage.py copies and verifies (job audit 3D-1 #10). Hours are set by certify.py afterwards."""
    c = json.loads((HERE / 'configs' / f'{name}.json').read_text())
    spec = dict(attempt=name, gpu='a100-80G', gpus=1, hours=1.0, mem='200G', cpus=8, production=True, mem_fraction='0.95',
                groups=[[dict(driver='e2run3d', config=f'configs/{name}.json')]])
    if c.get('cluster_inputs'):
        spec['cluster_inputs'] = c['cluster_inputs']
    w(spec)


if __name__ == '__main__':
    if sys.argv[1] in ('e2sel', 'e2ver', 'e2run3d'):
        {'e2sel': e2sel, 'e2ver': e2ver, 'e2run3d': e2run3d}[sys.argv[1]](sys.argv[2])
    else:
        {'e1': e1, 'e5': e5, 'e2train': e2train}[sys.argv[1]]()
