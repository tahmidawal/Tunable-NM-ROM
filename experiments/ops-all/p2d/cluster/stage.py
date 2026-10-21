"""Stage one p2d job into runs/<job>/ (its own submit directory, one job per directory, never reused).

  python cluster/stage.py train <job> --problem square --mesh 256 --family fno [--config fno.json] [--train-count N]
                          [--cg-batch B] [--gpu-constraint "a100-80G|h100-80G|h200-141G"]
  python cluster/stage.py panel <job> --config-json runs/<job>-panel.json [--gpu-constraint ...]

Code is copied flat-by-directory into runs/<job>/code and checksummed (MANIFEST.sha256, checked on the node).
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
NS = '/cluster/tufts/paralab/tawal01/opsall_20260924/p2d'
PY = '/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python'

HEAD = '''#!/bin/bash
#SBATCH --job-name=p2d___JOB__
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint="__CONSTRAINT__"
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=__MEM__
#SBATCH --time=__TIME__
#SBATCH --output=__REMOTE__/logs/%j.out
#SBATCH --error=__REMOTE__/logs/%j.err
set -euo pipefail
TASK_ROOT=__REMOTE__
PY=__PY__
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib" TMPDIR="$TASK_ROOT/tmp"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
echo "host=$(hostname) job=$SLURM_JOB_ID"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
"$PY" -c "import torch,sys; ok=torch.cuda.is_available(); print(f'torch_cuda={ok}',flush=True); sys.exit(0 if ok else 42)"
'''

TRAIN = '''cd "$TASK_ROOT/code/ops"
"$PY" train_ops.py --problem __PROBLEM__ --mesh __MESH__ --config ../configs/ops/__CFG__ --out "$TASK_ROOT/output" \\
    --wall-seconds 3000 --train-count __NTRAIN__ --val-count 16 --cg-batch __CGB__
cd "$TASK_ROOT"; rm -rf cache tmp
echo ALL-DONE
'''

PANEL = '''cd "$TASK_ROOT/code"
"$PY" panel.py --config panel.json --out "$TASK_ROOT/output"
cd "$TASK_ROOT"; rm -rf cache tmp
echo ALL-DONE
'''


def copy_tree(src, dst, patterns):
    dst.mkdir(parents=True, exist_ok=True)
    for pat in patterns:
        for p in sorted(src.glob(pat)):
            if p.is_file():
                shutil.copy2(p, dst / p.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('kind', choices=['train', 'panel'])
    ap.add_argument('job')
    ap.add_argument('--problem', choices=['square', 'lshape'])
    ap.add_argument('--mesh', type=int)
    ap.add_argument('--family')
    ap.add_argument('--config')
    ap.add_argument('--train-count', type=int, default=0)
    ap.add_argument('--cg-batch', type=int, default=32)
    ap.add_argument('--members', default='', help='co-scheduled networks on one GPU: "fno.json,transolver.json"')
    ap.add_argument('--mem-share', type=float, default=1.0)
    ap.add_argument('--checkpoint-blocks', action='store_true')
    ap.add_argument('--config-json')
    ap.add_argument('--gpu-constraint', default='a100-80G|h100-80G|h200-141G')
    ap.add_argument('--mem', default='128G')
    ap.add_argument('--time', default='02:45:00')
    a = ap.parse_args()
    assert a.job.replace('-', '').isalnum(), a.job
    run = HERE / 'runs' / a.job
    run.mkdir(parents=True, exist_ok=False)
    code = run / 'code'
    (run / 'logs').mkdir()
    remote = f'{NS}/{a.job}'
    copy_tree(HERE / 'ops', code / 'ops', ['*.py', 'NEURALOPERATOR-LICENSE'])
    copy_tree(HERE / 'configs' / 'ops', code / 'configs' / 'ops', ['*.json'])
    if a.kind == 'train' and a.members:
        members = a.members.split(',')
        stems = [m.replace('.json', '') for m in members]
        body = 'cd "$TASK_ROOT/code/ops"\nPIDS=""\n'
        for m, st in zip(members, stems):
            body += (f'"$PY" train_ops.py --problem {a.problem} --mesh {a.mesh} --config ../configs/ops/{m} '
                     f'--out "$TASK_ROOT/output/{st}" --wall-seconds 3000 --train-count {a.train_count} --val-count 16 '
                     f'--cg-batch {a.cg_batch} --mem-share {a.mem_share} --co-scheduled {",".join(stems)} '
                     f'{"--checkpoint-blocks " if a.checkpoint_blocks else ""}'
                     f'> "$TASK_ROOT/logs/{st}.log" 2>&1 &\nPIDS="$PIDS $!"\n')
        body += 'RC=0; for p in $PIDS; do wait $p || RC=1; done\ntail -n 3 "$TASK_ROOT"/logs/*.log\n'
        body += 'cd "$TASK_ROOT"; rm -rf cache tmp\necho "members rc=$RC"\necho ALL-DONE\n'
        meta = dict(kind='train', problem=a.problem, mesh=a.mesh, members=members, layout='output/<stem>/<stem>/best.pt',
                    train_count=a.train_count, cg_batch=a.cg_batch, mem_share=a.mem_share,
                    checkpoint_blocks=a.checkpoint_blocks)
    elif a.kind == 'train':
        body = TRAIN.replace('__PROBLEM__', a.problem).replace('__MESH__', str(a.mesh)).replace(
            '__CFG__', a.config or f'{a.family}.json').replace('__NTRAIN__', str(a.train_count)).replace('__CGB__', str(a.cg_batch))
        meta = dict(kind='train', problem=a.problem, mesh=a.mesh, family=a.family, config=a.config or f'{a.family}.json',
                    train_count=a.train_count, cg_batch=a.cg_batch, layout='output/<stem>/best.pt')
    else:
        cfg = json.loads(Path(a.config_json).read_text())
        shutil.copy2(HERE / 'panel.py', code / 'panel.py')
        sub = 'nmrom_sq' if cfg['problem'] == 'square' else 'nmrom_ls'
        copy_tree(HERE / sub, code / sub, ['*'])
        (code / 'panel.json').write_text(json.dumps(cfg, indent=1) + '\n')
        body = PANEL
        meta = dict(kind='panel', problem=cfg['problem'], mesh=cfg['intervals'])
    script = HEAD + body
    for k, v in (('__JOB__', a.job), ('__CONSTRAINT__', a.gpu_constraint), ('__MEM__', a.mem), ('__TIME__', a.time),
                 ('__REMOTE__', remote), ('__PY__', PY)):
        script = script.replace(k, v)
    (run / 'run.sbatch').write_text(script)
    meta.update(job=a.job, remote=remote, gpu_constraint=a.gpu_constraint, mem=a.mem, time=a.time)
    (run / 'STAGE.json').write_text(json.dumps(meta, indent=1) + '\n')
    lines = [f'{hashlib.sha256(q.read_bytes()).hexdigest()}  {q.relative_to(run)}'
             for q in sorted(run.rglob('*')) if q.is_file() and q.name != 'MANIFEST.sha256']
    (run / 'MANIFEST.sha256').write_text('\n'.join(lines) + '\n')
    print(run)


if __name__ == '__main__':
    main()
