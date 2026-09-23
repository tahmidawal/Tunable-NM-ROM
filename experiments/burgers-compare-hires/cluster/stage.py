"""Stage one burgers-compare-hires attempt into its own directory (one directory per job, never reused).

    python cluster/stage.py train <attempt> --mesh 1024 [--gpu a100-80G|h200] [--mem 240G] [--hours 8]
    python cluster/stage.py panel <attempt> --config config-1024.json --train-from <train attempt> [...]

Adapted from experiments/burgers-repanel/cluster/stage.py @ 219feb6e. Every staged file is checked byte for
byte against the Git blob at HEAD, so the job runs a committed tree. Raw sbatch with the protocol's
requirements: gpu partition, paralab venv, preflight exit 42, f64, JAX_DEFAULT_MATMUL_PRECISION=highest, all
output under this lane's namespace.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/bcmp_20260923'
LANE = 'experiments/burgers-compare-hires'
CHECKPOINT = 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
GRES = {'a100-80G': ('gpu:a100:1', '--constraint=a100-80G'), 'a100': ('gpu:a100:1', None),
        'h100': ('gpu:h100:1', None), 'h200': ('gpu:h200:1', None)}
LIBS = ['experiments/mr-burgers2d/engines.py', 'experiments/mr-burgers2d/iterative_paths.py',
        'experiments/mr-burgers2d/accuracy_paths.py', 'experiments/separable-decoder/sep_common.py',
        'experiments/head-ablation/arms.py', 'experiments/head-ablation/ladder.py',
        'experiments/head-ablation/ablation.py', 'experiments/b-ladder-top/topfix.py',
        'experiments/cheap-corrections/varpro.py', 'experiments/b-panel/speed/fast.py',
        'experiments/b-panel/speed/ladders.py', 'experiments/quadratic-manifold/qman.py']
OPS = [f'{LANE}/ops/{f}' for f in ('train.py', 'dataset.py', 'model.py', 'families.py', 'spectral_conv_f64.py',
                                   'NEURALOPERATOR-LICENSE', 'worker.py', 'optime.py')]
PYPATH = ['experiments/mr-burgers2d', 'experiments/separable-decoder', 'experiments/head-ablation',
          'experiments/cheap-corrections', 'experiments/b-ladder-top', 'experiments/b-panel/speed',
          'experiments/quadratic-manifold', f'{LANE}/lib', LANE]

HEADER = '''#!/bin/bash
#SBATCH --job-name=bcmp_{attempt}
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres={gres}
{constraint}
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem={mem}
#SBATCH --time={hours:02d}:00:00
#SBATCH --output={remote}/logs/%j.out
#SBATCH --error={remote}/logs/%j.err
set -euo pipefail
TASK_ROOT={remote}
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION={mem_fraction}
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}',flush=True); sys.exit(0 if b=='gpu' else 42)"
export PYTHONPATH="{pypath}"
'''

TRAIN = '''cd {lane}
"$PY" opdata.py --mesh {mesh} --out data
"$PY" -c "import torch,sys; ok=torch.cuda.is_available(); print('torch_cuda', ok, torch.cuda.get_device_name() if ok else None, flush=True); sys.exit(0 if ok else 42)"
"$PY" ops/worker.py {spec}
find out data -name '*.json' -o -name 'best.pt' | sort | xargs sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind', choices=('train', 'panel'))
    p.add_argument('attempt')
    p.add_argument('--mesh', type=int)
    p.add_argument('--config')
    p.add_argument('--train-from')
    p.add_argument('--spec', default='specs/train.json')
    p.add_argument('--late', default=None, help='<train attempt>:<arm,arm>:<max wait s>')
    p.add_argument('--gpu', default='a100-80G', choices=sorted(GRES))
    p.add_argument('--mem', default='240G')
    p.add_argument('--hours', type=int, default=8)
    p.add_argument('--mem-fraction', default='0.92')
    a = p.parse_args()
    assert a.attempt.isalnum(), a.attempt
    lane = ROOT / LANE
    if a.kind == 'train':
        assert a.mesh
        files = LIBS[:4] + OPS + [f'{LANE}/opdata.py', f'{LANE}/{a.spec}'] + \
            [f'{LANE}/inputs/pinned/{n}-index.json' for n in ('train', 'validation')] + \
            [str(x.relative_to(ROOT)) for x in sorted((lane / 'inputs/opconfigs').glob('*.json'))
             if not x.name.startswith('smoke')]
        body = TRAIN.format(lane=LANE, mesh=a.mesh, spec=a.spec)
        extra = []
    else:
        import panel_body                          # the panel job's body lives beside this script
        cfg = json.loads((lane / a.config).read_text())
        files, body, extra = panel_body.files_and_body(ROOT, LANE, LIBS, OPS, CHECKPOINT, cfg, a)
    files = list(dict.fromkeys(files))
    out = lane / 'runs' / a.attempt
    out.mkdir(parents=True, exist_ok=False)
    remote = f'{NAMESPACE}/{a.attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    proof = []
    for name in files:
        content = (ROOT / name).read_bytes()
        assert content == subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{name}']), f'not committed: {name}'
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        proof.append(dict(source=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), commit=commit))
    for src, name in extra:                       # uncommitted, hash-verified against a committed record
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(src).read_bytes())
        proof.append(dict(source=str(src), staged_as=name, bytes=dest.stat().st_size,
                          sha256=hashlib.sha256(dest.read_bytes()).hexdigest(), commit=None,
                          note='operator checkpoint, verified against the committed operators-<L>.json'))
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=2) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    (out / 'logs').mkdir()
    (out / LANE / 'logs').mkdir(parents=True, exist_ok=True)
    gres, constraint = GRES[a.gpu]
    script = HEADER.format(attempt=a.attempt, gres=gres, constraint=('#SBATCH ' + constraint) if constraint else '',
                           mem=a.mem, hours=a.hours, remote=remote, mem_fraction=a.mem_fraction,
                           pypath=':'.join('$TASK_ROOT/' + x for x in PYPATH)) + body
    (out / 'run.sbatch').write_text(script)
    manifest = [f'{hashlib.sha256(p_.read_bytes()).hexdigest()}  {p_.relative_to(out)}'
                for p_ in sorted(out.rglob('*')) if p_.is_file()]
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote)


if __name__ == '__main__':
    main()
