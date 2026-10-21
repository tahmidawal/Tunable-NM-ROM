"""Stage one burgers2d-speed attempt into its own directory (adapted from experiments/burgers-bank-knob/cluster/stage.py
@ b5c843ab). Every staged file is taken from the Git blob at HEAD (`git show`), so the job runs a committed tree even
for files outside this narrow sparse checkout; files present in the checkout must equal their blob byte for byte.

    python cluster/stage.py <attempt> <config> [--gpu a100|a100-80G|h100|l40s] [--mem 128G] [--hours 8]
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/b2speed_20260923'
LANE = 'experiments/burgers2d-speed'
BKL = 'experiments/burgers-bank-knob'
RP = 'experiments/burgers-repanel'
HB = 'experiments/hires-burgers'
CHECKPOINT = 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
FILES = [f'{LANE}/b2speed.py', f'{LANE}/b2fast.py', f'{LANE}/cluster/stage.py', f'{BKL}/bkfast.py',
         f'{BKL}/inputs/rotation_R512.npz', f'{RP}/xfast.py', f'{HB}/hops.py', f'{HB}/hfast.py',
         'experiments/b-panel/speed/fast.py', 'experiments/b-panel/speed/ladders.py',
         'experiments/b-panel/inputs/directions_qtd02.npz',
         'experiments/b-ladder-top/topfix.py', 'experiments/cheap-corrections/varpro.py',
         'experiments/head-ablation/arms.py', 'experiments/head-ablation/ladder.py',
         'experiments/head-ablation/ablation.py', 'experiments/mr-burgers2d/engines.py',
         'experiments/mr-burgers2d/iterative_paths.py', 'experiments/mr-burgers2d/accuracy_paths.py',
         'experiments/separable-decoder/sep_common.py', CHECKPOINT]
GRES = {'a100-80G': ('gpu:a100:1', '--constraint=a100-80G'), 'a100': ('gpu:a100:1', None),
        'h100': ('gpu:h100:1', None), 'l40s': ('gpu:l40s:1', None)}
PYPATH = ['experiments/mr-burgers2d', 'experiments/separable-decoder', 'experiments/head-ablation',
          'experiments/cheap-corrections', 'experiments/b-ladder-top', 'experiments/b-panel/speed', HB, RP, BKL, LANE]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('config')
    p.add_argument('--gpu', default='a100', choices=sorted(GRES))
    p.add_argument('--mem', default='128G')
    p.add_argument('--hours', type=int, default=8)
    p.add_argument('--mem-fraction', default='0.90')
    a = p.parse_args()
    assert a.attempt.isalnum(), a.attempt
    cfg = json.loads((ROOT / LANE / a.config).read_text())
    files = FILES + [f'{LANE}/{a.config}']
    files += [f"experiments/b-panel/inputs/{rs['file']}" for rs in cfg['rules'].values() if 'file' in rs]
    files = list(dict.fromkeys(files))
    out = ROOT / LANE / 'runs' / a.attempt
    out.mkdir(parents=True, exist_ok=False)
    remote = f'{NAMESPACE}/{a.attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain', '--', LANE], text=True)
    assert not [l for l in dirty.splitlines() if not l.startswith('?? ') or not l[3:].startswith(f'{LANE}/runs')], dirty
    proof = []
    for name in files:
        content = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{name}'])
        if (ROOT / name).exists():
            assert (ROOT / name).read_bytes() == content, f'working copy differs from HEAD: {name}'
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        proof.append(dict(source=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), commit=commit))
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=2) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    (out / 'logs').mkdir()
    gres, constraint = GRES[a.gpu]
    script = f'''#!/bin/bash
#SBATCH --job-name=b2sp_{a.attempt}
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres={gres}
{('#SBATCH ' + constraint) if constraint else ''}
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem={a.mem}
#SBATCH --time={a.hours:02d}:00:00
#SBATCH --output={remote}/logs/%j.out
#SBATCH --error={remote}/logs/%j.err
set -euo pipefail
TASK_ROOT={remote}
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION={a.mem_fraction}
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}',flush=True); sys.exit(0 if b=='gpu' else 42)"
export PYTHONPATH="{':'.join('$TASK_ROOT/' + x for x in PYPATH)}"
"$PY" {LANE}/b2speed.py --config {LANE}/{a.config} --checkpoint {CHECKPOINT} \\
  --rotation {BKL}/inputs/rotation_R512.npz \\
  --inputs experiments/b-panel/inputs --out output
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''
    (out / 'run.sbatch').write_text(script)
    manifest = [f'{hashlib.sha256(p_.read_bytes()).hexdigest()}  {p_.relative_to(out)}'
                for p_ in sorted(out.rglob('*')) if p_.is_file()]
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote)


if __name__ == '__main__':
    main()
