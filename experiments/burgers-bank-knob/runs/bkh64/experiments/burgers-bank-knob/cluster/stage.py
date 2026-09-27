"""Stage one burgers-bank-knob attempt (adapted from experiments/hires-burgers/cluster/stage.py @ 0ab60014) into an isolated directory (one attempt directory per job).

    python cluster/stage.py <attempt> <config> [--gpu h200|a100-80G|a100|h100] [--mem 240G] [--hours 20]

Every staged file is checked byte for byte against the Git blob at HEAD, so the job runs a
committed tree. Raw `sbatch` with the protocol's requirements (the ae-research tufts-*.sh
helpers are not on this machine): gpu partition, paralab venv, preflight exit 42, f64,
JAX_DEFAULT_MATMUL_PRECISION=highest, all output under the lane namespace.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/bbank_20260923'
LANE = 'experiments/burgers-bank-knob'
RP = 'experiments/burgers-repanel'
HB = 'experiments/hires-burgers'
CHECKPOINT = 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
FILES = [f'{LANE}/bankknob.py', f'{LANE}/bkfast.py', f'{LANE}/inputs/rotation_R512.npz', f'{RP}/xfast.py', f'{HB}/hops.py', f'{HB}/hfast.py', f'{LANE}/cluster/stage.py',
         'experiments/b-panel/speed/fast.py', 'experiments/b-panel/speed/ladders.py',
         'experiments/b-panel/inputs/directions_qtd02.npz',
         'experiments/b-ladder-top/topfix.py', 'experiments/cheap-corrections/varpro.py',
         'experiments/head-ablation/arms.py', 'experiments/head-ablation/ladder.py',
         'experiments/head-ablation/ablation.py', 'experiments/mr-burgers2d/engines.py',
         'experiments/mr-burgers2d/iterative_paths.py', 'experiments/mr-burgers2d/accuracy_paths.py',
         'experiments/separable-decoder/sep_common.py', CHECKPOINT]
GRES = {'a100-80G': ('gpu:a100:1', '--constraint=a100-80G'), 'a100': ('gpu:a100:1', None),
        'h100': ('gpu:h100:1', None), 'h200': ('gpu:h200:1', None)}
PYPATH = ['experiments/mr-burgers2d', 'experiments/separable-decoder', 'experiments/head-ablation',
          'experiments/cheap-corrections', 'experiments/b-ladder-top', 'experiments/b-panel/speed', HB, RP, LANE]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('config')
    p.add_argument('--gpu', default='a100-80G', choices=sorted(GRES))
    p.add_argument('--mem', default='240G')
    p.add_argument('--hours', type=int, default=20)
    p.add_argument('--mem-fraction', default='0.92')
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
    proof = []
    for name in files:
        content = (ROOT / name).read_bytes()
        assert content == subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{name}']), f'not committed: {name}'
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        proof.append(dict(source=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), commit=commit))
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=2) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    (out / 'logs').mkdir()
    gres, constraint = GRES[a.gpu]
    script = f'''#!/bin/bash
#SBATCH --job-name=bbk_{a.attempt}
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
"$PY" {LANE}/bankknob.py --config {LANE}/{a.config} --checkpoint {CHECKPOINT} \\
  --rotation {LANE}/inputs/rotation_R512.npz \\
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
