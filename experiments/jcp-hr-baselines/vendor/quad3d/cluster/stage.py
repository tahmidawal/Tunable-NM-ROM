"""Stage one quadrature-burgers3d job into runs/<attempt>/ (one attempt directory per job), from COMMITTED files only.

    python cluster/stage.py <attempt> --cmd "<python script + args, run from the code root>" [--gpu h200|a100-80G|h100]
          [--mem 240G] [--hours 8] [--extra path ...]

Raw sbatch (adapted from burgers3d-retry/cluster/stage.py): the ae-research tufts-submit.sh writes to its own
jobs/<worktree> directory and rsyncs the whole worktree, which cannot honour this lane's namespace. Every protocol
requirement is kept: gpu partition only, the paralab venv, the jax_backend=gpu preflight (exit 42), f64,
JAX_DEFAULT_MATMUL_PRECISION=highest, all output under the lane namespace, SHA256 manifest checked on the node.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/quad3d_20261001'
LANE = 'experiments/quadrature-burgers3d'
BASE = [f'{LANE}/offmesh.py', f'{LANE}/qpanel.py', f'{LANE}/refjob.py', f'{LANE}/cluster/stage.py',
        f'{LANE}/vendor/burgers3d-span/common.py', f'{LANE}/vendor/burgers3d-retry/tables.py',
        f'{LANE}/vendor/paper-b3d/vendor/b3d_common.py', f'{LANE}/inputs/model_M2/bank.pkl',
        f'{LANE}/rules/rules.npz']
GRES = {'a100-80G': ('gpu:a100:1', '--constraint=a100-80G'), 'h100': ('gpu:h100:1', None),
        'h200': ('gpu:h200:1', None)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('--cmd', required=True)
    p.add_argument('--gpu', default='h200', choices=sorted(GRES))
    p.add_argument('--mem', default='240G')
    p.add_argument('--hours', type=int, default=8)
    p.add_argument('--extra', nargs='*', default=[])
    a = p.parse_args()
    assert a.attempt.isalnum(), a.attempt
    files = list(dict.fromkeys(BASE + [x if x.startswith('experiments/') else f'{LANE}/{x}' for x in a.extra]))
    out = ROOT / LANE / 'runs' / a.attempt
    out.mkdir(parents=True, exist_ok=False)
    remote = f'{NAMESPACE}/{a.attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    proof = []
    for name in files:
        content = (ROOT / name).read_bytes()
        assert content == subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{name}']), f'not committed: {name}'
        dest = out / 'code' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        proof.append(dict(source=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), commit=commit))
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=2) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    (out / 'logs').mkdir()
    gres, constraint = GRES[a.gpu]
    script = f'''#!/bin/bash
#SBATCH --job-name=q3_{a.attempt}
#SBATCH --partition=gpu
#SBATCH --gres={gres}
{('#SBATCH ' + constraint) if constraint else ''}
#SBATCH --cpus-per-task=8
#SBATCH --mem={a.mem}
#SBATCH --time={a.hours:02d}:00:00
#SBATCH --output={remote}/logs/%j.out
#SBATCH --error={remote}/logs/%j.err
set -euo pipefail
TASK_ROOT={remote}
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt) SLURM_JOB_ID
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd "$TASK_ROOT/code"
{a.cmd}
cd "$TASK_ROOT"
find code/output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
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
