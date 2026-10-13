"""Stage ONE jcp-references job into runs/<attempt>/ from COMMITTED files only (adapted from vendor/quad3d/cluster).

    python cluster/stage.py <attempt> --config configs/<cfg>.json [--gpu a100-80G|h100|h200] [--mem 120G] [--hours 12]

Raw sbatch: gpu partition only, the paralab venv, the jax_backend=gpu preflight (exit 42), f64,
JAX_DEFAULT_MATMUL_PRECISION=highest, all output under /cluster/tufts/paralab/tawal01/jcp-ref/<attempt>/, SHA256
manifest checked on the node. Sealed configs reference the custody key and the encrypted seed file by absolute
cluster path (never staged, never in git).
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/jcp-ref'
LANE = 'experiments/jcp-references'
BASE = [f'{LANE}/{f}' for f in ('cohorts.py', 'refcore.py', 'pscore.py', 'verdict.py', 'gates.py', 'sealio.py',
                                 'refjob.py', 'checks/rng_golden.json', 'cluster/stage.py')]
BASE += ['experiments/mr-burgers2d/engines.py', 'experiments/separable-decoder/sep_common.py',
         f'{LANE}/vendor/quad3d/vendor/burgers3d-span/common.py',
         f'{LANE}/vendor/quad3d/vendor/paper-b3d/vendor/b3d_common.py']
GRES = {'a100-80G': ('gpu:a100:1', '--constraint=a100-80G'), 'h100': ('gpu:h100:1', None),
        'h200': ('gpu:h200:1', None)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('--config', required=True)
    p.add_argument('--gpu', default='a100-80G', choices=sorted(GRES))
    p.add_argument('--mem', default='120G')
    p.add_argument('--hours', type=int, default=12)
    a = p.parse_args()
    assert a.attempt.isalnum(), a.attempt
    cfg_rel = a.config if a.config.startswith('experiments/') else f'{LANE}/{a.config}'
    files = list(dict.fromkeys(BASE + [cfg_rel]))
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
#SBATCH --job-name=jr_{a.attempt}
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
which openssl
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd "$TASK_ROOT/code/{LANE}"
"$PY" refjob.py --config "$TASK_ROOT/code/{cfg_rel}" --out "$TASK_ROOT/output"
cd "$TASK_ROOT"
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
