"""Stage one ns2d attempt into an isolated paralab-bound directory (q-ridge mechanics).

    python experiments/ns2d/cluster/stage.py <attempt> <driver> <envfile> [gpu] [hours] [mem]

Each attempt gets its OWN submit directory and its OWN remote directory: one job per
directory, never two.  Every staged file is checked byte-for-byte against the committed
blob at HEAD before it is copied, so what runs on the cluster is exactly what is in Git.
<envfile> is experiments/ns2d/configs/<name>.env, sourced with `set -a` in the batch job.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/ns_20260917'
FILES = [
    'experiments/ns2d/ns2d_fom.py',
    'experiments/ns2d/ns2d_indep.py',
    'experiments/ns2d/ns2d_phase1.py',
    'experiments/ns2d/ns2d_decoder.py',
    'experiments/ns2d/ns2d_phase2.py',
    'experiments/ns2d/cluster/stage.py',
    'experiments/separable-decoder/sep_common.py',
]
OPTIONAL = [
    'experiments/ns2d/ns2d_rom.py',
    'experiments/ns2d/ns2d_phase3.py',
    'experiments/ns2d/configs/phase1-hashes.json',
]
DRIVERS = ('ns2d_phase1.py', 'ns2d_phase2.py', 'ns2d_phase3.py')
EXCLUDE = 'pax007'


def main():
    attempt, driver, envfile = sys.argv[1], sys.argv[2], sys.argv[3]
    gpu = sys.argv[4] if len(sys.argv) > 4 else 'a100'
    hours = sys.argv[5] if len(sys.argv) > 5 else '20:00:00'
    mem = sys.argv[6] if len(sys.argv) > 6 else '180G'
    assert attempt.isalnum(), attempt
    assert driver in DRIVERS, driver
    assert gpu in ('a100', 'h100', 'h200', 'l40s'), gpu
    files = FILES + [f'experiments/ns2d/configs/{envfile}']
    files += [f for f in OPTIONAL if (ROOT / f).exists()]
    out = ROOT / 'experiments/ns2d/runs' / attempt
    out.mkdir(parents=True, exist_ok=False)
    remote = f'{NAMESPACE}/{attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    proof = []
    for name in files:
        content = (ROOT / name).read_bytes()
        assert content == subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{name}']), name
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        proof.append(dict(source=name, bytes=len(content),
                          sha256=hashlib.sha256(content).hexdigest(), commit=commit))
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=2) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    (out / 'logs').mkdir()
    script = '''#!/bin/bash
#SBATCH --job-name=ns2d___ATTEMPT__
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:__GPU__:1
#SBATCH --exclude=__EXCLUDE__
#SBATCH --cpus-per-task=8
#SBATCH --mem=__MEM__
#SBATCH --time=__HOURS__
#SBATCH --output=__REMOTE__/logs/%j.out
#SBATCH --error=__REMOTE__/logs/%j.err
set -euo pipefail
TASK_ROOT=__REMOTE__
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
export TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
export PYTHONPATH="$TASK_ROOT/experiments/ns2d:$TASK_ROOT/experiments/separable-decoder"
set -a; source experiments/ns2d/configs/__ENVFILE__; set +a
export OUT=output
cd experiments/ns2d
"$PY" __DRIVER__
cd "$TASK_ROOT"
find experiments/ns2d/output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''
    for token, value in (('__ATTEMPT__', attempt), ('__REMOTE__', remote), ('__GPU__', gpu),
                         ('__HOURS__', hours), ('__MEM__', mem), ('__EXCLUDE__', EXCLUDE),
                         ('__DRIVER__', driver), ('__ENVFILE__', envfile)):
        script = script.replace(token, value)
    assert '__' not in script.replace('__pycache__', ''), script
    (out / 'run.sbatch').write_text(script)
    manifest = [f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}'
                for p in sorted(out.rglob('*')) if p.is_file()]
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote)


if __name__ == '__main__':
    main()
