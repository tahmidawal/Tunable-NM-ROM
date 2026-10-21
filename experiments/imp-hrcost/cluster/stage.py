"""Stage one imp-hrcost attempt (pattern of jcp-qualify cluster/stage.py): every staged file is the Git blob at HEAD;
files present in the checkout must equal their blob. One attempt = one config of hjob.py in one job.

    python cluster/stage.py <attempt> --config configs/<x>.json --hours H [--gpu h200|a100-80G|h100] [--mem 200G]
"""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/imp-hrcost'
LANE = 'experiments/imp-hrcost'
EXTRA = ['experiments/mr-burgers2d/engines.py', 'experiments/mr-burgers2d/iterative_paths.py',
         'experiments/separable-decoder/sep_common.py']
GRES = {'a100-80G': ('gpu:a100:1', '--constraint=a100-80G'), 'h100': ('gpu:h100:1', None), 'h200': ('gpu:h200:1', None)}
SKIP = ('runs/', 'audits/', 'checks/', 'plots/', 'report', 'HEARTBEAT.json', 'EXPORT.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('--config', required=True)
    p.add_argument('--hours', type=float, required=True)
    p.add_argument('--gpu', default='h200', choices=sorted(GRES))
    p.add_argument('--mem', default='200G')
    p.add_argument('--mem-fraction', default='0.92')
    a = p.parse_args()
    assert a.attempt.isalnum(), a.attempt
    lane = ROOT / LANE
    led = json.loads((lane / 'runs' / 'GPU-LEDGER.json').read_text()) if (lane / 'runs' / 'GPU-LEDGER.json').exists() \
        else dict(total_gpu_hours=0., reservations={})
    spent = float(led['total_gpu_hours']) + sum(led.get('reservations', {}).values())
    cfg = json.loads((lane / a.config).read_text())
    assert cfg['attempt'] == a.attempt, ('config attempt differs', cfg['attempt'], a.attempt)
    if a.attempt.startswith('smk'):
        assert a.hours <= 1.5 + 1e-9, 'smoke reservation is 1.5 h (DESIGN A3.5)'
    else:       # production: smoke-derived certificate bound to this exact config (code audit 1)
        c_ = json.loads((lane / 'checks' / f'resources-{a.attempt}.json').read_text())
        assert c_['ok'] and c_['config_sha256'] == hashlib.sha256((lane / a.config).read_bytes()).hexdigest(), 'certificate'
        acc = json.loads((lane / 'checks' / f'smoke-accept-{c_["smoke_attempt"]}.json').read_text())
        assert acc['accepted'], 'smoke not accepted'
        assert a.hours >= c_['projection_hours'] - 1e-9, ('time limit below projection', a.hours, c_['projection_hours'])
    assert spent + a.hours <= 3.0 + 1e-9, ('GPU-hour cap', spent, a.hours)
    out = ROOT / LANE / 'runs' / a.attempt
    out.mkdir(parents=True, exist_ok=False)
    remote = f'{NAMESPACE}/{a.attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain', '--', LANE], text=True)
    bad = [l for l in dirty.splitlines() if not any(l[3:].startswith(f'{LANE}/{s}') for s in ('runs', 'HEARTBEAT.json', 'audits'))]
    assert not bad, bad
    files = subprocess.check_output(['git', '-C', str(ROOT), 'ls-tree', '-r', '--name-only', commit, LANE + '/'], text=True).split()
    files = [f for f in files if not any(f[len(LANE) + 1:].startswith(s) for s in SKIP)] + EXTRA
    assert f'{LANE}/{a.config}' in files, f'config not committed: {a.config}'
    proof = []
    for name in files:
        content = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{name}'])
        if (ROOT / name).exists():
            assert (ROOT / name).read_bytes() == content, f'working copy differs from HEAD: {name}'
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        proof.append(dict(source=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), commit=commit))
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=1) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    (out / 'logs').mkdir()
    gres, constraint = GRES[a.gpu]
    h = math.ceil(a.hours * 4) / 4
    tl = f'{int(h):02d}:{int(round((h - int(h)) * 60)):02d}:00'
    script = f'''#!/bin/bash
#SBATCH --job-name=ihc_{a.attempt}
#SBATCH --partition=gpu
#SBATCH --gres={gres}
{('#SBATCH ' + constraint) if constraint else ''}
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem={a.mem}
#SBATCH --time={tl}
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
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd {LANE}
echo "STEP hjob {a.config} $(date -u +%FT%TZ)"
"$PY" hjob.py --config {a.config} --out "$TASK_ROOT/output/hjob"
cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''
    (out / 'run.sbatch').write_text(script)
    (out / 'RESERVATION.json').write_text(json.dumps(dict(attempt=a.attempt, gpu=a.gpu, time_limit_hours=h,
                                                          config=a.config, commit=commit), indent=1) + '\n')
    manifest = [f'{hashlib.sha256(p_.read_bytes()).hexdigest()}  {p_.relative_to(out)}'
                for p_ in sorted(out.rglob('*')) if p_.is_file() and p_.name != 'MANIFEST.sha256']
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote, 'time limit', tl)


if __name__ == '__main__':
    main()
