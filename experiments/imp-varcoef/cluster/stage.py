"""Stage one imp-varcoef job into its own directory (jcp-fom2d cluster/stage.py pattern). Every staged file is taken from
the Git blob at HEAD; checkout copies must equal their blob. One job may run several (driver, config) steps in sequence.

    python cluster/stage.py <job> --run vref:configs/x.json [--run vladder:configs/y.json ...]
                            --gpu h200|h100|a100|any --hours H [--mem 200G] [--mem-fraction 0.92]
Production jobs (name not starting with 'smk') need checks/resources-<job>.json (smoke-derived certificate, A3.20) with
hours >= 1.5 x projection and the exact configs bound by sha256.
"""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/imp-varcoef'
LANE = 'experiments/imp-varcoef'
DEPS = ['experiments/jcp-fom2d']
EXTRA = ['experiments/mr-burgers2d/engines.py', 'experiments/separable-decoder/sep_common.py',
         'experiments/jcp-references/verdict.py']
EXTRA_ROM = ['experiments/mr-burgers2d/iterative_paths.py',
             'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl']
DRIVERS = ('vref', 'vladder', 'vtime', 'checks_impl', 'vtest')
GRES = {'a100': 'gpu:a100:1', 'h100': 'gpu:h100:1', 'h200': 'gpu:h200:1', 'any': 'gpu:1'}
SKIP_LANE = ('runs/', 'audits/', 'plots/', 'report', 'HEARTBEAT.json', 'inputs/c24096-result.json')
SKIP_DEP = ('runs/', 'audits/', 'checks/', 'plots/', 'report', 'HEARTBEAT.json')


def tree(commit, d, skip):
    out = subprocess.check_output(['git', '-C', str(ROOT), 'ls-tree', '-r', '--name-only', commit, d + '/'], text=True)
    return [f for f in out.split() if not any(f[len(d) + 1:].startswith(s) for s in skip)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('job')
    p.add_argument('--run', action='append', required=True)
    p.add_argument('--gpu', required=True, choices=sorted(GRES))
    p.add_argument('--mem', default='200G')
    p.add_argument('--hours', type=float, required=True)
    p.add_argument('--mem-fraction', default='0.92')
    a = p.parse_args()
    assert a.job.replace('-', '').isalnum(), a.job
    runs = [tuple(r.split(':', 1)) for r in a.run]
    assert all(d in DRIVERS for d, _ in runs), runs
    if not a.job.startswith('smk'):
        c_ = json.loads((ROOT / LANE / 'checks' / f'resources-{a.job}.json').read_text())
        assert a.hours >= 1.5 * c_['projection_hours'] - 1e-9, (a.hours, c_['projection_hours'])
        for _, cf in runs:
            assert c_['configs'].get(cf) == hashlib.sha256((ROOT / LANE / cf).read_bytes()).hexdigest(), cf
        assert sorted(c_['runs']) == sorted(a.run), (c_['runs'], a.run)
        if any(d == 'vtime' for d, _ in runs):
            assert a.gpu == 'h200', 'timing needs the H200 (4096^2 bank, same GPU type as C2)'
    out = ROOT / LANE / 'runs' / a.job
    out.mkdir(parents=True, exist_ok=False)
    remote = f'{NAMESPACE}/{a.job}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain', '--', LANE], text=True)
    bad = [l for l in dirty.splitlines() if not any(l[3:].startswith(f'{LANE}/{s}') for s in ('runs', 'HEARTBEAT.json', 'audits'))]
    assert not bad, ('uncommitted lane changes', bad)
    files = tree(commit, LANE, SKIP_LANE) + [f for d in DEPS for f in tree(commit, d, SKIP_DEP)] + EXTRA
    if any(d == 'vtime' for d, _ in runs):
        files += EXTRA_ROM
    for _, cfg in runs:
        assert f'{LANE}/{cfg}' in files, f'config not committed: {cfg}'
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
    h = math.ceil(a.hours * 4) / 4
    tl = f'{int(h):02d}:{int(round((h - int(h)) * 60)):02d}:00'
    steps = '\n'.join(f'echo "STEP {drv} {cfg} $(date -u +%FT%TZ)"\n"$PY" {drv}.py --config {cfg} --out "$TASK_ROOT/output/{drv}_{Path(cfg).stem}"'
                      for drv, cfg in runs)
    script = f'''#!/bin/bash
#SBATCH --job-name=ivc_{a.job}
#SBATCH --partition=gpu
#SBATCH --gres={GRES[a.gpu]}
#SBATCH --exclude=pax007
#SBATCH --no-requeue
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
export JCPREF_STORE=/cluster/tufts/paralab/tawal01/jcp-ref/store
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}',flush=True); sys.exit(0 if b=='gpu' else 42)"
"$PY" -c "import jax; jax.config.update('jax_enable_x64', True); import jax.numpy as jnp, os; assert jnp.zeros(1).dtype == jnp.float64; assert os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'; print('x64+highest ok', flush=True)"
cd {LANE}
export JCPREF_META="$TASK_ROOT/experiments/jcp-fom2d/e0meta"
{steps}
cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''
    (out / 'run.sbatch').write_text(script)
    (out / 'RESERVATION.json').write_text(json.dumps(dict(job=a.job, gpu=a.gpu, time_limit_hours=h, runs=runs,
                                                          commit=commit), indent=1) + '\n')
    manifest = [f'{hashlib.sha256(p_.read_bytes()).hexdigest()}  {p_.relative_to(out)}'
                for p_ in sorted(out.rglob('*')) if p_.is_file()]
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote, 'time limit', tl)


if __name__ == '__main__':
    main()
