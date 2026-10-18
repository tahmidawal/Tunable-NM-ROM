"""Stage one jcp-fom2d attempt into its own directory (vendored quadrature-lane pattern). Every staged file is taken
from the Git blob at HEAD (`git show`), so the job runs a committed tree; files present in the checkout must equal their
blob byte for byte. One attempt may run several (driver, config) steps sequentially in one job.

    python cluster/stage.py <attempt> --run p1acc:configs/x.json [--run p1time:configs/y.json ...]
                            [--gpu a100-80G|h200|h100|a100] [--mem 200G] [--hours 2.5] [--mem-fraction 0.92]
"""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/jcp-fom2d'
LANE = 'experiments/jcp-fom2d'
EXTRA = ['experiments/mr-burgers2d/engines.py', 'experiments/separable-decoder/sep_common.py']   # engines imports sep_common
EXTRA_ROM = ['experiments/mr-burgers2d/iterative_paths.py',
             'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl']   # C2 (ROM) jobs only
DRIVERS = ('ladder', 'c2time')
GRES = {'a100-80G': ('gpu:a100:1', '--constraint=a100-80G'), 'a100': ('gpu:a100:1', None),
        'h100': ('gpu:h100:1', None), 'h200': ('gpu:h200:1', None)}
SKIP = ('runs/', 'audits/', 'checks/', 'plots/', 'report', 'HEARTBEAT.json')


def lane_files(commit):
    out = subprocess.check_output(['git', '-C', str(ROOT), 'ls-tree', '-r', '--name-only', commit, LANE + '/'], text=True)
    return [f for f in out.split() if not any(f[len(LANE) + 1:].startswith(s) for s in SKIP)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('--run', action='append', required=True)
    p.add_argument('--gpu', default='h200', choices=['h200'])          # DESIGN §7: H200 throughout
    p.add_argument('--mem', default='200G')
    p.add_argument('--hours', type=float, required=True)
    p.add_argument('--mem-fraction', default='0.92')
    a = p.parse_args()
    assert a.attempt.isalnum(), a.attempt
    cert = ROOT / LANE / 'checks' / f'resources-{a.attempt}.json'
    smoke = a.attempt.startswith('smk') and all(r.split(':', 1)[1].startswith('configs/smk') for r in a.run)
    if not smoke:
        # production attempts need a smoke-derived certificate; the time limit must cover projection x 1.5 (A1.11)
        c_ = json.loads(cert.read_text())
        assert a.hours >= 1.5 * c_['projection_hours'] - 1e-9, (a.hours, c_['projection_hours'])
        for drv, cf in [r.split(':', 1) for r in a.run]:     # the certificate is bound to the exact configs and runs
            assert c_['configs'].get(cf) == hashlib.sha256((ROOT / LANE / cf).read_bytes()).hexdigest(), cf
        assert sorted(c_['runs']) == sorted(a.run), (c_['runs'], a.run)
    runs = []
    for r in a.run:
        drv, cfg = r.split(':', 1)
        assert drv in DRIVERS, drv
        runs.append((drv, cfg))
    out = ROOT / LANE / 'runs' / a.attempt
    out.mkdir(parents=True, exist_ok=False)
    remote = f'{NAMESPACE}/{a.attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain', '--', LANE], text=True)
    bad = [l for l in dirty.splitlines() if not any(l[3:].startswith(f'{LANE}/{s}') for s in ('runs', 'HEARTBEAT.json', 'audits'))]
    assert not bad, bad
    files = lane_files(commit) + EXTRA + (EXTRA_ROM if any(d == 'c2time' for d, _ in runs) else [])
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
    gres, constraint = GRES[a.gpu]
    h = math.ceil(a.hours * 4) / 4
    tl = f'{int(h):02d}:{int(round((h - int(h)) * 60)):02d}:00'
    steps = '\n'.join(f'echo "STEP {drv} {cfg} $(date -u +%FT%TZ)"\n"$PY" {drv}.py --config {cfg} --out "$TASK_ROOT/output/{drv}_{Path(cfg).stem}"'
                      for drv, cfg in runs)
    script = f'''#!/bin/bash
#SBATCH --job-name=f2d_{a.attempt}
#SBATCH --partition=gpu
#SBATCH --gres={gres}
{('#SBATCH ' + constraint) if constraint else ''}
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
cd {LANE}
export JCPREF_META="$PWD/e0meta"
{steps}
cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''
    (out / 'run.sbatch').write_text(script)
    (out / 'RESERVATION.json').write_text(json.dumps(dict(attempt=a.attempt, gpu=a.gpu, time_limit_hours=h, runs=runs,
                                                          commit=commit), indent=1) + '\n')
    manifest = [f'{hashlib.sha256(p_.read_bytes()).hexdigest()}  {p_.relative_to(out)}'
                for p_ in sorted(out.rglob('*')) if p_.is_file()]
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote, 'time limit', tl)


if __name__ == '__main__':
    main()
