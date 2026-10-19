"""Stage one jcp-regimes attempt into its own directory (pattern of jcp-qualify's cluster/stage.py). Every staged file is
taken from the Git blob at HEAD (`git show`), so the job runs a committed tree; files present in the checkout must equal
their blob byte for byte. One attempt runs the unit gates, then its (driver, config) steps sequentially in one job.

    python cluster/stage.py <attempt> --run refjob:configs/x.json [--run qacc:configs/y.json ...] --hours 2
                            [--gpu h200|a100-80G] [--mem 200G] [--mem-fraction 0.92]
Production attempts (not smk*) need checks/resources-<attempt>.json bound to the exact configs and runs, and a time
limit >= 1.5 x its projection (DESIGN A2.6).
"""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/jcp-regimes'
LANE = 'experiments/jcp-regimes'
EXTRA = ['experiments/mr-burgers2d/engines.py', 'experiments/mr-burgers2d/iterative_paths.py',
         'experiments/separable-decoder/sep_common.py',
         'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl']
DRIVERS = ('qacc', 'refjob', 'refslim', 'fverify')
GRES = {'h200': ('gpu:h200:1', None)}                     # DESIGN A2.6: every job on H200
SKIP = ('runs/', 'audits/', 'plots/', 'report', 'HEARTBEAT.json', 'store/')


def lane_files(commit):
    out = subprocess.check_output(['git', '-C', str(ROOT), 'ls-tree', '-r', '--name-only', commit, LANE + '/'], text=True)
    return [f for f in out.split() if not any(f[len(LANE) + 1:].startswith(s) for s in SKIP)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('--run', action='append', required=True)
    p.add_argument('--gpu', default='h200', choices=sorted(GRES))
    p.add_argument('--mem', default='200G')
    p.add_argument('--hours', type=float, required=True)
    p.add_argument('--mem-fraction', default='0.92')
    a = p.parse_args()
    assert a.attempt.isalnum(), a.attempt
    assert math.isfinite(a.hours) and 0 < a.hours <= 48, a.hours
    if not a.attempt.startswith('smk'):            # A3.3: the job must be part of the certified production plan
        c_ = json.loads((ROOT / LANE / 'checks' / f'resources-{a.attempt}.json').read_text())
        plan = json.loads((ROOT / LANE / 'checks' / 'resources-plan.json').read_text())
        assert a.attempt in plan['attempts'], f'{a.attempt} is not in the certified plan'
        led = json.loads((ROOT / LANE / 'runs' / 'GPU-LEDGER.json').read_text())
        seen = {j['attempt'] for j in led['jobs']}
        assert a.attempt not in seen, 'attempt name already used in the ledger (a retry needs a new planned name)'
        assert plan['attempts'][a.attempt]['runs'] == c_['runs'] == a.run, 'plan / certificate / requested runs differ'
        remaining = sum(v['time_limit_hours'] for k, v in plan['attempts'].items() if k not in seen)
        assert led['spent_gpu_hours'] + led['reserved_gpu_hours'] + remaining <= 60 + 1e-9, 'plan exceeds the cap'
        assert abs(plan['attempts'][a.attempt]['time_limit_hours'] - a.hours) < 1e-9, 'time limit differs from the plan'
    if not a.attempt.startswith('smk'):
        assert a.hours >= 1.5 * c_['projection_hours'] - 1e-9, (a.hours, c_['projection_hours'])
        for drv, cf in [r.split(':', 1) for r in a.run]:
            assert c_['configs'].get(cf) == hashlib.sha256((ROOT / LANE / cf).read_bytes()).hexdigest(), cf
        assert sorted(c_['runs']) == sorted(a.run), (c_['runs'], a.run)
    else:
        assert a.hours <= 2.0, 'a smoke is capped at 2 GPU-h (COMMON-BRIEF gating)'
    runs = []
    for r in a.run:
        drv, cfg = r.split(':', 1)
        assert drv in DRIVERS, drv
        runs.append((drv, cfg))
    out = ROOT / LANE / 'runs' / a.attempt
    import sys
    sys.path.insert(0, str(ROOT / LANE))
    import guard
    guard.check_admin(out / 'run.sbatch', ROOT / LANE)
    out.mkdir(parents=True, exist_ok=False)
    remote = f'{NAMESPACE}/{a.attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain', '--', LANE], text=True)
    bad = [l for l in dirty.splitlines() if not any(l[3:].startswith(f'{LANE}/{s}') for s in ('runs', 'HEARTBEAT.json', 'audits'))]
    assert not bad, bad
    files = lane_files(commit) + EXTRA
    for _, cfg in runs:
        assert f'{LANE}/{cfg}' in files, f'config not committed: {cfg}'
    proof = []
    for name in files:
        content = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{name}'])
        if (ROOT / name).exists():
            assert (ROOT / name).read_bytes() == content, f'working copy differs from HEAD: {name}'
        dest = out / name
        guard.check_admin(dest, ROOT / LANE)
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
#SBATCH --job-name=jg_{a.attempt}
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
JAX_PLATFORMS=cpu "$PY" test_regimes.py "$TASK_ROOT/output/unit_gates.json"
{steps}
echo "STEP end - $(date -u +%FT%TZ)"
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
