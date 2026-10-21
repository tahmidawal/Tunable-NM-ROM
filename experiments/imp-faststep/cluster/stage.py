"""Stage one imp-faststep job into its own directory (p1-scaling / hr-baselines pattern). Every staged file is the Git
blob at HEAD (`git show`); a checked-out copy must equal its blob. The staged tree is import-checked on the local CPU
(every lane module must resolve inside the staged tree). One job per directory under the lane namespace.

    python cluster/stage.py <job> --driver fsrun|ecswjob --config configs/<x>.json [--gpu h200] [--mem 200G]
The time limit is the config's `time_limit_hours` (the registered reservation; the ledger admits on it).
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/imp-faststep'
E0_STORE = '/cluster/tufts/paralab/tawal01/jcp-ref/store'
LANE = 'experiments/imp-faststep'
P1, HRB, VQ, V3 = 'experiments/jcp-p1-scaling', 'experiments/jcp-hr-baselines', 'experiments/jcp-p1-scaling/vendor/quad2d', \
    'experiments/jcp-hr-baselines/vendor/quad3d'
LANE_FILES = ['fscore.py', 'fsrun.py', 'fsselect.py', 'fstiming.py', 'fscontrols.py', 'podarm.py', 'ecswjob.py']
EXTRA = [f'{P1}/p1core.py', f'{P1}/p1common.py', f'{VQ}/qcore.py', f'{VQ}/qstudy.py', f'{VQ}/vendor/arms.py',
         f'{VQ}/vendor/hops.py', f'{VQ}/vendor/hfast.py', f'{VQ}/vendor/bkfast.py', f'{VQ}/vendor/hari_quadrature.py',
         f'{VQ}/inputs/rotation_R512.npz', f'{VQ}/inputs/rule_q0_m1024_qrg304_reachable.npz',
         f'{HRB}/hrcore.py', f'{HRB}/rom.py', f'{HRB}/build.py', f'{HRB}/nnls.py', f'{HRB}/t2core.py',
         f'{HRB}/checks/FROZEN-STAGEB-c2d1024.json', f'{V3}/vendor/burgers3d-span/common.py',
         f'{V3}/vendor/paper-b3d/vendor/b3d_common.py',
         'experiments/jcp-references/refstore.py', 'experiments/jcp-references/verdict.py',
         'experiments/jcp-references/store/MANIFEST.sha256', 'experiments/jcp-references/store/STATUS.json',
         'experiments/mr-burgers2d/engines.py', 'experiments/mr-burgers2d/iterative_paths.py',
         'experiments/separable-decoder/sep_common.py',
         'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl']
GRES = {'a100-80G': ('gpu:a100:1', '--constraint=a100-80G'), 'h100': ('gpu:h100:1', None), 'h200': ('gpu:h200:1', None)}


def git(*a):
    return subprocess.check_output(['git', '-C', str(ROOT), *a])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('job')
    p.add_argument('--driver', required=True, choices=('fsrun', 'ecswjob'))
    p.add_argument('--config', required=True)
    p.add_argument('--gpu', default='h200', choices=sorted(GRES))
    p.add_argument('--mem', default='200G')
    p.add_argument('--extra', action='append', default=[], help='additional lane-relative committed inputs')
    a = p.parse_args()
    assert a.job.isalnum(), a.job
    hours = float(json.loads((ROOT / LANE / a.config).read_text())['time_limit_hours'])
    assert 0 < hours <= 3.0, hours
    commit = git('rev-parse', 'HEAD').decode().strip()
    dirty = git('status', '--porcelain', '--', LANE).decode()
    bad = [l for l in dirty.splitlines() if not any(l[3:].startswith(f'{LANE}/{s}') for s in ('runs', 'HEARTBEAT.json', 'audits'))]
    assert not bad, bad
    out = ROOT / LANE / 'runs' / a.job / 'stage'
    out.mkdir(parents=True, exist_ok=False)
    files = [f'{LANE}/{f}' for f in LANE_FILES + [a.config] + a.extra] + EXTRA
    proof = []
    for name in files:
        content = git('show', f'{commit}:{name}')
        if (ROOT / name).exists():
            assert (ROOT / name).read_bytes() == content, f'working copy differs from HEAD: {name}'
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        proof.append(dict(source=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), commit=commit))
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=1) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    env = dict(os.environ, JAX_PLATFORMS='cpu', JAX_DEFAULT_MATMUL_PRECISION='highest', PYTHONPATH='', PYTHONDONTWRITEBYTECODE='1')
    mods = 'fsrun, fscore, fsselect, fstiming, fscontrols, podarm' if a.driver == 'fsrun' else 'ecswjob'
    chk = (f'import sys, {mods}; staged = {str(out)!r}; '
           'ms = [m.__file__ for m in list(sys.modules.values()) if getattr(m, "__file__", None) and "/experiments/" in m.__file__]; '
           'bad = [f for f in ms if not f.startswith(staged)]; assert not bad, bad; print(len(ms), "lane modules from the staged tree")')
    slot = '/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports/2026-10-21-improvement-campaign/campaign/jaxslot.sh'
    r_ = subprocess.run(['bash', slot, '900', 'jaxrun', '/home/tahmid/Dev/.venv/bin/python', '-c', chk], cwd=out / LANE, env=env,
                        capture_output=True, text=True)
    assert r_.returncode == 0, r_.stderr[-3000:]
    print(r_.stdout.strip())
    remote = f'{NAMESPACE}/{a.job}'
    gres, constraint = GRES[a.gpu]
    h = math.ceil(hours * 4) / 4
    tl = f'{int(h):02d}:{int(round((h - int(h)) * 60)):02d}:00'
    script = f'''#!/bin/bash
#SBATCH --job-name=ifs_{a.job}
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
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export JCPREF_STORE={E0_STORE}
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
echo "STEP {a.driver} {a.config} $(date -u +%FT%TZ)"
set +e
timeout --signal=TERM --kill-after=30 $(( {int(h * 60)} * 60 - 240 )) "$PY" {a.driver}.py --config {a.config} --out "$TASK_ROOT/output/{a.driver}_{a.job}"
rc=$?
echo "driver_exit=$rc"
set -e
cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
if [ $rc -eq 0 ] && [ ! -f "output/{a.driver}_{a.job}/COMPLETE" ]; then echo "no COMPLETE marker"; rc=5; fi
echo "ALL-DONE rc=$rc"
exit $rc
'''
    (out / 'run.sbatch').write_text(script)
    (out / 'RESERVATION.json').write_text(json.dumps(dict(job=a.job, gpu=a.gpu, time_limit_hours=h, driver=a.driver,
                                                          config=a.config, commit=commit), indent=1) + '\n')
    (out / 'logs').mkdir()
    manifest = [f'{hashlib.sha256(p_.read_bytes()).hexdigest()}  {p_.relative_to(out)}'
                for p_ in sorted(out.rglob('*')) if p_.is_file() and p_.name != 'MANIFEST.sha256']
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote, 'time limit', tl)


if __name__ == '__main__':
    main()
