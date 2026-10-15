"""Stage one jcp-speed3d attempt into runs/<attempt>/ (pattern of jcp-combined cluster/stage.py). Code files are taken
from the Git blob at HEAD (the checkout must match byte for byte); the first-order 3D reference (needed by fom / mesh
modes for continuity scores) is copied from the 3D quadrature lane and checked against its sha256. The E0 store is read
on the cluster at JCPREF_STORE (pinned store version and STATUS sha checked in job).

    python experiments/jcp-speed3d/cluster/stage.py <attempt> [--gpu h200] [--mem 200G] [--hours 24]
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LANE = 'experiments/jcp-speed3d'
NAMESPACE = '/cluster/tufts/paralab/tawal01/jcp-speed3d'
E0STORE = '/cluster/tufts/paralab/tawal01/jcp-ref/store'
REF1 = ROOT.parent / '2026-10-01-quadrature-burgers3d/experiments/quadrature-burgers3d/runs/ref1/code/output/ref_923801.npz'
REF1_SHA = '36e84e3dbac2f5cf1b3350c311c1f16d17ecfcd4dc217ded836ba30613c6cc8b'
V3 = f'{LANE}/vendor/quad3d'
FILES = [f'{LANE}/{f}' for f in ('c3common.py', 'fs3.py', 'fs3p.py', 'qn3.py', 't2core.py', 'timing.py', 's3common.py',
                                 's3ctx.py', 's3run.py', 's3modes.py', 'replay.py', 'bigtables.py', 'fomcore.py', 'fomrun.py',
                                 'pscore.py', 'psjax.py', 'test_s3.py', 'test_fom.py', 'e0loader/refstore.py',
                                 'e0loader/verdict.py', 'inputs/model_W1024/bank.pkl', 'rules3d/rules_extra.npz')] + \
    [f'{V3}/offmesh.py', f'{V3}/qpanel.py', f'{V3}/vendor/burgers3d-span/common.py', f'{V3}/vendor/burgers3d-retry/tables.py',
     f'{V3}/vendor/paper-b3d/vendor/b3d_common.py', f'{V3}/inputs/model_M2/bank.pkl', f'{V3}/rules/rules.npz']
GRES = {'a100-80G': ('gpu:a100:1', '--constraint=a100-80G'), 'h100': ('gpu:h100:1', None), 'h200': ('gpu:h200:1', None)}


def git_show(commit, name):
    return subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{name}'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('--gpu', default='h200', choices=['h200'])            # DESIGN §9: every lane timing on H200
    p.add_argument('--mem', default='200G')
    p.add_argument('--hours', type=int, default=24)
    p.add_argument('--cmd', default=None, help='alternative command (smoke tests): run inside the lane directory')
    a = p.parse_args()
    assert a.attempt.isalnum(), a.attempt
    out = ROOT / LANE / 'runs' / a.attempt
    assert not out.exists(), f'{out} exists'
    remote = f'{NAMESPACE}/{a.attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain', '--', LANE], text=True)
    assert not [l for l in dirty.splitlines() if not l[3:].startswith(f'{LANE}/runs')], dirty
    cfgname = f'{LANE}/configs/{a.attempt}.json'
    cfg0 = json.loads(git_show(commit, cfgname))
    assert cfg0['attempt'] == a.attempt
    if not cfg0.get('smoke'):
        res = json.loads(git_show(commit, f'{LANE}/checks/resources-{a.attempt}.json'))
        assert res['config_sha256'] == hashlib.sha256(git_show(commit, cfgname)).hexdigest(), 'resource certificate is for another config'
        assert res['gpu'] == a.gpu and res['hours_allocated'] == a.hours and res['projected_hours_x1.5'] <= a.hours, res
    files = list(FILES)
    if cfg0['mode'] == 'rep':                                        # A1.11: the immutable freeze is staged and pinned
        fz = f'{LANE}/checks/FROZEN-SELECTION.json'
        assert hashlib.sha256(git_show(commit, fz)).hexdigest() == cfg0['frozen_sha256'], 'freeze pin mismatch'
        files.append(fz)
    for name in files + [cfgname]:
        assert (ROOT / name).read_bytes() == git_show(commit, name), name
    out.mkdir(parents=True)
    proof = []
    for name in files + [cfgname]:
        content = git_show(commit, name)
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        proof.append(dict(source=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), commit=commit))
    (out / 'refs').mkdir()
    assert hashlib.sha256(REF1.read_bytes()).hexdigest() == REF1_SHA
    shutil.copy2(REF1, out / 'refs' / REF1.name)
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=2) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    (out / 'logs').mkdir()
    gres, constraint = GRES[a.gpu]
    cmd = a.cmd or f'"$PY" s3run.py --config configs/{a.attempt}.json --out "$TASK_ROOT/output"'
    script = f'''#!/bin/bash
#SBATCH --job-name=sp_{a.attempt}
#SBATCH --partition=gpu
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
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export JCPREF_STORE={E0STORE} JCPREF_META={E0STORE}
export REF1_PATH="$TASK_ROOT/refs/{REF1.name}"
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
{cmd}
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
