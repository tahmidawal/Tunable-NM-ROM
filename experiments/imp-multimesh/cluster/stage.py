"""Stage one imp-multimesh attempt into runs/<attempt>/ (pattern of jcp-p1-scaling / jcp-hr-baselines cluster/stage.py).
Every file comes from the Git blob at HEAD (the working copy must match it). The E0 store is read on the cluster through
JCPREF_STORE. The job runs the config's "commands" in order (each under a timeout), from the lane directory.

    python experiments/imp-multimesh/cluster/stage.py <attempt> --config configs/<x>.json [--gpu h200] [--mem 240G]
        [--minutes 60]
"""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LANE = 'experiments/imp-multimesh'
NAMESPACE = '/cluster/tufts/paralab/tawal01/imp-multimesh'
E0_STORE = '/cluster/tufts/paralab/tawal01/jcp-ref/store'
HRB = 'experiments/jcp-hr-baselines'
P1 = 'experiments/jcp-p1-scaling'
V3 = f'{HRB}/vendor/quad3d'
HRB_FILES = [f'{HRB}/{f}' for f in ('hrcore.py', 'rom.py', 'build.py', 'nnls.py', 'pipeline.py', 'score.py', 't2core.py',
                                     'e0/refcore.py', 'e0/refstore.py', 'e0/verdict.py', 'e0/cohorts.py',
                                     'e0/store/MANIFEST.sha256', 'e0/store/STATUS.json')] + \
    [f'{V3}/vendor/burgers3d-span/common.py', f'{V3}/vendor/burgers3d-retry/tables.py',
     f'{V3}/vendor/paper-b3d/vendor/b3d_common.py']
EXTRA = ['experiments/mr-burgers2d/engines.py', 'experiments/mr-burgers2d/iterative_paths.py',
         'experiments/separable-decoder/sep_common.py',
         'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl']
SKIP = ('runs/', 'audits/', 'checks/', 'plots/', 'report', 'HEARTBEAT.json', 'probe/')
GRES = {'a100-80G': ('gpu:a100:1', '--constraint=a100-80G'), 'h100': ('gpu:h100:1', None), 'h200': ('gpu:h200:1', None)}


def tree_files(commit, d, skip=SKIP):
    out = subprocess.check_output(['git', '-C', str(ROOT), 'ls-tree', '-r', '--name-only', commit, d + '/'], text=True)
    return [f for f in out.split() if not any(f[len(d) + 1:].startswith(s) for s in skip)]


def git_show(commit, name):
    return subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{name}'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('--config', required=True)
    p.add_argument('--gpu', default='h200', choices=sorted(GRES))
    p.add_argument('--mem', default='240G')
    p.add_argument('--minutes', type=int, default=60)
    a = p.parse_args()
    assert a.attempt.isalnum(), a.attempt
    out = ROOT / LANE / 'runs' / a.attempt
    assert not out.exists(), f'{out} exists'
    remote = f'{NAMESPACE}/{a.attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    cfgname = f'{LANE}/{a.config}'
    cfg = json.loads(git_show(commit, cfgname))
    files = sorted(set(tree_files(commit, LANE) + HRB_FILES + tree_files(commit, P1, SKIP + ('vendor/quad3d/',)) + EXTRA
                       + cfg.get('extra_files', [])))
    assert cfgname in files
    proof = []
    out.mkdir(parents=True)
    for name in files:
        content = git_show(commit, name)
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
    hh, mm = divmod(a.minutes, 60)
    budget_s = a.minutes * 60 - 300
    tot = sum(int(c.get('timeout_s', budget_s)) for c in cfg['commands'])
    assert tot <= budget_s, f'command timeouts {tot}s exceed the allocation {budget_s}s (--minutes {a.minutes})'
    cmds = []
    for i, c in enumerate(cfg['commands']):
        t = int(c.get('timeout_s', budget_s))
        (out / 'cmds').mkdir(exist_ok=True)
        (out / 'cmds' / f'{i:02d}_{c["name"]}.sh').write_text('set -uo pipefail\n' + c['cmd'] + '\n')
        gate = '' if c.get('always_run') else 'if [ $FAIL -ne 0 ]; then echo "skip ' + c['name'] + ' (earlier failure)"; else\n'
        cmds.append(f'echo "=== $(date -u +%FT%TZ) {c["name"]}"\n' + gate +
                    f'timeout --signal=TERM --kill-after=30 {t} bash "$TASK_ROOT/cmds/{i:02d}_{c["name"]}.sh"; rc=$?\n'
                    f'echo "step_exit[{c["name"]}]=$rc"\n'
                    + ('' if c.get('continue_on_fail') else '[ $rc -eq 0 ] || FAIL=1\n')
                    + ('' if c.get('always_run') else 'fi\n'))
    body = '\n'.join(cmds)
    script = f'''#!/bin/bash
#SBATCH --job-name=imm_{a.attempt}
#SBATCH --partition=gpu
#SBATCH --gres={gres}
{('#SBATCH ' + constraint) if constraint else ''}
#SBATCH --cpus-per-task=8
#SBATCH --mem={a.mem}
#SBATCH --time={hh:02d}:{mm:02d}:00
#SBATCH --output={remote}/logs/%j.out
#SBATCH --error={remote}/logs/%j.err
set -uo pipefail
export TASK_ROOT={remote}
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 JAX_ENABLE_COMPILATION_CACHE=false
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
unset JCPREF_META
export JCPREF_STORE={E0_STORE}
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp" CUDA_CACHE_PATH="$TASK_ROOT/cache/cuda"
export PYTHONDONTWRITEBYTECODE=1 SOURCE_COMMIT=$(cat $TASK_ROOT/COMMIT.txt) OUT="$TASK_ROOT/output"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$OUT"
cd "$TASK_ROOT"
finalize() {{ cd "$TASK_ROOT"; mkdir -p output; echo "{{\"early_exit\": $1, \"reason\": \"$2\", \"all_passed\": false}}" > output/EARLY_EXIT.json
  find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256; echo "early_exit=$1 ($2)"; exit $1; }}
sha256sum -c MANIFEST.sha256 --quiet || finalize 3 manifest; echo "stage MANIFEST OK"
echo "host=$(hostname) source_commit=$(cat COMMIT.txt)"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}',flush=True); sys.exit(0 if b=='gpu' else 42)" || finalize 42 gpu-preflight
cd {LANE}
FAIL=0
{body}
cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256 || FAIL=1
echo "job_fail=$FAIL"
echo ALL-DONE
exit $FAIL
'''
    (out / 'run.sbatch').write_text(script)
    man = subprocess.check_output(['bash', '-c', 'find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | '
                                   'xargs -0 sha256sum'], cwd=out, text=True)
    (out / 'MANIFEST.sha256').write_text(man)
    (out / 'RESERVATION.json').write_text(json.dumps(dict(attempt=a.attempt, time_limit_hours=a.minutes / 60,
                                                          gpu=a.gpu, commit=commit, config=a.config)) + '\n')
    env = dict(os.environ, JAX_PLATFORMS='cpu', JAX_DEFAULT_MATMUL_PRECISION='highest', PYTHONPATH='',
               PYTHONDONTWRITEBYTECODE='1')
    for mod in cfg.get('import_check', ['mmcore']):
        stmt = mod if (';' in mod or mod.startswith('import ')) else f'import {mod}'
        chk = (f'import sys; sys.path.insert(0, "../jcp-p1-scaling"); {stmt}; staged = {str(out)!r}; '
               'mods = [m.__file__ for m in list(sys.modules.values()) if getattr(m, "__file__", None) and "/experiments/" in m.__file__]; '
               'bad = [f for f in mods if not f.startswith(staged)]; assert not bad, bad; print(len(mods), "modules staged")')
        r_ = subprocess.run(['/home/tahmid/Dev/.venv/bin/python', '-c', chk], cwd=out / LANE, env=env,
                            capture_output=True, text=True)
        assert r_.returncode == 0, (mod, r_.stderr[-3000:])
        print(mod, r_.stdout.strip())
    print('staged', out, len(files), 'files; commit', commit[:9])


if __name__ == '__main__':
    main()
