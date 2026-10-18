"""Stage one jcp-hr-baselines attempt into runs/<attempt>/ (pattern of jcp-combined cluster/stage.py). Code files are
taken from the Git blob at HEAD (the checkout must match byte for byte); an isolated import check proves the driver
imports only staged files. The E0 store is read on the cluster through JCPREF_STORE (not copied).

    python experiments/jcp-hr-baselines/cluster/stage.py <attempt> --driver smoke --config configs/smoke.json
        [--gpu h200|a100-80G|h100|a100] [--mem 200G] [--minutes 60]
"""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LANE = 'experiments/jcp-hr-baselines'
NAMESPACE = '/cluster/tufts/paralab/tawal01/jcp-hrb'
E0_STORE = '/cluster/tufts/paralab/tawal01/jcp-ref/store'
VQ, V3 = f'{LANE}/vendor/quad2d', f'{LANE}/vendor/quad3d'
FILES = [f'{LANE}/{f}' for f in ('hrcore.py', 'rom.py', 'build.py', 'nnls.py', 'pipeline.py', 'score.py', 'stats.py',
                                  'smoke.py', 'ours.py', 't2core.py', 's3common.py', 'qn3.py', 'fs3p.py', 'c2common.py', 'c3common.py', 'fs3.py',
                                  'e0/refcore.py', 'e0/refstore.py', 'e0/verdict.py', 'e0/cohorts.py',
                                  'e0/store/MANIFEST.sha256', 'e0/store/STATUS.json', 'inputs/model_W1024/bank.pkl',
                                  'rules3d/rules_extra.npz', 'rules3d/rules_gl96.npz', 'rules3d/rules_gl112.npz',
                                  'smoke_accept.py', 'tests/test_core.py', 'checks/test_core.json', 'tune.py', 'stageb.py',
                                  'p1core.py', 'p1common.py', 'inputs/p1_analysis.json', 'inputs/p1_EXPORT.json',
                                  'inputs/speed3d_EXPORT.json', 'inputs/speed3d_batch1_e0.json')] + \
    [f'{VQ}/qcore.py', f'{VQ}/qstudy.py', f'{VQ}/vendor/arms.py', f'{VQ}/vendor/hops.py', f'{VQ}/vendor/hfast.py',
     f'{VQ}/vendor/bkfast.py', f'{VQ}/vendor/hari_quadrature.py', f'{VQ}/inputs/rotation_R512.npz',
     f'{VQ}/inputs/rule_q0_m1024_qrg304_reachable.npz', f'{V3}/offmesh.py', f'{V3}/inputs/model_M2/bank.pkl', f'{V3}/qpanel.py', f'{V3}/rules/rules.npz',
     f'{V3}/vendor/burgers3d-span/common.py', f'{V3}/vendor/burgers3d-retry/tables.py',
     f'{V3}/vendor/paper-b3d/vendor/b3d_common.py', 'experiments/mr-burgers2d/engines.py',
     'experiments/mr-burgers2d/iterative_paths.py',
     'experiments/separable-decoder/sep_common.py',
     'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl']
GRES = {'a100-80G': ('gpu:a100:1', '--constraint=a100-80G'), 'a100': ('gpu:a100:1', None),
        'h100': ('gpu:h100:1', None), 'h200': ('gpu:h200:1', None)}


def git_show(commit, name):
    return subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{name}'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('--driver', required=True, choices=['smoke', 'tune', 'stageb'])
    p.add_argument('--config', required=True)
    p.add_argument('--gpu', default='h200', choices=sorted(GRES))
    p.add_argument('--mem', default='200G')
    p.add_argument('--minutes', type=int, default=60)
    a = p.parse_args()
    assert a.attempt.isalnum(), a.attempt
    out = ROOT / LANE / 'runs' / a.attempt
    assert not out.exists(), f'{out} exists'
    remote = f'{NAMESPACE}/{a.attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    cfgname = f'{LANE}/{a.config}'
    files = FILES + [cfgname] + [f'{LANE}/{x}' for x in json.loads(git_show(commit, cfgname)).get('stage_files', [])]
    for name in files:
        assert (ROOT / name).read_bytes() == git_show(commit, name), f'{name} differs from HEAD'
    if a.driver in ('tune', 'stageb'):                 # A26.4 / A32.5: the wall limit is the GPU-hour bound
        c0 = json.loads(git_show(commit, cfgname))
        assert a.minutes == int(round(60 * c0['wall_hours'])), ('--minutes must equal the config wall_hours', a.minutes)
        reserve = 7.5 if a.driver == 'tune' else 0.0
        RES_ARGS = [a.attempt, a.driver, a.config, a.gpu, commit, str(c0['wall_hours']), str(reserve)]
    out.mkdir(parents=True)
    proof = []
    for name in files:
        content = git_show(commit, name)
        if name == cfgname:
            cfg = json.loads(content)
            cfg['commit'] = commit
            content = (json.dumps(cfg, indent=1) + '\n').encode()
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        proof.append(dict(source=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(), commit=commit))
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=2) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    (out / 'logs').mkdir()
    gres, constraint = GRES[a.gpu]
    hh, mm = divmod(a.minutes, 60)
    script = f'''#!/bin/bash
#SBATCH --job-name=hrb_{a.attempt}
#SBATCH --partition=gpu
#SBATCH --gres={gres}
{('#SBATCH ' + constraint) if constraint else ''}
#SBATCH --cpus-per-task=8
#SBATCH --mem={a.mem}
#SBATCH --time={hh:02d}:{mm:02d}:00
#SBATCH --output={remote}/logs/%j.out
#SBATCH --error={remote}/logs/%j.err
set -euo pipefail
TASK_ROOT={remote}
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
source /cluster/tufts/paralab/tawal01/ae-research/venv/bin/activate
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.92
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
unset JCPREF_META
export JCPREF_STORE={E0_STORE}
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp" CUDA_CACHE_PATH="$TASK_ROOT/cache/cuda"
export JAX_COMPILATION_CACHE_DIR="$TASK_ROOT/cache/jax" PYTHONDONTWRITEBYTECODE=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$CUDA_CACHE_PATH" "$JAX_COMPILATION_CACHE_DIR"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
echo "host=$(hostname) source_commit=$(cat COMMIT.txt)"
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}',flush=True); sys.exit(0 if b=='gpu' else 42)"
cd {LANE}
# hard driver deadline: {a.minutes} min wall minus 4 min reserved for checksumming the (per-phase checkpointed) evidence
set +e
timeout --signal=TERM --kill-after=30 $(( {a.minutes} * 60 - 240 )) "$PY" {a.driver}.py {a.config} "$TASK_ROOT/output"
echo "driver_exit=$?"
set -e
cd "$TASK_ROOT"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE   # job completion only; acceptance is smoke_accept.py's ALL (offline)
'''
    (out / 'run.sbatch').write_text(script)
    env = dict(os.environ, JAX_PLATFORMS='cpu', JAX_DEFAULT_MATMUL_PRECISION='highest', PYTHONPATH='',
               PYTHONDONTWRITEBYTECODE='1')
    chk = (f'import sys, {a.driver}, ours, c2common, c3common, p1common, s3common; staged = {str(out)!r}; '
           'mods = [m.__file__ for m in list(sys.modules.values()) if getattr(m, "__file__", None) and "/experiments/" in m.__file__]; '
           'bad = [f for f in mods if not f.startswith(staged)]; assert not bad, bad; print(len(mods), "lane modules from the staged tree")')
    r_ = subprocess.run(['/home/tahmid/Dev/.venv/bin/python', '-c', chk], cwd=out / LANE, env=env, capture_output=True, text=True)
    assert r_.returncode == 0, r_.stderr[-3000:]
    print(r_.stdout.strip())
    if a.driver == 'stageb':                       # A33.3 preflight: remote tuning dependencies + resource admission
        c0 = json.loads(git_show(commit, cfgname))
        fz = json.loads(git_show(commit, f"{LANE}/{c0['frozen']}"))
        assert hashlib.sha256(git_show(commit, f"{LANE}/{c0['frozen']}")).hexdigest() == c0['frozen_sha256']
        need = dict(fz['artifacts'], **{'tune.json': fz['tune_json_sha256']})
        lst = ' '.join(need)
        r3 = subprocess.run(['ssh', '-o', 'ConnectTimeout=30', 'tufts-login', f"cd {c0['tune_dir']} && sha256sum {lst}"],
                            capture_output=True, text=True, timeout=600)
        assert r3.returncode == 0, r3.stderr[-2000:]
        got = {ln.split()[1]: ln.split()[0] for ln in r3.stdout.splitlines()}
        bad = [k for k, h in need.items() if got.get(k) != h]
        assert not bad, ('remote tuning dependencies differ from the frozen hashes', bad[:5])
        proj = fz['verify']['projection']['hours_certified']
        usable = c0['wall_hours'] - 4 / 60                 # driver deadline = wall - 4 min (OUTPUTS checksums)
        assert proj <= usable and fz['verify']['projection']['memory_ok'], ('Stage-B admission', proj, usable)
        print(f'stageb preflight: {len(need)} remote dependencies verified; projection {proj:.2f} h <= {c0["wall_hours"]} h')
    if a.driver == 'tune':                         # every lazily loaded asset the configuration needs (code audit 19)
        chk2 = (f'import json, ours; cfg = json.load(open({str(out / cfgname)!r})); print(ours.check_assets(cfg))')
        r2 = subprocess.run(['/home/tahmid/Dev/.venv/bin/python', '-c', chk2], cwd=out / LANE, env=env,
                            capture_output=True, text=True)
        assert r2.returncode == 0, r2.stderr[-3000:]
        print(r2.stdout.strip()[-2000:])
    import shutil
    for pc in list(out.rglob('__pycache__')):                    # job audit 1: no unmanifested bytecode in a bundle
        shutil.rmtree(pc)
    manifest = [f'{hashlib.sha256(p_.read_bytes()).hexdigest()}  {p_.relative_to(out)}'
                for p_ in sorted(out.rglob('*')) if p_.is_file() and '__pycache__' not in str(p_)]
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    if a.driver in ('tune', 'stageb'):            # A34.2: atomic reservation under the ledger lock (fails if over cap)
        r4 = subprocess.run(['/home/tahmid/Dev/.venv/bin/python', str(ROOT / LANE / 'cluster' / 'ledger.py'), 'reserve']
                            + RES_ARGS, capture_output=True, text=True)
        assert r4.returncode == 0, r4.stderr[-2000:]
        print(r4.stdout.strip())
    print(out)
    print(remote)


if __name__ == '__main__':
    main()
