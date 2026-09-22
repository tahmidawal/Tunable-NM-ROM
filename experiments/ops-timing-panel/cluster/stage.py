"""Stage one ops-timing-panel attempt into an isolated paralab-bound directory.

    python cluster/stage.py <attempt> <config-file-name> [--gpu a100-80G|a100|h100|h200|l40s]
                            [--mem 180G] [--hours 6]

b-panel's `cluster/stage.py` (exp/2026-09-17-b-panel @ 25434a27) with three changes, all
recorded in DESIGN.md:

1. the lane's own directory and namespace, and the imported JAX/torch modules flattened into
   `lib/` instead of six sibling experiment directories on PYTHONPATH;
2. the FNO phase becomes an OPERATOR phase over every checkpoint in `operators.json` -- the
   FNO plus the archived U-Net and Transolver capacities of the no-second lane -- each timed
   by the unmodified `fno_panel.py` as its own process inside this one allocation;
3. every operator checkpoint is re-hashed here against the SHA256 the job that produced it
   recorded, and staging aborts on a mismatch.

One attempt directory per job, never two. Every Git-tracked lane file is checked byte for byte
against the blob at HEAD before it is copied. The checkpoints live under `.gitignore`d `runs/`
trees in other worktrees and have no Git object, so their SHA256 and the producing worktree's
HEAD go into `PROVENANCE.json` instead.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

LANE = 'experiments/ops-timing-panel'
ROOT = Path(__file__).resolve().parents[3]
WORKTREES = ROOT.parent
NAMESPACE = '/cluster/tufts/paralab/tawal01/opstime_20260922'
FILES = [
    f'{LANE}/panel.py',
    f'{LANE}/cluster/stage.py',
    f'{LANE}/operators.json',
    f'{LANE}/COPIED-FROM.json',
    f'{LANE}/inputs/PROVENANCE.json',
    f'{LANE}/inputs/directions_qtd02.npz',
    'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl',
]
LIB = ['engines.py', 'iterative_paths.py', 'accuracy_paths.py', 'arms.py', 'ladder.py', 'ablation.py',
       'varpro.py', 'topfix.py', 'eqcert.py', 'fast.py', 'ladders.py', 'sep_common.py',
       'fno_panel.py', 'model.py', 'families.py', 'dataset.py', 'spectral_conv_f64.py']
TRAIN_INDEX = '/cluster/tufts/paralab/tawal01/no_burgers_20260914/pilot-data01/train/index.json'
CHECKPOINT = 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
EXCLUDE = 'pax007'
GRES = {'a100-80G': ('gpu:a100:1', '--constraint=a100-80G'), 'a100': ('gpu:a100:1', None),
        'h100': ('gpu:h100:1', None), 'h200': ('gpu:h200:1', None), 'l40s': ('gpu:l40s:1', None)}
PYPATH = [f'{LANE}/lib', LANE]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('config')
    p.add_argument('--gpu', default='a100-80G', choices=sorted(GRES))
    p.add_argument('--mem', default='180G')
    p.add_argument('--hours', type=int, default=6)
    p.add_argument('--mem-fraction', default='0.90')
    p.add_argument('--reps', type=int, default=5)
    p.add_argument('--burn-in', type=int, default=5)
    a = p.parse_args()
    attempt, config = a.attempt, a.config
    assert attempt.isalnum(), attempt
    cfg = json.loads((ROOT / LANE / config).read_text())
    files = FILES + [f'{LANE}/{config}'] + [f'{LANE}/lib/{m}' for m in LIB]
    files += [f"{LANE}/inputs/rules/{v['file']}" for v in cfg['rules'].values()]
    for es in cfg.get('extra_rule_sets', []):
        files += [f"{LANE}/inputs/{es.get('subdir', 'rules')}/{v['file']}" for v in es['rules'].values()]
    files = list(dict.fromkeys(files))
    out = ROOT / LANE / 'runs' / attempt
    out.mkdir(parents=True, exist_ok=False)
    remote = f'{NAMESPACE}/{attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain', LANE], text=True)
    assert not [l for l in dirty.splitlines() if not l.endswith('runs/')], f'uncommitted lane files:\n{dirty}'
    proof = []
    for name in files:
        content = (ROOT / name).read_bytes()
        assert content == subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{name}']), name
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        proof.append(dict(source=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(),
                          commit=commit, provenance='git object of this worktree'))

    # ---- operator checkpoints: hash-verified against the job that produced each one ----
    ops = json.loads((ROOT / LANE / 'operators.json').read_text())['checkpoints']
    (out / 'opsckpt').mkdir()
    staged = []
    for spec in ops:
        rel = spec['path'].removeprefix('worktrees/')
        src = WORKTREES / rel
        assert src.exists(), src
        blob = src.read_bytes()
        got = hashlib.sha256(blob).hexdigest()
        assert got == spec['sha256'], (spec['name'], got, spec['sha256'])
        (out / 'opsckpt' / f"{spec['name']}.pt").write_bytes(blob)
        other = subprocess.check_output(
            ['git', '-C', str(WORKTREES / rel.split('/experiments/')[0]), 'rev-parse', 'HEAD'],
            text=True).strip()
        proof.append(dict(source=spec['path'], staged_as=f"opsckpt/{spec['name']}.pt", bytes=len(blob),
                          sha256=got, expected_sha256=spec['sha256'], sha256_verified=True, commit=other,
                          source_job=spec['source_job'], role=spec['role'], family=spec['family'],
                          provenance="out-of-band copy; the checkpoint is under a .gitignore'd runs/ tree "
                                     "and has no Git object"))
        staged.append(spec['name'])

    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=2) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    (out / 'logs').mkdir()
    gres, constraint = GRES[a.gpu]
    ops_block = '\necho "OPERATOR PHASE"\n' + ''.join(
        f'''set +e
"$PY" {LANE}/lib/fno_panel.py \\
  --checkpoint "$TASK_ROOT/opsckpt/{name}.pt" \\
  --index output/fno-cohort/index.json \\
  --out output --name {name} --repetitions {a.reps} --burn-in {a.burn_in} > logs/op-{name}.log 2>&1
echo "op_exit {name}=$?"
set -e
tail -3 logs/op-{name}.log || true
''' for name in staged)

    script = '''#!/bin/bash
#SBATCH --job-name=opt___ATTEMPT__
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=__GRES__
__CONSTRAINT__
#SBATCH --exclude=__EXCLUDE__
#SBATCH --cpus-per-task=8
#SBATCH --mem=__MEM__
#SBATCH --time=__HOURS__:00:00
#SBATCH --output=__REMOTE__/logs/%j.out
#SBATCH --error=__REMOTE__/logs/%j.err
set -euo pipefail
TASK_ROOT=__REMOTE__
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=__MEMFRAC__
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
export TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT"
nvidia-smi --query-gpu=name,memory.total,uuid --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
export PYTHONPATH="__PYPATH__"
"$PY" __LANE__/panel.py \\
  --config __LANE__/__CONFIG__ \\
  --checkpoint __CKPT__ \\
  --inputs __LANE__/inputs \\
  --fno-train-index __TRAININDEX__ \\
  --out output
__OPSBLOCK__
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''
    for token, value in (('__ATTEMPT__', attempt), ('__REMOTE__', remote), ('__CKPT__', CHECKPOINT),
                         ('__LANE__', LANE), ('__CONFIG__', config), ('__HOURS__', f'{a.hours:02d}'),
                         ('__EXCLUDE__', EXCLUDE), ('__GRES__', gres),
                         ('__CONSTRAINT__', f'#SBATCH {constraint}' if constraint else ''),
                         ('__MEM__', a.mem), ('__MEMFRAC__', a.mem_fraction),
                         ('__PYPATH__', ':'.join(f'$TASK_ROOT/{x}' for x in PYPATH)),
                         ('__TRAININDEX__', TRAIN_INDEX), ('__OPSBLOCK__', ops_block)):
        script = script.replace(token, value)
    assert '__' not in script.replace('__pycache__', ''), script
    (out / 'run.sbatch').write_text(script)
    manifest = [f'{hashlib.sha256(p_.read_bytes()).hexdigest()}  {p_.relative_to(out)}'
                for p_ in sorted(out.rglob('*')) if p_.is_file()]
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote)
    print('gpu', a.gpu, 'mem', a.mem, 'operators', len(staged), ' '.join(staged))


if __name__ == '__main__':
    main()
