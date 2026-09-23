"""Stage one quadratic-manifold attempt into an isolated paralab-bound directory.

    python cluster/stage.py <attempt> <config-file-name> [--gpu a100-80G|a100|h100|h200|l40s]
                            [--mem 180G] [--hours 20]

One attempt directory per job, never two. Every Git-tracked file is checked byte for byte
against the blob at HEAD before it is copied. b-panel's `cluster/stage.py` (exp/2026-09-17-b-panel @ 25434a27) with the lane's own
directory and namespace, `qman.py` added to the staged files, and the FNO phase removed: this
lane compares a trial map against the NM-ROM, POD-LSPG and the full-order ladder, and the
neural-operator arms belong to the sibling ops-timing-panel lane.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
LANE = 'experiments/quadratic-manifold'
NAMESPACE = '/cluster/tufts/paralab/tawal01/qman_20260922'
FILES = [
    f'{LANE}/panel.py',
    f'{LANE}/qman.py',
    f'{LANE}/cluster/stage.py',
    f'{LANE}/speed/fast.py',
    f'{LANE}/speed/ladders.py',
    f'{LANE}/inputs/PROVENANCE.json',
    f'{LANE}/inputs/directions_qtd02.npz',
    'experiments/b-ladder-top/topfix.py',
    'experiments/q-ridge/eqcert.py',
    'experiments/cheap-corrections/varpro.py',
    'experiments/head-ablation/arms.py',
    'experiments/head-ablation/ladder.py',
    'experiments/head-ablation/ablation.py',
    'experiments/mr-burgers2d/engines.py',
    'experiments/mr-burgers2d/iterative_paths.py',
    'experiments/mr-burgers2d/accuracy_paths.py',
    'experiments/separable-decoder/sep_common.py',
    'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl',
]
CHECKPOINT = 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
EXCLUDE = 'pax007'
GRES = {'a100-80G': ('gpu:a100:1', '--constraint=a100-80G'), 'a100': ('gpu:a100:1', None),
        'h100': ('gpu:h100:1', None), 'h200': ('gpu:h200:1', None), 'l40s': ('gpu:l40s:1', None)}
PYPATH = ['experiments/mr-burgers2d', 'experiments/separable-decoder', 'experiments/head-ablation',
          'experiments/cheap-corrections', 'experiments/b-ladder-top', 'experiments/q-ridge',
          LANE, f'{LANE}/speed']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('attempt')
    p.add_argument('config')
    p.add_argument('--gpu', default='a100-80G', choices=sorted(GRES))
    p.add_argument('--mem', default='180G')
    p.add_argument('--hours', type=int, default=12)
    p.add_argument('--mem-fraction', default='0.90')
    a = p.parse_args()
    attempt, config = a.attempt, a.config
    assert attempt.isalnum(), attempt
    cfg = json.loads((ROOT / LANE / config).read_text())
    files = FILES + [f'{LANE}/{config}']
    files += [f"{LANE}/inputs/rules/{v['file']}" for v in cfg['rules'].values()]
    for es in cfg.get('extra_rule_sets', []):          # DESIGN A5.1 / A8: e.g. b-eqtop's set under rules-eqtop/
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
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=2) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    (out / 'logs').mkdir()
    gres, constraint = GRES[a.gpu]
    script = '''#!/bin/bash
#SBATCH --job-name=qmn___ATTEMPT__
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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
export PYTHONPATH="__PYPATH__"
"$PY" __LANE__/panel.py \\
  --config __LANE__/__CONFIG__ \\
  --checkpoint __CKPT__ \\
  --inputs __LANE__/inputs \\
  --out output
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''
    for token, value in (('__ATTEMPT__', attempt), ('__REMOTE__', remote), ('__CKPT__', CHECKPOINT),
                         ('__LANE__', LANE), ('__CONFIG__', config), ('__HOURS__', f'{a.hours:02d}'), ('__EXCLUDE__', EXCLUDE),
                         ('__GRES__', gres), ('__CONSTRAINT__', f'#SBATCH {constraint}' if constraint else ''),
                         ('__MEM__', a.mem), ('__MEMFRAC__', a.mem_fraction),                         ('__PYPATH__', ':'.join(f'$TASK_ROOT/{x}' for x in PYPATH)),
                         ):
        script = script.replace(token, value)
    assert '__' not in script.replace('__pycache__', ''), script
    (out / 'run.sbatch').write_text(script)
    manifest = [f'{hashlib.sha256(p_.read_bytes()).hexdigest()}  {p_.relative_to(out)}'
                for p_ in sorted(out.rglob('*')) if p_.is_file()]
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote)
    print('gpu', a.gpu, 'mem', a.mem)


if __name__ == '__main__':
    main()
