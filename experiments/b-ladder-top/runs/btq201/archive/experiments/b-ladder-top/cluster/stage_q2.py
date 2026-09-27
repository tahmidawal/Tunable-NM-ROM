"""Stage Q2 (the combined envelope) into an isolated paralab-bound attempt.

Two kinds of input are staged and they are provenance-tracked differently:

* Git-tracked files of THIS worktree, asserted byte-identical to the commit being staged;
* out-of-band files from the `2026-09-14-no-audit` lane — the trained FNO checkpoint and
  the three modules that define its architecture, normalisation and data contract. The
  checkpoint lives under a `.gitignore`d `runs/` tree there, so it cannot be staged from
  a Git object; it is copied directly, its SHA256 recorded, and the source worktree's
  HEAD recorded beside it.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/b_ladder_top_20260916'
NOAUDIT = ROOT.parent / '2026-09-14-no-audit/experiments/neural-operator-audit'
FILES = [
    'experiments/b-ladder-top/q2_envelope.py',
    'experiments/b-ladder-top/fno_panel.py',
    'experiments/b-ladder-top/topfix.py',
    'experiments/b-ladder-top/config-q2.json',
    'experiments/b-ladder-top/cluster/stage_q2.py',
    'experiments/cheap-corrections/varpro.py',
    'experiments/cheap-corrections/directions.py',
    'experiments/head-ablation/arms.py',
    'experiments/head-ablation/ladder.py',
    'experiments/head-ablation/ablation.py',
    'experiments/mr-burgers2d/engines.py',
    'experiments/mr-burgers2d/iterative_paths.py',
    'experiments/mr-burgers2d/accuracy_paths.py',
    'experiments/separable-decoder/sep_common.py',
    'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl',
]
FNO_CODE = ['model.py', 'dataset.py', 'spectral_conv_f64.py']
FNO_MODEL = 'fno-large'
FNO_CKPT = f'runs/fno_burgers02/fno_burgers02/out/{FNO_MODEL}/best.pt'
FNO_TRAIN_INDEX = '/cluster/tufts/paralab/tawal01/no_burgers_20260914/pilot-data01/train/index.json'
CHECKPOINT = 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
HOURS = '20:00:00'
EXCLUDE = 'pax007'


def main():
    attempt = sys.argv[1]
    assert attempt.isalnum(), attempt
    out = ROOT / 'experiments/b-ladder-top/runs' / attempt
    out.mkdir(parents=True, exist_ok=False)
    remote = f'{NAMESPACE}/{attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    proof = []
    for name in FILES:
        content = (ROOT / name).read_bytes()
        assert content == subprocess.check_output(['git', '-C', str(ROOT), 'show', f'{commit}:{name}']), name
        dest = out / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        proof.append(dict(source=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(),
                          commit=commit, provenance='git object of this worktree'))

    have_fno = NOAUDIT.exists() and (NOAUDIT / FNO_CKPT).exists()
    if have_fno:
        other = subprocess.check_output(['git', '-C', str(NOAUDIT), 'rev-parse', 'HEAD'], text=True).strip()
        (out / 'fnocode').mkdir()
        for name in FNO_CODE:
            content = (NOAUDIT / name).read_bytes()
            (out / 'fnocode' / name).write_bytes(content)
            proof.append(dict(source=f'2026-09-14-no-audit/experiments/neural-operator-audit/{name}',
                              bytes=len(content), sha256=hashlib.sha256(content).hexdigest(),
                              commit=other, provenance='out-of-band copy from the no-audit worktree'))
        (out / 'fnockpt').mkdir()
        blob = (NOAUDIT / FNO_CKPT).read_bytes()
        (out / 'fnockpt' / 'best.pt').write_bytes(blob)
        proof.append(dict(source=f'2026-09-14-no-audit/.../{FNO_CKPT}', bytes=len(blob),
                          sha256=hashlib.sha256(blob).hexdigest(), commit=other,
                          provenance=('out-of-band copy; the checkpoint is under a .gitignore\'d '
                                      'runs/ tree in that worktree and has no Git object')))
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=2) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    (out / 'logs').mkdir()
    fno_block = '''
echo "FNO PHASE"
export PYTHONPATH="$TASK_ROOT/fnocode:$TASK_ROOT/experiments/b-ladder-top"
set +e
"$PY" experiments/b-ladder-top/fno_panel.py \\
  --checkpoint "$TASK_ROOT/fnockpt/best.pt" \\
  --index output/fno-cohort/index.json \\
  --out output --name MODEL --repetitions 3 --burn-in 3 > logs/fno.log 2>&1
echo "fno_exit=$?"
set -e
tail -20 logs/fno.log || true
''' if have_fno else '\necho "FNO PHASE SKIPPED: no checkpoint staged"\n'
    script = '''#!/bin/bash
#SBATCH --job-name=btq2_ATTEMPT
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100:1
#SBATCH --exclude=EXCLUDE
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=HOURS
#SBATCH --output=REMOTE/logs/%j.out
#SBATCH --error=REMOTE/logs/%j.err
set -euo pipefail
TASK_ROOT=REMOTE
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
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
export PYTHONPATH="$TASK_ROOT/experiments/mr-burgers2d:$TASK_ROOT/experiments/head-ablation:$TASK_ROOT/experiments/cheap-corrections:$TASK_ROOT/experiments/b-ladder-top"
"$PY" experiments/b-ladder-top/q2_envelope.py \\
  --config experiments/b-ladder-top/config-q2.json \\
  --checkpoint CKPT \\
  --fno-train-index TRAININDEX \\
  --out output
FNOBLOCK
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''
    script = (script.replace('ATTEMPT', attempt).replace('REMOTE', remote)
              .replace('CKPT', CHECKPOINT).replace('TRAININDEX', FNO_TRAIN_INDEX)
              .replace('FNOBLOCK', fno_block).replace('MODEL', FNO_MODEL)
              .replace('HOURS', HOURS).replace('EXCLUDE', EXCLUDE))
    (out / 'run.sbatch').write_text(script)
    manifest = [f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}'
                for p in sorted(out.rglob('*')) if p.is_file()]
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote)
    print('fno_staged', have_fno)


if __name__ == '__main__':
    main()
