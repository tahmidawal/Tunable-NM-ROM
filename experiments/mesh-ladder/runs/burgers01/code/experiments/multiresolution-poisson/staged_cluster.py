"""Immutable private staging; reuse checksum collection and exact cleanup."""
from cluster import *


def stage(label):
    commit=run(['git','rev-parse','HEAD'],cwd=TREE).strip();dest=CELL/'stages'/label
    dest.mkdir(parents=True,exist_ok=False)
    for name in ['code','in','out','logs']:(dest/name).mkdir()
    names=['core.py','speed_core.py','kernel_solver.py','pilot.py','iterative_core.py','tuning_core.py','training_core.py','followup.py','staged_training.py','staged_oracles.py','staged_accuracy.py','test_staged_training.py','config-staged-accuracy.json']
    files=[CELL/x for x in names]+[TREE/'experiments/separable-decoder/sep_common.py',TREE/'experiments/cost-to-tolerance/ctol_tol.py',TREE/'experiments/wave2d-rom-latent-stepping/deps/multistage-precision/ms_parametric.py']
    hashes={}
    for source in files:
        payload=subprocess.check_output(['git','show',f'{commit}:{source.relative_to(TREE)}'],cwd=TREE)
        assert payload==source.read_bytes();target=dest/'code'/source.name;target.write_bytes(payload)
        hashes[str(target.relative_to(dest))]=digest(target)
    config=json.loads((dest/'code/config-staged-accuracy.json').read_text());ck=config['checkpoint']
    payload=subprocess.check_output(['git','show',f'{commit}:{ck}'],cwd=TREE);assert hashlib.sha256(payload).hexdigest()==config['checkpoint_sha256']
    (dest/'in/model.pkl').write_bytes(payload)
    write_json(dest/'in/ORIGIN.json',dict(checkpoint_path=ck,checkpoint_source_commit=commit,checkpoint_sha256=config['checkpoint_sha256']))
    remote=NAMESPACE+'/'+label
    batch=f'''#!/bin/bash
#SBATCH --job-name=ctol_mr_poisson_{label}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output={remote}/logs/%j.out
#SBATCH --error={remote}/logs/%j.err
set -euo pipefail
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_DEFAULT_MATMUL_PRECISION=highest
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OPENBLAS_NUM_THREADS=8
export OMP_NUM_THREADS=8
export COMMIT={commit}
cd {remote}
sha256sum -c MANIFEST.sha256 --quiet
finish() {{
 task_exit=$?
 printf '%s\\n' "$task_exit" > EXIT_CODE
 find out -type f -print0 | sort -z | xargs -0 -r sha256sum > RESULTS.sha256
 exit "$task_exit"
}}
trap finish EXIT
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}',flush=True); sys.exit(0 if b=='gpu' else 42)"
"$PY" -u code/test_staged_training.py
"$PY" -u code/staged_accuracy.py --config code/config-staged-accuracy.json --checkpoint in/model.pkl --out out/smoke --smoke
"$PY" -u code/staged_accuracy.py --config code/config-staged-accuracy.json --checkpoint in/model.pkl --out out/pilot
'''
    (dest/'job.sbatch').write_text(batch)
    info=dict(label=label,source_commit=commit,source_hashes=hashes,remote=remote,staged_at=datetime.now(timezone.utc).isoformat())
    write_json(dest/'CONFIG.json',info)
    (dest/'MANIFEST.sha256').write_text(''.join(f'{digest(p)}  {p.relative_to(dest)}\n' for p in sorted(dest.rglob('*')) if p.is_file()))
    record=CELL/'runs'/label;record.mkdir(parents=True,exist_ok=False);write_json(record/'submission.json',info)
    print(json.dumps(info,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage','submit','collect']);parser.add_argument('label');args=parser.parse_args()
    assert re.fullmatch(r'staged_accuracy[0-9a-z_]+',args.label)
    globals()[args.action](args.label)
