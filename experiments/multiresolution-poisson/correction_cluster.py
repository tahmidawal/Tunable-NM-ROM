"""Immutable staging and checked collection for the final correction family."""
from cluster import *


def stage(label):
    commit=run(['git','rev-parse','HEAD'],cwd=TREE).strip();dest=CELL/'stages'/label;dest.mkdir(parents=True,exist_ok=False)
    for name in ['code','in','out','logs']:(dest/name).mkdir()
    names=['core.py','speed_core.py','kernel_solver.py','pilot.py','iterative_core.py','tuning_core.py','correction_core.py','correction_accuracy.py','test_corrections.py','config-correction-accuracy.json']
    files=[CELL/name for name in names]+[TREE/'experiments/separable-decoder/sep_common.py',TREE/'experiments/cost-to-tolerance/ctol_tol.py',TREE/'experiments/wave2d-rom-latent-stepping/deps/multistage-precision/ms_parametric.py'];hashes={}
    for source in files:
        payload=subprocess.check_output(['git','show',f'{commit}:{source.relative_to(TREE)}'],cwd=TREE);assert payload==source.read_bytes();target=dest/'code'/source.name;target.write_bytes(payload);hashes[str(target.relative_to(dest))]=digest(target)
    cfg=json.loads((dest/'code/config-correction-accuracy.json').read_text());inputs={}
    for key,target_name in [('checkpoint','head.pkl'),('original_checkpoint','original.pkl'),('basis','basis.npz')]:
        source=cfg[key];payload=subprocess.check_output(['git','show',f'{commit}:{source}'],cwd=TREE);assert hashlib.sha256(payload).hexdigest()==cfg[key+'_sha256'];(dest/'in'/target_name).write_bytes(payload);inputs['in/'+target_name]=dict(source_path=source,sha256=cfg[key+'_sha256'])
    write_json(dest/'in/ORIGIN.json',dict(source_commit=commit,inputs=inputs));remote=NAMESPACE+'/'+label
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
"$PY" -u code/test_corrections.py --mode algebra
"$PY" -u code/test_corrections.py --mode solve
"$PY" -u code/correction_accuracy.py --config code/config-correction-accuracy.json --checkpoint in/head.pkl --original in/original.pkl --basis in/basis.npz --out out/smoke --smoke
"$PY" -u code/correction_accuracy.py --config code/config-correction-accuracy.json --checkpoint in/head.pkl --original in/original.pkl --basis in/basis.npz --out out/pilot
'''
    (dest/'job.sbatch').write_text(batch);info=dict(label=label,source_commit=commit,source_hashes=hashes,inputs=inputs,remote=remote,staged_at=datetime.now(timezone.utc).isoformat());write_json(dest/'CONFIG.json',info)
    (dest/'MANIFEST.sha256').write_text(''.join(f'{digest(p)}  {p.relative_to(dest)}\n' for p in sorted(dest.rglob('*')) if p.is_file()))
    record=CELL/'runs'/label;record.mkdir(parents=True,exist_ok=False);write_json(record/'submission.json',info);print(json.dumps(info,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage','submit','collect']);parser.add_argument('label');a=parser.parse_args();assert re.fullmatch(r'correction_accuracy[0-9a-z_]+',a.label);globals()[a.action](a.label)
