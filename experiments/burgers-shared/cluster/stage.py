"""Stage one pinned preparation job; submission and ledger changes remain coordinator actions."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
LANE='experiments/burgers-shared'
NS='/cluster/tufts/paralab/tawal01/bshared-20261028'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('attempt');a=ap.parse_args();assert a.attempt.isalnum()
    commit=subprocess.check_output(['git','-c','gc.auto=0','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    dest=ROOT/LANE/'runs'/a.attempt;dest.mkdir(parents=True,exist_ok=False);remote=f'{NS}/{a.attempt}'
    paths=[f'{LANE}/{x}' for x in ('train.py','regenerate.py','make_reference.py','DESIGN.md','checks/training-reference.json','checks/legacy-data-manifest.sha256','vendor/PROVENANCE.json','vendor/patch_bf_precond.py','cluster/stage.py')]
    paths+=['experiments/separable-decoder/sep_common.py','experiments/separable-decoder/sep_coeff_extract.py',
            'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl',
            'experiments/burgers2d-rom-latent-stepping/blat_common.py','experiments/nonlinear-decoder-architecture/nda_arch.py',
            'experiments/jcp-heads/sep_hfit.py','experiments/mr-burgers2d/engines.py']
    mapping={p:p for p in paths}
    for name in ('burgers2d-coord-rom/burgers2d_film.py','multistage-precision/ms_parametric.py','multistage-precision/ms_autodecoder.py'):
        mapping['experiments/burgers2d-rom-latent-stepping/deps/'+name]=(f'{LANE}/vendor/burgers2d_film.py'
            if name.endswith('burgers2d_film.py') else 'experiments/wave2d-rom-latent-stepping/deps/'+name)
    proof=[]
    for target,source in mapping.items():
        b=subprocess.check_output(['git','-c','gc.auto=0','-C',str(ROOT),'show',f'{commit}:{source}'])
        if (ROOT/source).is_file():assert b==(ROOT/source).read_bytes(),f'dirty source {source}'
        p=dest/target;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
        proof.append(dict(path=target,source_path=source,commit=commit,sha256=hashlib.sha256(b).hexdigest(),bytes=len(b)))
    (dest/'PROVENANCE.json').write_text(json.dumps(proof,indent=2)+'\n');(dest/'COMMIT.txt').write_text(commit+'\n')
    script=f'''#!/bin/bash
#SBATCH --job-name=bshared_{a.attempt}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --exclude=pax007
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output={remote}/logs/%j.out
#SBATCH --error={remote}/logs/%j.err
set -euo pipefail
TASK_ROOT={remote}
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" TMPDIR="$TASK_ROOT/tmp" PYTHONUNBUFFERED=1
export PREP_DEADLINE_EPOCH=$(( $(date +%s) + 3540 ))
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output/extracted"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={{b}}',flush=True); sys.exit(0 if b=='gpu' else 42)"
trap 'rc=$?; cd "$TASK_ROOT"; find output -type f -print0 | sort -z | xargs -0 -r sha256sum > OUTPUTS.sha256; exit "$rc"' EXIT
env N=256 K=16 R=512 MAX_SNAPS=131072 T_EARLY=5 N_TEST=8 SEED0=0 LOOSE=0 N_TRAIN=512 N_VAL=64 SEED=0 TEST_SEED=1 NEWTON_ITERS=8 LIN_TOL=1e-10 LIN_MAXITER=2000 \\
 EXTRA_SEED=1000 EXTRA_TRAJ=4032 GEN_CHUNK=64 PROJ_CHUNK=256 IDENT_ROWS=64 \\
 CKPT="$TASK_ROOT/experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl" \\
 OUT_PREFIX="$TASK_ROOT/output/extracted/" \\
 "$PY" {LANE}/regenerate.py
"$PY" {LANE}/train.py \\
 --data "$TASK_ROOT/output/extracted/sep_coeff_N256_K16_R512.npz" \\
 --checkpoint "$TASK_ROOT/experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl" \\
 --reference "$TASK_ROOT/{LANE}/checks/training-reference.json" \\
 --deadline-epoch "$PREP_DEADLINE_EPOCH" --out "$TASK_ROOT/output/trained"
echo ALL-DONE
'''
    (dest/'run.sbatch').write_text(script);(dest/'logs').mkdir()
    (dest/'RESERVATION.json').write_text(json.dumps(dict(attempt=a.attempt,remote=remote,reserved_seconds=3600,commit=commit),indent=2)+'\n')
    (dest/'MANIFEST.sha256').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(dest)}\n' for p in sorted(dest.rglob('*')) if p.is_file()))
    print(dest);print(remote)


if __name__=='__main__':main()
