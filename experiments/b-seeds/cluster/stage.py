"""Stage one b-seeds attempt into an isolated paralab-bound directory.

    python experiments/b-seeds/cluster/stage.py s1            # seed job, SEED0=1
    python experiments/b-seeds/cluster/stage.py final         # the sealed-cohort job

Each attempt gets its OWN submit directory and its OWN remote directory: one job per
directory, never two. Every staged source file is checked byte-for-byte against the
committed blob at HEAD before it is copied, so what runs on the cluster is exactly what is
in Git; the three generator dependencies are staged from this lane's `deps/` (their hashes
equal the incumbent's own staged manifest entries, see PROVENANCE-COPIES.json) into the
layout `blat_common` expects. The sealed job additionally stages the three seed checkpoints
from `experiments/b-seeds/checkpoints/`, which are committed produced artifacts whose
SHA256s must equal the ones the seed jobs recorded.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/b_seeds_20260917'
HOURS_SEED = '16:00:00'
HOURS_FINAL = '10:00:00'
EXCLUDE = 'pax007'
INCUMBENT = 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'

# --- the ladder / EQ-certification machinery (q-trajdirs + q-ridge staging lists) ---
LADDER_FILES = [
    'experiments/b-seeds/seeds_run.py',
    'experiments/b-seeds/trajdirs.py',
    'experiments/b-seeds/cluster/stage.py',
    'experiments/q-ridge/q_eqcert.py',
    'experiments/q-ridge/eqcert.py',
    'experiments/q-ridge/ridge.py',
    'experiments/b-ladder-top/topfix.py',
    'experiments/cheap-corrections/varpro.py',
    'experiments/cheap-corrections/directions.py',
    'experiments/head-ablation/arms.py',
    'experiments/head-ablation/ladder.py',
    'experiments/mr-burgers2d/engines.py',
    'experiments/mr-burgers2d/iterative_paths.py',
    'experiments/mr-burgers2d/accuracy_paths.py',
    'experiments/separable-decoder/sep_common.py',
    INCUMBENT,
]
# --- the incumbent's own training recipe: exactly the dn256b / push_r3a manifest set ---
TRAIN_FILES = [
    'experiments/separable-decoder/sep_burgers_r3.py',
    'experiments/separable-decoder/sep_coeff_extract.py',
    'experiments/separable-decoder/sep_hfit_run.py',
    'experiments/separable-decoder/sep_hfit.py',
    'experiments/separable-decoder/sep_solvers.py',
    'experiments/cost-to-tolerance/ctol_eq.py',
    'experiments/cost-to-tolerance/ctol_tol.py',
    'experiments/burgers2d-rom-latent-stepping/blat_common.py',
    'experiments/burgers2d-rom-latent-stepping/blat_train_ad.py',
    'experiments/burgers2d-rom-latent-stepping/followup/fu_common.py',
    'experiments/burgers2d-rom-latent-stepping/followup/fu_style.py',
    'experiments/nonlinear-decoder-architecture/nda_arch.py',
    'experiments/poisson2d-rom-objective/pro_common.py',
    'experiments/poisson2d-rom-objective/followup/fu_eq.py',
    'experiments/poisson2d-rom-objective/followup/fu_style.py',
    'experiments/poisson2d-rom-objective/followup/fu_train.py',
]
# source in Git -> destination in the staged tree (the layout blat_common bootstraps)
RELOCATED = {
    'experiments/b-seeds/deps/burgers2d-coord-rom/burgers2d_film.py':
        'experiments/burgers2d-rom-latent-stepping/deps/burgers2d-coord-rom/burgers2d_film.py',
    'experiments/b-seeds/deps/multistage-precision/ms_autodecoder.py':
        'experiments/burgers2d-rom-latent-stepping/deps/multistage-precision/ms_autodecoder.py',
    'experiments/b-seeds/deps/multistage-precision/ms_parametric.py':
        'experiments/burgers2d-rom-latent-stepping/deps/multistage-precision/ms_parametric.py',
    'experiments/b-seeds/deps/multistage-precision/ms_autodecoder.py#poisson':
        'experiments/poisson2d-rom-objective/deps/ms_autodecoder.py',
    'experiments/b-seeds/deps/multistage-precision/ms_parametric.py#poisson':
        'experiments/poisson2d-rom-objective/deps/ms_parametric.py',
}

PREAMBLE = '''#!/bin/bash
#SBATCH --job-name=bsd___ATTEMPT__
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:__GPU__:1
#SBATCH --exclude=__EXCLUDE__
#SBATCH --cpus-per-task=8
#SBATCH --mem=180G
#SBATCH --time=__HOURS__
#SBATCH --output=__REMOTE__/logs/%j.out
#SBATCH --error=__REMOTE__/logs/%j.err
set -euo pipefail
TASK_ROOT=__REMOTE__
PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
export XDG_CACHE_HOME="$TASK_ROOT/cache" MPLCONFIGDIR="$TASK_ROOT/cache/matplotlib"
export TMPDIR="$TASK_ROOT/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TASK_ROOT/output"
cd "$TASK_ROOT"
sha256sum -c MANIFEST.sha256 --quiet
export SOURCE_COMMIT=$(cat COMMIT.txt)
echo "host=$(hostname) source_commit=$SOURCE_COMMIT attempt=__ATTEMPT__ started=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /cluster/tufts/paralab | tail -1
"$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}',flush=True); sys.exit(0 if b=='gpu' else 42)"
LADDER_PATH="$TASK_ROOT/experiments/mr-burgers2d:$TASK_ROOT/experiments/head-ablation:$TASK_ROOT/experiments/cheap-corrections:$TASK_ROOT/experiments/b-ladder-top:$TASK_ROOT/experiments/q-ridge:$TASK_ROOT/experiments/b-seeds"
'''

SEED_BODY = '''
# ---------------------------------------------------------------- stage A: the bank ----
# run_r3a.sbatch (job 2835788) verbatim except SEED0, TRAIN_ONLY=1 and TIME_CAP=0 (DESIGN.md D1)
mkdir -p "$TASK_ROOT/output/train"
cd "$TASK_ROOT/experiments/separable-decoder"
echo "STAGE A bank seed=__SEED__ $(date -Is)"
env ROUND=3 K=16 LR=1e-3 P_SUB=4096 WD=1e-5 EMA_DECAY=0.999 LAM_ORTH=1e-4 \\
  MAX_SNAPS=16384 T_EARLY=5 FULLROWS=64 \\
  N_FF=128 FF_SCALE=4.0 H_HIDDEN=256 \\
  EQ_MS=64,256 EQ_CAND_CAP=65536 STEP_TOLS=1e-9,1e-6 \\
  N_TEST=8 REPS=5 WARM=2 PAIR_REPS=3 TR_FACTOR=0.01 SEED0=__SEED__ \\
  IC_TOP=12 IC_ENC_BUDGET=50 ENC_STEPS=12000 EXTRAP=1.0 \\
  ORACLE_BUDGET=150 SSTEP_TS=1,2,3,5,10,25,50 SSTEP_BUDGET=120 \\
  NEWTON_TOLS=3e-1,1e-1,3e-2,1e-2,3e-3,1e-3,1e-4 LIN_FRACS=0.05,0.5 \\
  MAX_NEWTON=20 BATCHED=1 \\
  N=256 R=512 G_HIDDEN=1024 SNAP_NORM=0 STEPS=300000 TIME_CAP=0 FULL_LAST=10000 POOL=0 MESH_ARM=1 \\
  TRAIN_ONLY=1 OUT_PREFIX="$TASK_ROOT/output/train/" \\
  "$PY" sep_burgers_r3.py
BANK="$TASK_ROOT/output/train/sep_burgers_r3_N256_K16_R512.pkl"
test -s "$BANK"

# --------------------------------------------- stage B: coefficient extraction ----
# run_dn256b.sbatch (job 2837431) extraction verbatim except SEED0 (the state pick)
echo "STAGE B extract seed=__SEED__ $(date -Is)"
env N=256 K=16 R=512 MAX_SNAPS=131072 T_EARLY=5 N_TEST=8 SEED0=__SEED__ LOOSE=1 \\
  EXTRA_SEED=1000 EXTRA_TRAJ=4032 GEN_CHUNK=64 PROJ_CHUNK=256 IDENT_ROWS=64 \\
  CKPT="$BANK" OUT_PREFIX="$TASK_ROOT/output/train/" \\
  "$PY" sep_coeff_extract.py
NPZ="$TASK_ROOT/output/train/sep_coeff_N256_K16_R512.npz"
test -s "$NPZ"

# ------------------------------------------------- stage C: the head refit ('mid') ----
# run_dn256b.sbatch hfit-full verbatim except SEED0, ARMS=mid only, TIME_CAP=0 (DESIGN.md D1)
echo "STAGE C hfit seed=__SEED__ $(date -Is)"
env NPZ="$NPZ" CKPT="$BANK" OUT="$TASK_ROOT/output/train/hfit_full.json" \\
  ARMS=mid STEPS=200000 BATCH=4096 LR=1e-3 TIME_CAP=0 \\
  ORACLE_ITERS=150 CODEDIAG_N=2048 CODEDIAG_ITERS=60 ENC_STEPS=12000 SEED0=__SEED__ \\
  EMIT=mid EMIT_PATH="$TASK_ROOT/output/train/sep_hfit_seed__SEED__.pkl" \\
  "$PY" sep_hfit_run.py
SEEDCK="$TASK_ROOT/output/train/sep_hfit_seed__SEED__.pkl"
test -s "$SEEDCK"
sha256sum "$BANK" "$NPZ" "$SEEDCK" | tee "$TASK_ROOT/output/train/TRAIN-SHA256.txt"
cd "$TASK_ROOT"

# ------------------------------------------- stage D: the seed's development ladder ----
echo "STAGE D ladder seed=__SEED__ $(date -Is)"
PYTHONPATH="$LADDER_PATH" "$PY" experiments/b-seeds/seeds_run.py \\
  --config experiments/b-seeds/config-dev-seed__SEED__.json \\
  --checkpoint "$SEEDCK" --out output/ladder_seed

# -------------------------------- stage E: the incumbent's development ladder, same job ----
echo "STAGE E ladder incumbent $(date -Is)"
PYTHONPATH="$LADDER_PATH" "$PY" experiments/b-seeds/seeds_run.py \\
  --config experiments/b-seeds/config-dev-incumbent.json \\
  --checkpoint __INCUMBENT__ --out output/ladder_incumbent

# ------------------------------ stage F: EQ certification, q <= 64 (non-fatal, optional) ----
echo "STAGE F eqcert seed=__SEED__ $(date -Is)"
set +e
PYTHONPATH="$LADDER_PATH" "$PY" experiments/q-ridge/q_eqcert.py \\
  --config experiments/b-seeds/config-eqcert-seed__SEED__.json \\
  --checkpoint "$SEEDCK" --out output/eqcert_seed
rc=$?
set -e
if [ "$rc" -ne 0 ]; then echo "EQCERT FAILED rc=$rc $(date -Is)" | tee output/EQCERT-FAILED; fi

echo "STAGES DONE $(date -Is)"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''

FINAL_BODY = '''
# The sealed cohort is opened here and only here (DESIGN.md section 3). Four sequential
# invocations of the SAME driver and configuration shape as the development ladders: the
# incumbent, then seeds 1, 2, 3. Each writes its own output directory.
for CK in incumbent seed1 seed2 seed3; do
  echo "SEALED LADDER $CK $(date -Is)"
  case "$CK" in
    incumbent) CKPT=__INCUMBENT__ ;;
    *) CKPT="experiments/b-seeds/checkpoints/sep_hfit_$CK.pkl" ;;
  esac
  sha256sum "$CKPT"
  PYTHONPATH="$LADDER_PATH" "$PY" experiments/b-seeds/seeds_run.py \\
    --config "experiments/b-seeds/config-sealed-$CK.json" \\
    --checkpoint "$CKPT" --out "output/sealed_$CK"
done
echo "STAGES DONE $(date -Is)"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''


def main():
    attempt = sys.argv[1]
    gpu = sys.argv[2] if len(sys.argv) > 2 else 'a100'
    assert attempt.isalnum(), attempt
    assert gpu in ('a100', 'h100', 'h200', 'l40s'), gpu
    if attempt.startswith('s') and attempt[1:].isdigit():
        seed = int(attempt[1:])
        kind = 'seed'
        files = LADDER_FILES + TRAIN_FILES + [
            f'experiments/b-seeds/config-dev-seed{seed}.json',
            'experiments/b-seeds/config-dev-incumbent.json',
            f'experiments/b-seeds/config-eqcert-seed{seed}.json']
        hours = HOURS_SEED
    elif attempt.startswith('final'):
        seed = None
        kind = 'final'
        files = LADDER_FILES + ['experiments/b-seeds/config-sealed-incumbent.json'] + [
            f'experiments/b-seeds/config-sealed-seed{s}.json' for s in (1, 2, 3)] + [
            f'experiments/b-seeds/checkpoints/sep_hfit_seed{s}.pkl' for s in (1, 2, 3)]
        hours = HOURS_FINAL
    else:
        raise SystemExit(f'attempt must be s<seed> or final*, got {attempt}')
    # STAGE_ROOT lets the local smoke stage into scratch without touching runs/<attempt>
    out = Path(os.environ.get('STAGE_ROOT', ROOT / 'experiments/b-seeds/runs')) / attempt
    out.mkdir(parents=True, exist_ok=False)
    remote = f'{NAMESPACE}/{attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True)
    assert not dirty.strip(), f'working tree not clean:\n{dirty}'
    proof = []

    def stage_one(name, dest_rel):
        content = (ROOT / name).read_bytes()
        assert content == subprocess.check_output(
            ['git', '-C', str(ROOT), 'show', f'{commit}:{name}']), f'uncommitted: {name}'
        dest = out / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        proof.append(dict(source=name, staged_as=dest_rel, bytes=len(content),
                          sha256=hashlib.sha256(content).hexdigest(), commit=commit))

    for name in files:
        stage_one(name, name)
    if kind == 'seed':
        for src, dest_rel in RELOCATED.items():
            stage_one(src.split('#')[0], dest_rel)
    if kind == 'final':
        # the seed checkpoints must be the ones the seed jobs recorded
        for s in (1, 2, 3):
            rec = ROOT / f'experiments/b-seeds/checkpoints/sep_hfit_seed{s}.sha256'
            want = rec.read_text().split()[0]
            got = hashlib.sha256((ROOT / f'experiments/b-seeds/checkpoints/sep_hfit_seed{s}.pkl').read_bytes()).hexdigest()
            assert want == got, (s, want, got)
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=2) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    (out / 'logs').mkdir()
    script = PREAMBLE + (SEED_BODY if kind == 'seed' else FINAL_BODY)
    for token, value in (('__ATTEMPT__', attempt), ('__REMOTE__', remote), ('__GPU__', gpu),
                         ('__HOURS__', hours), ('__EXCLUDE__', EXCLUDE),
                         ('__INCUMBENT__', INCUMBENT), ('__SEED__', str(seed))):
        script = script.replace(token, value)
    assert '__' not in script.replace('__pycache__', ''), script
    (out / 'run.sbatch').write_text(script)
    manifest = [f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(out)}'
                for p in sorted(out.rglob('*')) if p.is_file()]
    (out / 'MANIFEST.sha256').write_text('\n'.join(manifest) + '\n')
    print(out)
    print(remote)
    print(f'rsync -a {out}/ tufts-login:{remote}/')
    print(f"ssh tufts-login 'cd {remote} && sha256sum -c MANIFEST.sha256 --quiet && sbatch run.sbatch'")


if __name__ == '__main__':
    main()
