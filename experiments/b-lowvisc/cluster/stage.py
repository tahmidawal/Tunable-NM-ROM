"""Stage one b-lowvisc attempt into an isolated paralab-bound directory.

    python experiments/b-lowvisc/cluster/stage.py lvg01 [a100]

Mechanics copied from `b-seeds/cluster/stage.py` (itself q-ridge's): every staged file is
checked byte-for-byte against the committed blob at HEAD before it is copied, so what runs on
the cluster is exactly what is in Git; `COMMIT.txt`, `PROVENANCE.json` and `MANIFEST.sha256`
are written beside it; the sbatch activates the paralab venv, exports `JAX_ENABLE_X64=true`
and `JAX_DEFAULT_MATMUL_PRECISION=highest`, runs the GPU preflight (`jax_backend=gpu` or exit
42), writes `OUTPUTS.sha256` and ends with `ALL-DONE`.  One job per attempt directory, ever.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
NAMESPACE = '/cluster/tufts/paralab/tawal01/b_lowvisc_20260917'
EXCLUDE = 'pax007'

GATE_FILES = [
    'experiments/b-lowvisc/gate.py',
    'experiments/b-lowvisc/lv_common.py',
    'experiments/b-lowvisc/audit_gate.py',
    'experiments/b-lowvisc/cluster/stage.py',
    'experiments/b-lowvisc/config-gate.json',
    'experiments/mr-burgers2d/engines.py',
    'experiments/mr-burgers2d/iterative_paths.py',
    'experiments/head-ablation/ladder.py',
    'experiments/head-ablation/ablation.py',
    'experiments/head-ablation/arms.py',
    'experiments/separable-decoder/sep_common.py',
]

PREAMBLE = '''#!/bin/bash
#SBATCH --job-name=lv___ATTEMPT__
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:__GPU__:1
__CONSTRAINT__#SBATCH --exclude=__EXCLUDE__
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
LANE_PATH="$TASK_ROOT/experiments/mr-burgers2d:$TASK_ROOT/experiments/head-ablation:$TASK_ROOT/experiments/b-lowvisc"
'''

GATE_BODY = '''
# ---------------------------------------------------------------- the gate job ----
# Both viscosity families, one allocation, one GPU, no trained model (DESIGN.md section 5).
echo "GATE $(date -Is)"
PYTHONPATH="$LANE_PATH" "$PY" experiments/b-lowvisc/gate.py \\
  --config experiments/b-lowvisc/config-gate.json --out output/gate

echo "STAGES DONE $(date -Is)"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''


# --- the incumbent's own training recipe: the b-seeds stage A/B/C manifest set, verbatim ---
TRAIN_FILES = [
    'experiments/b-lowvisc/check_generator_parity.py',
    'experiments/separable-decoder/sep_burgers_r3.py',
    'experiments/separable-decoder/sep_coeff_extract.py',
    'experiments/separable-decoder/sep_hfit_run.py',
    'experiments/separable-decoder/sep_hfit.py',
    'experiments/separable-decoder/sep_solvers.py',
    'experiments/separable-decoder/sep_common.py',
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
    'experiments/mr-burgers2d/engines.py',
    'experiments/mr-burgers2d/iterative_paths.py',
    'experiments/b-lowvisc/lv_common.py',
    'experiments/b-lowvisc/cluster/stage.py',
]
# source in Git -> destination in the staged tree (the layout blat_common bootstraps).
# The generator is THIS LANE's copy: the incumbent's file with the viscosity bounds lifted into
# BURGERS_NU_LO / BURGERS_NU_HI, proved bit-identical under the defaults by
# check_generator_parity.py (checks/generator-parity.json), which the job reruns before stage A.
RELOCATED = {
    'experiments/b-lowvisc/deps/burgers2d-coord-rom/burgers2d_film.py':
        'experiments/burgers2d-rom-latent-stepping/deps/burgers2d-coord-rom/burgers2d_film.py',
    'experiments/b-lowvisc/deps/multistage-precision/ms_autodecoder.py':
        'experiments/burgers2d-rom-latent-stepping/deps/multistage-precision/ms_autodecoder.py',
    'experiments/b-lowvisc/deps/multistage-precision/ms_parametric.py':
        'experiments/burgers2d-rom-latent-stepping/deps/multistage-precision/ms_parametric.py',
    'experiments/b-lowvisc/deps/multistage-precision/ms_autodecoder.py#poisson':
        'experiments/poisson2d-rom-objective/deps/ms_autodecoder.py',
    'experiments/b-lowvisc/deps/multistage-precision/ms_parametric.py#poisson':
        'experiments/poisson2d-rom-objective/deps/ms_parametric.py',
    'experiments/b-seeds/deps/burgers2d-coord-rom/burgers2d_film.py':
        'experiments/b-seeds/deps/burgers2d-coord-rom/burgers2d_film.py',
    'experiments/b-lowvisc/deps/burgers2d-coord-rom/burgers2d_film.py#lane':
        'experiments/b-lowvisc/deps/burgers2d-coord-rom/burgers2d_film.py',
}

TRAIN_BODY = '''
# The generator this job runs is the incumbent's with the viscosity bounds lifted into the
# environment; prove that here, on this machine, before a single training step (DESIGN.md A4).
echo "GENERATOR PARITY $(date -Is)"
PYTHONPATH="$LANE_PATH" "$PY" experiments/b-lowvisc/check_generator_parity.py \\
  | tee output/generator-parity.txt
cp experiments/b-lowvisc/checks/generator-parity.json output/generator-parity.json

export BURGERS_NU_LO=0.001 BURGERS_NU_HI=0.01
echo "viscosity family nu ~ logU($BURGERS_NU_LO, $BURGERS_NU_HI)"

# ---------------------------------------------------------------- stage A: the bank ----
# b-seeds run_r3a body verbatim (job 2835788's recipe) except the viscosity family.
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
echo "STAGE B extract seed=__SEED__ $(date -Is)"
env N=256 K=16 R=512 MAX_SNAPS=131072 T_EARLY=5 N_TEST=8 SEED0=__SEED__ LOOSE=1 \\
  EXTRA_SEED=1000 EXTRA_TRAJ=4032 GEN_CHUNK=64 PROJ_CHUNK=256 IDENT_ROWS=64 \\
  CKPT="$BANK" OUT_PREFIX="$TASK_ROOT/output/train/" \\
  "$PY" sep_coeff_extract.py
NPZ="$TASK_ROOT/output/train/sep_coeff_N256_K16_R512.npz"
test -s "$NPZ"

# ------------------------------------------------- stage C: the head refit ('mid') ----
echo "STAGE C hfit seed=__SEED__ $(date -Is)"
env NPZ="$NPZ" CKPT="$BANK" OUT="$TASK_ROOT/output/train/hfit_full.json" \\
  ARMS=mid STEPS=200000 BATCH=4096 LR=1e-3 TIME_CAP=0 \\
  ORACLE_ITERS=150 CODEDIAG_N=2048 CODEDIAG_ITERS=60 ENC_STEPS=12000 SEED0=__SEED__ \\
  EMIT=mid EMIT_PATH="$TASK_ROOT/output/train/sep_hfit_lowvisc__SEED__.pkl" \\
  "$PY" sep_hfit_run.py
CK="$TASK_ROOT/output/train/sep_hfit_lowvisc__SEED__.pkl"
test -s "$CK"
sha256sum "$BANK" "$NPZ" "$CK" | tee "$TASK_ROOT/output/train/TRAIN-SHA256.txt"
cd "$TASK_ROOT"

echo "STAGES DONE $(date -Is)"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''


# --- the stage-3 panel (DESIGN A3/A8): b-panel's staged module set with this lane's driver ---
PANEL_CKPT = 'experiments/b-lowvisc/checkpoints/sep_hfit_lowvisc0.pkl'
PANEL_FILES = [
    'experiments/b-lowvisc/lv_panel.py',
    'experiments/b-lowvisc/lv_directions.py',
    'experiments/b-lowvisc/lv_common.py',
    'experiments/b-lowvisc/config-panel.json',
    'experiments/b-lowvisc/cluster/stage.py',
    'experiments/b-lowvisc/deps/b-panel-speed/fast.py',
    'experiments/b-lowvisc/deps/b-panel-speed/ladders.py',
    PANEL_CKPT,
    'experiments/b-ladder-top/topfix.py',
    'experiments/q-ridge/eqcert.py',
    'experiments/cheap-corrections/varpro.py',
    'experiments/cheap-corrections/directions.py',
    'experiments/head-ablation/arms.py',
    'experiments/head-ablation/ladder.py',
    'experiments/head-ablation/ablation.py',
    'experiments/mr-burgers2d/engines.py',
    'experiments/mr-burgers2d/iterative_paths.py',
    'experiments/mr-burgers2d/accuracy_paths.py',
    'experiments/separable-decoder/sep_common.py',
]
PANEL_BODY = '''
# bpn301's memory fraction on an 80 GB card; the panel holds every reduced query resident.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export PYTHONPATH="$TASK_ROOT/experiments/mr-burgers2d:$TASK_ROOT/experiments/separable-decoder:$TASK_ROOT/experiments/head-ablation:$TASK_ROOT/experiments/cheap-corrections:$TASK_ROOT/experiments/b-ladder-top:$TASK_ROOT/experiments/q-ridge:$TASK_ROOT/experiments/b-lowvisc:$TASK_ROOT/experiments/b-lowvisc/deps/b-panel-speed"
CKPT="$TASK_ROOT/__CKPT__"
echo "checkpoint $(sha256sum "$CKPT")"

# ------------------------------- stage 1: the checkpoint's own correction directions ----
echo "DIRECTIONS $(date -Is)"
"$PY" experiments/b-lowvisc/lv_directions.py \\
  --config experiments/b-lowvisc/config-panel.json --checkpoint "$CKPT" --out output/directions

# ---------------------------------------------------------- stage 2: the panel ----
echo "PANEL $(date -Is)"
"$PY" experiments/b-lowvisc/lv_panel.py \\
  --config experiments/b-lowvisc/config-panel.json --checkpoint "$CKPT" \\
  --inputs output/directions --out output/panel

echo "STAGES DONE $(date -Is)"
find output -type f -print0 | sort -z | xargs -0 sha256sum > OUTPUTS.sha256
echo ALL-DONE
'''

BODIES = {'lvg': (GATE_FILES, GATE_BODY, '6:00:00'),
          'lvt': (TRAIN_FILES, TRAIN_BODY, '16:00:00'),
          'lvp': (PANEL_FILES, PANEL_BODY, '8:00:00')}


def main():
    attempt = sys.argv[1]
    gpu = sys.argv[2] if len(sys.argv) > 2 else 'a100'
    assert attempt.isalnum(), attempt
    assert gpu in ('a100', 'h100', 'h200', 'l40s'), gpu
    kind = attempt[:3]
    assert kind in BODIES, f'unknown attempt kind {kind!r}; known: {sorted(BODIES)}'
    files, body, hours = BODIES[kind]
    # training jobs never run on L40S: 300000 f64 steps would not fit the wall clock (b-seeds A1.4)
    assert kind != 'lvt' or gpu in ('a100', 'h100', 'h200'), f'training runs on a100/h100/h200, not {gpu}'
    seed = int(os.environ.get('SEED0', '0'))
    # the panel needs an 80 GB card (b-qxm: several large-M subjects do not fit 40 GB); h100/h200 are 80+ GB
    constraint = '#SBATCH --constraint=a100-80G\n' if (kind == 'lvp' and gpu == 'a100') else ''
    if kind == 'lvp':
        assert gpu != 'l40s', 'the panel needs 80 GB'
        rec = (ROOT / PANEL_CKPT).with_suffix('.sha256').read_text().split()[0]
        got = hashlib.sha256((ROOT / PANEL_CKPT).read_bytes()).hexdigest()
        assert got == rec, f'checkpoint differs from the training job record: {got} != {rec}'

    out = Path(os.environ.get('STAGE_ROOT', ROOT / 'experiments/b-lowvisc/runs')) / attempt
    out.mkdir(parents=True, exist_ok=False)
    remote = f'{NAMESPACE}/{attempt}'
    commit = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True)
    assert not dirty.strip(), f'working tree not clean:\n{dirty}'
    proof = []
    for name in list(files) + (list(RELOCATED) if kind == 'lvt' else []):
        dest_rel = RELOCATED.get(name, name) if kind == 'lvt' else name
        name = name.split('#')[0]
        content = (ROOT / name).read_bytes()
        assert content == subprocess.check_output(
            ['git', '-C', str(ROOT), 'show', f'{commit}:{name}']), f'uncommitted: {name}'
        dest = out / dest_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
        proof.append(dict(source=name, staged_as=dest_rel, bytes=len(content),
                          sha256=hashlib.sha256(content).hexdigest(), commit=commit))
    (out / 'PROVENANCE.json').write_text(json.dumps(proof, indent=2) + '\n')
    (out / 'COMMIT.txt').write_text(commit + '\n')
    (out / 'logs').mkdir()
    script = PREAMBLE + body
    for token, value in (('__ATTEMPT__', attempt), ('__REMOTE__', remote), ('__GPU__', gpu),
                         ('__HOURS__', hours), ('__EXCLUDE__', EXCLUDE), ('__SEED__', str(seed)),
                         ('__CONSTRAINT__', constraint), ('__CKPT__', PANEL_CKPT)):
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
