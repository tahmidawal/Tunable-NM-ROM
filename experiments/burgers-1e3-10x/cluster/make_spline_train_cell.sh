#!/usr/bin/env bash
# make_spline_train_cell.sh CELL ARM SEED S0_CELL [PRIOR_TRAIN_CELL ...]
set -euo pipefail
cell="$1"
arm="$2"
seed="$3"
s0_cell="$4"
shift 4
prior_cells=("$@")
command -v jq >/dev/null || { echo "jq is required" >&2; exit 2; }
[[ "$cell" =~ ^spline_[abc]_s(11|29|47)_r[0-9]+$ ]] || {
  echo "invalid spline trainer cell" >&2; exit 2;
}
[[ "$arm" =~ ^[ABC]$ ]] || { echo "invalid arm" >&2; exit 2; }
[[ "$seed" =~ ^(11|29|47)$ ]] || { echo "invalid seed" >&2; exit 2; }
[[ "$cell" == spline_"${arm,,}"_s"${seed}"_r* ]] || {
  echo "cell name does not match arm/seed" >&2; exit 2;
}
[[ "$s0_cell" =~ ^s0_spline_r[0-9]+$ ]] || { echo "invalid S0 cell" >&2; exit 2; }
for prior in "${prior_cells[@]}"; do
  [[ "$prior" =~ ^spline_[abc]_s(11|29|47)_r[0-9]+$ ]] || {
    echo "invalid prior cell: $prior" >&2; exit 2;
  }
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP="$(dirname "$HERE")"
WORKTREE="$(cd "$EXP/../.." && pwd)"
ROOT="$(cd "$WORKTREE/../.." && pwd)"
STAGE="$HERE/stage/$cell"
REMOTE="/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/$cell"
S0_RUN="$EXP/runs/$s0_cell"
COMMIT="$(git -C "$WORKTREE" rev-parse HEAD)"
[[ -z "$(git -C "$WORKTREE" status --porcelain -- "$EXP")" ]] || {
  echo "experiment tree is dirty; commit exact sources before staging" >&2; exit 3;
}
[[ -f "$S0_RUN/LOCAL.sha256" ]] || { echo "missing pulled S0 run" >&2; exit 4; }
(cd "$S0_RUN" && sha256sum -c LOCAL.sha256)
s0_json_sha="$(sha256sum "$S0_RUN/out/s0.json" | cut -d' ' -f1)"
s0_npz_sha="$(sha256sum "$S0_RUN/out/s0.npz" | cut -d' ' -f1)"
[[ "$(jq -r .status "$S0_RUN/out/AUDIT.json")" == pass ]] || {
  echo "S0 independent audit did not pass" >&2; exit 4;
}
[[ "$(jq -r .source_json_sha256 "$S0_RUN/out/AUDIT.json")" == "$s0_json_sha" ]] || {
  echo "S0 JSON/audit binding mismatch" >&2; exit 4;
}
[[ "$(jq -r .source_npz_sha256 "$S0_RUN/out/AUDIT.json")" == "$s0_npz_sha" ]] || {
  echo "S0 NPZ/audit binding mismatch" >&2; exit 4;
}

# Replacing a local staging copy is recoverable and cannot touch a pulled run.
rm -rf "$STAGE"
mkdir -p "$STAGE/logs" "$STAGE/out" "$STAGE/code/deps/s0" \
  "$STAGE/code/deps/burgers2d-coord-rom"
cp "$EXP"/b10_*.py "$STAGE/code/"
cp "$WORKTREE/experiments/burgers-hybrid-1024/bh_common.py" "$STAGE/code/"
cp "$ROOT/worktrees/2026-08-14-burgers2d-coord-rom/experiments/burgers2d-coord-rom/burgers2d_film.py" \
  "$STAGE/code/deps/burgers2d-coord-rom/"
cp "$S0_RUN/out/s0.json" "$S0_RUN/out/s0.npz" "$S0_RUN/out/AUDIT.json" \
  "$STAGE/code/deps/s0/"

prior_args=()
for index in "${!prior_cells[@]}"; do
  prior="${prior_cells[$index]}"
  run="$EXP/runs/$prior"
  [[ -f "$run/LOCAL.sha256" ]] || { echo "missing pulled prior: $prior" >&2; exit 4; }
  (cd "$run" && sha256sum -c LOCAL.sha256)
  json_sha="$(sha256sum "$run/out/train.json" | cut -d' ' -f1)"
  npz_sha="$(sha256sum "$run/out/train.npz" | cut -d' ' -f1)"
  checkpoint_sha="$(sha256sum "$run/out/checkpoint.pkl" | cut -d' ' -f1)"
  [[ "$(jq -r .status "$run/out/AUDIT.json")" == pass ]] || {
    echo "prior independent audit did not pass: $prior" >&2; exit 4;
  }
  [[ "$(jq -r .source_json_sha256 "$run/out/AUDIT.json")" == "$json_sha" \
     && "$(jq -r .source_npz_sha256 "$run/out/AUDIT.json")" == "$npz_sha" \
     && "$(jq -r .source_checkpoint_sha256 "$run/out/AUDIT.json")" == "$checkpoint_sha" ]] || {
    echo "prior artifact/audit binding mismatch: $prior" >&2; exit 4;
  }
  target="$STAGE/code/deps/prior_$index"
  mkdir -p "$target"
  cp "$run/out/train.json" "$run/out/train.npz" "$run/out/checkpoint.pkl" \
    "$run/out/AUDIT.json" "$target/"
  prior_args+=("deps/prior_$index/train.json")
done

job_name="ctol_b10_$cell"
{
  printf '%s' '$PY b10_spline_train.py deps/s0/s0.json '
  printf '%q ' "$arm" '../out/train.json' '../out/train.npz' '../out/checkpoint.pkl'
  if ((${#prior_args[@]})); then
    printf '%q ' "${prior_args[@]}"
  fi
  printf '\n'
} > "$STAGE/command.txt"
command="$(cat "$STAGE/command.txt")"
cat > "$STAGE/run.sbatch" <<EOF
#!/bin/bash
#SBATCH -J $job_name
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o $REMOTE/logs/%j.out
#SBATCH -e $REMOTE/logs/%j.err
set -euo pipefail
cd "$REMOTE"
echo "stage_manifest_file_sha256=\$(sha256sum MANIFEST.sha256 | cut -d' ' -f1)"
sha256sum -c MANIFEST.sha256
cd code
export JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PY=/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python
export B10_COMMIT=$COMMIT
export TRAIN_SEED=$seed
export SMOKE=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host=\$(hostname) gpu=\$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "commit=$COMMIT cell=$cell arm=$arm seed=$seed"
\$PY - <<'PRE' || { echo "GPU PREFLIGHT FAILED"; exit 42; }
import jax, sys
backend = jax.default_backend()
print(f"jax_backend={backend}", jax.devices()[0], flush=True)
sys.exit(0 if backend == "gpu" else 42)
PRE
$command
echo ALL-DONE
EOF
rm "$STAGE/command.txt"
(cd "$STAGE" && find . -type f -not -name MANIFEST.sha256 -exec sha256sum {} \; | sort > MANIFEST.sha256)
echo "stage=$STAGE"
echo "commit=$COMMIT"
echo "manifest_file_sha256=$(sha256sum "$STAGE/MANIFEST.sha256" | cut -d' ' -f1)"
