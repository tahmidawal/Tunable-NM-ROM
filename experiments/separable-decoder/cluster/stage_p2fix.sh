#!/bin/bash
# Stage a phase-2 GATE re-run for one finished cell with the corrected D1/G0a controls (retractions 5-6):
# fresh code, and the finished job's cache (data, bank, certified heads) COPIED on the cluster side so the
# heads are loaded, not retrained.  New directory (one job per directory).  Usage: ./stage_p2fix.sh <N> <BC>
set -euo pipefail
NN=${1:?N}; BCX=${2:?BC}
SRC=n${NN}_${BCX}; JOB=p2fix_n${NN}_${BCX}
HERE=$(cd "$(dirname "$0")" && pwd)
SEP=$(dirname "$HERE")
DST=$HERE/stage/$JOB
rm -rf "$DST"; mkdir -p "$DST/code" "$DST/out" "$DST/logs"
cp "$SEP"/wav2d_common.py "$SEP"/wav2d_bank.py "$SEP"/wav2d_head.py "$SEP"/wav2d_head_gates.py \
   "$SEP"/wav2d_rom.py "$SEP"/wav2d_rom_gates.py "$SEP"/stk2d_head.py "$DST/code/"
git -C "$SEP" rev-parse HEAD > "$DST/COMMIT.txt"
( cd "$DST" && find code COMMIT.txt -type f \( -name '*.py' -o -name 'COMMIT.txt' \) -exec sha256sum {} + | sort -k2 > MANIFEST.sha256 )
MEM=$([ "$NN" -ge 128 ] && echo 120G || echo 64G)
cat > "$HERE/run_wav2d_${JOB}.sbatch" <<EOS
#!/bin/bash
#SBATCH --job-name=wav2d_${JOB}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=${MEM}
#SBATCH --time=04:00:00
#SBATCH --output=/cluster/tufts/paralab/tawal01/wav2d/${JOB}/logs/%j.out
#SBATCH --error=/cluster/tufts/paralab/tawal01/wav2d/${JOB}/logs/%j.err
# Phase-2 gate re-run (corrected D1/G0a controls) on the cached heads of ${SRC}; then phase 3 again on the same cache.
set -euo pipefail
ROOT=/cluster/tufts/paralab/tawal01/wav2d/${JOB}
VENV=/cluster/tufts/paralab/tawal01/ae-research/venv
PY="\$VENV/bin/python"
export JAX_DEFAULT_MATMUL_PRECISION=highest
echo "host=\$(hostname) gpu=\$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
"\$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)"
( cd "\$ROOT" && sha256sum -c MANIFEST.sha256 --quiet && echo "stage MANIFEST OK" )
echo "commit=\$(cat \$ROOT/COMMIT.txt)"
rm -rf "\$ROOT/cache"; cp -r /cluster/tufts/paralab/tawal01/wav2d/${SRC}/cache "\$ROOT/cache"
cd "\$ROOT/code"
"\$PY" wav2d_head_gates.py N=${NN} BC=${BCX} R=64 STEPS=40000 OUT="\$ROOT/out" CACHE="\$ROOT/cache"
"\$PY" wav2d_rom_gates.py  N=${NN} BC=${BCX} R=64 STEPS=40000 M=64 RS=8,20,40 OUT="\$ROOT/out" CACHE="\$ROOT/cache"
cd "\$ROOT/out" && sha256sum *.json > RESULTS.sha256
echo ALL-DONE
EOS
echo "staged -> $DST and wrote run_wav2d_${JOB}.sbatch"
