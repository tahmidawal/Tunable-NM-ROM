#!/bin/bash
# jcp-robust E2 3D: the W1024 bank recipe (train2w.py, configs/train_r1024.json) for one recipe seed
# (DESIGN.md 3.1, 3.3, A1.7). Only model_seed and subset_seed differ from the incumbent (make_config.py).
#
#   PY=<python> bash experiments/jcp-robust/train3d/run_seed.sh <seed|pilot> <outdir>
#
# <seed> = 1, 2, 3 -> configs/train_r1024_s<seed>.json;  pilot -> configs/train_r1024_pilot.json (200 bank steps).
# Must be launched with the job root (<jobroot>/experiments/jcp-robust/...) as the layout; it cd's to <jobroot>
# because the config's compare_bank is a repo-relative path. The incumbent job j3b ran on an H200 with --mem=420G.
# Optional env: TRAIN3D_CONFIG=<path> overrides the config (CPU chaining test only; production never sets it).
set -euo pipefail
[ $# -eq 2 ] || { echo "usage: PY=<python> $0 <seed|pilot> <outdir>" >&2; exit 2; }
WHICH=$1
OUTDIR=$2
: "${PY:?set PY to the python interpreter (cluster: /cluster/tufts/paralab/tawal01/ae-research/venv/bin/python)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"            # <jobroot>/experiments/jcp-robust/train3d
JOBROOT="$(cd "$HERE/../../.." && pwd)"
case "$WHICH" in
  pilot) CFG="$HERE/configs/train_r1024_pilot.json";;
  1|2|3|4|5) CFG="$HERE/configs/train_r1024_s$WHICH.json";;
  *) echo "seed must be 1..5 or pilot (seed 0 = the incumbent j3b)" >&2; exit 2;;
esac
CFG="${TRAIN3D_CONFIG:-$CFG}"
[ -f "$CFG" ] || { echo "missing config $CFG" >&2; exit 2; }
mkdir -p "$OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"
[ ! -e "$OUTDIR/SEED-COMPLETE" ] || { echo "$OUTDIR/SEED-COMPLETE exists: refusing to rerun" >&2; exit 3; }
log() { echo "[train3d $WHICH $(date -Is)] $*"; }

# environment of the incumbent job (wide-bank runs/j3b/run.sbatch)
export JAX_ENABLE_X64=true JAX_DEFAULT_MATMUL_PRECISION=highest
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUTDIR/tmp" "$OUTDIR/cache"
export TMPDIR="$OUTDIR/tmp" XDG_CACHE_HOME="$OUTDIR/cache"

log "host=$(hostname) config=$CFG jobroot=$JOBROOT commit=${SOURCE_COMMIT:-unset}"
{ nvidia-smi --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "nvidia-smi unavailable"; } \
  | tee "$OUTDIR/device.txt"
if [ -z "${TRAIN3D_CONFIG:-}" ]; then
  "$PY" -c "import jax,sys; b=jax.default_backend(); print(f'jax_backend={b}', flush=True); sys.exit(0 if b=='gpu' else 42)" \
    | tee -a "$OUTDIR/device.txt" || exit 42
fi
"$PY" -I - "$HERE" "$CFG" <<'EOF'
import hashlib, json, sys
from pathlib import Path
here, cfg = Path(sys.argv[1]), Path(sys.argv[2])
bad = [f['path'] for f in json.loads((here / 'PROVENANCE.json').read_text())['files']
       if hashlib.sha256((here.parent / f['path']).read_bytes()).hexdigest() != f['sha256']]
print('PROVENANCE sha256 ' + ('OK' if not bad else 'MISMATCH ' + ' '.join(bad)), flush=True)
c, b = json.loads(cfg.read_text()), json.loads((here / 'configs/train_r1024.json').read_text())
if 'local_smoke' not in c:
    d = sorted(k for k in b if c.get(k) != b[k])
    assert set(c) == set(b) and set(d) <= {'model_seed', 'subset_seed', 'bank_steps', 'warmup'}, d
    print('config differs from the incumbent only in', d, flush=True)
sys.exit(1 if bad else 0)
EOF

# compare_bank = "experiments/jcp-wide-bank/vendor/quad3d/inputs/model_M2" (repo-relative, kept verbatim in every
# config). The same bank (sha256 6687f259...) is vendored in this lane; link it at the path the config names.
CB="$JOBROOT/experiments/jcp-wide-bank/vendor/quad3d/inputs/model_M2"
if [ ! -e "$CB/bank.pkl" ]; then
  mkdir -p "$(dirname "$CB")"
  ln -s ../../../../jcp-robust/vendor/quad3d/inputs/model_M2 "$CB"
fi
echo "6687f259947ec08a986732db157db69c66f1110ab28125a029de17b4aa389af9  $CB/bank.pkl" | sha256sum -c --quiet

cd "$JOBROOT"
TIMECMD=""; [ -x /usr/bin/time ] && TIMECMD="/usr/bin/time -v -o $OUTDIR/time_v.txt"
log "train2w.py (via memwrap.py: unchanged script, device peak recorded)"
set +e
$TIMECMD "$PY" -u "$HERE/memwrap.py" "$OUTDIR/memory.json" "$HERE/train2w.py" --config "$CFG" --out "$OUTDIR" \
  > "$OUTDIR/train.out" 2> "$OUTDIR/train.err" &
TPID=$!
( while kill -0 $TPID 2>/dev/null; do C=$(pgrep -P $TPID | head -1); P=${C:-$TPID}
    echo "RSS_SAMPLE $(date -Is) $(ps -o rss= -p $P 2>/dev/null) kB | $(tail -n 1 "$OUTDIR/train.out" | cut -c1-160)" >> "$OUTDIR/rss.log"
    sleep 30; done ) &
SPID=$!
wait $TPID; RC=$?
kill $SPID 2>/dev/null; set -e
cat "$OUTDIR/train.out"; cat "$OUTDIR/train.err" >&2
[ $RC -eq 0 ] || { log "train2w.py failed rc=$RC"; exit $RC; }
grep -q "TRAINING COMPLETE" "$OUTDIR/train.out"

"$PY" -I - "$OUTDIR" "$WHICH" "$CFG" <<'EOF'
import json, sys
from pathlib import Path
o, which, cfg = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
t = json.load(open(o / 'training.json'))
m = json.load(open(o / 'memory.json'))
assert t.get('complete') is True
rec = dict(which=which, config=cfg, model_seed=t['config']['model_seed'], subset_seed=t['config']['subset_seed'],
           bank_steps=t['config']['bank_steps'], selected_step=t['bank']['selected_step'],
           selection_rule='lowest worst bank-validation projection error over the three groups (train2w.train_bank); '
                          'incumbent j3b selected 11000', validation_worst_over_groups=t['bank']['validation_worst_over_groups'],
           condition_65=t['bank'].get('condition_65'), ordering_inverse_check=t['ordering']['inverse_check'],
           gpu=t.get('gpu'), backend=t.get('backend'), seconds=t.get('seconds'),
           device_peak_GiB=m.get('peak_GiB'), device_bytes_limit=m.get('memory_stats', {}).get('bytes_limit'),
           process_maxrss_kB=m.get('ru_maxrss_kB'), device=(o / 'device.txt').read_text().strip().splitlines(),
           note='host MaxRSS of the job: sacct -j $SLURM_JOB_ID --format=JobID,MaxRSS,ReqMem,Elapsed')
(o / 'run_record.json').write_text(json.dumps(rec, indent=1, default=float) + '\n')
print('RUN RECORD', json.dumps({k: rec[k] for k in ('selected_step', 'bank_steps', 'device_peak_GiB', 'process_maxrss_kB')}))
EOF
rm -rf "$OUTDIR/tmp" "$OUTDIR/cache"
( cd "$OUTDIR" && find . -type f ! -name OUTPUTS.sha256 ! -name SEED-COMPLETE -printf '%P\n' | sort \
    | xargs sha256sum > OUTPUTS.sha256 )
date -Is > "$OUTDIR/SEED-COMPLETE"
log "SEED-COMPLETE ($OUTDIR)"
