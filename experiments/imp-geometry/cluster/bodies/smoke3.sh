OUT=$(dirname $(pwd))/out; mkdir -p $OUT
run() { local name=$1; shift; echo "=== $name: $*"; local t0=$(date +%s); timeout 900 "$@" > $OUT/$name.log 2>&1; rc=$?; echo "=== $name rc=$rc $(( $(date +%s)-t0 ))s"; tail -12 $OUT/$name.log; }
for v in v3 v4; do run inter_${v}_0.25 $PY interaction.py --out $OUT/interaction --n 256 --ncases 16 --nsteps 2000 --variant $v --T 0.25; done
