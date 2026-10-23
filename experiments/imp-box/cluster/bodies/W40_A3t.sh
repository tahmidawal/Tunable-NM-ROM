OUT=$(dirname $(pwd))/out; mkdir -p $OUT/cell
run() { local name=$1; shift; echo "=== $name: $*"; local t0=$(date +%s); "$@" > $OUT/$name.log 2>&1; rc=$?; [ $rc -ne 0 ] && FAILED=$((FAILED+1)); echo "=== $name rc=$rc $(( $(date +%s)-t0 ))s"; tail -5 $OUT/$name.log; }
# A16 restart: restore the archived cell (ref/data/pod/bank/ladder done) and run its time and controls stages
run restore bash -c "cp -a /cluster/tufts/paralab/tawal01/burgers-archive/imp-box/W40_A3/. $OUT/cell/ && cd $OUT/cell && sha256sum -c --quiet SHA256SUMS && rm -rf _job SHA256SUMS .done_time .done_controls final.json frozen.json && ls .done_ladder"
for st in time controls; do
  [ $FAILED -eq 0 ] && run $st $PY cell.py --W W40 --lev A3 --out $OUT/cell --stages $st
done
