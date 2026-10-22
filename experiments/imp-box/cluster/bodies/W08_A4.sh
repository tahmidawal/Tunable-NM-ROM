OUT=$(dirname $(pwd))/out; mkdir -p $OUT
run() { local name=$1; shift; echo "=== $name: $*"; local t0=$(date +%s); "$@" > $OUT/$name.log 2>&1; rc=$?; [ $rc -ne 0 ] && FAILED=$((FAILED+1)); echo "=== $name rc=$rc $(( $(date +%s)-t0 ))s"; tail -5 $OUT/$name.log; }
for st in ref data pod bank ladder time controls; do
  run $st $PY cell.py --W W08 --lev A4 --out $OUT/cell --stages $st 
  [ $FAILED -ne 0 ] && break
done
