OUT=$(dirname $(pwd))/out; mkdir -p $OUT/cell
run() { local name=$1; shift; echo "=== $name: $*"; local t0=$(date +%s); "$@" > $OUT/$name.log 2>&1; rc=$?; [ $rc -ne 0 ] && FAILED=$((FAILED+1)); echo "=== $name rc=$rc $(( $(date +%s)-t0 ))s"; tail -5 $OUT/$name.log; }
run reserve mkdir -p /cluster/tufts/paralab/tawal01/burgers-archive/imp-box/.test-opened
[ $FAILED -eq 0 ] && run open_once mkdir /cluster/tufts/paralab/tawal01/burgers-archive/imp-box/.test-opened/W40_A1
[ $FAILED -eq 0 ] && run restore bash -c "test ! -e /cluster/tufts/paralab/tawal01/burgers-archive/imp-box/W40_A1/test.json && cp -a /cluster/tufts/paralab/tawal01/burgers-archive/imp-box/W40_A1/. $OUT/cell/ && cd $OUT/cell && sha256sum -c --quiet SHA256SUMS && rm -rf _job SHA256SUMS"
[ $FAILED -eq 0 ] && run test $PY cell.py --W W40 --lev A1 --out $OUT/cell --stages test --frozen $(pwd)/runs/frozen/W40_A1.json
