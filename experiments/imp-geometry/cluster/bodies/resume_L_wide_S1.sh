OUT=$(dirname $(pwd))/out; mkdir -p $OUT/cell
run() { local name=$1; shift; echo "=== $name: $*"; local t0=$(date +%s); "$@" > $OUT/$name.log 2>&1; rc=$?; [ $rc -ne 0 ] && FAILED=$((FAILED+1)); echo "=== $name rc=$rc $(( $(date +%s)-t0 ))s"; tail -5 $OUT/$name.log; }
ARCH=/cluster/tufts/paralab/tawal01/burgers-archive/imp-geometry/L_wide_S1
run restore bash -c "cp -a $ARCH/. $OUT/cell/ && cd $OUT/cell && sha256sum -c --quiet SHA256SUMS && rm SHA256SUMS"
for st in ladder inter time diag freeze; do
  [ $FAILED -ne 0 ] && break
  run $st $PY run_cell.py --dom L --fam wide --lev S1 --out $OUT/cell --stages $st --bank-steppers lspg_cnr
done
