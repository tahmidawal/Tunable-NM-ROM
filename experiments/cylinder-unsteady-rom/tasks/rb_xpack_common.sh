# Second rebuttal push, PROTOCOL.md A9(a): extra references (CPU): settle -> 0.7 s at 1/3200 -> record.
# Expects PAIRS="<Re>:<training case whose recorded end state is the start> ...".  The runs are packed into one
# allocation (submitted with 16 CPUs per run), 16 threads each; reference generation only, no timing is taken.
set -euo pipefail
NS=/cluster/tufts/paralab/tawal01/cyl-unsteady-20261103
pids=()
for P in $PAIRS; do
  RE=${P%%:*}; FROM=${P##*:}
  SRC=$NS/r1-record-$FROM/results/record_${FROM}_p4
  ( "$PY" ref_cfg.py --base $SRC/config.json --Re $RE --initial $SRC/record.npz --kind settle --out results/settle_re${RE}_cfg.json
    "$PY" vendor/ref/transient.py --config results/settle_re${RE}_cfg.json --out results/settle_re${RE}_p4
    "$PY" ref_cfg.py --base $SRC/config.json --Re $RE --initial results/settle_re${RE}_p4/settle.npz --kind record --out results/record_re${RE}_cfg.json
    "$PY" vendor/ref/transient.py --config results/record_re${RE}_cfg.json --out results/record_re${RE}_p4 ) > results/ref_re$RE.log 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
tail -3 results/ref_re*.log
exit $rc
