# Second rebuttal push, PROTOCOL.md A9(c): continue the Re 95 and Re 105 records for another 3.4 s (same step, refresh
# and saves).  Two CPU runs packed into one allocation, 16 threads each (submitted with 32 CPUs) (reference generation only; no timing is taken).
set -euo pipefail
NS=/cluster/tufts/paralab/tawal01/cyl-unsteady-20261103
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 MKL_NUM_THREADS=16
pids=()
for C in re95 re105; do
  SRC=$NS/r1-record-$C/results/record_${C}_p4
  ( "$PY" ref_cfg.py --base $SRC/config.json --Re ${C#re} --initial $SRC/record.npz --kind cont --threads 16 --out results/cont_${C}_cfg.json
    "$PY" vendor/ref/transient.py --config results/cont_${C}_cfg.json --out results/cont_${C}_p4 ) > results/cont_$C.log 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
tail -3 results/cont_re95.log results/cont_re105.log
exit $rc
