# Job command of the comparators on the fresh cohort (operators and the CoLoRA re-implementation, models and online
# stages unchanged). Sourced inside the job script written by cluster/stage.sh.
D=$NS/data/fresh
(export PYTHONPATH=$TASK_ROOT/lane/baselines/ops
 "$PY" lane/baselines/fresh_refs.py --seed 20261114 --count 64 --name fresh64 --out $D
 "$PY" lane/baselines/ops_manifest.py /cluster/tufts/paralab/tawal01/rebuttal-burgers-ops/selection_manifest.json $TASK_ROOT/ops_manifest.json
 "$PY" lane/baselines/ops/evaluate.py --manifest $TASK_ROOT/ops_manifest.json --cohort-dir $D --cohort fresh64 --out-dir $TASK_ROOT/out-ops --skip-1024)
(export PYTHONPATH=$TASK_ROOT/lane/baselines/nfrom NF_GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
 "$PY" lane/baselines/fresh_colora.py --seed 20261114 --sha 1116b7080643b2b2ea96662f6bd3592d5c51b963b3e2e3c45fc5fb22711db2f8 --keep L16-w512:accurate --selection /cluster/tufts/paralab/tawal01/rebuttal-nfrom/selection.json --out-selection $TASK_ROOT/colora_selection.json -- --models /cluster/tufts/paralab/tawal01/rebuttal-nfrom --test $D/fresh64-L256 --out $TASK_ROOT/out-colora)
