set -uo pipefail
$PY mmcompile.py comp 1024 PM-1024 1024 /cluster/tufts/paralab/tawal01/burgers-archive/imp-multimesh/j0/output/src1024 /cluster/tufts/paralab/tawal01/burgers-archive/imp-multimesh/j3/output 384 8 $OUT/compile/comp_1024_PM-1024_r384k8.json
