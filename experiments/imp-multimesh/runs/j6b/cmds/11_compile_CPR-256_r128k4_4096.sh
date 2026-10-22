set -uo pipefail
$PY mmcompile.py comp 4096 CPR-256 256 /cluster/tufts/paralab/tawal01/burgers-archive/imp-multimesh/j0/output/src256 /cluster/tufts/paralab/tawal01/burgers-archive/imp-multimesh/j4/output 128 4 $OUT/compile/comp_4096_CPR-256_r128k4.json
