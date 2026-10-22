set -uo pipefail
$PY mmcompile.py comp 256 PM-256 256 /cluster/tufts/paralab/tawal01/burgers-archive/imp-multimesh/j0/output/src256 /cluster/tufts/paralab/tawal01/burgers-archive/imp-multimesh/j1/output 128 4 $OUT/compile/comp_256_PM-256_r128k4.json
