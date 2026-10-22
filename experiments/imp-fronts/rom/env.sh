# imp-fronts rom/: put the modules the copied closure imports from sibling experiments on PYTHONPATH
# (engines.py + iterative_paths.py from mr-burgers2d, sep_common.py from separable-decoder; see PROVENANCE.json note).
# Source from anywhere:  source experiments/imp-fronts/rom/env.sh
_ROM="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ROOT="$(cd "$_ROM/../../.." && pwd)"
export PYTHONPATH="$_ROM/deps:$_ROOT/experiments/mr-burgers2d:$_ROOT/experiments/separable-decoder${PYTHONPATH:+:$PYTHONPATH}"   # deps first: arm (c) sep_common
export JAX_ENABLE_X64=true
unset _ROM _ROOT
