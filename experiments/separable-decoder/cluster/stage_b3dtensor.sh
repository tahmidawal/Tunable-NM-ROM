#!/bin/bash
# Stage one Burgers-3D job directory: cluster/stage/<jobname>/{code,in,out,logs}.
# The code is the self-contained b3d_* set plus the 2D reference FOM for gate F8;
# the persisted parameter tables travel in code/runs/b3dtensor/tables (they are
# inputs, sha256-fingerprinted in every JSON).  Extra payloads (checkpoints,
# kernel npz) go to in/.  Writes COMMIT.txt and MANIFEST.sha256.
# Usage: ./stage_b3dtensor.sh <jobname> [payload ...]
set -euo pipefail
JOB=${1:?jobname}; shift
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$(cd "$HERE/.." && pwd)
ST="$HERE/stage/$JOB"
rm -rf "$ST"; mkdir -p "$ST/code/runs/b3dtensor/tables" "$ST/code/deps/burgers2d-coord-rom" "$ST/in" "$ST/out" "$ST/logs"
cp "$SRC"/b3d_common.py "$SRC"/b3d_tensor_common.py "$SRC"/b3d_fom_gates.py "$SRC"/sep_b3d_tensor.py "$SRC"/sep_b3d_kernels.py "$ST/code/"
cp "$SRC/../wave2d-rom-latent-stepping/deps/burgers2d-coord-rom/burgers2d_film.py" "$ST/code/deps/burgers2d-coord-rom/"
cp "$SRC"/runs/b3dtensor/tables/*.npz "$ST/code/runs/b3dtensor/tables/"
for p in "$@"; do cp "$p" "$ST/in/"; done
git -C "$SRC" rev-parse HEAD > "$ST/COMMIT.txt"
( cd "$ST" && find code in COMMIT.txt -type f | sort | xargs sha256sum > MANIFEST.sha256 )
echo "staged $ST ($(wc -l < "$ST/MANIFEST.sha256") files, commit $(cat "$ST/COMMIT.txt"))"
