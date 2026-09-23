#!/bin/bash
# usage: cluster/collect.sh <attempt>  -- checksum-verified pull, then remote delete.
set -euo pipefail
A=$1
HERE=$(cd "$(dirname "$0")/.." && pwd)
NS=/cluster/tufts/paralab/tawal01/pbank_20260923
ssh tufts-login "test -f $NS/$A/OUTPUTS.sha256 && grep -l ALL-DONE $NS/$A/logs/*.out"
mkdir -p "$HERE/runs/$A/archive"
for O in output2 output3; do if ssh tufts-login "test -d $NS/$A/$O"; then rsync -a "tufts-login:$NS/$A/$O" "$HERE/runs/$A/archive/"; fi; done
rsync -a "tufts-login:$NS/$A/output" "tufts-login:$NS/$A/logs" "tufts-login:$NS/$A/OUTPUTS.sha256" "$HERE/runs/$A/archive/"
(cd "$HERE/runs/$A/archive" && sha256sum -c OUTPUTS.sha256 --quiet && echo CHECKSUMS-OK)
grep -h "jax_backend" "$HERE/runs/$A/archive/logs/"*.out
ssh tufts-login "rm -rf $NS/$A && ls $NS"
