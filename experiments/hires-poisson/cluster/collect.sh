#!/bin/bash
# usage: cluster/collect.sh <attempt>  -- checksum-verified pull, then remote delete.
set -euo pipefail
A=$1
HERE=$(cd "$(dirname "$0")/.." && pwd)
NS=/cluster/tufts/paralab/tawal01/hires_p_20260920
ssh tufts-login "test -f $NS/$A/OUTPUTS.sha256 && grep -l ALL-DONE $NS/$A/logs/*.out"
mkdir -p "$HERE/runs/$A/archive"
ssh tufts-login "test -d $NS/$A/output2" && rsync -a "tufts-login:$NS/$A/output2" "$HERE/runs/$A/archive/"
rsync -a "tufts-login:$NS/$A/output" "tufts-login:$NS/$A/logs" "tufts-login:$NS/$A/OUTPUTS.sha256" "$HERE/runs/$A/archive/"
(cd "$HERE/runs/$A/archive" && sha256sum -c OUTPUTS.sha256 --quiet && echo CHECKSUMS-OK)
grep -h "jax_backend" "$HERE/runs/$A/archive/logs/"*.out
ssh tufts-login "rm -rf $NS/$A && ls $NS"
