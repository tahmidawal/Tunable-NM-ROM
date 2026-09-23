#!/bin/bash
# usage: cluster/collect.sh <attempt>  -- checksum-verified pull (fields already deleted by the audit), then remote delete.
set -euo pipefail
A=$1
HERE=$(cd "$(dirname "$0")/.." && pwd)
NS=/cluster/tufts/paralab/tawal01/specfom_20260923
ssh tufts-login "test -f $NS/$A/OUTPUTS.sha256 && grep -l ALL-DONE $NS/$A/logs/*.out"
mkdir -p "$HERE/runs/$A/archive"
rsync -a "tufts-login:$NS/$A/logs" "tufts-login:$NS/$A/OUTPUTS.sha256" "$HERE/runs/$A/archive/"
rsync -a --include='output*/***' --exclude='*' "tufts-login:$NS/$A/" "$HERE/runs/$A/archive/"
(cd "$HERE/runs/$A/archive" && sha256sum -c OUTPUTS.sha256 --quiet && echo CHECKSUMS-OK)
grep -h "jax_backend\|AUDIT\|GATES" "$HERE/runs/$A/archive/logs/"*.out
ssh tufts-login "rm -rf $NS/$A && ls $NS"
