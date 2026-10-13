#!/bin/bash
# jcp-references: pull ONE finished attempt (logs, OUTPUTS.sha256, result and non-sealed outputs) into runs/<attempt>/.
# Sealed jobs: only result.json (QC aggregates + ciphertext hashes), logs and OUTPUTS.sha256 come back; the *.enc
# artifacts stay on the cluster. Then verifies the pulled files against OUTPUTS.sha256.
set -euo pipefail
A="$1"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NS=/cluster/tufts/paralab/tawal01/jcp-ref
LOCAL="$ROOT/experiments/jcp-references/runs/$A"
rsync -a --exclude '*.enc' --exclude 'cache/' --exclude 'tmp/' "tufts-login:$NS/$A/logs" "tufts-login:$NS/$A/OUTPUTS.sha256" "$LOCAL/"
rsync -a --exclude '*.enc' "tufts-login:$NS/$A/output" "$LOCAL/"
cd "$LOCAL" && grep -v '\.enc$' OUTPUTS.sha256 | sha256sum -c --quiet && echo "verified $(grep -vc '\.enc$' OUTPUTS.sha256) files"
