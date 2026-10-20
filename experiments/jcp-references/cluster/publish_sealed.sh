#!/bin/bash
# jcp-references: after a sealed fd job has been pulled (cluster/pull.sh <attempt>: result.json with QC aggregates and
# ciphertext hashes only), verify every ciphertext on the cluster against the hashes in that result.json, move the
# ciphertexts and the encrypted manifest into the read-only sealed store, and delete the job directory.
set -euo pipefail
A="$1"; [[ "$A" =~ ^[a-zA-Z0-9]+$ ]]
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
NS=/cluster/tufts/paralab/tawal01/jcp-ref
R="$ROOT/experiments/jcp-references/runs/$A/output/result.json"
LIST=$(/home/tahmid/Dev/.venv/bin/python -c "
import json,sys; r=json.load(open('$R')); m=r['sealed_manifest']
print(m['cipher_sha256']+'  manifest.json.enc')
for c in m['ciphertexts']: print(c['cipher']+'  '+c['file'])")
COH=$(/home/tahmid/Dev/.venv/bin/python -c "import json; r=json.load(open('$R')); print(f\"{r['config']['dim']}d/{r['config']['cohort']}\")")
echo "$LIST" | ssh tufts-login "cd $NS/$A/output && sha256sum -c --quiet - && echo verified"
ssh tufts-login "set -e; D=$NS/sealed/$COH/$A; mkdir -p $NS/sealed/$COH && mkdir \$D && cd $NS/$A/output && cp -a --parents \$(find . -name '*.enc') \$D/ && cp result.json \$D/ && chmod -R a-w \$D && rm -rf $NS/$A && echo published \$D"
