#!/bin/bash
# jcp-references: copy the collected NON-sealed store (case files, manifest, status) to the cluster read-only copy
# /cluster/tufts/paralab/tawal01/jcp-ref/store (for lanes that score inside cluster jobs: JCPREF_STORE=<that path>),
# then verify the remote copy against the manifest.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
S="$ROOT/experiments/jcp-references/store"
NS=/cluster/tufts/paralab/tawal01/jcp-ref
ssh tufts-login "mkdir -p $NS/store && chmod -R u+w $NS/store"
rsync -aL --delete "$S/" "tufts-login:$NS/store/"
ssh tufts-login "cd $NS/store && sha256sum -c --quiet MANIFEST.sha256 && chmod -R a-w $NS/store && echo published store \$(sha256sum MANIFEST.sha256 | cut -c1-12)"
