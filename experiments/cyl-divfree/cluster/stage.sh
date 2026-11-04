#!/usr/bin/env bash
# Stage committed lane code into a fresh cluster job directory: bash cluster/stage.sh <job-name>
# Copies HEAD's experiments/cyl-divfree and experiments/ns-obstacle-rom code (sparse checkout, no archives)
# via git archive (committed content only), writes CODE-SHA256SUMS, scp's directly into the paralab path.
set -euo pipefail
NAME=$1; NS=/cluster/tufts/paralab/tawal01/cyl-divfree-20261103; DEST=$NS/$NAME
ROOT=$(git rev-parse --show-toplevel); cd "$ROOT"
STAGE=$(mktemp -d); trap 'rm -rf "$STAGE"' EXIT
git -c gc.auto=0 archive HEAD experiments/cyl-divfree $(cd experiments/ns-obstacle-rom && git -c gc.auto=0 ls-files | grep -v -E '^(artifacts|staging|checks|datasets|retention|retention_delta|collections)/' | sed 's#^#experiments/ns-obstacle-rom/#') | tar -x -C "$STAGE"
(cd "$STAGE" && find . -type f ! -name CODE-SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > CODE-SHA256SUMS && git -C "$ROOT" rev-parse HEAD > SOURCE_COMMIT.txt)
ssh tufts-login "test ! -e $DEST && mkdir -p $DEST"
scp -q -r "$STAGE" tufts-login:$DEST/code
scp -q experiments/cyl-divfree/cluster/job.sh tufts-login:$DEST/job.sh
ssh tufts-login "cd $DEST/code && sha256sum -c --quiet CODE-SHA256SUMS && echo staged $DEST commit \$(cat SOURCE_COMMIT.txt) files \$(wc -l < CODE-SHA256SUMS)"
