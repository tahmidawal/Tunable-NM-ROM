#!/usr/bin/env bash
# Preserve private raw outputs, numerical witnesses, accounting, inputs and exact source.
set -euo pipefail
task=$1
jobid=$2
[[ $task =~ ^[a-zA-Z0-9_-]+$ && $jobid =~ ^[0-9]+$ ]]
scratch=/dev/shm/jcp-sealed-closeout/$task
local_ledger=$(cd "$(dirname "$0")" && pwd)/receipts/$task
src=/cluster/tufts/paralab/tawal01/jcp-sealed-closeout-20261115/$task
cold=/cluster/tufts/paralab/tawal01/burgers-archive/jcp-sealed-closeout-20261115/$task
mkdir -p "$scratch" "$local_ledger"
ssh tufts-login "set -eu; test -d '$src'; test -f '$src/$jobid.out'; cd '$src'; sacct -X -j '$jobid' -n -P -o jobid,state,exitcode,elapsed,nodeList > accounting.txt; /usr/bin/grep -q '^$jobid[|]COMPLETED[|]0:0[|]' accounting.txt; mkdir -p -m 700 '$cold'; rsync -a --exclude=cache --exclude=tmp ./ '$cold'/; cd '$cold'; sha256sum -c SOURCE.sha256 --quiet; sha256sum -c OUTPUTS.sha256 --quiet; find . -type f ! -name ARCHIVE.sha256 -print0 | sort -z | xargs -0 sha256sum > ARCHIVE.sha256"
rsync -aq "tufts-login:$cold/" "$scratch/"
(cd "$scratch" && sha256sum --quiet -c ARCHIVE.sha256)
cp "$scratch"/{accounting.txt,ARCHIVE.sha256,OUTPUTS.sha256,SOURCE.sha256} "$scratch/$jobid."{out,err} "$local_ledger/"
cd "$scratch" && sha256sum ARCHIVE.sha256 | ssh tufts-login "cd '$cold' && sha256sum --quiet -c - && sha256sum --quiet -c ARCHIVE.sha256 && rm -rf -- '$src'"
