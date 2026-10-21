#!/usr/bin/env bash
# Move a finished job's local raw archive to the campaign archive on the paralab share, verified.
# Usage: offload.sh <lane> <job> <local_archive_dir>
# Copies with rsync, compares a full sha256 listing local vs remote, then deletes the local copy and
# leaves ARCHIVE-LOCATION.txt (remote path + listing hash). Aborts without deleting on any mismatch.
set -euo pipefail
export LC_ALL=C
lane=$1; job=$2; src=${3%/}
R=/cluster/tufts/paralab/tawal01/burgers-archive/$lane/$job
[ -d "$src" ] || { echo "no dir $src"; exit 1; }
timeout 60 ssh -o ConnectTimeout=20 tufts-login "mkdir -p $R"
timeout 7200 rsync -a --partial "$src/" "tufts-login:$R/"
L=$(cd "$src" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum | cut -d' ' -f1)
RM=$(timeout 3600 ssh -o ConnectTimeout=20 tufts-login "cd $R && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum" | cut -d' ' -f1)
[ "$L" = "$RM" ] || { echo "MISMATCH $src local=$L remote=$RM — not deleting"; exit 2; }
(cd "$src" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > "$src.sha256-listing"
rm -rf "$src"; mkdir -p "$src"
printf 'Offloaded %s by the campaign coordinator.\nremote: tufts-login:%s\nlisting sha256: %s\nper-file listing: %s.sha256-listing\n' "$(date -u +%FT%TZ)" "$R" "$L" "$(basename "$src")" > "$src/ARCHIVE-LOCATION.txt"
echo "offloaded $src -> $R ($L)"
