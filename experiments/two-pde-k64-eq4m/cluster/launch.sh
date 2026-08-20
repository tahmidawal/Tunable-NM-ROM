#!/usr/bin/env bash
# Stage directly to paralab, verify checksums, and submit exactly one job.
set -euo pipefail

cell="ctol_k64_eq4m_r1"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
stage="$here/stage/$cell"
remote="/cluster/tufts/paralab/tawal01/k64-eq4m/$cell"
[[ -d "$stage" ]] || { echo "missing stage: $stage" >&2; exit 2; }

echo "queue before submission:"
ssh tufts-login "squeue -u tawal01 -o '%.18i %.28j %.10T %.10M %.20R'"
queued="$(ssh tufts-login "squeue -u tawal01 -h -o '%j' | grep -cx '$cell' || true")"
[[ "$queued" == "0" ]] || { echo "$queued existing jobs named $cell" >&2; exit 3; }
exists="$(ssh tufts-login "if [ -e '$remote' ]; then echo yes; else echo no; fi")"
[[ "$exists" == "no" ]] || { echo "remote job directory already exists: $remote" >&2; exit 4; }

ssh tufts-login "mkdir -p '/cluster/tufts/paralab/tawal01/k64-eq4m' && mkdir '$remote'"
scp -q -r "$stage"/. "tufts-login:$remote/"
local_sum="$(cd "$stage" && sha256sum MANIFEST.sha256 | cut -d' ' -f1)"
remote_sum="$(ssh tufts-login "cd '$remote' && sha256sum MANIFEST.sha256 | cut -d' ' -f1")"
[[ "$local_sum" == "$remote_sum" ]] || { echo "manifest transfer mismatch" >&2; exit 5; }
ssh tufts-login "cd '$remote' && sha256sum -c MANIFEST.sha256 >/dev/null"

jid="$(ssh tufts-login "cd '$remote' && sbatch --parsable run.sbatch")"
echo "queue after submission:"
ssh tufts-login "squeue -u tawal01 -o '%.18i %.28j %.10T %.10M %.20R'"
after="$(ssh tufts-login "squeue -u tawal01 -h -j '$jid' -o '%j' | grep -cx '$cell' || true")"
[[ "$after" == "1" ]] || { echo "submitted job $jid is not queued as $cell" >&2; exit 6; }
echo "job_id=$jid remote=$remote manifest_sha256=$local_sum"
