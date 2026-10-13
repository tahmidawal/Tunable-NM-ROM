#!/bin/bash
# jcp-references: pull every finished staged attempt (remote OUTPUTS.sha256 present, job no longer queued) that has not
# been pulled yet; non-sealed: verify and delete the remote job directory; sealed: publish into the sealed store.
cd "$(dirname "$0")/.."
Q=$(ssh tufts-login 'squeue -u $USER -h -o "%j"')
for d in runs/*/; do
  A=$(basename "$d"); [ -f "$d/run.sbatch" ] || continue
  [ -f "$d/output/result.json" ] && continue
  echo "$Q" | grep -qx "jr_$A" && continue
  ssh tufts-login "test -f /cluster/tufts/paralab/tawal01/jcp-ref/$A/OUTPUTS.sha256" || continue
  bash cluster/pull.sh "$A" || { echo "PULL FAILED $A"; continue; }
  if grep -q '"sealed_manifest"' "$d/output/result.json"; then
    bash cluster/publish_sealed.sh "$A" && echo "HARVESTED sealed $A"
  else
    ssh tufts-login "rm -rf /cluster/tufts/paralab/tawal01/jcp-ref/$A" && echo "HARVESTED $A"
  fi
done
