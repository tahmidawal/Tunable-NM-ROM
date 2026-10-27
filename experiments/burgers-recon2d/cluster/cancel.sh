#!/usr/bin/env bash
# Narrow adaptation of cost-to-tolerance/cluster/cancel.sh for one invalid own job.
# Numeric ID, shared-account ownership, exact job name AND work directory required.
# No name/user selectors are ever passed to scancel.
set -euo pipefail
[[ $# -eq 1 ]] || { echo 'usage: cancel.sh 6568278 (one numeric ID only)' >&2; exit 1; }
job_id="$1"
[[ "$job_id" =~ ^[0-9]+$ && "$job_id" == 6568278 ]] || { echo 'REFUSING unexpected job ID' >&2; exit 2; }
ssh tufts-login "squeue -u tawal01 -o '%.18i %.26j %.9T %.10M'"
owned=$(ssh tufts-login "squeue -u tawal01 -h -o '%i %j %u'")
row=$(awk -v id="$job_id" '$1==id {print $0}' <<<"$owned")
read -r actual_id actual_name actual_user <<<"$row"
[[ "$actual_id" == 6568278 && "$actual_name" == b2dr_screen && "$actual_user" == tawal01 ]] || { echo 'REFUSING: exact account/job/name tuple not in queue' >&2; exit 3; }
info=$(ssh tufts-login "scontrol show job $job_id -o")
work_dir=$(awk '{for(i=1;i<=NF;i++)if($i~/^WorkDir=/){sub(/^WorkDir=/,"",$i);print $i}}' <<<"$info")
[[ "$work_dir" == /cluster/tufts/paralab/tawal01/b2d-recon-20261027/dev01 ]] || { echo 'REFUSING unexpected job work directory' >&2; exit 4; }
[[ " $info " == *' JobId=6568278 '* && " $info " == *' JobName=b2dr_screen '* && " $info " == *' UserId=tawal01('* ]] || { echo 'REFUSING unexpected scontrol identity' >&2; exit 5; }
echo "Verified own invalid development job $job_id at $work_dir; canceling only this numeric ID."
ssh tufts-login "scancel -- $job_id"
sleep 3
ssh tufts-login "squeue -u tawal01 -o '%.18i %.26j %.9T %.10M'"
