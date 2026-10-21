#!/bin/bash
# usage: submit.sh <jobname> <gpu> <mem> <time> <bodyfile>
# stages the lane code (git HEAD of the worktree) into /cluster/.../imp-geometry/<jobname>/code and submits through the
# campaign lock (<= 8 jobs account-wide). Prints the job id.
set -eu
NAME=$1; GPU=$2; MEM=$3; TIME=$4; BODY=$5
LANE=$(cd "$(dirname "$0")/.." && pwd)
NS=/cluster/tufts/paralab/tawal01/imp-geometry
DIR=$NS/$NAME
COMMIT=$(git -C "$LANE" rev-parse --short HEAD)
timeout 60 ssh -o ConnectTimeout=20 tufts-login "test ! -e $DIR/submitted || { echo 'job dir already used'; exit 3; }; mkdir -p $DIR/code"
tar -C "$LANE" -czf - --exclude=runs --exclude=audits --exclude=__pycache__ . | timeout 300 ssh -o ConnectTimeout=20 tufts-login "tar -C $DIR/code -xzf -"
sed -e "s#__NAME__#$NAME#g; s#__GPU__#$GPU#g; s#__MEM__#$MEM#g; s#__TIME__#$TIME#g; s#__DIR__#$DIR#g; s#__COMMIT__#$COMMIT#g" "$LANE/cluster/job.sbatch.tmpl" > /tmp/igeo-$NAME.sbatch
python3 - "$BODY" /tmp/igeo-$NAME.sbatch <<'PY'
import sys
body=open(sys.argv[1]).read(); f=sys.argv[2]; s=open(f).read().replace('__BODY__', body); open(f,'w').write(s)
PY
cat /tmp/igeo-$NAME.sbatch | timeout 60 ssh -o ConnectTimeout=20 tufts-login "cat > $DIR/run.sbatch"
timeout 60 ssh -o ConnectTimeout=20 tufts-login 'squeue -u tawal01 -h | wc -l' | sed 's/^/before: queued+running=/'
timeout 700 ssh -o ConnectTimeout=20 tufts-login "flock -w 600 /cluster/tufts/paralab/tawal01/.campaign-sbatch.lock bash -c 'n=\$(squeue -u tawal01 -h | wc -l); [ \$n -lt 8 ] || exit 75; cd $DIR && sbatch --parsable run.sbatch > submitted && cat submitted'"
timeout 60 ssh -o ConnectTimeout=20 tufts-login "squeue -u tawal01 -h -o '%i %j %T' ; ls $DIR"
