"""Checksum collect a finished exact attempt. Removal is a separate explicit step."""
import argparse
import json
import shlex
import subprocess
from pathlib import Path

EXP=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('attempt');a=p.parse_args()
run=EXP/'runs'/a.attempt
s=json.loads((run/'submission.json').read_text());remote=s['remote'];job=s['job_id']
queued=subprocess.check_output(['ssh','tufts-login',shlex.join(['squeue','-h','-j',job,'-o','%i'])],text=True).strip()
assert not queued,'Job remains queued/running'
target=run/'collected';target.mkdir(exist_ok=False)
subprocess.run(['scp','-qr','tufts-login:'+remote+'/.',str(target)],check=True)
manifest=target/'OUTPUTS.sha256'
if not manifest.exists():
    print('Partial job: generating a remote checksum manifest before acceptance',flush=True)
    cmd='cd '+shlex.quote(remote)+" && find code out -type f ! -path '*/__pycache__/*' -print0 | sort -z | xargs -0 sha256sum"
    manifest.write_bytes(subprocess.check_output(['ssh','tufts-login',cmd]))
subprocess.run(['sha256sum','--quiet','-c','OUTPUTS.sha256'],cwd=target,check=True)
print(json.dumps(dict(collected=str(target),checksums_passed=True,remote_removal_pending=remote),indent=2))
