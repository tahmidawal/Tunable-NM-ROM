"""One-shot exact archived-job cleanup after independent audit and durable mirror."""
from pathlib import Path
import datetime,hashlib,json,shlex,subprocess
B=Path(__file__).resolve().parents[1];REPO=B.parents[1];prefix='/cluster/tufts/paralab/tawal01/ns-obstacle-2d-20261027/'
mirrors={'g2-solverquad01':'20b812ae3','g2-domain-upstream':'aef32dde0','g2-domain-downstream':'aef32dde0','g2-domain-symmetric':'aef32dde0','g2-export-batch01':'caee29610'}
for name,mirror in mirrors.items():
 folder=B/'artifacts'/name;retention=json.loads((folder/'RETENTION.json').read_text());receipt=B/'staging'/name/'REMOTE-CLEANUP.json'
 if receipt.exists():raise FileExistsError('inspect authoritative cleanup receipt; do not retry blindly: '+name)
 for chunk in retention['chunks']:
  path=folder/chunk['path'];relative=str(path.relative_to(REPO));blob=subprocess.check_output(['git','-c','gc.auto=0','-C',str(REPO),'rev-parse',mirror+':'+relative],text=True).strip()
  with path.open('rb') as stream:
   if hashlib.file_digest(stream,'sha256').hexdigest()!=chunk['sha256']:raise ValueError('local retained chunk changed')
  local_blob=subprocess.check_output(['git','-c','gc.auto=0','-C',str(REPO),'hash-object',str(path)],text=True).strip()
  if local_blob!=blob:raise ValueError('durable mirror chunk differs')
 directory=prefix+name;archive=directory+'-collection.tar.gz';expected=retention['archive_sha256']
 record={'requested_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':retention['job_id'],'archive_sha256':expected,'durable_mirror':mirror,'exact_directory':directory,'exact_archive':archive,'status':'submitted_unknown_until_receipt'};receipt.write_text(json.dumps(record,indent=2)+'\n')
 command='set -eu; test -d '+shlex.quote(directory)+'; test -f '+shlex.quote(archive)+'; actual=$(sha256sum '+shlex.quote(archive)+'); test "${actual%% *}" = '+shlex.quote(expected)+'; rm -rf -- '+shlex.quote(directory)+' '+shlex.quote(archive)+'; test ! -e '+shlex.quote(directory)+'; test ! -e '+shlex.quote(archive)+'; date -u +%FT%TZ'
 result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','tufts-login',command],capture_output=True,text=True)
 record.update(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,status='checksum_verified_removed' if result.returncode==0 else 'uncertain_inspect_before_any_retry');receipt.write_text(json.dumps(record,indent=2)+'\n')
 if result.returncode:raise RuntimeError('remote cleanup uncertain: '+name)
 retention.update(remote_cleaned=True,remote_cleanup_receipt=str(receipt.relative_to(B)));(folder/'RETENTION.json').write_text(json.dumps(retention,indent=2)+'\n');print(name,'removed',flush=True)
