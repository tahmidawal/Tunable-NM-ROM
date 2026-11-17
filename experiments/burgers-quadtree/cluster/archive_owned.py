"""Move only this completed campaign's checksummed jobs into its new owned archive."""
from pathlib import Path
import json,subprocess,shlex
H=Path(__file__).resolve().parents[1];p=H/'cluster/ledger.json';rows=json.loads(p.read_text())
ns='/cluster/tufts/paralab/tawal01/burgers-quadtree-20261117'
q=subprocess.check_output(['ssh','tufts-login','squeue -u tawal01 -h -o "%i %j %Z"'],text=True)
assert not any(str(r['id']) in q.split() for r in rows)
for r in rows:
 assert r['state']=='COMPLETED'
 original=r['remote'];archive=ns+'/archive/'+r['name'];assert original==ns+'/'+r['name']
 if r.get('remote_archive'):continue
 local=H/'results'/r['name'];subprocess.run(['sha256sum','-c','OUTPUTS.sha256','--quiet'],cwd=local,check=True)
 cmd=f'cd {shlex.quote(original)} && sha256sum -c OUTPUTS.sha256 --quiet && sha256sum -c MANIFEST.sha256 --quiet && test ! -e {shlex.quote(archive)} && mkdir -p {shlex.quote(ns+"/archive")} && mv -- {shlex.quote(original)} {shlex.quote(archive)}'
 subprocess.run(['ssh','tufts-login',cmd],check=True)
 r['remote_archive']=archive
 p.write_text(json.dumps(rows,indent=2)+'\n')
 man=H/'evidence'/f"{r['name']}-retention.json";m=json.loads(man.read_text());m['original_submission_directory']=original;m['remote']=archive;m['archived_after_local_and_remote_sha256_checks']=True;man.write_text(json.dumps(m,indent=2)+'\n')
 print(r['name'],'archived',flush=True)
