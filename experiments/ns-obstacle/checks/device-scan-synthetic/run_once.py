"""Single under-one-minute synthetic GPU smoke with retained execution evidence."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import shlex
import subprocess
import time

out = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert shutil.disk_usage(out).free > 10*2**30 + 256*2**20 + 64*2**20
units = subprocess.check_output(['systemctl','--user','list-units','--type=scope','--state=running','--no-legend','--plain'],text=True)
active=[]
for row in units.splitlines():
    unit=row.split()[0]
    limit=subprocess.check_output(['systemctl','--user','show',unit,'--property=MemoryMax','--value'],text=True).strip()
    if limit==str(36*2**30): active.append(unit)
assert len(active)<3
assert not any((out/name).exists() for name in ('CLAIM.json','stdout.txt','stderr.txt','EXECUTION.json','RESULT.json','arrays.npz'))
record=dict(source_sha256=sha(out/'check.py'), wrapper_sha256=sha(Path(__file__)),start_epoch=time.time(),existing_36GiB_scopes=active,automatic_retry=False,scope='synthetic GPU equivalence only')
with (out/'CLAIM.json').open('x') as f:json.dump(record,f,indent=2)
env=dict(os.environ,JAX_ENABLE_X64='True',JAX_DEFAULT_MATMUL_PRECISION='highest',JAXRUN_MAX='36G',JAXRUN_HIGH='30G',PYTHONDONTWRITEBYTECODE='1')
command=['/usr/bin/timeout','--signal=TERM','--kill-after=1s','58s','/home/tahmid/.local/bin/jaxrun','/usr/bin/timeout','--signal=TERM','--kill-after=3s','50s','/home/tahmid/Dev/.venv/bin/python',str(out/'check.py')]
with (out/'stdout.txt').open('x') as so,(out/'stderr.txt').open('x') as se:
    run=subprocess.run(['/bin/bash','-c','source /etc/profile.d/jax-mem.sh\nexec '+shlex.join(command)],env=env,stdout=so,stderr=se)
record.update(returncode=run.returncode,finish_epoch=time.time(),stdout_sha256=sha(out/'stdout.txt'),stderr_sha256=sha(out/'stderr.txt'))
(out/'EXECUTION.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
run.check_returncode()
