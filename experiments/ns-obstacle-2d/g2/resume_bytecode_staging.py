"""One-time resume of the rejected pre-submit Re20-02 bytecode-only staging."""
from pathlib import Path
import datetime,hashlib,json,shlex,shutil,subprocess
B=Path(__file__).resolve().parents[1];S=B/'staging/g2-population-re20-02';P=S/'payload'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,r):p.write_text(json.dumps(r,indent=2)+'\n')
r=json.loads((S/'STAGING.json').read_text());remote=r['remote_directory']
if r['submitted']or(S/'SUBMISSION.json').exists():raise ValueError('submission already exists; refuse repeat')
ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','tufts-login']
def call(command,name):
 p=subprocess.run(ssh+[command],capture_output=True,text=True);(S/name).write_text(p.stdout+p.stderr);p.check_returncode();return p.stdout
expected={line.split('  ',1)[1]for line in(P/'SOURCE-SHA256SUMS').read_text().splitlines()}|{'SOURCE-SHA256SUMS'}
actual={str(p.relative_to(P))for p in P.rglob('*')if p.is_file()};extras=actual-expected
if len(extras)!=7 or expected-actual or any('__pycache__'not in Path(p).parts or not p.endswith('.cpython-312.pyc')for p in extras):raise ValueError('unexpected local stage changes')
remote_files=set(call('cd '+shlex.quote(remote)+' && find . -type f -printf "%P\\n"','RESUME-REMOTE-FILE-INVENTORY.txt').splitlines())
if remote_files!=actual:raise ValueError('remote file inventory differs')
extra_hashes={p:sha(P/p)for p in sorted(extras)}
text=call('cd '+shlex.quote(remote)+' && sha256sum '+ ' '.join(shlex.quote(p)for p in sorted(extras)),'RESUME-BYTECODE-HASHES.txt')
if {line.split('  ',1)[1]:line.split('  ',1)[0]for line in text.splitlines()}!=extra_hashes:raise ValueError('remote extra hashes differ')
dump(S/'PRE-SUBMIT-BYTECODE-REJECTION.json',{'failure':'exact file count173 versus166 after successful source hashes','submitted':False,'PDE_solves':0,'identified_generated_extras':extra_hashes,'source_inputs_changed':False})
call('cd '+shlex.quote(remote)+' && rm -- '+' '.join(shlex.quote(p)for p in sorted(extras)),'RESUME-REMOVE-BYTECODE.txt')
for path in extras:(P/path).unlink()
p=subprocess.run(['/home/tahmid/Dev/.venv/bin/python','-B',str(P/'g2/validate_population_inputs.py'),'--root',str(P),'--plan',str(P/'g2/population01/PLAN.json'),'--group','re20','--out',str(S/'RESUME-IMMUTABLE-INPUT-PREFLIGHT.json')],capture_output=True,text=True);(S/'RESUME-IMMUTABLE-INPUT-PREFLIGHT.log').write_text(p.stdout+p.stderr);p.check_returncode()
if {str(p.relative_to(P))for p in P.rglob('*')if p.is_file()}!=expected:raise ValueError('preflight changed staged files')
verified=call('cd '+shlex.quote(remote)+' && sha256sum -c SOURCE-SHA256SUMS && find . -type f | wc -l','RESUME-REMOTE-VERIFICATION.txt')
if int(verified.splitlines()[-1])!=r['expected_regular_files']:raise ValueError('remote count differs')
queue=call('squeue -u tawal01 -h -o "%i %j %T"','RESUME-PRE-SUBMIT-QUEUE.txt')
if len([x for x in queue.splitlines()if'nsobs'in x])>=2:raise RuntimeError('both slots occupied')
free=shutil.disk_usage(B).free
if free<=r['required_local_free_bytes']:raise MemoryError('post-stage local reserve')
space=call('df -B1 --output=avail '+shlex.quote(remote),'RESUME-REMOTE-DISK.txt')
if int(space.splitlines()[-1])<=10*1024**3:raise MemoryError('remote reserve')
r.update(resume_helper_sha256=sha(Path(__file__)),pre_submit_local_free_bytes=free,staged_actual_reader_revalidated=True,generated_bytecode_removed=extra_hashes);dump(S/'STAGING.json',r)
command='cd '+shlex.quote(remote)+' && sha256sum -c SOURCE-SHA256SUMS >/dev/null && sbatch --parsable --gres=gpu:h200:1 --chdir='+shlex.quote(remote)+' g2/population-re20-02/run.sh'
p=subprocess.run(ssh+[command],capture_output=True,text=True);receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':command,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'source_commit':r['source_commit'],'remote_directory':remote,'single_submission':True,'release':'coordinator d931edeeb reviews/RELEASE-POPULATION-RE20-02.json'};dump(S/'SUBMISSION.json',receipt);print(json.dumps(receipt),flush=True)
if p.returncode:raise RuntimeError('submission failed or uncertain; never retry')
r.update(submitted=True,job_id=int(p.stdout.strip().split(';')[0]),submitted_utc=receipt['utc'],chosen_GPU_type='h200',all_runtime_hashes_verified=True);dump(S/'STAGING.json',r)
print(call('squeue -u tawal01 -o "%.18i %.35j %.12T %.10M"','POST-SUBMIT-QUEUE.txt'),flush=True)
