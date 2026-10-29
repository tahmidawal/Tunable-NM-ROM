"""Stage one explicitly released immutable package and submit once; never retry."""
from pathlib import Path
import argparse,datetime,hashlib,json,re,shlex,shutil,subprocess
from source_inputs import SourceInputs
B=Path(__file__).resolve().parents[1];REPO=B.parents[1];PREFIX='/cluster/tufts/paralab/tawal01/ns-obstacle-2d-20261027/'
def sha(data):return hashlib.sha256(data).hexdigest()
def dump(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--candidate',required=True);p.add_argument('--candidate-sha',required=True);p.add_argument('--attempt',required=True);p.add_argument('--release',required=True);p.add_argument('--batch',required=True);a=p.parse_args()
    if not re.fullmatch(r'g2-[a-z0-9-]+',a.attempt):raise ValueError('explicit lane attempt required')
    git=['git','-C',str(REPO)];commit=subprocess.check_output(git+['rev-parse',a.commit],text=True).strip();S=B/'staging'/a.attempt;P=S/'payload';P.mkdir(parents=True,exist_ok=False);remote=PREFIX+a.attempt
    def committed(relative):return subprocess.check_output(git+['show',commit+':'+str((B/relative).relative_to(REPO))])
    raw=committed(a.candidate)
    if sha(raw)!=a.candidate_sha:raise ValueError('released candidate hash mismatch')
    candidate=json.loads(raw)
    with SourceInputs(REPO,str(B.relative_to(REPO)),commit) as sources:
        for row in candidate['files']:
            data=sources.read(row);destination=P/row['destination']
            if not destination.resolve().is_relative_to(P.resolve()) or sha(data)!=row['sha256']:raise ValueError('input path/hash mismatch')
            destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data)
    (P/'SOURCE_COMMIT.txt').write_text(commit+'\n');dump(P/'INPUT-MANIFEST.json',{'source_commit':commit,'candidate_sha256':a.candidate_sha,'files':candidate['files'],'release':a.release})
    files=sorted(x for x in P.rglob('*') if x.is_file());(P/'SOURCE-SHA256SUMS').write_text(''.join(sha(x.read_bytes())+'  '+str(x.relative_to(P))+'\n' for x in files))
    free=shutil.disk_usage(B).free
    if free<=22*1024**3:raise MemoryError('local reserve plus working envelope not available')
    metadata=subprocess.run(['/home/tahmid/Dev/.venv/bin/python',str(P/'dependencies.py'),'--validate-metadata-only','--target',remote+'/deps'],capture_output=True,text=True,check=True);(S/'DEPENDENCY-METADATA-PREFLIGHT.json').write_text(metadata.stdout)
    record={'attempt':a.attempt,'source_commit':commit,'remote_directory':remote,'manifest_sha256':sha((P/'INPUT-MANIFEST.json').read_bytes()),'source_checksums_sha256':sha((P/'SOURCE-SHA256SUMS').read_bytes()),'candidate_sha256':a.candidate_sha,'expected_regular_files':len(files)+1,'local_free_bytes':free,'dependency_metadata_preflight_passed':True,'submitted':False,'staging_helper_sha256':sha(Path(__file__).read_bytes())};dump(S/'STAGING.json',record)
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','tufts-login']
    def remote_call(command):return subprocess.run(ssh+[command],capture_output=True,text=True)
    def queue(path):
        result=remote_call('squeue -u tawal01 -h -o "%i %j %T"');path.write_text(result.stdout+result.stderr)
        if result.returncode or len([line for line in result.stdout.splitlines() if 'nsobs' in line])>=2:raise RuntimeError('queue unavailable or both campaign slots occupied')
    queue(S/'PRE-STAGING-QUEUE.txt')
    space=remote_call('df -B1 --output=avail '+shlex.quote(PREFIX));(S/'REMOTE-DISK.txt').write_text(space.stdout+space.stderr)
    if space.returncode or int(space.stdout.splitlines()[-1])<=10*1024**3:raise RuntimeError('remote disk reserve unavailable')
    create=remote_call('test ! -e '+shlex.quote(remote)+' && mkdir '+shlex.quote(remote));(S/'REMOTE-CREATE.txt').write_text(create.stdout+create.stderr)
    if create.returncode:raise RuntimeError('fresh directory creation failed or uncertain; inspect, never retry blindly')
    subprocess.run(['scp','-r','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',str(P)+'/.','tufts-login:'+remote+'/'],check=True)
    verified=remote_call('cd '+shlex.quote(remote)+' && sha256sum -c SOURCE-SHA256SUMS && find . -type f | wc -l');(S/'REMOTE-VERIFICATION.txt').write_text(verified.stdout+verified.stderr)
    if verified.returncode or int(verified.stdout.splitlines()[-1])!=record['expected_regular_files']:raise ValueError('remote source verification failed')
    record['pre_submit_local_free_bytes']=shutil.disk_usage(B).free;dump(S/'STAGING.json',record)
    if record['pre_submit_local_free_bytes']<=22*1024**3:raise MemoryError('local reserve changed during staging; no submission')
    queue(S/'PRE-SUBMIT-QUEUE.txt')
    command='cd '+shlex.quote(remote)+' && sha256sum -c SOURCE-SHA256SUMS >/dev/null && sbatch --parsable --gres=gpu:h200:1 --chdir='+shlex.quote(remote)+' '+shlex.quote(a.batch)
    result=remote_call(command);receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':command,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'source_commit':commit,'remote_directory':remote,'single_submission':True,'release':a.release};dump(S/'SUBMISSION.json',receipt);print(json.dumps(receipt),flush=True)
    if result.returncode:raise RuntimeError('submission failed or uncertain; receipt retained, no retry')
    job=int(result.stdout.strip().split(';')[0]);record.update(submitted=True,job_id=job,submitted_utc=receipt['utc'],chosen_GPU_type='h200',all_runtime_hashes_verified=True);dump(S/'STAGING.json',record)
    post=remote_call('squeue -u tawal01 -o "%.18i %.35j %.12T %.10M"');(S/'POST-SUBMIT-QUEUE.txt').write_text(post.stdout+post.stderr);print(post.stdout,flush=True)
if __name__=='__main__':main()
