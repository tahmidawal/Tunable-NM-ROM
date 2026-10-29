"""Stage one explicitly released immutable package and submit once; never retry."""
from pathlib import Path
import argparse,datetime,hashlib,json,re,shlex,shutil,subprocess
from source_inputs import SourceInputs
B=Path(__file__).resolve().parents[1];REPO=B.parents[1];PREFIX='/cluster/tufts/paralab/tawal01/ns-obstacle-2d-20261027/'
def sha(data):return hashlib.sha256(data).hexdigest()
def dump(path,data):path.write_text(json.dumps(data,indent=2)+'\n')
def compute_only_bounds(candidate,commit,candidate_sha,release,committed,helper_sha,other):
    if type(other)is not int or other<0:raise ValueError('nonnegative concurrent reservation required')
    if release.get('schema')!='ns-obstacle-compute-only-stage-release-v1'or release.get('compute_only')is not True:raise ValueError('explicit compute-only operational release required')
    if release.get('source_commit')!=commit or candidate_sha not in release.get('candidate_sha256s',[])or release.get('stage_helper_sha256')!=helper_sha:raise ValueError('operational release source/package binding differs')
    reserve=release.get('minimum_local_reserve_bytes')
    if type(reserve)is not int or reserve<10*1024**3:raise ValueError('untouched reserve cannot be reduced')
    archives={}
    for row in candidate['files']:
        declaration=row.get('archive_source')
        if not declaration:continue
        raw=committed(declaration['manifest_source'])
        if sha(raw)!=declaration['manifest_sha256']:raise ValueError('cold source manifest differs')
        data=json.loads(raw);size=data['archive_bytes']
        if type(size)is not int or size<=0:raise ValueError('invalid cold archive size')
        archives[declaration['manifest_sha256']]=size
    inputs=sum(row['bytes']for row in candidate['files']);margin=256*1024**2
    return {'cold_source_archive_bytes':sum(archives.values()),'staged_input_bytes':inputs,'other_active_reservations_bytes':other,'metadata_margin_bytes':margin,'minimum_local_reserve_bytes':reserve,'required_before_staging_bytes':reserve+other+sum(archives.values())+inputs+margin,'required_after_staging_bytes':reserve+other+margin}

def main():
    p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--candidate',required=True);p.add_argument('--candidate-sha',required=True);p.add_argument('--attempt',required=True);p.add_argument('--release',required=True);p.add_argument('--batch',required=True);p.add_argument('--compute-only-release',type=Path);p.add_argument('--other-reservations-bytes',type=int,default=0);a=p.parse_args()
    if a.other_reservations_bytes<0:raise ValueError('reservations cannot be negative')
    if not re.fullmatch(r'g2-[a-z0-9-]+',a.attempt):raise ValueError('explicit lane attempt required')
    git=['git','-C',str(REPO)];commit=subprocess.check_output(git+['rev-parse',a.commit],text=True).strip();S=B/'staging'/a.attempt;P=S/'payload';remote=PREFIX+a.attempt
    def committed(relative):return subprocess.check_output(git+['show',commit+':'+str((B/relative).relative_to(REPO))])
    raw=committed(a.candidate)
    if sha(raw)!=a.candidate_sha:raise ValueError('released candidate hash mismatch')
    candidate=json.loads(raw)
    bounds=candidate.get('bounds',{});required_free=max(22*1024**3,int((bounds.get('minimum_local_reserve_GiB',10)+bounds.get('local_collection_planning_GiB',0))*1024**3)+int(bounds.get('other_local_reservations_bytes',0)))
    operational=None
    if a.compute_only_release:
        raw_release=a.compute_only_release.read_bytes();operational=compute_only_bounds(candidate,commit,a.candidate_sha,json.loads(raw_release),committed,sha(Path(__file__).read_bytes()),a.other_reservations_bytes)
        operational['release_sha256']=sha(raw_release)
        if shutil.disk_usage(B).free<=operational['required_before_staging_bytes']:raise MemoryError('compute-only staging plus cold recovery/concurrent reserve does not fit')
        required_free=operational['required_after_staging_bytes']
    elif a.other_reservations_bytes:
        required_free+=a.other_reservations_bytes
    P.mkdir(parents=True,exist_ok=False)
    with SourceInputs(REPO,str(B.relative_to(REPO)),commit) as sources:
        sources.materialize(candidate['files'],P)
    (P/'SOURCE_COMMIT.txt').write_text(commit+'\n');dump(P/'INPUT-MANIFEST.json',{'source_commit':commit,'candidate_sha256':a.candidate_sha,'files':candidate['files'],'release':a.release})
    files=sorted(x for x in P.rglob('*') if x.is_file());(P/'SOURCE-SHA256SUMS').write_text(''.join(sha(x.read_bytes())+'  '+str(x.relative_to(P))+'\n' for x in files))
    free=shutil.disk_usage(B).free
    if free<=required_free:raise MemoryError('local reserve plus working envelope not available')
    metadata=subprocess.run(['/home/tahmid/Dev/.venv/bin/python',str(P/'dependencies.py'),'--validate-metadata-only','--target',remote+'/deps'],capture_output=True,text=True,check=True);(S/'DEPENDENCY-METADATA-PREFLIGHT.json').write_text(metadata.stdout)
    if candidate.get('population_group'):
        population_check=subprocess.run(['/home/tahmid/Dev/.venv/bin/python','-B',str(P/'g2/validate_population_inputs.py'),'--root',str(P),'--plan',str(P/'g2/population01/PLAN.json'),'--group',candidate['population_group'],'--out',str(S/'IMMUTABLE-INPUT-METADATA-PREFLIGHT.json')],capture_output=True,text=True)
        (S/'IMMUTABLE-INPUT-METADATA-PREFLIGHT.stdout').write_text(population_check.stdout);(S/'IMMUTABLE-INPUT-METADATA-PREFLIGHT.stderr').write_text(population_check.stderr)
        population_check.check_returncode()
    record={'attempt':a.attempt,'source_commit':commit,'remote_directory':remote,'manifest_sha256':sha((P/'INPUT-MANIFEST.json').read_bytes()),'source_checksums_sha256':sha((P/'SOURCE-SHA256SUMS').read_bytes()),'candidate_sha256':a.candidate_sha,'expected_regular_files':len(files)+1,'local_free_bytes':free,'required_local_free_bytes':required_free,'dependency_metadata_preflight_passed':True,'submitted':False,'staging_helper_sha256':sha(Path(__file__).read_bytes()),'operational_compute_only_reservation':operational};dump(S/'STAGING.json',record)
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
    if record['pre_submit_local_free_bytes']<=required_free:raise MemoryError('local reserve changed during staging; no submission')
    queue(S/'PRE-SUBMIT-QUEUE.txt')
    command='cd '+shlex.quote(remote)+' && sha256sum -c SOURCE-SHA256SUMS >/dev/null && sbatch --parsable --gres=gpu:h200:1 --chdir='+shlex.quote(remote)+' '+shlex.quote(a.batch)
    result=remote_call(command);receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':command,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'source_commit':commit,'remote_directory':remote,'single_submission':True,'release':a.release};dump(S/'SUBMISSION.json',receipt);print(json.dumps(receipt),flush=True)
    if result.returncode:raise RuntimeError('submission failed or uncertain; receipt retained, no retry')
    job=int(result.stdout.strip().split(';')[0]);record.update(submitted=True,job_id=job,submitted_utc=receipt['utc'],chosen_GPU_type='h200',all_runtime_hashes_verified=True);dump(S/'STAGING.json',record)
    post=remote_call('squeue -u tawal01 -o "%.18i %.35j %.12T %.10M"');(S/'POST-SUBMIT-QUEUE.txt').write_text(post.stdout+post.stderr);print(post.stdout,flush=True)
if __name__=='__main__':main()
