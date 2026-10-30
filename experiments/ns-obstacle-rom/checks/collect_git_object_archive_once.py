"""Prospective Git-object-first collector. Requires separate exact release.

Never run against campaign data before independent review of this new protocol.
Old working-chunk collectors remain unchanged. Failure preserves remote originals,
completed loose objects and any already-created source/mirror commits.
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,tempfile,resource,signal
from retention.git_object_archive import git,ingest,verify_archive,verify_complete_stream,commit_virtual_chunks,storage_bound,tree_allocation_bound,CHUNK
from retention.build_mirror_index import updates

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def textgit(repo,args,**kw):return git(repo,args,stdout=subprocess.PIPE,**kw).stdout.decode().strip()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--release',type=Path,required=True);ap.add_argument('--release-sha256',required=True);a=ap.parse_args()
    B=Path(__file__).resolve().parents[1];repo=B.parents[1]
    if sha(a.release)!=a.release_sha256:raise ValueError('release hash differs')
    r=json.loads(a.release.read_text());measurement=Path(r['measurement_path']);m=json.loads(measurement.read_text())
    expected={'schema':'ns-obstacle-git-object-first-collection-release-v1','authorized':True,'single_execution':True,'automatic_retry':False,'collector_sha256':sha(__file__),'archive_helper_sha256':sha(B/'retention/git_object_archive.py'),'mirror_helper_sha256':sha(B/'retention/build_mirror_index.py'),'chunk_bytes':CHUNK,'transport_timeout_seconds':300,'transport_kill_grace_seconds':5,'measurement_sha256':sha(measurement),'archive_sha256':m['archive_sha256'],'archive_bytes':m['archive_bytes'],'job_id':m['job_id'],'regular_files':m['regular_files'],'ledger_sha256':m['ledger_sha256']}
    if not all(type(r.get(k)) is type(v) and r[k]==v for k,v in expected.items()):raise ValueError('typed release contract differs')
    if type(r['metadata_limit_bytes']) is not int or m['metadata_extraction_bytes']>r['metadata_limit_bytes']:raise ValueError('metadata cap differs')
    terminal=[x.split('|') for x in m['terminal_status'].splitlines() if x.split('|')[0]==str(m['job_id'])]
    if len(terminal)!=1 or terminal[0][1]!='COMPLETED' or terminal[0][3]!='0:0':raise ValueError('successful authoritative terminal required')
    branch=textgit(repo,['symbolic-ref','HEAD']);parent=textgit(repo,['rev-parse',branch])
    if parent!=r['expected_parent_commit'] or textgit(repo,['status','--porcelain']):raise ValueError('frozen clean lane required')
    for n in ['checks/collect_git_object_archive_once.py','retention/git_object_archive.py','retention/build_mirror_index.py']:
        raw=git(repo,['show',r['source_commit']+':'+str((B/n).relative_to(repo))],stdout=subprocess.PIPE).stdout
        if hashlib.sha256(raw).hexdigest()!=sha(B/n):raise ValueError('source history differs')
    mirror_ref=r['mirror_ref'];oldmirror=r['expected_mirror_commit']
    if mirror_ref!='refs/heads/codeonly/'+branch.removeprefix('refs/heads/'):raise ValueError('wrong lane mirror')
    advertised=subprocess.check_output(['git','-C',str(repo),'ls-remote','origin',mirror_ref],text=True).split()
    if not advertised or advertised[0]!=oldmirror:raise ValueError('advertised mirror moved; no fetch permitted')
    index=Path(textgit(repo,['rev-parse','--git-path','index']));index=index if index.is_absolute() else repo/index
    index_size=index.stat().st_size;initial_index_sha=sha(index)
    # Exact remote metadata files plus a bounded generated-receipt allowance.
    metadata=list(m['metadata_files'])+[{'path':'owner/'+n,'bytes':size} for n,size in [('ORIGINAL-FILE-INVENTORY.json',256*2**10),('RETENTION.json',128*2**10),('COLD-RECOVERY.json',128*2**10),('ATTEMPT.json',256*2**10),('MEASUREMENT.json',128*2**10),('RELEASE.json',128*2**10),('SUCCESS.json',128*2**10),('TRANSPORT.stderr.txt',2**20),('ATTEMPT-PREVIOUS-VERSION.json',256*2**10)]]
    artifact_preview=Path(r['artifact_relative_path'])
    prospective_paths=[str(artifact_preview/'metadata'/x['path']) for x in m['metadata_files']]+[str(artifact_preview/x['path'].split('/')[-1]) for x in metadata[len(m['metadata_files']):]]+[str(artifact_preview/'archive-chunks'/('collection.tar.gz.part-%03d'%i)) for i in range((m['archive_bytes']+CHUNK-1)//CHUNK)]
    if len(prospective_paths)>512 or sum(len(x.encode())+256 for x in prospective_paths)>2**20:raise ValueError('prospective index growth cap exceeded')
    trees=3*(tree_allocation_bound(repo,parent,prospective_paths)+tree_allocation_bound(repo,oldmirror,prospective_paths))+6*os.statvfs(repo).f_frsize
    if trees>4*2**20:raise ValueError('all source/mirror/cold tree+commit allocations exceed4MiB cap')
    budget=storage_bound(m['archive_bytes'],metadata,index_size,os.statvfs(repo).f_frsize)
    refs_bound=2*os.statvfs(repo).f_frsize*((m['archive_bytes']+CHUNK-1)//CHUNK)+64*os.statvfs(repo).f_frsize
    budget['recovery_refs_and_reflogs_bound']=refs_bound;budget['required_free_bytes']+=refs_bound
    if r['required_local_free_bytes']!=budget['required_free_bytes'] or shutil.disk_usage(repo).free<budget['required_free_bytes']:raise ValueError('fresh exact storage guard failed')
    artifact=Path(r['artifact_relative_path']);out=repo/artifact
    if artifact.is_absolute() or '..' in artifact.parts or not str(artifact).startswith(str(B.relative_to(repo))+'/artifacts/'):raise ValueError('artifact outside lane')
    out.mkdir(exist_ok=False);state={'phase':'claimed','budget':budget,'source_parent':parent,'mirror_parent':oldmirror,'release_sha256':a.release_sha256,'remote_originals_preserved':True,'automatic_retry':False}
    def record(update):
        for i,row in enumerate(update.get('chunks',[])):
            ref='refs/retention/'+out.name+'/chunk-%03d'%i
            q=subprocess.run(['git','-C',str(repo),'rev-parse','--verify',ref],capture_output=True,text=True)
            if q.returncode==0:
                if q.stdout.strip()!=row['oid']:raise ValueError('retention ref collision')
            else:git(repo,['update-ref',ref,row['oid'],'0'*40])
        state.update(update);payload=json.dumps(state,indent=2)+'\n'
        if len(payload.encode())>256*2**10:raise ValueError('attempt receipt cap exceeded')
        (out/'ATTEMPT.json').write_text(payload)
    caps={x['path'].split('/')[-1]:x['bytes'] for x in metadata[len(m['metadata_files']):]}
    def check_generated():
        for f in out.iterdir():
            if f.is_file() and (f.name not in caps or f.stat().st_size>caps[f.name]):raise ValueError('generated per-file cap exceeded: '+f.name)
    record({});shutil.copyfile(measurement,out/'MEASUREMENT.json');shutil.copyfile(a.release,out/'RELEASE.json')
    try:
        # Remote command only reads the exact released regular archive; no wildcard.
        remote=m['archive_path']
        if not remote.startswith('/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./-' for c in remote):raise ValueError('unsafe archive path')
        def limit_stderr():resource.setrlimit(resource.RLIMIT_FSIZE,(2**20,2**20))
        with (out/'TRANSPORT.stderr.txt').open('xb') as transport_err:
            process=subprocess.Popen(['/usr/bin/timeout','--signal=TERM','--kill-after=5s','300s','ssh','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','tufts-login','cat','--',remote],stdout=subprocess.PIPE,stderr=transport_err,start_new_session=True,preexec_fn=limit_stderr)
            try:chunks=ingest(repo,process.stdout,m['archive_bytes'],m['archive_sha256'],record)
            finally:
                process.stdout.close()
                try:status=process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:status=process.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);status=process.wait()
        err=(out/'TRANSPORT.stderr.txt').read_bytes();record({'transport_returncode':status,'transport_stderr_sha256':sha(out/'TRANSPORT.stderr.txt')})
        if status!=0 or err:raise ValueError('archive transport failed; stderr retained')
        verified=verify_archive(repo,chunks,m,out/'metadata',r['metadata_limit_bytes'])
        (out/'ORIGINAL-FILE-INVENTORY.json').write_text(json.dumps(verified['files'],indent=2)+'\n')
        retention={'schema':'ns-obstacle-git-object-first-retention-v1','status':'all_originals_verified_before_source_and_mirror','job_id':m['job_id'],'archive_bytes':m['archive_bytes'],'archive_sha256':m['archive_sha256'],'original_files':m['regular_files'],'ledger_sha256':m['ledger_sha256'],'chunks':chunks,'working_archive_or_chunks_created':False,'remote_originals_preserved':True,'metadata_extracted_bytes':verified['metadata_extracted_bytes'],'collection_release_sha256':a.release_sha256}
        (out/'RETENTION.json').write_text(json.dumps(retention,indent=2)+'\n');record({'phase':'verified_before_virtual_commit'})
        paths=[str(f.relative_to(repo)) for f in out.rglob('*') if f.is_file()]
        # Bound extra index entries before write, and actual generated files.
        if len(paths)+len(chunks)>512 or sum(len(x.encode())+256 for x in paths)+len(chunks)*512>2**20:raise ValueError('index growth envelope exceeded')
        check_generated()
        source_metadata={n:{'bytes':(repo/n).stat().st_size,'sha256':sha(repo/n)} for n in paths}
        source=commit_virtual_chunks(repo,parent,branch,str(artifact),chunks,paths,'Retain complete archive directly in Git objects',index_size+2**20)
        record({'phase':'source_commit_created','archive_source_commit':source})
        def mirror(source,previous):
            actual=subprocess.check_output(['git','-C',str(repo),'ls-remote','origin',mirror_ref],text=True).split()
            if not actual or actual[0]!=previous:raise ValueError('mirror changed; no fetch/retry')
            with tempfile.TemporaryDirectory(prefix='archive-mirror-index-',dir=repo) as tmp:
                idx=Path(tmp)/'index';env=dict(os.environ,GIT_INDEX_FILE=str(idx))
                git(repo,['read-tree',r['mirror_base_commit']],env=env)
                # Exact existing lane filter; source paths/large-blob policy unchanged.
                old=os.getcwd();os.chdir(repo)
                try:payload,_,skipped=updates(source,r['source_base_commit'],50*2**20)
                finally:os.chdir(old)
                if any(str(artifact).encode() in x for x in skipped):raise ValueError('new archive excluded by mirror')
                git(repo,['update-index','-z','--index-info'],input=payload,env=env)
                if idx.stat().st_size>index_size+2**20:raise ValueError('mirror index cap exceeded')
                tree=textgit(repo,['write-tree'],env=env);commit=textgit(repo,['commit-tree',tree,'-p',previous],input=('mirror of '+source+'\n').encode())
                git(repo,['push','--force-with-lease='+mirror_ref+':'+previous,'origin',commit+':'+mirror_ref])
                actual=subprocess.check_output(['git','-C',str(repo),'ls-remote','origin',mirror_ref],text=True).split()
                if not actual or actual[0]!=commit:raise ValueError('mirror advertisement differs')
                return commit
        def verify_metadata(rev,expected):
            for n,row in expected.items():
                raw=git(repo,['show',rev+':'+n],stdout=subprocess.PIPE).stdout
                if len(raw)!=row['bytes'] or hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('committed metadata differs: '+n)
        mirror1=mirror(source,oldmirror)
        verify_metadata(source,source_metadata);verify_metadata(mirror1,source_metadata)
        for rev in [source,mirror1]:
            recovered=[]
            for i,row in enumerate(chunks):
                name=str(artifact/'archive-chunks'/('collection.tar.gz.part-%03d'%i));oid=textgit(repo,['rev-parse',rev+':'+name])
                if oid!=row['oid']:raise ValueError('source/mirror chunk differs')
                recovered.append(dict(row,oid=oid))
            verify_complete_stream(repo,recovered,m['archive_bytes'],m['archive_sha256'])
        cold={'schema':'git-object-first-cold-recovery-v1','archive_source_commit':source,'advertised_archive_mirror_commit':mirror1,'artifact_relative_path':str(artifact),'archive_sha256':m['archive_sha256'],'archive_bytes':m['archive_bytes'],'chunks':chunks,'all_originals_verified_before_commit':True,'both_histories_full_stream_recovered':True,'working_chunks_never_created':True}
        (out/'COLD-RECOVERY.json').write_text(json.dumps(cold,indent=2)+'\n');record({'phase':'archive_source_and_mirror_recovered','archive_mirror_commit':mirror1})
        check_generated()
        cold_metadata={str(f.relative_to(repo)):{'bytes':f.stat().st_size,'sha256':sha(f)} for f in out.rglob('*') if f.is_file()}
        # Explicit cold lifecycle commit removes only virtual chunk paths from
        # current tree. Immutable source+mirror histories retain every byte.
        with tempfile.TemporaryDirectory(prefix='archive-cold-index-',dir=repo) as tmp:
            idx=Path(tmp)/'index';env=dict(os.environ,GIT_INDEX_FILE=str(idx));git(repo,['read-tree',source],env=env)
            for i in range(len(chunks)):git(repo,['update-index','--force-remove',str(artifact/'archive-chunks'/('collection.tar.gz.part-%03d'%i))],env=env)
            git(repo,['add','-f','--',str(artifact)],env=env)
            if idx.stat().st_size>index_size+2**20:raise ValueError('cold index cap exceeded')
            final=textgit(repo,['commit-tree',textgit(repo,['write-tree'],env=env),'-p',source],input=b'Record verified Git-only archive cold lifecycle\n')
            git(repo,['update-ref',branch,final,source])
        mirror2=mirror(final,mirror1)
        verify_metadata(final,cold_metadata);verify_metadata(mirror2,cold_metadata)
        success={'status':'complete_archive_and_cold_metadata_verified_in_advertised_histories','archive_source':source,'archive_mirror':mirror1,'cold_source':final,'cold_mirror':mirror2,'remote_originals_preserved':True}
        (out/'SUCCESS.json').write_text(json.dumps(success,indent=2)+'\n');check_generated()
        # Persist completed-phase pins; its own commit ID is returned externally.
        completed=commit_virtual_chunks(repo,final,branch,str(artifact),[],[str(out.relative_to(repo))],'Retain completed Git-only collection proof',index_size+2**20)
        mirror3=mirror(completed,mirror2)
        expected_final={str(f.relative_to(repo)):{'bytes':f.stat().st_size,'sha256':sha(f)} for f in out.rglob('*') if f.is_file()}
        verify_metadata(completed,expected_final);verify_metadata(mirror3,expected_final)
        if sha(index)!=initial_index_sha or textgit(repo,['symbolic-ref','HEAD'])!=branch or textgit(repo,['rev-parse',branch])!=completed:raise ValueError('real index or branch changed concurrently; no sync')
        git(repo,['read-tree',completed])
        print(json.dumps({'source':completed,'mirror':mirror3,'archive_source':source,'archive_mirror':mirror1,'archive_sha256':m['archive_sha256'],'all_originals':m['regular_files'],'remote_originals_preserved':True},indent=2))
    except BaseException as exc:
        record({'failure':repr(exc),'no_retry':True,'recoverable_objects_and_commits_preserved':True});raise
if __name__=='__main__':main()
