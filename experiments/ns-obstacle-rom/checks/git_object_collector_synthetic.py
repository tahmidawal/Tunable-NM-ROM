"""Exercise complete collector ONLY in fresh temporary Git repos/local bare mirror.

The fake SSH executable reads an external synthetic archive; no network/cluster
command is issued. No campaign Git index, branch, objects or remote are changed.
"""
from pathlib import Path
import hashlib,io,json,os,random,subprocess,tarfile,tempfile,sys
from retention.git_object_archive import storage_bound
def sh(c,**kw):return subprocess.check_output(c,**kw).decode().strip()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    B=Path(__file__).resolve().parents[1];out=B/'checks/git-object-collector-synthetic02';out.mkdir(exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='complete-collector-synthetic-') as tmp:
        t=Path(tmp);repo=t/'repo';repo.mkdir();remote=t/'remote.git'
        sh(['git','init','-q',str(repo)]);sh(['git','init','-q','--bare',str(remote)])
        g=['git','-C',str(repo)];sh(g+['config','user.name','Synthetic']);sh(g+['config','user.email','synthetic@example.invalid']);sh(g+['checkout','-qb','test'])
        b=repo/'experiments/ns-obstacle-rom';(b/'checks').mkdir(parents=True);(b/'retention').mkdir();(b/'artifacts').mkdir()
        for n in ['checks/collect_git_object_archive_once.py','retention/git_object_archive.py','retention/build_mirror_index.py']:(b/n).write_bytes((B/n).read_bytes())
        files={'data.bin':random.Random(923).randbytes(2*2**20+3),'FINAL.json':b'{"synthetic":true}\n'}
        ledger=''.join(hashlib.sha256(v).hexdigest()+'  '+k+'\n' for k,v in files.items()).encode();files['COLLECTION-SHA256SUMS']=ledger
        archive=t/'synthetic.tar.gz'
        with tarfile.open(archive,'w:gz') as tf:
            for n,data in files.items():
                x=tarfile.TarInfo('synthetic/'+n);x.size=len(data);x.mtime=0;tf.addfile(x,io.BytesIO(data))
        measured=[{'path':n,'bytes':len(v)} for n,v in files.items() if n!='data.bin']
        m={'job_id':999001,'terminal_status':'999001|COMPLETED|1|0:0\n','archive_path':'/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/synthetic.tar.gz','archive_top_directory':'synthetic','archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),'regular_files':3,'expanded_bytes':sum(map(len,files.values())),'ledger_sha256':hashlib.sha256(ledger).hexdigest(),'metadata_files':measured,'metadata_extraction_bytes':sum(x['bytes'] for x in measured)}
        measurement=b/'checks/MEASUREMENT.json';measurement.write_text(json.dumps(m,indent=2)+'\n')
        sh(g+['add','.']);sh(g+['commit','-qm','synthetic frozen source']);parent=sh(g+['rev-parse','HEAD']);sh(g+['remote','add','origin',str(remote)]);sh(g+['push','-q','origin',parent+':refs/heads/codeonly/test'])
        records=measured+[{'path':'owner/'+n,'bytes':size} for n,size in [('ORIGINAL-FILE-INVENTORY.json',256*2**10),('RETENTION.json',128*2**10),('COLD-RECOVERY.json',128*2**10),('ATTEMPT.json',256*2**10),('MEASUREMENT.json',128*2**10),('RELEASE.json',128*2**10),('SUCCESS.json',128*2**10),('TRANSPORT.stderr.txt',2**20),('ATTEMPT-PREVIOUS-VERSION.json',256*2**10)]]
        index=repo/'.git/index';block=os.statvfs(repo).f_frsize;budget=storage_bound(m['archive_bytes'],records,index.stat().st_size,block);budget['required_free_bytes']+=2*block*((m['archive_bytes']+8*2**20-1)//(8*2**20))+64*block
        r={'schema':'ns-obstacle-git-object-first-collection-release-v1','authorized':True,'single_execution':True,'automatic_retry':False,'collector_sha256':sha(b/'checks/collect_git_object_archive_once.py'),'archive_helper_sha256':sha(b/'retention/git_object_archive.py'),'mirror_helper_sha256':sha(b/'retention/build_mirror_index.py'),'chunk_bytes':8*2**20,'transport_timeout_seconds':300,'transport_kill_grace_seconds':5,'measurement_path':str(measurement),'measurement_sha256':sha(measurement),'archive_sha256':m['archive_sha256'],'archive_bytes':m['archive_bytes'],'job_id':m['job_id'],'regular_files':3,'ledger_sha256':m['ledger_sha256'],'metadata_limit_bytes':2**20,'expected_parent_commit':parent,'source_commit':parent,'mirror_ref':'refs/heads/codeonly/test','expected_mirror_commit':parent,'source_base_commit':parent,'mirror_base_commit':parent,'artifact_relative_path':'experiments/ns-obstacle-rom/artifacts/synthetic','required_local_free_bytes':budget['required_free_bytes']}
        release=t/'RELEASE.json';release.write_text(json.dumps(r,indent=2)+'\n')
        fake=t/'bin';fake.mkdir();ssh=fake/'ssh';ssh.write_text('#!'+sys.executable+'\nimport os,sys\nfrom pathlib import Path\nassert sys.argv[-2:]==["--","/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/synthetic.tar.gz"]\nsys.stdout.buffer.write(Path(os.environ["SYNTHETIC_ARCHIVE"]).read_bytes())\n');ssh.chmod(0o700)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE="1",PYTHONPATH=str(b),PATH=str(fake)+os.pathsep+os.environ['PATH'],SYNTHETIC_ARCHIVE=str(archive))
        run=subprocess.run([sys.executable,str(b/'checks/collect_git_object_archive_once.py'),'--release',str(release),'--release-sha256',sha(release)],cwd=repo,env=env,text=True,capture_output=True,timeout=35)
        (out/'stdout.txt').write_text(run.stdout);(out/'stderr.txt').write_text(run.stderr)
        result={'returncode':run.returncode,'source_files':{n:sha(B/n) for n in ['checks/collect_git_object_archive_once.py','retention/git_object_archive.py','retention/build_mirror_index.py']},'scope':'temporary synthetic repo plus local bare mirror; fake SSH; no campaign data/Git or network','budget':budget}
        (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
        run.check_returncode();receipt=json.loads(run.stdout);artifact=b/'artifacts/synthetic';cold=json.loads((artifact/'COLD-RECOVERY.json').read_text())
        assert not (artifact/'archive-chunks').exists() and sh(g+['status','--porcelain'])==''
        assert sh(g+['ls-remote','origin','refs/heads/codeonly/test']).split()[0]==receipt['mirror']
        for rev in [cold['archive_source_commit'],cold['advertised_archive_mirror_commit']]:
            parts=[subprocess.check_output(g+['show',rev+':'+cold['artifact_relative_path']+'/archive-chunks/collection.tar.gz.part-%03d'%i]) for i in range(len(cold['chunks']))]
            assert hashlib.sha256(b''.join(parts)).hexdigest()==m['archive_sha256']
        result.update(passed=True,receipt=receipt,clean_final_index=True,both_original_histories_exact=True,no_working_chunks=True);(out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'scope':result['scope']}))
if __name__=='__main__':main()
