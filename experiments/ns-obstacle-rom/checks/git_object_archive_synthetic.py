"""Bounded synthetic Git-only retention test; no campaign archive or remote."""
from pathlib import Path
import hashlib,io,json,os,random,subprocess,tarfile,tempfile
from retention.git_object_archive import ingest,BlobReader,verify_archive,commit_virtual_chunks,git,storage_bound

def main():
    out=Path(__file__).with_name('git-object-archive-synthetic01');out.mkdir(exist_ok=False)
    payload=random.Random(419).randbytes(9*2**20+17)
    original={'data.bin':payload,'history.jsonl':b'{"step":1}\n','FINAL.json':b'{"done":true}\n'}
    ledger=''.join(hashlib.sha256(b).hexdigest()+'  '+n+'\n' for n,b in original.items()).encode()
    files=dict(original,**{'COLLECTION-SHA256SUMS':ledger});buf=io.BytesIO()
    with tarfile.open(fileobj=buf,mode='w:gz') as tf:
        for name,b in files.items():
            info=tarfile.TarInfo('synthetic/'+name);info.size=len(b);info.mtime=0;tf.addfile(info,io.BytesIO(b))
    archive=buf.getvalue();sha=hashlib.sha256(archive).hexdigest();events=[]
    with tempfile.TemporaryDirectory(prefix='synthetic-git-archive-') as tmp:
        repo=Path(tmp);git(repo,['init','-q']);git(repo,['config','user.name','Synthetic']);git(repo,['config','user.email','synthetic@example.invalid'])
        tree=git(repo,['mktree'],input=b'',stdout=subprocess.PIPE).stdout.decode().strip()
        parent=git(repo,['commit-tree',tree],input=b'synthetic base\n',stdout=subprocess.PIPE).stdout.decode().strip();git(repo,['update-ref','refs/heads/test',parent]);git(repo,['symbolic-ref','HEAD','refs/heads/test']);git(repo,['read-tree',parent])
        chunks=ingest(repo,io.BytesIO(archive),len(archive),sha,events.append)
        m={'archive_top_directory':'synthetic','regular_files':4,'expanded_bytes':sum(map(len,files.values())),'ledger_sha256':hashlib.sha256(ledger).hexdigest(),'metadata_extraction_bytes':len(ledger)+len(files['FINAL.json']),'metadata_policy':'JSONL retained only in complete archive; JSON/source/log metadata expanded'}
        artifact=repo/'artifact';artifact.mkdir();verified=verify_archive(repo,chunks,m,artifact/'metadata',2**20)
        restored=io.BufferedReader(BlobReader(repo,chunks)).read();assert restored==archive and not list(artifact.rglob('*.jsonl'))
        commit=commit_virtual_chunks(repo,parent,'refs/heads/test','artifact',chunks,['artifact/metadata'],'synthetic virtual archive',2**20)
        mirror=git(repo,['commit-tree',git(repo,['rev-parse',commit+'^{tree}'],stdout=subprocess.PIPE).stdout.decode().strip(),'-p',parent],input=b'synthetic mirror\n',stdout=subprocess.PIPE).stdout.decode().strip()
        for rev in [commit,mirror]:
            for i,r in enumerate(chunks):
                raw=git(repo,['show',rev+':artifact/archive-chunks/collection.tar.gz.part-%03d'%i],stdout=subprocess.PIPE).stdout
                assert len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256']
        assert not (artifact/'archive-chunks').exists()
        failed=[]
        try:ingest(repo,io.BytesIO(archive[:-1]),len(archive),sha,failed.append)
        except ValueError as e:failure=str(e)
        else:raise AssertionError('truncation not rejected')
        result={'passed':True,'scope':'synthetic archive/Git temporary repository only; no remote mirror qualification','archive_bytes':len(archive),'archive_sha256':sha,'chunks':chunks,'verified_files':len(verified['files']),'all_archive_bytes_recovered_from_both_synthetic_commits':True,'no_working_archive_chunks':True,'truncated_attempt_completed_chunk_records':failed,'truncation_failure':failure,'events':events,'example_bound':storage_bound(156681968,2*2**20,4420385),'source_sha256':hashlib.sha256((Path(__file__).parents[1]/'retention/git_object_archive.py').read_bytes()).hexdigest()}
        (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['passed','archive_bytes','verified_files','scope']}))
if __name__=='__main__':main()
