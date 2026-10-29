"""Retain a verified immutable collected job without touching remote state."""
from pathlib import Path
import argparse,hashlib,json,re,shutil,subprocess
B=Path(__file__).resolve().parents[1]
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--job',required=True,type=int);p.add_argument('--archive-sha',required=True);p.add_argument('--files',required=True,type=int);p.add_argument('--move-canonical',action='store_true');a=p.parse_args()
    if not re.fullmatch(r'g2-[a-z0-9-]+',a.name) or a.job<=0:raise ValueError('explicit namespace and numeric job required')
    archive=B/'runs'/(a.name+'-collection.tar.gz');R=B/'runs'/a.name;A=B/'artifacts'/a.name
    if sha(archive)!=a.archive_sha:raise ValueError('archive hash mismatch')
    lines=(R/'COLLECTION-SHA256SUMS').read_text().splitlines()
    if len(lines)!=a.files:raise ValueError('inventory count mismatch')
    for line in lines:
        digest,name=line.split('  ',1);path=(R/name).resolve()
        if not path.is_relative_to(R.resolve()) or sha(path)!=digest:raise ValueError('retained file hash/path mismatch: '+name)
    A.mkdir(exist_ok=False)
    for path in R.iterdir():
        if path.name in {'deps','wheels','cache','tmp'}:continue
        if a.move_canonical:shutil.move(str(path),str(A/path.name))
        elif path.is_dir():shutil.copytree(path,A/path.name)
        else:shutil.copy2(path,A/path.name)
    chunks=[]
    with archive.open('rb') as f:
        while data:=f.read(48*1024**2):
            path=A/f'collection.tar.gz.part-{len(chunks):03d}';path.write_bytes(data);chunks.append({'path':path.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
    (A/'CHECKSUM-VERIFICATION.txt').write_text(f'Archive SHA256 {a.archive_sha} and all {a.files} regular file SHA256 values verified.\n')
    record={'status':'complete_all_regular_files_verified','job_id':a.job,'archive_sha256':a.archive_sha,'archive_bytes':archive.stat().st_size,'verified_regular_files':a.files,'chunks':chunks,'full_local_archive':str(archive),'full_local_extraction':None if a.move_canonical else str(R),'canonical_expanded_artifacts':str(A),'scientific_extraction_moved_without_copy':a.move_canonical,'residual_runtime_extraction':str(R) if a.move_canonical else None,'recovery':'Concatenate ordered chunks, verify archive SHA, extract, then enter job directory and sha256sum -c COLLECTION-SHA256SUMS. Full source, dependencies, inputs and scientific arrays retained.','reference_data_admitted':False,'remote_cleanup':'pending_durable_mirror'}
    (A/'RETENTION.json').write_text(json.dumps(record,indent=2)+'\n')
    command=['ssh','-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3','tufts-login',f'sacct -j {a.job} --format=JobID,State,ExitCode,ElapsedRaw,NodeList,MaxRSS -P']
    result=subprocess.run(command,capture_output=True,text=True,check=True);(A/'SACCT.txt').write_text(result.stdout)
    print(a.name,'retained',a.files,'files in',len(chunks),'archive chunks',flush=True)
if __name__=='__main__':main()
