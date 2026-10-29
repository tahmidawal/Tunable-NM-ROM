"""Retire verified working duplicates after row review and Git-only archive proof.

Original retention/candidate records never change. Adopted runtime stays warm;
all other original bytes remain exactly recoverable through delta and base history.
"""
from pathlib import Path
import argparse,hashlib,json,subprocess
B=Path(__file__).resolve().parents[1];REPO=B.parents[1]
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
def require(condition,message):
 if not condition:raise ValueError(message)
def main():
 p=argparse.ArgumentParser();p.add_argument('--group',required=True);a=p.parse_args();frozen=B/'g2/population01/frozen'/a.group;candidate=frozen/'QUALIFICATION-CANDIDATE.json';review=frozen/'INDEPENDENT-ROW-REVIEW.json';audit=read(review);data=read(candidate);require(audit['passed']is True and audit['candidate_sha256']==sha(candidate)and len(data['rows'])==1,'row review differs');require(audit['Re']==data['rows'][0]['Re']and audit['split']==data['rows'][0]['split'],'row scope differs')
 closure=read(frozen/'ADOPTED-RUNTIME-CLOSURE.json');case=B/closure['case_directory'];adopted=data['rows'][0]['adopted_reference'];require(closure['case_directory']==adopted['directory'] and closure['export_manifest_sha256']==adopted['export_manifest_sha256']==sha(case/'EXPORT-MANIFEST.json'),'reviewed adopted case differs');export=read(case/'EXPORT-MANIFEST.json');expected={**export['runtime_files'],'EXPORT-MANIFEST.json':sha(case/'EXPORT-MANIFEST.json')};actual={r['destination']:r['sha256']for r in closure['files']};require(len(actual)==len(closure['files'])and actual==expected,'incomplete adopted runtime closure')
 for row in closure['files']:require(B/row['source']==case/row['destination']and sha(B/row['source'])==row['sha256'],'runtime source differs')
 artifact=next(x for x in case.parents if(x/'RETENTION.json').is_file());retention=read(artifact/'RETENTION.json');cold_path=artifact/'GIT-COLD-ARCHIVE.json';cold=read(cold_path);require(cold['source_retention_sha256']==sha(artifact/'RETENTION.json')and cold['archive_sha256']==retention['archive_sha256']and cold['git_only_stream_verification']['archive_sha256']==retention['archive_sha256'],'cold archive binding differs');require(sha(artifact/'DELTA-RETENTION-MAP.json')==retention['retention_map_sha256'],'retention map differs');subprocess.run(['git','-c','gc.auto=0','merge-base','--is-ancestor',cold['source_commit'],'HEAD'],check=True)
 advertised=subprocess.check_output(['git','-c','gc.auto=0','ls-remote','origin',cold['mirror_ref']],text=True).split();require(len(advertised)==2 and advertised[1]==cold['mirror_ref'],'mirror ref unavailable');subprocess.run(['git','-c','gc.auto=0','merge-base','--is-ancestor',cold['mirror_commit'],advertised[0]],check=True)
 keep={str((B/r['source']).relative_to(artifact))for r in closure['files']};original=read(artifact/'DELTA-RETENTION-MAP.json');removed=[]
 for name,row in original['files'].items():
  path=artifact/name
  if not path.resolve().is_relative_to(artifact.resolve())or path.is_symlink():raise ValueError('unexpected source path')
  if not path.is_file()or path.stat().st_size!=row['bytes']or sha(path)!=row['sha256']:raise ValueError('working original differs:'+name)
  small=(name.startswith('results/')or'/'not in name)and path.suffix in ('.json','.txt','.stdout','.stderr','.out','.err')and row['bytes']<=1024**2
  if name in keep or small or name=='COLLECTION-SHA256SUMS':continue
  removed.append({'path':name,'bytes':row['bytes'],'sha256':row['sha256']})
 for row in cold['chunks']:
  path=REPO/row['repository_path']
  if path.stat().st_size!=row['bytes']or sha(path)!=row['sha256']:raise ValueError('chunk working duplicate differs')
  removed.append({'path':str(path.relative_to(artifact)),'bytes':row['bytes'],'sha256':row['sha256']})
 record={'schema':'ns-obstacle-reviewed-population-working-copy-retirement-v1','group':a.group,'verified_advertised_mirror_tip':advertised[0],'candidate_sha256':sha(candidate),'independent_row_review_sha256':sha(review),'git_cold_manifest_sha256':sha(cold_path),'delta_retention_map_sha256':sha(artifact/'DELTA-RETENTION-MAP.json'),'reuse_index_sha256':retention['reuse_index_sha256'],'source_and_mirror_history_preserved':True,'historical_retention_record_unchanged':True,'all_adopted_runtime_kept_warm':True,'chunk_storage':'git_history_only','removed':removed,'removed_bytes':sum(r['bytes']for r in removed),'retirement_complete':False};out=artifact/'WORKING-COPY-LIFECYCLE.json';out.write_text(json.dumps(record,indent=2)+'\n')
 for row in removed:(artifact/row['path']).unlink()
 record['retirement_complete']=True;out.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'group':a.group,'retired_duplicate_bytes':record['removed_bytes'],'all_adopted_runtime_kept_warm':True}))
if __name__=='__main__':main()
