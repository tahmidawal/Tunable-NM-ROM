"""Declare only adopted runtime Git inputs; shared immutable bytes use old full archives.

This does not delete, commit, admit, or rewrite scientific files. The returned rows
are consumed by the existing SourceInputs reader, never by an implicit delta reader.
"""
from pathlib import Path
import argparse,hashlib,json
B=Path(__file__).resolve().parents[1]
def sha(p):
 with p.open('rb')as f:return hashlib.file_digest(f,'sha256').hexdigest()
def declare(case,index):
 case=Path(case).resolve();export=json.loads((case/'EXPORT-MANIFEST.json').read_text());catalog=json.loads(Path(index).read_text());files={**export['runtime_files'],'EXPORT-MANIFEST.json':sha(case/'EXPORT-MANIFEST.json')};rows=[];tracked=[]
 for name,digest in files.items():
  p=case/name
  if not p.resolve().is_relative_to(case)or sha(p)!=digest:raise ValueError('adopted runtime bytes differ')
  row={'source':str(p.relative_to(B)),'destination':name,'sha256':digest,'bytes':p.stat().st_size};entry=catalog['entries'].get(digest+':'+str(p.stat().st_size))
  if entry:
   base=catalog['archives'][entry['archive_id']];manifest=Path(base['cold_manifest_path']);prefix=Path('experiments/ns-obstacle-2d')
   row['archive_source']={'manifest_source':str(manifest.relative_to(prefix)),'manifest_sha256':base['cold_manifest_sha256'],'member':entry['member']}
  else:
   if p.stat().st_size>48*1024**2:raise ValueError('new oversized adopted runtime needs separately reviewed source closure')
   tracked.append(row['source'])
  rows.append(row)
 return {'schema':'ns-obstacle-adopted-runtime-source-closure-v1','case_directory':str(case.relative_to(B)),'export_manifest_sha256':sha(case/'EXPORT-MANIFEST.json'),'reuse_index_sha256':sha(Path(index)),'files':rows,'direct_git_paths':tracked,'new_warm_git_bytes':sum(r['bytes']for r in rows if 'archive_source'not in r),'warm_runtime_bytes':sum(r['bytes']for r in rows),'all_consumed_inputs_direct_Git_or_existing_full_archive':True,'admission':False}
def main():
 p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--index',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();result=declare(a.case,a.index);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'new_warm_git_bytes':result['new_warm_git_bytes'],'warm_runtime_bytes':result['warm_runtime_bytes']}))
if __name__=='__main__':main()
