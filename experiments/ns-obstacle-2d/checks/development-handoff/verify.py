"""Read-only development-only runtime closure verification; no native execution."""
from pathlib import Path
import hashlib,json,subprocess
W=Path(__file__).resolve().parents[4];B=W/'experiments/ns-obstacle-2d';R=W.parent/'2026-10-27-ns-obstacle-rom/experiments/ns-obstacle-rom'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
admission_path=B/'g2/population01/complete15/REFERENCE-ADMISSION.json';admission=read(admission_path)
assert sha(admission_path)=='5c9402af9ddb2af0ad8925ef224a43ae72cb36209df99865908617c63b5b506f'
candidate=read(B/'g2/population01/complete15/QUALIFICATION-CANDIDATE.json');cases=[]
for re,label in [(16.875,'re16p875'),(23.125,'re23p125')]:
 frozen=B/'g2/population01/frozen'/label;cp=frozen/'ADOPTED-RUNTIME-CLOSURE.json';closure=read(cp)
 row=next(x for x in candidate['rows'] if x['Re']==re);a=next(x for x in admission['admitted_references'] if x['Re']==re)
 assert row['split']==a['split']=='development'
 for k in ('config_sha256','export_manifest_sha256','mesh_sha256','layout_sha256','solution_sha256'):assert row['adopted_reference'][k]==a[k]
 folder=B/closure['case_directory'];assert closure['case_directory']==row['adopted_reference']['directory']
 ex=read(folder/'EXPORT-MANIFEST.json');primary=read(folder/'PRIMARY-REFERENCE.json');cfg=read(folder/'config.json')
 assert sha(folder/'EXPORT-MANIFEST.json')==closure['export_manifest_sha256']==a['export_manifest_sha256']
 assert sha(folder/'config.json')==a['config_sha256'] and sha(folder/'mesh.vol.bin')==a['mesh_sha256']
 assert read(folder/'fe_layout.json')['sha256']==a['layout_sha256'] and sha(folder/'solution.sol')==a['solution_sha256']
 files=[]
 for f in closure['files']:
  p=B/f['source'];assert p.is_file() and not p.is_symlink() and p.stat().st_size==f['bytes'] and sha(p)==f['sha256'],str(p)
  proof=dict(f,warm_verified=True)
  if 'archive_source' in f:
   ap=B/f['archive_source']['manifest_source'];assert sha(ap)==f['archive_source']['manifest_sha256'];proof['archive_manifest_verified']=True
  else:
   blob=subprocess.check_output(['git','-C',str(W),'show','HEAD:'+str(p.relative_to(W))]);assert hashlib.sha256(blob).hexdigest()==f['sha256'];proof['committed_bytes_verified']=True
  files.append(proof)
 assert {f['destination'] for f in files}==set(ex['runtime_files'])|{'EXPORT-MANIFEST.json'}
 for name,h in ex['producer_source_sha256'].items():assert sha(R/'primary_native'/name)==h,name
 for name,h in primary['source_sha256'].items():assert sha(R/'primary_native'/name)==h,name
 review=read(frozen/'INDEPENDENT-ROW-REVIEW.json');assert review['passed'] is True and review['Re']==re and review['split']=='development'
 cases.append({'Re':re,'split':'development','adopted_reference':row['adopted_reference'],'closure_path':str(cp),'closure_sha256':sha(cp),'row_review_sha256':sha(frozen/'INDEPENDENT-ROW-REVIEW.json'),'original_config_role':{k:cfg.get(k) for k in ('release','split','population_split')},'primary_record_sha256':sha(folder/'PRIMARY-REFERENCE.json'),'producer_source_sha256':ex['producer_source_sha256'],'versions':ex['versions'],'files':files,'input_bytes':sum(f['bytes'] for f in files)})
out={'schema':'ns-obstacle-development-input-identity-review-v1','passed':True,'scope':'two development references only; no native replay and no validation fields accessed','source_commit':subprocess.check_output(['git','-C',str(W),'rev-parse','HEAD'],text=True).strip(),'reference_admission_sha256':sha(admission_path),'cases':cases,'input_bytes':sum(c['input_bytes'] for c in cases),'role_handling':'Use existing handoff2d.population / handoff_primary_worker development route. Never call the training-only materialization entry or relabel development as training. Keep original producer exports unchanged; write fresh derived population descriptors with actual content hashes.'}
Path(__file__).with_name('RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'passed':True,'cases':len(cases),'input_bytes':out['input_bytes']}))
