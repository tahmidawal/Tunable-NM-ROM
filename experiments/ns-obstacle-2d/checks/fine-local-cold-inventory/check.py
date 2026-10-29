"""Read-only presence audit; no extraction, deletion or numerical-array access."""
from pathlib import Path
import hashlib,json,subprocess
W=Path(__file__).resolve().parents[4];R=W.parent/'2026-10-27-ns-obstacle-rom';relative='experiments/ns-obstacle-rom/artifacts/fine-preparation01';b=R/relative
source='f775b07afda415181b9242dc47f9f076a7ce7645';mirror='4e80bc7dc4a01a1f3e0d74a5dac12aff429ba445'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
meta={}
for name in ('RETENTION.json','COMPOSITION-R2.json','DELTA-RETENTION-MAP.json'):
 digest=sha(b/name)
 for commit in (source,mirror):
  data=subprocess.check_output(['git','-c','gc.auto=0','-C',str(R),'show',commit+':'+relative+'/'+name]);assert hashlib.sha256(data).hexdigest()==digest
 meta[name]=digest
composition=json.loads((b/'COMPOSITION-R2.json').read_text());retention=json.loads((b/'RETENTION.json').read_text());rows=[]
for name,expected in composition['compact_scientific_files'].items():
 p=b/'results/preparation'/name;exists=p.is_file();row={'archive_scientific_relative_path':name,'expected_sha256':expected['sha256'],'expected_bytes':expected['bytes'],'local_path':str(p),'exists':exists,'payload_container':p.suffix in ('.npz','.npy'),'eligible_for_retirement':False}
 if exists:
  assert not p.is_symlink() and p.stat().st_size==expected['bytes'];assert not row['payload_container'];assert sha(p)==expected['sha256'];row['reason']='present metadata retained'
 else:row['reason']='not locally materialized; no bytes reclaimable'
 rows.append(row)
allfiles=[p for p in b.rglob('*') if p.is_file()];assert not any(p.suffix in ('.npz','.npy') for p in allfiles)
out={'schema':'ns-obstacle-fine-local-cold-noop-inventory-v1','passed':True,'scientific_compact_file_count':len(rows),'logical_compact_scientific_bytes':sum(x['expected_bytes'] for x in rows),'actual_local_artifact_regular_files':len(allfiles),'actual_local_artifact_regular_bytes':sum(p.stat().st_size for p in allfiles),'exact_scientific_paths':rows,'eligible_payload_paths':[],'reclaimable_payload_bytes':0,'metadata_source_commit':source,'metadata_mirror_commit':mirror,'metadata_sha256':meta,'archive_sha256':retention['archive_sha256'],'archive_bytes':retention['archive_bytes'],'archive_replay_performed':False,'reason_archive_replay_not_needed':'no proposed local scientific payload exists; prior archive acceptance remains untouched, no new recovery proof claimed','deletions_performed':False,'numerical_arrays_opened':False,'scope_correction':'1.313GB is logical compact scientific closure retained archive-only, not local warm duplicate footprint; different training-materialize01 candidate requires separate scope authorization','owner_confirmation':'ROM owner confirms no local fine scientific expansion; all active jobs use remote immutable trees'}
Path(__file__).with_name('RESULT.json').write_text(json.dumps(out,indent=2)+'\n');print({k:out[k] for k in ('scientific_compact_file_count','actual_local_artifact_regular_bytes','reclaimable_payload_bytes')})
