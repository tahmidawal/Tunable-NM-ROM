"""Read-only independent saved-byte audit; no native FE evaluation or solve."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
base=Path(sys.argv[1]); p=base/'results/preparation'; v=p/'prepared/volume'; r=p/'restored_volume'; primary=p/'primary_volume'
def sha(f):
 with f.open('rb') as s:return hashlib.file_digest(s,'sha256').hexdigest()
def digest(*arrays):
 h=hashlib.sha256()
 for a in arrays:
  a=np.ascontiguousarray(a);h.update(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes())
 return h.hexdigest()
def arrays(f):
 with np.load(f,allow_pickle=False) as z:d={k:z[k] for k in z.files}
 assert all(a.dtype==np.float64 and np.isfinite(a).all() for a in d.values()),str(f)
 return d
m=json.loads((v/'manifest.json').read_text());pm=json.loads((primary/'manifest.json').read_text());replay=json.loads((p/'primary_replay.json').read_text())
assert sha(v/'manifest.json')==sha(r/'manifest.json')==sha(primary/'volume-manifest.json')==replay['restoration']['manifest_sha256']
assert sha(primary/'manifest.json')==replay['primary_manifest_sha256']
assert m['points']==pm['points']==1036497 and m['all_original_points'] is True
assert m['units']==pm['units']=={'values':'dimensionless','gradients':'d/d(x/D)','weights':'dx/D**2'}
files={}
for name,h in m['files'].items():
 assert sha(v/name)==sha(r/name)==h,name
 files[name]={'sha256':h,'bytes':(v/name).stat().st_size}
assert sum(x['bytes'] for x in files.values())==m['actual_compressed_bytes']
for root in (v,r):assert {q.name for q in root.iterdir()}==set(files)|{'manifest.json'}
for name,h in pm['files'].items():assert sha(primary/name)==h,name
sources={n[7:]:h for n,h in pm['files'].items() if n.startswith('source/')}
assert hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()==pm['source_sha256']
geom=arrays(v/'geometry.npz');assert sha(v/'geometry.npz')==sha(primary/'geometry.npz')
assert digest(*(geom[k] for k in sorted(geom)))==pm['geometry_array_sha256']
n=m['points'];shapes={'points_physical':(n,2),'points_dimensionless':(n,2),'weights_dimensionless':(n,),'jacobians_physical':(n,2,2)}
assert set(geom)==set(shapes) and all(geom[k].shape==s for k,s in shapes.items())
assert np.all(geom['weights_dimensionless']>0)
j=geom['jacobians_physical'];assert np.all(j[:,0,0]*j[:,1,1]-j[:,0,1]*j[:,1,0]>0)
coeff=arrays(primary/'coefficients.npz');assert set(coeff)==set(m['fields'])==set(pm['fields'])
fields={};previous=None
for name,f in m['fields'].items():
 pf=pm['fields'][name];assert digest(coeff[name])==pf['coefficient_array_sha256'];assert f['vector']==pf['vector'] and f['columns']==pf['columns']
 cursor=0;ranges=[]
 assert len(f['blocks'])==len(pf['blocks'])
 for row,pr in zip(f['blocks'],pf['blocks']):
  a,b=row['start'],row['end'];assert a==cursor and a<b<=n;cursor=b;ranges.append((a,b))
  assert {k:row[k] for k in ('start','end','array_sha256')}==pr
  d=arrays(v/row['file']);assert set(d)=={'values','gradients'}
  c=f['columns'];vs=(b-a,2,c) if f['vector'] else (b-a,c);gs=(b-a,2,2,c) if f['vector'] else (b-a,2,c)
  assert d['values'].shape==vs and d['gradients'].shape==gs
  assert digest(d['values'],d['gradients'])==row['array_sha256']
 assert cursor==n and (previous is None or ranges==previous);previous=ranges
 fields[name]={'blocks':len(ranges),'points':cursor,'columns':f['columns'],'vector':f['vector'],'coefficient_array_sha256':pf['coefficient_array_sha256']}
prepared=json.loads((p/'prepared/manifest.json').read_text())
for name,h in prepared['coefficient_files'].items():assert sha(p/'prepared'/name)==h,name
assert json.loads((primary/'fe_layout.json').read_text())['sha256']==pm['layout_sha256']
assert replay['restoration']['exact_archive_and_array_bytes'] is True and replay['omission_authorized'] is False
assert all(it['completed'] and it['points_returned']==n for it in replay['replay_accounting']['iterations'])
allfiles={str(q.relative_to(p)):{'sha256':sha(q),'bytes':q.stat().st_size} for q in sorted(p.rglob('*')) if q.is_file() and not str(q.relative_to(p)).startswith(('prepared/volume/','restored_volume/'))}
print(json.dumps({'schema':'ns-obstacle-independent-fine-primary-byte-review-v1','passed':True,'job_id':'6633574','remote_root':str(base),'scope':'saved arrays, exact original/restored bytes, primary and nonvolume closure; no native recomputation','result_sha256':sha(p/'result.json'),'replay_sha256':sha(p/'primary_replay.json'),'primary_manifest_sha256':sha(primary/'manifest.json'),'volume_manifest_sha256':sha(v/'manifest.json'),'primary_source_sha256':pm['source_sha256'],'fields':fields,'volume_files':files,'compact_scientific_files':allfiles,'compact_scientific_bytes':sum(x['bytes'] for x in allfiles.values()),'omission_authorized':False},indent=2))
