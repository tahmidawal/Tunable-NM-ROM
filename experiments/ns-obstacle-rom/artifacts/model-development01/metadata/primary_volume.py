"""Prospective coefficient-primary FE volume replay; never confers admission.

The complete physical rule, exact coefficients/mesh/layout, source closure and
per-field block byte hashes are primary. Regeneration is offline work and each
returned block must reproduce the original float64 bytes exactly.
"""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,platform,resource,shutil,time,sys
from functools import lru_cache
import numpy as np
from fe_algebra import array_digest
from point_store import read_npz,sha

VECTORS={'lift_u','pod_velocity','supremizers','momentum_tests'}
FIELDS=VECTORS|{'lift_p_per_nu','pressure','continuity_tests'}
SOURCES=('primary_volume.py','point_store.py','fe_blocks.py','fe_adapter.py','fe_algebra.py',
 'fe_pod_a6.py','fe_controls_a6.py','component_factor.py','qualification.py','FD1-SCOPE-CONTRACT.json',
 'spaces.py','weak.py','train_bank.py','oriented_normals.py','scalars.py','MODEL-CONTRACT.json')

def source_digest():
 root=Path(__file__).parent
 return hashlib.sha256(json.dumps({n:sha(root/n) for n in SOURCES},sort_keys=True).encode()).hexdigest()

def versions(ng):
 import scipy
 return {'python':platform.python_version(),'machine':platform.machine(),
         'ngsolve':ng.__version__,'numpy':np.__version__,'scipy':scipy.__version__}

@lru_cache(maxsize=1)
def runtime_identity():
 """Hash executable package code/data; omit install paths/cache-only metadata."""
 import importlib.metadata
 packages={}
 for name in ('ngsolve','netgen-mesher','netgen-occt','ngsolve-openblas','numpy','scipy'):
  distribution=importlib.metadata.distribution(name);files={}
  for relative in distribution.files or ():
   label=str(relative)
   if label.endswith('.pyc') or '/__pycache__/' in label:continue
   if '.dist-info/' in label and relative.name not in ('METADATA','WHEEL'):continue
   path=Path(distribution.locate_file(relative))
   if path.is_file():files[label]=sha(path)
   else:raise ValueError('declared runtime dependency missing: '+label)
  if not files:raise ValueError('empty runtime dependency inventory')
  packages[name]={'version':distribution.version,'files':files}
 return {'packages':packages,'python_executable_sha256':sha(Path(sys.executable)),
         'libc':list(platform.libc_ver()),'excluded':'bytecode caches and installer/path bookkeeping; executable sources/native libraries/data are hashed'}

def host_guard(reserved_bytes):
 from fe_adapter import CAPS
 peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
 if peak+reserved_bytes>CAPS['process_rss_bytes']:
  raise MemoryError('primary replay RSS plus prospective workspace exceeds unchanged process cap')
 return {'prior_peak_rss_bytes':peak,'reserved_bytes':reserved_bytes}

def geometry_arrays(adapter):
 return {'points_physical':adapter.volume['points'],'points_dimensionless':adapter.volume['points']/adapter.D,
         'weights_dimensionless':adapter.volume['weights']/adapter.D**2,'jacobians_physical':adapter.volume['jacobians']}

def write_primary(adapter,directory,source_case,volume_directory,block_points=4096):
 from point_store import PointVolumeStore
 from fe_adapter import CAPS,MESH_FILENAME
 started=time.perf_counter();directory=Path(directory);source_case=Path(source_case)
 if directory.exists():raise FileExistsError('fresh primary volume required')
 if set(adapter.coefficients)!=FIELDS:raise ValueError('complete declared FE coefficient set required')
 volume_directory=Path(volume_directory)
 original=PointVolumeStore(volume_directory,sha(volume_directory/'manifest.json'))
 if set(original.manifest['fields'])!=FIELDS or original.manifest['block_points']!=block_points:
  raise ValueError('original complete field set/partition required')
 host_guard(sum(np.asarray(c).nbytes for c in adapter.coefficients.values())+2**28)
 geometry=geometry_arrays(adapter);coefficients=adapter.coefficients;n=len(geometry['weights_dimensionless'])
 if any(not np.array_equal(geometry[k],original.geometry[k]) for k in geometry):raise ValueError('original complete rule differs')
 expanded=sum(a.nbytes for a in geometry.values())+sum(a.nbytes for a in coefficients.values())
 originals={MESH_FILENAME:source_case/MESH_FILENAME,'config.json':source_case/'config.json','fe_layout.json':source_case/'fe_layout.json',
            'volume-manifest.json':volume_directory/'manifest.json','geometry.npz':volume_directory/'geometry.npz'}
 projected=expanded+sum(p.stat().st_size for p in originals.values())+32*2**20
 if n>CAPS['volume_points'] or expanded>CAPS['archive_bytes']:raise MemoryError('primary volume payload cap')
 if shutil.disk_usage(directory.parent).free<10*2**30+projected:raise OSError('primary volume disk reserve')
 directory.mkdir();(directory/'source').mkdir()
 for name,path in originals.items():shutil.copyfile(path,directory/name)
 for name in SOURCES:shutil.copyfile(Path(__file__).with_name(name),directory/'source'/name)
 np.savez_compressed(directory/'coefficients.npz',**coefficients)
 fields={}
 for name,c in coefficients.items():
  vector=name in VECTORS;columns=1 if c.ndim==1 else c.shape[1];field=original.manifest['fields'][name]
  if field['vector']!=vector or field['columns']!=columns:raise ValueError('original field coefficient descriptor differs')
  rows=[]
  for row in field['blocks']:
   digest=row.get('array_sha256')
   if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
    raise ValueError('original array byte hash required')
   rows.append({k:row[k] for k in ('start','end','array_sha256')})
  fields[name]={'vector':vector,'columns':columns,'blocks':rows,'coefficient_array_sha256':array_digest(c)}
 files={str(p.relative_to(directory)):sha(p) for p in sorted(directory.rglob('*')) if p.is_file()}
 manifest={'schema':'ns-obstacle-coefficient-primary-volume-v1','physical_admission':False,
  'source_sha256':source_digest(),'versions':versions(adapter.ng),'runtime_identity':runtime_identity(),
  'files':files,'points':n,'block_points':block_points,'fields':fields,'layout_sha256':adapter.layout_sha256,
  'units':{'values':'dimensionless','gradients':'d/d(x/D)','weights':'dx/D**2'},
  'complete_rule':'all original mapped VOL points, weights and Jacobians, in native enumeration',
  'geometry_array_sha256':array_digest(*(geometry[k] for k in sorted(geometry))),
  'expanded_primary_payload_bytes':expanded,'projected_output_bytes':projected,
  'creation_seconds':time.perf_counter()-started,
  'source_role':'derived FE field retention only; training inputs, fitting evidence and actual query outputs are separate mandatory artifacts'}
 (directory/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
 return manifest

class PrimaryVolumeStore:
 def __init__(self,directory,expected_manifest_sha256,ng):
  from fe_adapter import CAPS,MESH_FILENAME,ESSENTIAL,layout_record,TrainingFEAdapter
  started=time.perf_counter();self.directory=Path(directory);path=self.directory/'manifest.json'
  if sha(path)!=expected_manifest_sha256:raise ValueError('primary manifest hash mismatch')
  m=json.loads(path.read_text());self.manifest=m
  if (m['schema']!='ns-obstacle-coefficient-primary-volume-v1' or m['physical_admission'] is not False or
      m['source_sha256']!=source_digest() or m['versions']!=versions(ng) or m.get('runtime_identity')!=runtime_identity()):
   raise ValueError('primary schema/source/runtime differs; no approximate replay fallback')
  if m['units']!={'values':'dimensionless','gradients':'d/d(x/D)','weights':'dx/D**2'}:raise ValueError('primary units differ')
  expected_files={'geometry.npz','coefficients.npz',MESH_FILENAME,'config.json','fe_layout.json','volume-manifest.json'}|{'source/'+n for n in SOURCES}
  if set(m['files'])!=expected_files:raise ValueError('primary dependency closure differs')
  if sum((self.directory/name).stat().st_size for name in m['files'])>CAPS['archive_bytes']:
   raise MemoryError('primary dependency payload cap')
  for name,digest in m['files'].items():
   file=self.directory/name
   if not file.resolve().is_relative_to(self.directory.resolve()) or sha(file)!=digest:raise ValueError('primary dependency hash/path')
  for name in SOURCES:
   if m['files']['source/'+name]!=sha(Path(__file__).with_name(name)):raise ValueError('primary source closure differs')
  if type(m['points']) is not int or not 0<m['points']<=CAPS['volume_points']:raise MemoryError('primary point cap')
  if type(m['block_points']) is not int or not 1<=m['block_points']<=16384:raise ValueError('primary block cap')
  if type(m['expanded_primary_payload_bytes']) is not int or not 0<m['expanded_primary_payload_bytes']<=CAPS['archive_bytes']:raise MemoryError('primary declared payload cap')
  setup_guard=host_guard(3*m['expanded_primary_payload_bytes']+2*2**30)
  self.original_manifest=json.loads((self.directory/'volume-manifest.json').read_text())
  if self.original_manifest['schema']!='ns-obstacle-point-volume-v1' or self.original_manifest['points']!=m['points'] or self.original_manifest['files'].get('geometry.npz')!=m['files']['geometry.npz']:
   raise ValueError('original volume manifest binding differs')
  original=self.original_manifest
  if (original.get('all_original_points') is not True or original.get('units')!=m['units'] or
      set(original['fields'])!=FIELDS or original['block_points']!=m['block_points'] or
      not 0<original['projected_expanded_bytes']<=8*2**30 or not 0<original['actual_compressed_bytes']<=8*2**30):
   raise ValueError('original volume coverage/units/budget differs')
  original_files={'geometry.npz'}
  for name,field in m['fields'].items():
   source=original['fields'][name]
   if source['vector']!=field['vector'] or source['columns']!=field['columns'] or len(source['blocks'])!=len(field['blocks']):raise ValueError('original field declaration differs')
   for old,new in zip(source['blocks'],field['blocks']):
    if {k:old[k] for k in ('start','end','array_sha256')}!=new or Path(old['file']).name!=old['file'] or original['files'].get(old['file'])!=old['sha256']:
     raise ValueError('original block/hash closure differs')
    original_files.add(old['file'])
  if set(original['files'])!=original_files:raise ValueError('original file closure differs')
  self.point_count=m['points'];self.geometry=read_npz(self.directory/'geometry.npz',2**28)
  shapes={'points_physical':(self.point_count,2),'points_dimensionless':(self.point_count,2),
          'weights_dimensionless':(self.point_count,),'jacobians_physical':(self.point_count,2,2)}
  if set(self.geometry)!=set(shapes) or any(self.geometry[k].shape!=v for k,v in shapes.items()):raise ValueError('primary complete geometry shapes')
  if np.min(self.geometry['weights_dimensionless'])<=0 or np.min(np.linalg.det(self.geometry['jacobians_physical']))<=0:raise ValueError('primary positive geometry')
  if array_digest(*(self.geometry[k] for k in sorted(self.geometry)))!=m['geometry_array_sha256']:raise ValueError('primary geometry byte identity')
  self.coefficients=read_npz(self.directory/'coefficients.npz',CAPS['archive_bytes'])
  if set(self.coefficients)!=FIELDS or set(m['fields'])!=FIELDS:raise ValueError('primary complete coefficient fields')
  expanded=sum(a.nbytes for a in self.geometry.values())+sum(a.nbytes for a in self.coefficients.values())
  if expanded!=m['expanded_primary_payload_bytes'] or expanded>CAPS['archive_bytes']:raise MemoryError('primary expanded payload differs/exceeds cap')
  cfg=json.loads((self.directory/'config.json').read_text());mesh=ng.Mesh(str(self.directory/MESH_FILENAME))
  if mesh.dim!=2 or mesh.ne>CAPS['elements'] or int(mesh.GetCurveOrder())!=cfg['geometry_order']:raise ValueError('primary immutable curved mesh')
  V=ng.VectorH1(mesh,order=cfg['velocity_order'],dirichlet=ESSENTIAL);Q=ng.H1(mesh,order=cfg['pressure_order'])
  if V.ndof+Q.ndof>CAPS['mixed_dofs']:raise MemoryError('primary DOF cap')
  layout,digest=layout_record(mesh,V,Q,ng)
  if digest!=m['layout_sha256'] or json.loads((self.directory/'fe_layout.json').read_text())!={'layout':layout,'sha256':digest}:raise ValueError('primary native layout differs')
  a=SimpleNamespace(ng=ng,mesh=mesh,V=V,Q=Q,cfg=cfg,D=cfg['D'],U=cfg['U'])
  a.volume=TrainingFEAdapter.mapped_rule(a,False)
  mapped=geometry_arrays(a)
  if array_digest(*(mapped[k] for k in sorted(mapped)))!=m['geometry_array_sha256']:raise ValueError('complete regenerated rule differs')
  self.adapter=a;self.ranges=None
  for name,c in self.coefficients.items():
   field=m['fields'][name];vector=name in VECTORS;space=V if vector else Q
   valid_shapes=[(space.ndof,field['columns'])]
   if field['columns']==1:valid_shapes.append((space.ndof,))
   if (type(field['vector']) is not bool or field['vector']!=vector or type(field['columns']) is not int or
       field['columns']<=0 or c.shape not in valid_shapes or array_digest(c)!=field['coefficient_array_sha256']):raise ValueError('primary coefficient descriptor differs')
   cursor=0;ranges=[]
   for row in field['blocks']:
    if row['start']!=cursor or not cursor<row['end']<=self.point_count:raise ValueError('primary block coverage')
    ranges.append((cursor,row['end']));cursor=row['end']
   if cursor!=self.point_count or (self.ranges is not None and ranges!=self.ranges):raise ValueError('primary common full partition')
   self.ranges=ranges
  self.replay_accounting={'setup_seconds':time.perf_counter()-started,'setup_guard':setup_guard,'iterations':[],'physical_admission':False,
                          'cost_scope':'generation is included in enclosing offline phase, not added twice; iterator elapsed includes consumer work'}
 def __getitem__(self,key):return self.geometry[key]
 def iter_blocks(self):
  from fe_blocks import evaluated_blocks_batched
  started=time.perf_counter();seen=0;complete=False;generation=0.
  scalars=sum((6 if name in VECTORS else 3)*field['columns'] for name,field in self.manifest['fields'].items())
  guard=host_guard(sum(c.nbytes for c in self.coefficients.values())+4*self.manifest['block_points']*scalars*8+2**26)
  streams={name:iter(evaluated_blocks_batched(self.adapter,c,name in VECTORS,False,self.manifest['block_points'])) for name,c in self.coefficients.items()}
  try:
   for index,(start,end) in enumerate(self.ranges):
    block_started=time.perf_counter()
    try:
     arrays={k:a[start:end] for k,a in self.geometry.items()}
     for name,stream in streams.items():
      actual_start,actual_end,values,gradients=next(stream);gradients=gradients*self.adapter.D
      if (actual_start,actual_end)!=(start,end) or array_digest(values,gradients)!=self.manifest['fields'][name]['blocks'][index]['array_sha256']:
       raise ValueError('regenerated field bytes differ; no numerical-tolerance substitution')
      arrays[name]=values;arrays[name+'_gradient_dimensionless']=gradients
    finally:generation+=time.perf_counter()-block_started
    seen+=end-start
    yield start,end,arrays
   for stream in streams.values():
    try:next(stream)
    except StopIteration:pass
    else:raise ValueError('extra regenerated points')
   complete=True
  finally:
   cleanup_started=time.perf_counter()
   for stream in streams.values():stream.close()
   generation+=time.perf_counter()-cleanup_started
   self.replay_accounting['iterations'].append({'iterator_elapsed_seconds_including_consumer':time.perf_counter()-started,
       'generation_and_verification_seconds':generation,'points_returned':seen,'workspace_guard':guard,
       'completed':complete,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})


def materialize(store,directory):
 """Restore the exact original point-store files; count all work as offline."""
 started=time.perf_counter();directory=Path(directory);m=store.original_manifest
 if directory.exists():raise FileExistsError('fresh materialization directory required')
 required=m['projected_expanded_bytes']+(len(m['files'])+1)*2**20
 if shutil.disk_usage(directory.parent).free<10*2**30+required:raise OSError('materialization disk reserve')
 for filename in m['files']:
  if Path(filename).name!=filename:raise ValueError('unsafe original field filename')
 directory.mkdir();shutil.copyfile(store.directory/'geometry.npz',directory/'geometry.npz')
 for index,(start,end,arrays) in enumerate(store.iter_blocks()):
  for name,field in m['fields'].items():
   row=field['blocks'][index]
   if (start,end)!=(row['start'],row['end']):raise ValueError('restored partition mismatch')
   path=directory/row['file']
   np.savez_compressed(path,values=arrays[name],gradients=arrays[name+'_gradient_dimensionless'])
   if sha(path)!=row['sha256']:raise ValueError('restored archive bytes differ; raw counterexample retained')
 shutil.copyfile(store.directory/'volume-manifest.json',directory/'manifest.json')
 from point_store import PointVolumeStore
 PointVolumeStore(directory,sha(directory/'manifest.json'))
 return {'seconds':time.perf_counter()-started,'restored_files':len(m['files'])+1,
         'manifest_sha256':sha(directory/'manifest.json'),'exact_archive_and_array_bytes':True,
         'offline_cost_includes_regeneration_verification_and_writes':True}
