"""Exact point-block field archives, bounded per payload and total disk use.

Each coefficient field is evaluated on every original point. Geometry is shared once;
readers iterate one complete block and never concatenate a full gradient bank.
"""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile
import numpy as np
from fe_algebra import array_digest


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read_npz(path,max_bytes):
    with zipfile.ZipFile(path) as archive:
        if sum(row.file_size for row in archive.infolist())>max_bytes:raise MemoryError('expanded block cap')
    with np.load(path,allow_pickle=False) as archive:
        result={key:archive[key].copy() for key in archive.files}
    if any(a.dtype!=np.dtype('float64') or not np.isfinite(a).all() for a in result.values()):
        raise ValueError('finite original float64 blocks required')
    return result


def write_volume(adapter,directory,block_points=4096,max_payload_bytes=2**28,
                 max_total_bytes=8*2**30,reserve_bytes=10*2**30):
    from fe_blocks import evaluated_blocks_batched as evaluated_blocks
    directory=Path(directory)
    if directory.exists():raise FileExistsError('fresh point-store directory required')
    n=len(adapter.volume['points'])
    projected_geometry=sum(np.asarray(adapter.volume[key]).nbytes for key in ('points','weights','jacobians'))+n*2*8
    field_specs={}
    for name,c in adapter.coefficients.items():
        vector=name in ('lift_u','pod_velocity','supremizers','momentum_tests')
        coefficient=np.asarray(c)
        if coefficient.ndim not in (1,2):raise ValueError('coefficient vector or column matrix required')
        count=1 if coefficient.ndim==1 else coefficient.shape[1]
        field_specs[name]={'vector':vector,'columns':count,'scalars_per_point':(6 if vector else 3)*count}
    projected=projected_geometry+n*8*sum(s['scalars_per_point'] for s in field_specs.values())
    # NPZ headers/file systems add small overhead; reserve 1 MiB per field block.
    conservative=projected+(1+(n+block_points-1)//block_points*len(field_specs))*2**20
    if conservative>max_total_bytes:raise MemoryError('total point-store payload budget')
    parent=directory.parent
    if not parent.is_dir():raise FileNotFoundError('point-store parent must exist')
    if shutil.disk_usage(parent).free<conservative+reserve_bytes:raise OSError('retention disk reserve')
    directory.mkdir()
    geometry={'points_physical':adapter.volume['points'],'points_dimensionless':adapter.volume['points']/adapter.D,
              'weights_dimensionless':adapter.volume['weights']/adapter.D**2,
              'jacobians_physical':adapter.volume['jacobians']}
    if sum(a.nbytes for a in geometry.values())>max_payload_bytes:raise MemoryError('geometry payload cap')
    np.savez_compressed(directory/'geometry.npz',**geometry)
    files={'geometry.npz':sha(directory/'geometry.npz')};fields={};actual=(directory/'geometry.npz').stat().st_size
    for name,c in adapter.coefficients.items():
        rows=[]
        for start,end,values,gradients in evaluated_blocks(adapter,c,field_specs[name]['vector'],False,block_points):
            gradients=gradients*adapter.D
            if values.nbytes+gradients.nbytes>max_payload_bytes:raise MemoryError('field block payload cap')
            filename=f'{name}-{start:09d}.npz';path=directory/filename
            np.savez_compressed(path,values=values,gradients=gradients)
            digest=sha(path);files[filename]=digest;actual+=path.stat().st_size
            if actual>max_total_bytes:raise MemoryError('actual point-store disk budget')
            rows.append({'start':start,'end':end,'file':filename,'sha256':digest,
                         'array_sha256':array_digest(values,gradients)})
        fields[name]=dict(field_specs[name],blocks=rows)
    manifest={'schema':'ns-obstacle-point-volume-v1','points':n,'block_points':block_points,
        'fields':fields,'files':files,'geometry_file':'geometry.npz','all_original_points':True,
        'units':{'values':'dimensionless','gradients':'d/d(x/D)','weights':'dx/D**2'},
        'projected_expanded_bytes':projected,'actual_compressed_bytes':actual,
        'max_payload_bytes':max_payload_bytes,'max_total_bytes':max_total_bytes,
        'reserve_bytes':reserve_bytes,'physical_admission':False}
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


class PointVolumeStore:
    def __init__(self,directory,expected_manifest_sha256,max_payload_bytes=2**28,max_total_bytes=8*2**30):
        self.directory=Path(directory);path=self.directory/'manifest.json'
        if sha(path)!=expected_manifest_sha256:raise ValueError('point-store manifest hash')
        self.manifest=json.loads(path.read_text());m=self.manifest
        if m['schema']!='ns-obstacle-point-volume-v1' or m['all_original_points'] is not True:raise ValueError('point-store schema')
        if type(m['points']) is not int or m['points']<=0 or type(m['block_points']) is not int or m['block_points']<=0:
            raise ValueError('positive integer point counts required')
        if m['units']!={'values':'dimensionless','gradients':'d/d(x/D)','weights':'dx/D**2'}:
            raise ValueError('point-store units differ')
        self.max_payload_bytes=max_payload_bytes;self.point_count=m['points']
        if m['actual_compressed_bytes']>max_total_bytes or m['projected_expanded_bytes']>max_total_bytes:raise MemoryError('point-store total size cap')
        actual=0
        for name,digest in m['files'].items():
            if Path(name).name!=name:raise ValueError('unsafe point-store file path')
            path=self.directory/name
            if sha(path)!=digest:raise ValueError('point-store file hash')
            actual+=path.stat().st_size
        if actual!=m['actual_compressed_bytes'] or actual>max_total_bytes:raise ValueError('point-store size differs')
        if m['geometry_file'] not in m['files']:raise ValueError('unbound geometry file')
        self.geometry=read_npz(self.directory/m['geometry_file'],max_payload_bytes)
        shapes={'points_physical':(self.point_count,2),'points_dimensionless':(self.point_count,2),
                'weights_dimensionless':(self.point_count,),'jacobians_physical':(self.point_count,2,2)}
        if set(self.geometry)!=set(shapes) or any(self.geometry[k].shape!=shape for k,shape in shapes.items()):
            raise ValueError('geometry keys/shapes differ')
        if np.min(self.geometry['weights_dimensionless'])<=0 or np.min(np.linalg.det(self.geometry['jacobians_physical']))<=0:
            raise ValueError('positive mapped weights/Jacobians required')
        expected_ranges=None
        for name,field in m['fields'].items():
            if type(field['columns']) is not int or field['columns']<=0 or type(field['vector']) is not bool:
                raise ValueError('positive field columns and explicit vector flag required')
            cursor=0;ranges=[]
            for row in field['blocks']:
                if row['start']!=cursor or not cursor<row['end']<=self.point_count:raise ValueError('incomplete/reordered field points')
                if m['files'].get(row['file'])!=row['sha256']:raise ValueError('field block hash binding')
                ranges.append((cursor,row['end']));cursor=row['end']
            if cursor!=self.point_count:raise ValueError('incomplete field point set')
            if expected_ranges is not None and ranges!=expected_ranges:raise ValueError('different field partitions')
            expected_ranges=ranges
        self.ranges=expected_ranges

    def __getitem__(self,key):
        return self.geometry[key]

    def iter_blocks(self):
        for index,(start,end) in enumerate(self.ranges):
            arrays={key:value[start:end] for key,value in self.geometry.items()}
            for name,field in self.manifest['fields'].items():
                row=field['blocks'][index];data=read_npz(self.directory/row['file'],self.max_payload_bytes)
                count=field['columns'];shape=(end-start,2,count) if field['vector'] else (end-start,count)
                gshape=(end-start,2,2,count) if field['vector'] else (end-start,2,count)
                if set(data)!={'values','gradients'} or data['values'].shape!=shape or data['gradients'].shape!=gshape:raise ValueError('field block shape')
                if 'array_sha256' in row and array_digest(data['values'],data['gradients'])!=row['array_sha256']:
                    raise ValueError('field block array byte hash')
                arrays[name]=data['values'];arrays[name+'_gradient_dimensionless']=data['gradients']
            yield start,end,arrays
