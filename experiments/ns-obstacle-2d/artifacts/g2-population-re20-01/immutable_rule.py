"""Hash-bound immutable geometry and certified common-point native readout caches.

Only field values depend on the changing coefficients. Native transformations/rules
remain owned by each live cache; no native pointer is serialized. Construction and
validation are offline and must be measured separately from subsequent evaluations.
"""
from pathlib import Path
import hashlib,json,time,zipfile
import numpy as np
import reference_export as rex

SCHEMA='ns-obstacle-immutable-rule-cache-v1'


def array_digest(value):
    a=np.ascontiguousarray(value);h=hashlib.sha256(json.dumps({'dtype':a.dtype.str,'shape':list(a.shape)},sort_keys=True).encode());h.update(memoryview(a).cast('B'));return h.hexdigest()


def _f64(a,shape,name):
    a=np.asarray(a)
    if a.dtype!=np.dtype('float64') or a.shape!=shape or not np.isfinite(a).all():raise ValueError('finite f64 cache shape mismatch: '+name)
    return a


def write_cache(folder,metadata,arrays,support_files=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=False);archive=folder/'arrays.npz'
    np.savez_compressed(archive,**arrays)
    support={}
    for name,data in (support_files or {}).items():
        if Path(name).name!=name:raise ValueError('cache support must be beside manifest')
        (folder/name).write_bytes(data);support[name]=rex.sha_file(folder/name)
    record={**metadata,'support_files':support,'schema':SCHEMA,'arrays_file':'arrays.npz','arrays_sha256':rex.sha_file(archive),'arrays':{k:{'shape':list(v.shape),'dtype':v.dtype.str,'content_sha256':array_digest(v)} for k,v in arrays.items()},'physical_admission':False,'primary_geometry_arrays_retained_losslessly':True}
    rex.dump(folder/'cache.json',record)
    return {'manifest_path':str((folder/'cache.json').resolve()),'manifest_sha256':rex.sha_file(folder/'cache.json')}


def load_cache(descriptor,mesh_path,verified_layout_sha256,cfg,ng):
    path=Path(descriptor['manifest_path']).resolve()
    if rex.sha_file(path)!=descriptor['manifest_sha256']:raise ValueError('immutable cache manifest hash mismatch')
    m=json.loads(path.read_text())
    if (m.get('schema')!=SCHEMA or m['mesh_sha256']!=rex.sha_file(mesh_path)
        or m['layout_sha256']!=verified_layout_sha256 or m['geometry']!=rex.geometry_signature(cfg)
        or m['ngsolve_version']!=ng.__version__):raise ValueError('immutable cache mesh/layout/geometry/runtime mismatch')
    for name,digest in m.get('support_files',{}).items():
        if Path(name).name!=name or rex.sha_file(path.parent/name)!=digest:raise ValueError('immutable cache support evidence mismatch')
    name=m['arrays_file']
    if Path(name).name!=name:raise ValueError('cache array path must remain beside manifest')
    archive=path.parent/name
    if rex.sha_file(archive)!=m['arrays_sha256'] or archive.stat().st_size>2**30:raise ValueError('cache array hash/size mismatch')
    with zipfile.ZipFile(archive) as z:
        if sum(x.file_size for x in z.infolist())>2**30:raise MemoryError('cache expanded array cap')
    with np.load(archive,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    if set(arrays)!=set(m['arrays']):raise ValueError('cache array inventory mismatch')
    for key,value in arrays.items():
        expected=m['arrays'][key]
        if (list(value.shape)!=expected['shape'] or value.dtype.str!=expected['dtype'] or array_digest(value)!=expected['content_sha256']):raise ValueError('cache array identity mismatch')
        value.setflags(write=False)
    count=len(arrays['weights']);_f64(arrays['points'],(count,2),'points');_f64(arrays['weights'],(count,),'weights')
    cap=rex.CAPS['boundary_points'] if m['region']=='boundary' else rex.volume_point_cap(cfg)
    if not 0<count<=cap or np.any(arrays['weights']<=0):raise ValueError('complete positive cache rule/cap required')
    if m['point_count']!=count:raise ValueError('cache point count changed')
    return m,arrays


class _FieldReadout:
    def _field(self,cf,shape):
        a=np.asarray(cf(self.mapped))
        if a.dtype!=np.dtype('float64') or a.size!=int(np.prod(shape)) or not np.isfinite(a).all():raise ValueError('native cached field is not finite f64')
        value=np.array(a.reshape(shape),copy=True)
        if self.native_to_original is not None:
            reordered=np.empty_like(value);reordered[self.native_to_original]=value;value=reordered
        return value

    def evaluate(self,state,*,gradient=True):
        if state.space is not self.owner_space or state.space.mesh is not self.mesh:raise ValueError('changing state does not own the cached native mesh/space')
        velocity,pressure=state.components
        start=time.perf_counter();n=len(self.arrays['weights']);fields={'velocity':self._field(velocity,(n,2)),'pressure':self._field(pressure,(n,))}
        if gradient:
            derivative=self.ng.Grad(velocity)
            if self.metadata['region']=='boundary':derivative=self.ng.BoundaryFromVolumeCF(derivative)
            fields['velocity_gradient']=self._field(derivative,(n,2,2))
        return fields,{'evaluation_seconds':time.perf_counter()-start,'point_count':n,'all_points_evaluated':True,'field_content_sha256':{k:array_digest(v) for k,v in fields.items()}}

    def _certify_coordinates(self,cfg):
        n=len(self.arrays['weights']);coordinates=self._field(self.ng.CF((self.ng.x,self.ng.y)),(n,2));errors=np.linalg.norm(coordinates-self.arrays['points'],axis=1)
        self.reconstruction_errors=errors
        if np.any(errors>rex.RELOAD_TOL*cfg['D']):raise ValueError('cached native map fails unchanged physical-coordinate certificate')
        self.certificate={'all_points_preserved':True,'point_adjustments':0,'discarded_points':0,'point_count':n,'max_reconstruction_error':float(errors.max()),'physical_tolerance':rex.RELOAD_TOL*cfg['D'],'layout_sha256':self.metadata['layout_sha256'],'mesh_sha256':self.metadata['mesh_sha256']}


class NativeRuleCache(_FieldReadout):
    """Original element/rule order with immutable points/weights/Jacobians/normals."""
    def __init__(self,ng,mesh,mesh_path,verified_layout_sha256,cfg,descriptor,*,space,normal=None,normal_factors=None):
        started=time.perf_counter();self.ng,self.mesh=ng,mesh;self.owner_space=space
        if space.mesh is not mesh:raise ValueError('cache space and native mesh are different live objects')
        self.metadata,self.arrays=load_cache(descriptor,mesh_path,verified_layout_sha256,cfg,ng)
        m=self.metadata
        if m['mode']!='native_element_rule' or m['region'] not in ('volume','boundary') or m['order']!=cfg['diagnostic_order']:raise ValueError('native cache rule/order mismatch')
        boundary=m['region']=='boundary';n=len(self.arrays['weights']);self.native_to_original=None;self.owners=[];batches=[];element_record=[];labels=[];built_count=0
        if boundary:
            if normal is None or normal_factors!=m['normal_orientation_factors']:raise ValueError('ordinary-boundary normal orientation identity changed')
            _f64(self.arrays['normals'],(n,2),'normals');_f64(self.arrays['raw_normals'],(n,2),'raw normals')
        else:
            jac=_f64(self.arrays['jacobians'],(n,2,2),'jacobians')
            if np.any(np.linalg.det(jac)<=0):raise ValueError('nonpositive immutable mapped Jacobian')
        for el in mesh.Elements(ng.BND if boundary else ng.VOL):
            tr=mesh.GetTrafo(el);ir=ng.IntegrationRule(el.type,m['order'])
            if built_count+len(ir)>n:raise MemoryError('native rule exceeds recorded point cap before growth')
            mapped=tr(ir);built_count+=len(ir)
            self.owners.append((el,tr,ir,mapped));batches.append(mapped);element_record.append([int(el.nr),str(el.type),len(ir)])
            if boundary:labels.extend([str(el.mat)]*len(ir))
        if element_record!=m['element_rule_record']:raise ValueError('immutable element/rule enumeration changed')
        self.mapped=np.concatenate(batches);self.owners.append(self.mapped)
        if len(self.mapped)!=n:raise ValueError('native cache lost rule points')
        if boundary:
            if labels!=self.arrays['labels'].tolist():raise ValueError('native boundary labels changed')
            for actual,key in ((self._field(normal,(n,2)),'normals'),(self._field(ng.specialcf.normal(2),(n,2)),'raw_normals')):
                if np.max(np.abs(actual-self.arrays[key]))>rex.RELOAD_TOL:raise ValueError('cached boundary normal differs')
        self._certify_coordinates(cfg);self.offline_seconds=time.perf_counter()-started


class CommonPointCache(_FieldReadout):
    """Replay existing certified reference coordinates; no new lookup or nudges."""
    def __init__(self,ng,mesh,mesh_path,verified_layout_sha256,cfg,descriptor,*,space):
        started=time.perf_counter();self.ng,self.mesh=ng,mesh;self.owner_space=space
        if space.mesh is not mesh:raise ValueError('cache space and native mesh are different live objects')
        self.metadata,self.arrays=load_cache(descriptor,mesh_path,verified_layout_sha256,cfg,ng)
        m=self.metadata;n=len(self.arrays['weights'])
        if m['mode']!='certified_common_points' or m['region']!='volume':raise ValueError('common-point cache role mismatch')
        ref=_f64(self.arrays['reference_coordinates'],(n,3),'reference coordinates');bary=_f64(self.arrays['barycentric_coordinates'],(n,3),'barycentrics');elements=self.arrays['element_numbers'];valid=self.arrays['valid'];eps=64*np.finfo(np.float64).eps
        if elements.dtype!=np.dtype('int64') or elements.shape!=(n,) or valid.dtype!=np.dtype('bool') or valid.shape!=(n,) or not valid.all():raise ValueError('invalid original mapping record')
        reconstructed_bary=np.column_stack((ref[:,0],ref[:,1],1-ref[:,0]-ref[:,1]))
        if not np.array_equal(bary,reconstructed_bary) or np.min(bary)<-eps or np.max(np.abs(ref[:,2]))>eps:raise ValueError('strict original triangle membership failed')
        lookup={int(e.nr):e for e in mesh.Elements(ng.VOL)};order=np.argsort(elements,kind='stable');sorted_elements=elements[order];starts=np.r_[0,np.flatnonzero(np.diff(sorted_elements))+1,n];self.owners=[];batches=[]
        for lo,hi in zip(starts[:-1],starts[1:]):
            ids=order[lo:hi];nr=int(sorted_elements[lo]);el=lookup.get(nr)
            if el is None or str(el.type)!='ET.TRIG':raise ValueError('saved common point does not name an actual triangle')
            ir=ng.IntegrationRule([tuple(x) for x in ref[ids,:2]],[1.]*len(ids));tr=mesh.GetTrafo(el);mapped=tr(ir)
            self.owners.append((el,tr,ir,mapped,ids));batches.append(mapped)
        self.mapped=np.concatenate(batches);self.owners.append(self.mapped);self.native_to_original=order
        if len(self.mapped)!=n:raise ValueError('common cache lost points')
        self._certify_coordinates(cfg);self.certificate.update(reference_membership_roundoff=eps,original_lookup_evidence=m['original_lookup_evidence'],native_to_original_sha256=array_digest(order));self.offline_seconds=time.perf_counter()-started
