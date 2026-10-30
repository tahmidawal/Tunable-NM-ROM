"""Exact retained TRAIN cylinder POD gradients; no FE reconstruction or flow labels."""
from pathlib import Path
import json
import numpy as np
from fe_adapter import sha_file
from accuracy_v2.train import digest
from accuracy_v2.boundary_traction_objective import teacher_normalization


def load(prepared,metadata,identity,directory):
    out=Path(directory);out.mkdir(exist_ok=False)
    path=Path(prepared)/'boundary.npz'
    if metadata['split']!='training' or metadata['Re']!=[15+1.25*j for j in range(9)] or identity['Re']!=metadata['Re'] or identity['split']!='training':raise ValueError('nine actual TRAIN cases required')
    if sha_file(path)!=metadata['coefficient_files']['boundary.npz']:raise ValueError('accepted boundary file changed')
    if metadata['gradient_axes']!='point, component, derivative_coordinate, column' or metadata['units']['all_exported_field_gradients']!='derivative with respect to x/D' or metadata['units']['boundary_weights']!='ds/D' or metadata['scales']['D']!=.1:raise ValueError('accepted boundary units/axes required')
    with np.load(path,allow_pickle=False) as f:
        labels=f['labels'];mask=labels=='obstacle';indices=np.flatnonzero(mask)
        full={k:f[k] for k in ('points_physical','points_dimensionless','weights_dimensionless','normals_fluid','pod_velocity_gradient_dimensionless')}
    if not len(indices):raise ValueError('empty cylinder subset')
    x=full['points_physical'][mask];xd=full['points_dimensionless'][mask];w=full['weights_dimensionless'][mask];normal=-full['normals_fluid'][mask];g=full['pod_velocity_gradient_dimensionless'][mask,...,:4]
    if x.shape!=(len(w),2) or g.shape!=(len(w),2,2,4) or not np.array_equal(x/.1,xd) or any(not np.isfinite(a).all() for a in (x,xd,w,normal,g)):raise ValueError('finite exact cylinder layout/scales required')
    sigma=np.asarray(identity['singular_values'],dtype=np.float64)
    if not np.array_equal(sigma,metadata['pod']['velocity']['small_R_singular_values_padded'][:4]):raise ValueError('TRAIN teacher column ordering changed')
    arrays={'indices':indices,'points_physical':x,'points_dimensionless':xd,'weights_dimensionless':w,'normals_body':normal,'teacher_gradient_dimensionless':g,'singular_values':sigma}
    # Keep the actual selected arrays before normalizer acceptance.
    np.savez_compressed(out/'cylinder-teacher.npz',**arrays)
    norm=teacher_normalization(g,normal,w,sigma)
    record={'schema':'ns-obstacle-training-cylinder-traction-teacher-v1','split':'training','Re':identity['Re'],'boundary_file_sha256':sha_file(path),'prepared_manifest_sha256':identity['prepared_manifest_sha256'],'volume_teacher_identity':identity,'mask_definition':"labels == 'obstacle'; original increasing indices",'normal_definition':'negative accepted outward-fluid normal','gradient_units':'d(u/U)/d(x/D)','weight_units':'ds/D','Re_in_target':False,'pressure_in_target':False,'source_sha256':sha_file(__file__),'selected_arrays_sha256':sha_file(out/'cylinder-teacher.npz'),'arrays':{k:digest(v) for k,v in arrays.items()},'normalization':{k:np.asarray(v).tolist() for k,v in norm.items()}}
    (out/'RESULT.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    return arrays,record
