"""Read actual training POD teachers through the existing complete point-store closure."""
from pathlib import Path
import numpy as np
from point_store import PointVolumeStore
from pilot2d import read_preparation_metadata
from capacity_context2d import consumer_caps

def load(prepared,release,rank=4):
    metadata,r=read_preparation_metadata(prepared,release)
    from fe_adapter import CAPS
    caps=consumer_caps(metadata,release,r,CAPS)
    if rank!=4:raise ValueError('first reviewed candidate has four teachers')
    store=PointVolumeStore(Path(prepared)/'volume',metadata['coefficient_files']['volume/manifest.json'])
    n=store.point_count
    if n>caps['volume_points']:raise MemoryError('point count exceeds approved capacity')
    v=np.empty((n,2,rank));g=np.empty((n,2,2,rank))
    for start,end,block in store.iter_blocks():
        v[start:end]=block['pod_velocity'][...,:rank]
        g[start:end]=block['pod_velocity_gradient_dimensionless'][...,:rank]
    from accuracy_v2.train import digest
    from qualification import require_qualification
    x=store['points_physical'];w=store['weights_dimensionless']
    identity={'split':metadata['split'],'Re':metadata['Re'],'reference_qualification':require_qualification(metadata['reference_qualification']),'prepared_manifest_sha256':r['adapter_manifest_sha256'],'reference_admission_sha256':metadata['admission_evidence']['reference_admission_sha256'],'volume_manifest_sha256':metadata['coefficient_files']['volume/manifest.json'],'training_array_sha256':metadata['training_array_sha256'],'arrays':{'points':digest(x),'weights':digest(w),'values':digest(v),'gradients':digest(g)}}
    return x,w,v,g,metadata,identity
