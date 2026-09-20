"""Independent NumPy/SciPy checks of retained POD bases and Galerkin fields."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import cho_factor, cho_solve


def laplacian_columns(basis,n):
    shape=((n-1),)*3+(basis.shape[1],)
    field=basis.reshape(shape)
    lap=6*n*n*field.copy()
    for axis in range(3):
        lo=[slice(None)]*4;hi=lo.copy();lo[axis]=slice(None,-1);hi[axis]=slice(1,None)
        lap[tuple(lo)]-=n*n*field[tuple(hi)]
        lap[tuple(hi)]-=n*n*field[tuple(lo)]
    return lap.reshape(basis.shape)


def prolong_independently(basis,n,m):
    padded=np.pad(basis.reshape(((n-1),)*3+(basis.shape[1],)),((1,1),(1,1),(1,1),(0,0)))
    axis=np.arange(n+1,dtype=np.float64)/n
    interpolate=RegularGridInterpolator((axis,axis,axis),padded,method='linear')
    fine=np.arange(1,m,dtype=np.float64)/m
    points=np.stack(np.meshgrid(fine,fine,fine,indexing='ij'),axis=-1).reshape(-1,3)
    result=np.empty((len(points),basis.shape[1]),dtype=np.float64)
    for start in range(0,len(points),4096):result[start:start+4096]=interpolate(points[start:start+4096])
    return result


def audit(out,record):
    out=Path(out);cfg=record['config'];checks=[];bases={};orthogonality=[]
    for mesh in record['meshes']:
        n=mesh['intervals'];basis=np.load(out/f'offline/pod_N{n}.npy',allow_pickle=False)
        info=json.loads((out/f'offline/pod_N{n}.json').read_text())
        assert info==mesh['pod'] and basis.dtype==np.float64
        assert basis.shape==((n-1)**3,info['retained_rank'])
        defect=float(np.max(np.abs(basis.T@basis-np.eye(basis.shape[1]))))
        assert defect<1e-6
        orthogonality.append(dict(intervals=n,maximum_gram_defect=defect))
        bases[n]=basis
        fields=[(name,meta) for name,meta in mesh['methods'].items() if meta['kind']=='pod']
        transfer=None
        for name,meta in fields:
            if meta.get('frozen_mesh_transfer'):
                native=meta['training_intervals']
                if transfer is None:transfer=prolong_independently(bases[native],native,n)
                matrix=transfer[:,:meta['rank']]
            else:matrix=basis[:,:meta['rank']]
            operator=matrix.T@laplacian_columns(matrix,n)
            reference=np.load(out/'fields'/f'N{n}_case0_reference.npz')['forcing'].reshape(-1)
            coefficients=cho_solve(cho_factor(operator,lower=True),matrix.T@reference)
            predicted=matrix@coefficients
            stored=np.load(out/'fields'/f'N{n}_case0_{name}.npz')['prediction'].reshape(-1)
            difference=float(np.linalg.norm(predicted-stored)/np.linalg.norm(stored))
            assert difference<1e-8,(n,name,difference)
            checks.append(dict(intervals=n,method=name,case=0,relative_field_difference=difference))
    return dict(passed=True,checks=checks,basis_orthogonality=orthogonality,
                tolerance=1e-8,final_assets_frozen=record['final_cohort_opened'],
                scope='One saved field per POD method/mesh reconstructed through independent finite-difference Galerkin algebra; transferred bases use SciPy interpolation. All saved fields separately receive reference/error/hash audits.')
