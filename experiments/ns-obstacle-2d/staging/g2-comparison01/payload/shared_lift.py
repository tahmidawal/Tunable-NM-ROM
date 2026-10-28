"""One common Stokes lift: offline export/load, cheap cached viscosity scaling.

No solver runs here. The future tuned query runner must start from state_for_nu,
not re-factor the identical Stokes problem. Reference generation stays separate.
"""
from pathlib import Path
import hashlib
import json
import math
import time
import numpy as np


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def split_common_lift(coefficients,velocity_ndof,nu):
    c=np.asarray(coefficients)
    if c.dtype!=np.dtype('float64') or c.ndim!=1 or not np.isfinite(c).all():raise ValueError('finite float64 mixed Stokes vector required')
    if not math.isfinite(nu) or nu<=0 or not 0<velocity_ndof<len(c):raise ValueError('invalid Stokes scaling or block size')
    return c[:velocity_ndof].copy(),c[velocity_ndof:].copy()/nu


def export_common_lift(out,cfg,mesh_sha256,layout,coefficients,linear_control,reference_stokes_seconds):
    if cfg['kind']!='dfg' or cfg['outlet']!='gradient':raise ValueError('common family lift supports unforced gradient-outlet DFG geometry only')
    if linear_control['value']>cfg['linear_relative_tolerance']:raise ArithmeticError('base Stokes solve was not certified')
    out=Path(out);u,p_per_nu=split_common_lift(coefficients,layout['layout']['velocity_ndof'],cfg['nu'])
    archive=out/'common_stokes_lift.npz'
    np.savez_compressed(archive,velocity_coefficients=u,pressure_coefficients_per_nu=p_per_nu)
    record={'schema':'ns-obstacle-common-stokes-v1','physical_admission':False,'coordinator_admission_required':True,
            'mesh_sha256':mesh_sha256,'layout_sha256':layout['sha256'],'U':cfg['U'],'D':cfg['D'],'base_nu':cfg['nu'],
            'velocity_ndof':len(u),'pressure_ndof':len(p_per_nu),'outlet':'gradient','pressure_gauge':'natural outlet, unshifted',
            'linear_control':linear_control,'linear_tolerance':cfg['linear_relative_tolerance'],
            'reference_stokes_phase_seconds':reference_stokes_seconds,
            'timing_scope':'reference linear assembly/factor/solve/check only; preceding mesh/space/form setup recorded separately; no tuned offline timing claim',
            'coefficient_file':'common_stokes_lift.npz','coefficient_sha256':sha(archive),
            'query_contract':'cache these arrays offline; each query copies velocity and scales pressure by its viscosity; no repeated Stokes factorization'}
    (out/'common_stokes_lift.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    return record


class CommonStokesLift:
    """Cached initialization only, not a FOM query solver or a timing result."""
    @classmethod
    def from_admission(cls,path,expected_mesh_sha256,expected_layout_sha256,expected_U,expected_D):
        started=time.perf_counter();path=Path(path);admission=json.loads(path.read_text())
        if (admission.get('schema')!='ns-obstacle-common-stokes-admission-v1' or admission.get('gate')!='G2-steady'
            or admission.get('passed') is not True):raise RuntimeError('genuine independently reviewed G2 common-lift admission required')
        for key in ('reference_admission_sha256','review_sha256'):
            value=admission.get(key,'')
            if len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('missing admission evidence digest')
        folder=path.parent/admission['directory'];manifest=folder/'common_stokes_lift.json';archive=folder/'common_stokes_lift.npz'
        if sha(manifest)!=admission['manifest_sha256'] or sha(archive)!=admission['coefficient_sha256']:
            raise ValueError('common-lift artifact hash mismatch')
        meta=json.loads(manifest.read_text())
        if (meta['schema']!='ns-obstacle-common-stokes-v1' or meta['mesh_sha256']!=expected_mesh_sha256
            or meta['layout_sha256']!=expected_layout_sha256 or meta['coefficient_sha256']!=sha(archive)
            or meta['U']!=expected_U or meta['D']!=expected_D or meta['outlet']!='gradient'
            or meta['pressure_gauge']!='natural outlet, unshifted'):
            raise ValueError('common lift geometry/space identity mismatch')
        if archive.stat().st_size>2**30:raise MemoryError('common lift archive exceeds fixed cap')
        import zipfile
        with zipfile.ZipFile(archive) as z:
            if sum(row.file_size for row in z.infolist())>2**30:raise MemoryError('uncompressed lift exceeds fixed cap')
        with np.load(archive,allow_pickle=False) as z:
            if set(z.files)!={'velocity_coefficients','pressure_coefficients_per_nu'}:raise ValueError('unexpected common lift arrays')
            u=z['velocity_coefficients'];p=z['pressure_coefficients_per_nu']
        if (u.shape!=(meta['velocity_ndof'],) or p.shape!=(meta['pressure_ndof'],)
            or any(v.dtype!=np.dtype('float64') or not np.isfinite(v).all() for v in (u,p))):
            raise ValueError('invalid common lift arrays')
        self=cls();self.velocity=u.copy();self.pressure_per_nu=p.copy()
        self.velocity.setflags(write=False);self.pressure_per_nu.setflags(write=False)
        self.metadata=meta;self.admission_sha256=sha(path);self.offline_load_seconds=time.perf_counter()-started
        return self

    def state_for_nu(self,nu):
        if not math.isfinite(nu) or nu<=0:raise ValueError('positive finite viscosity required')
        # The returned owning array may be mutated by Newton; cached lift stays fixed.
        return np.concatenate((self.velocity,nu*self.pressure_per_nu))
