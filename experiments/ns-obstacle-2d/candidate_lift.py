"""Reviewed FOM candidate initialization, distinct from G2 truth admission.

This loader never writes eligibility or admission. Coordinator-owned records bind
native scaling evidence and an exact query configuration. Calibration is restricted
to the two registered development parameters; a frozen comparator requires a
separate selection digest. Native-equivalence controls use Re=20 only.
"""
from pathlib import Path
import hashlib,json,math,time,zipfile
import numpy as np
from shared_lift import CommonStokesLift,sha


def configuration_sha256(cfg):
    return hashlib.sha256(json.dumps(cfg,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def digest(value):
    return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)


def load_npz(path,keys):
    if path.stat().st_size>2**30:raise MemoryError('candidate lift archive cap')
    with zipfile.ZipFile(path) as archive:
        if sum(row.file_size for row in archive.infolist())>2**30:raise MemoryError('candidate lift expanded archive cap')
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files)!=set(keys):raise ValueError('candidate lift archive schema mismatch')
        return {k:archive[k] for k in keys}


class CandidateStokesLift(CommonStokesLift):
    @classmethod
    def from_eligibility(cls,path,expected_mesh_sha256,expected_layout_sha256,expected_U,expected_D,
                         expected_configuration,*,expected_truth_admission_sha256=None,expected_selection_sha256=None):
        started=time.perf_counter();path=Path(path);eligibility=json.loads(path.read_text())
        if (eligibility.get('schema')!='ns-obstacle-fom-candidate-lift-eligibility-v1'
            or eligibility.get('native_control_reviewed') is not True or eligibility.get('physical_reference_admitted') is not False):
            raise ValueError('distinct reviewed candidate eligibility required; no coarse truth admission')
        role=eligibility.get('role')
        if role not in ('native_equivalence_control','development_calibration_only','frozen_fom_comparator'):
            raise ValueError('unknown FOM candidate role')
        if not digest(eligibility.get('review_sha256')):raise ValueError('native control review digest required')
        configuration_digest=configuration_sha256(expected_configuration)
        if eligibility.get('configuration_sha256')!=configuration_digest:raise ValueError('candidate query configuration mismatch')
        if role!='native_equivalence_control':
            if (not digest(expected_truth_admission_sha256)
                or eligibility.get('truth_reference_admission_sha256')!=expected_truth_admission_sha256):
                raise ValueError('calibration/comparator must bind the independently admitted truth')
        if role=='frozen_fom_comparator':
            if (eligibility.get('selection_frozen') is not True or not digest(expected_selection_sha256)
                or eligibility.get('selection_sha256')!=expected_selection_sha256
                or not digest(eligibility.get('selection_review_sha256'))):
                raise ValueError('independently frozen development-only comparator selection required')
        folder=path.parent/eligibility['directory'];manifest=folder/'common_stokes_lift.json';archive=folder/'common_stokes_lift.npz'
        control_path=folder/'control.json';arrays_path=folder/'lift_scaling_arrays.npz'
        for p,key in ((manifest,'manifest_sha256'),(archive,'coefficient_sha256'),(control_path,'native_control_sha256'),(arrays_path,'native_arrays_sha256')):
            if sha(p)!=eligibility.get(key):raise ValueError('candidate lift evidence hash mismatch: '+key)
        meta=json.loads(manifest.read_text());control=json.loads(control_path.read_text())
        if (meta.get('schema')!='ns-obstacle-common-stokes-v1' or meta.get('physical_admission') is not False
            or meta['mesh_sha256']!=expected_mesh_sha256 or meta['layout_sha256']!=expected_layout_sha256
            or meta['coefficient_sha256']!=sha(archive) or meta['U']!=expected_U or meta['D']!=expected_D
            or meta['outlet']!='gradient' or meta['pressure_gauge']!='natural outlet, unshifted'):
            raise ValueError('candidate lift mesh/layout/scaling mismatch')
        if (control.get('passed') is not True or control.get('physical_admission') is not False
            or control['layout_sha256']!=expected_layout_sha256
            or control['mesh_origin']['mesh_sha256']!=expected_mesh_sha256):
            raise ValueError('native scaling control is not bound to this candidate mesh/layout')
        cache=load_npz(archive,('velocity_coefficients','pressure_coefficients_per_nu'));u=cache['velocity_coefficients'];p=cache['pressure_coefficients_per_nu'];nv=meta['velocity_ndof'];nq=meta['pressure_ndof'];n=nv+nq
        if u.shape!=(nv,) or p.shape!=(nq,) or any(v.dtype!=np.dtype('float64') or not np.isfinite(v).all() for v in (u,p)):
            raise ValueError('finite owning candidate coefficients required')
        raw=load_npz(arrays_path,('reference_coefficients','states','residuals','free_mask','row_scale','forcing_vectors','trace_coefficients'))
        free=raw['free_mask'];scale=raw['row_scale'];states=raw['states'];residuals=raw['residuals'];forcing=raw['forcing_vectors'];trace=raw['trace_coefficients']
        if free.dtype!=np.dtype('bool') or free.shape!=(n,) or not free.any():raise ValueError('native scaling mask mismatch')
        for name in ('reference_coefficients','states','residuals','row_scale','forcing_vectors','trace_coefficients'):
            if raw[name].dtype!=np.dtype('float64') or not np.isfinite(raw[name]).all():raise ValueError('native scaling arrays must already be finite float64')
        if (scale.shape!=(n,) or trace.shape!=(n,) or raw['reference_coefficients'].shape!=(n,)
            or states.shape!=(3,n) or residuals.shape!=(3,n) or forcing.shape!=(3,n)):
            raise ValueError('native scaling array shape mismatch')
        expected_scale=np.r_[np.full(nv,expected_U**2*expected_D),np.full(nq,expected_U*expected_D)]
        if not np.array_equal(scale,expected_scale):raise ValueError('native scaling row units mismatch')
        tolerance=meta['linear_tolerance']
        if not math.isfinite(tolerance) or tolerance<=0 or meta['linear_control']['value']>tolerance:
            raise ValueError('original Stokes linear solve not certified')
        if not np.allclose(raw['reference_coefficients'],np.r_[u,meta['base_nu']*p],rtol=64*np.finfo(float).eps,atol=0):
            raise ValueError('cached lift differs from native base coefficients')
        records=control['records']
        if len(records)!=3 or [row['Re'] for row in records]!=[15.,20.,25.]:raise ValueError('registered native scaling controls missing')
        for index,row in enumerate(records):
            nu=expected_U*expected_D/row['Re']
            if row['nu']!=nu or not np.array_equal(states[index],np.r_[u,nu*p]):raise ValueError('cached viscosity scaling differs from native control')
            numerator=float(np.linalg.norm((residuals[index]/scale)[free]));denominator=max(float(np.linalg.norm((forcing[index]/scale)[free])),1e-30);value=numerator/denominator
            if (value>tolerance or not np.isclose(value,row['relative_residual'],rtol=1e-12,atol=0)
                or not np.isclose(numerator,row['numerator'],rtol=1e-12,atol=0)
                or not np.isclose(denominator,row['denominator'],rtol=1e-12,atol=0)):
                raise ValueError('native scaled residual replay failed')
            if not np.allclose(states[index,~free],trace[~free],rtol=0,atol=1e-12*expected_U):raise ValueError('candidate essential trace mismatch')
        self=cls();self.velocity=u.copy();self.pressure_per_nu=p.copy();self.velocity.setflags(write=False);self.pressure_per_nu.setflags(write=False)
        self.metadata=meta;self.usage_role=role;self.eligibility_sha256=sha(path);self.configuration_sha256=configuration_digest;self.offline_load_seconds=time.perf_counter()-started
        self.truth_reference_admission_sha256=eligibility.get('truth_reference_admission_sha256')
        self.selection_sha256=eligibility.get('selection_sha256')
        self.selection_review_sha256=eligibility.get('selection_review_sha256')
        self.selection_frozen=eligibility.get('selection_frozen') is True
        return self

    def allows_Re(self,Re):
        if self.usage_role=='native_equivalence_control':return Re==20.
        if self.usage_role=='development_calibration_only':return Re in (16.875,23.125)
        return self.usage_role=='frozen_fom_comparator' and 15<=Re<=25
