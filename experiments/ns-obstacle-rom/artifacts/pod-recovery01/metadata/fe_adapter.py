"""Gated, bounded 2D NGSolve-to-ROM bridge. No FE operation runs on import.

This source has NOT passed a pinned-library FE integration test. A future reviewed job
must supply a genuine FD1-scoped training manifest; current G1 control files are rejected.
Only this module owns the adapter; it never imports or mutates the FOM solver module.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import resource
import shutil
import time
import zipfile
import numpy as np
from fe_algebra import array_digest, reference_inf_sup
from fe_pod_a6 import (coefficient_pod_a6, validate_actual_pod, orient_clusters,
                       expand_components, check_component_blocks)
from fe_controls_a6 import boundary_control, gradient_control
from qualification import require_qualification, EXPORT_RELEASE, SCOPE_SHA256
from spaces import select_test_columns
from oriented_normals import fluid_outward_normal

SCHEMA = 'ns-obstacle-fe-training-v1'
REYNOLDS = [15 + 1.25*j for j in range(9)]
ESSENTIAL = 'inlet|wall.*|obstacle'
MESH_FILENAME = 'mesh.vol.bin'
MESH_FORMAT = 'netgen-binary-archive-v6.2.2608'
RELOAD_POLICY = 'saved_curved_binary_verified_fresh_process_no_recurve'
FILES = ('config.json', MESH_FILENAME, 'full_rule.npz', 'boundary_rule.json',
         'solution.sol', 'result.json', 'fe_layout.json', 'reload_checks.json',
         'mesh_identity.json', 'normal_orientation.json', 'reload_normal_orientation.json',
         'EXPORT-MANIFEST.json')
CAPS = {'mixed_dofs': 100000, 'elements': 20000, 'volume_points': 100000,
        'boundary_points': 30000, 'matrix_nnz': 12000000,
        'single_array_bytes': 2**30, 'archive_bytes': 2**30,
        'factor_estimate_bytes': 8*2**30, 'process_rss_bytes': 16*2**30,
        'vector_eigen_candidates': 96, 'velocity_scalar_eigen_candidates': 48, 'scalar_eigen_candidates': 32,
        'eigensolver_iterations': 2000}


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def adapter_digest():
    root = Path(__file__).parent
    return hashlib.sha256(json.dumps({name: sha_file(root/name) for name in
        ('fe_adapter.py', 'fe_blocks.py', 'component_factor.py', 'qualification.py', 'FD1-SCOPE-CONTRACT.json', 'point_store.py', 'fe_algebra.py', 'fe_pod_a6.py', 'fe_controls_a6.py', 'spaces.py', 'weak.py', 'train_bank.py', 'oriented_normals.py', 'scalars.py', 'MODEL-CONTRACT.json')}, sort_keys=True).encode()).hexdigest()


def verify_mesh_contract(record, policy_key='policy'):
    """Require the declared binary format; no text-mesh or automatic re-curve fallback."""
    if (record.get('mesh_filename') != MESH_FILENAME or record.get('mesh_format') != MESH_FORMAT
            or record.get(policy_key) != RELOAD_POLICY):
        raise ValueError('explicit binary mesh/fresh-process reload contract required')


def verify_normal_certificate(record):
    if (record.get('schema') != 'ns-obstacle-2d-fluid-outward-normal-v1' or
            record.get('passed') is not True or
            record.get('helper_sha256') != sha_file(Path(__file__).with_name('oriented_normals.py')) or
            record.get('returned_CF_point_checks', {}).get('passed') is not True):
        raise ValueError('reviewed A8 actual returned-normal certificate required')
    return record['factors']


def read_shared_lift_admission(path, expected_sha256):
    path = Path(path)
    if sha_file(path) != expected_sha256: raise ValueError('common lift admission hash mismatch')
    admission = json.loads(path.read_text())
    qualification = require_qualification(admission)
    if admission.get('schema') != 'ns-obstacle-common-stokes-admission-v1':
        raise RuntimeError('genuine scoped common Stokes lift admission required')
    for key in ('reference_admission_sha256', 'review_sha256'):
        digest = admission.get(key, '')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('common lift review/reference digest missing')
    folder = (path.parent/admission['directory']).resolve()
    meta_file = folder/'common_stokes_lift.json'; array_file = folder/'common_stokes_lift.npz'
    if (max(meta_file.stat().st_size, array_file.stat().st_size) > CAPS['archive_bytes'] or
        sha_file(meta_file) != admission['manifest_sha256'] or
        sha_file(array_file) != admission['coefficient_sha256']):
        raise ValueError('common lift artifact size/hash mismatch')
    meta = json.loads(meta_file.read_text())
    if (meta.get('schema') != 'ns-obstacle-common-stokes-v1' or
        meta.get('coefficient_sha256') != admission['coefficient_sha256'] or
        meta.get('outlet') != 'gradient' or meta.get('pressure_gauge') != 'natural outlet, unshifted' or
        not math.isfinite(meta['base_nu']) or meta['base_nu'] <= 0 or
        not math.isfinite(meta['linear_control']['value']) or meta['linear_control']['value'] < 0 or
        not math.isfinite(meta['linear_tolerance']) or meta['linear_tolerance'] <= 0 or
        meta['linear_control']['value'] > meta['linear_tolerance']):
        raise ValueError('invalid common lift configuration/linear control')
    return {'metadata': meta, 'archive': str(array_file), 'admission_sha256': expected_sha256,
            'reference_qualification': qualification,
            'reference_admission_sha256': admission['reference_admission_sha256']}


def read_training_manifest(path):
    """Check admission before opening any mesh/state archive or importing NGSolve."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    qualification = require_qualification(manifest)
    if manifest.get('schema') != SCHEMA or manifest.get('split') != 'training':
        raise RuntimeError('reviewed scoped TRAINING manifest required; controls are not admission')
    if manifest.get('adapter_source_sha256') != adapter_digest():
        raise RuntimeError('review must bind the exact adapter source')
    for name in ('adapter_review_sha256', 'reference_admission_sha256', 'campaign_sha256'):
        value = manifest.get(name, '')
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'missing SHA256 evidence: {name}')
    verify_mesh_contract(manifest, 'mesh_reload')
    contract = json.loads(Path(__file__).with_name('MODEL-CONTRACT.json').read_text())
    if manifest['campaign_sha256'] != contract['central_campaign_sha256']:
        raise ValueError('manifest does not match the frozen central campaign contract')
    cases = manifest.get('cases', [])
    if [c.get('Re') for c in cases] != REYNOLDS or any(c.get('split') != 'training' for c in cases):
        raise ValueError('exact nine-state training population/order required')
    mesh_hashes = []
    for case in cases:
        folder = (path.parent / case['directory']).resolve()
        # Snapshot manifests are local file evidence; do not follow undeclared files.
        if set(case['files']) != set(FILES):
            raise ValueError('incomplete or extra declared FE case files')
        for name in FILES:
            file = folder / name
            if file.stat().st_size > CAPS['archive_bytes'] or sha_file(file) != case['files'][name]:
                raise ValueError(f'FE case artifact size/hash mismatch: {file}')
        mesh_hashes.append(case['files'][MESH_FILENAME])
        case['resolved_directory'] = str(folder)
    if len(set(mesh_hashes)) != 1:
        raise ValueError('training adapter requires one byte-identical admitted curved mesh')
    common = manifest.get('common_stokes_lift')
    if not common: raise RuntimeError('A6 requires one admitted shared Stokes lift identity')
    manifest['resolved_common_stokes_lift'] = read_shared_lift_admission(
        path.parent/common['admission_path'], common['admission_sha256'])
    shared = manifest['resolved_common_stokes_lift']
    if (shared['reference_qualification'] != qualification or
        shared['reference_admission_sha256'] != manifest['reference_admission_sha256']):
        raise ValueError('training and shared lift qualification differ')
    manifest['manifest_sha256'] = sha_file(path)
    return manifest


def load_npz_bounded(path):
    # Inspect ORIGINAL NPY headers before loading/converting any payload. This both
    # preserves f64 provenance and prevents f32->f64 conversion from hiding size growth.
    with zipfile.ZipFile(path) as z:
        entries = z.infolist()
        if sum(x.file_size for x in entries) > CAPS['archive_bytes']:
            raise MemoryError('uncompressed FE archive exceeds bound')
        if len({x.filename for x in entries}) != len(entries):
            raise ValueError('duplicate FE archive member')
        for entry in entries:
            if not entry.filename.endswith('.npy'):
                raise ValueError('FE archive contains a non-NPY member')
            with z.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError('unsupported NPY header version')
            if dtype.kind != 'f' or dtype.itemsize != 8:
                raise TypeError(f'FE array {entry.filename} must be stored float64, got {dtype}')
            check_allocation(shape)
    with np.load(path, allow_pickle=False) as z:
        result = {}
        for key in z.files:
            original = z[key]
            if original.dtype.kind != 'f' or original.dtype.itemsize != 8:
                raise TypeError(f'FE array {key} changed dtype after header verification')
            if not np.isfinite(original).all():
                raise ValueError(f'nonfinite FE array {key}')
            # Only byte order/layout may change here; original precision is already f64.
            result[key] = np.array(original, dtype=np.float64, copy=True)
    return result


def strict_scalar(value, label='FE scalar'):
    """Finite real scalar or singleton tuple/array; never select from a larger CF."""
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in 'fiu':
        raise ValueError(f'{label}: expected one real scalar, got shape {array.shape}, dtype {array.dtype}')
    scalar = float(array.reshape(-1)[0])
    if not math.isfinite(scalar):
        raise ValueError(f'{label}: nonfinite value')
    return scalar


def check_allocation(shape):
    count = math.prod(shape) * 8
    if count > CAPS['single_array_bytes']:
        raise MemoryError(f'proposed array exceeds adapter cap: {shape}')
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 + count > CAPS['process_rss_bytes']:
        raise MemoryError('adapter RSS plus proposed array exceeds cap')


def relative_gap(actual, expected, scale=1.):
    a = np.asarray(actual); b = np.asarray(expected)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('round-trip shape/nonfinite mismatch')
    return float(np.max(np.abs(a-b), initial=0) / max(float(np.max(np.abs(b), initial=0)), scale))


def layout_record(mesh, velocity_space, pressure_space, ng):
    """Canonical JSON digest includes ordered VOL elements and local FE DOF numbers."""
    record = {'schema': 'ns-obstacle-fe-layout-v1', 'dimension': int(mesh.dim),
              'velocity_ndof': int(velocity_space.ndof), 'pressure_ndof': int(pressure_space.ndof),
              'velocity_free': [int(bool(v)) for v in velocity_space.FreeDofs()],
              'pressure_free': [int(bool(v)) for v in pressure_space.FreeDofs()], 'elements': []}
    for el in mesh.Elements(ng.VOL):
        record['elements'].append({'number': int(el.nr), 'type': str(el.type),
            'velocity_dofs': [int(i) for i in velocity_space.GetDofNrs(el)],
            'pressure_dofs': [int(i) for i in pressure_space.GetDofNrs(el)]})
    digest = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return record, digest


def _sparse(form, size):
    from scipy.sparse import coo_matrix
    if form.mat.nze > CAPS['matrix_nnz']:
        raise MemoryError('sparse nonzero bound')
    rows, cols, vals = form.mat.COO()
    matrix = coo_matrix((np.asarray(vals), (np.asarray(rows), np.asarray(cols))),
                        shape=(size, size)).tocsc()
    matrix.sum_duplicates(); matrix.sort_indices()
    gap = matrix-matrix.T
    if not np.isfinite(matrix.data).all() or (gap.nnz and np.max(np.abs(gap.data)) > 1e-12*max(np.max(np.abs(matrix.data)), 1e-300)):
        raise ArithmeticError('assembled FE metric is not finite symmetric')
    return matrix


def sparse_digest(matrix):
    a = matrix.tocsc(copy=True); a.sum_duplicates(); a.sort_indices()
    return array_digest(np.asarray(a.shape, dtype=np.int64), a.indptr, a.indices, a.data)


def bounded_factor(matrix):
    from scipy.sparse.linalg import splu
    # Same conservative fill model as FOM; a prospective guard, not a memory theorem.
    estimate = matrix.nnz*16*100 + matrix.shape[0]*8*32
    if (matrix.nnz > CAPS['matrix_nnz'] or estimate > CAPS['factor_estimate_bytes'] or
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 + estimate > CAPS['process_rss_bytes']):
        raise MemoryError('FE adapter factorization estimate exceeds fixed cap')
    started = time.perf_counter()
    factor = splu(matrix, permc_spec='COLAMD')
    return factor, {'estimate_bytes': int(estimate), 'seconds': time.perf_counter()-started,
                    'matrix_sha256': sparse_digest(matrix), 'backend': 'scipy SuperLU COLAMD'}


def bounded_eigenmodes(h1, mass, solve_h1, count):
    """H1 = M+K; generalized eigenmodes also diagonalize scalar Laplace K.

    One look-ahead eigenpair protects the truncation from splitting a degenerate block.
    No retry, larger candidate bank, or dense fallback is selected from field errors.
    """
    from scipy.sparse.linalg import eigsh, LinearOperator
    n = h1.shape[0]
    if n <= count+2:
        raise ValueError('not enough free DOFs for frozen eigenbank plus look-ahead')
    v0 = np.random.default_rng(27102026).normal(size=n)
    inv = LinearOperator(h1.shape, matvec=solve_h1, dtype=np.float64)
    lam, vec = eigsh(h1, k=count+1, M=mass, sigma=0., which='LM', OPinv=inv,
                    v0=v0, tol=1e-12, maxiter=CAPS['eigensolver_iterations'],
                    ncv=min(n-1, 2*(count+1)+1))
    idx = np.argsort(lam, kind='stable'); lam = lam[idx]; vec = vec[:, idx]
    if abs(lam[count]-lam[count-1]) <= 1e-10*max(1., abs(lam[count])):
        raise ValueError('candidate cap splits eigenvalue cluster; stop for review')
    look_ahead = float(lam[count]); boundary_gap = float(lam[count]-lam[count-1])
    lam = lam[:count]; vec = vec[:, :count]
    av = h1 @ vec; mv = mass @ vec
    errors = np.linalg.norm(av-mv*lam, axis=0) / np.maximum(
        np.linalg.norm(av, axis=0)+np.abs(lam)*np.linalg.norm(mv, axis=0), 1e-300)
    if np.max(errors) > 1e-9:
        raise ArithmeticError('generalized eigenpair residual gate')
    if np.linalg.norm(vec.T@mv-np.eye(count), 2) > 1e-10:
        raise ArithmeticError('scalar eigenvectors fail mass Gram check')
    groups = []; start = 0
    while start < count:
        end = start+1
        while end < count and abs(lam[end]-lam[end-1]) <= 1e-10*max(1., abs(lam[end]), abs(lam[end-1])):
            end += 1
        groups.append((start, end)); start = end
    vec, orientation = orient_clusters(vec, groups)
    av = h1 @ vec; mv = mass @ vec
    oriented_errors = np.linalg.norm(av-mv*lam, axis=0) / np.maximum(
        np.linalg.norm(av, axis=0)+np.abs(lam)*np.linalg.norm(mv, axis=0), 1e-300)
    if np.max(oriented_errors) > 1e-9:
        raise ArithmeticError('oriented eigenpair residual gate')
    return vec, {'eigenvalues_H1': lam.tolist(), 'eigenvalues_Laplace': (lam-1).tolist(),
                 'residuals': errors.tolist(), 'oriented_residuals': oriented_errors.tolist(),
                 'orientation': orientation,
                 'h1_sha256': sparse_digest(h1), 'mass_sha256': sparse_digest(mass),
                 'eigensolver': 'scipy eigsh shift-invert sigma0 tol1e-12 seed27102026',
                 'gap_checked': True, 'scalar_spectral_count_verified': False, 'look_ahead_eigenvalue': look_ahead,
                 'boundary_eigenvalue_gap': boundary_gap}


class TrainingFEAdapter:
    @classmethod
    def open(cls, manifest_path, evidence_directory=None):
        admission = read_training_manifest(manifest_path)
        if evidence_directory is None:
            raise ValueError('fresh evidence directory required for native adapter controls')
        evidence_directory = Path(evidence_directory)
        evidence_directory.mkdir(parents=True, exist_ok=False)
        import ngsolve as ng  # Deliberately below all data admission checks.
        return cls(admission, ng, evidence_directory)

    def __init__(self, admission, ng, evidence_directory):
        self.ng = ng; self.admission = admission; self.started = time.perf_counter()
        self.evidence_directory = Path(evidence_directory)
        self.cases = []
        for case in admission['cases']:
            folder = Path(case['resolved_directory'])
            cfg = json.loads((folder/'config.json').read_text())
            result = json.loads((folder/'result.json').read_text())
            reload_checks = json.loads((folder/'reload_checks.json').read_text())
            if (cfg.get('kind') != 'dfg' or cfg.get('outlet') != 'gradient' or
                cfg.get('release') != EXPORT_RELEASE or cfg.get('scope_contract_sha256') != SCOPE_SHA256 or
                cfg.get('precision') != 'float64'):
                raise ValueError('only declared 2D homogeneous-outlet steady training exports supported')
            if cfg['campaign_sha256'] != admission['campaign_sha256']:
                raise ValueError('campaign mismatch')
            if (cfg['U'], cfg['D'], cfg['xmin'], cfg['xmax'], cfg['H'], cfg['radius'], cfg['center']) != (
                    .2, .1, 0., 2.2, .41, .05, [.2, .2]):
                raise ValueError('not the fixed 2D training geometry/scales')
            if abs(cfg['nu']-.02/case['Re']) > 1e-15:
                raise ValueError('Re/viscosity mismatch')
            if result['versions']['ngsolve'] != ng.__version__:
                raise ValueError('NGSolve version differs from exported coefficients')
            verify_mesh_contract(reload_checks)
            execution = reload_checks.get('execution', {})
            if (reload_checks.get('passed') is not True or execution.get('fresh_interpreter') is not True
                    or execution.get('pid') == execution.get('producer_pid')):
                raise RuntimeError('exporter reload admission missing')
            identity = json.loads((folder/'mesh_identity.json').read_text())
            exporter = json.loads((folder/'EXPORT-MANIFEST.json').read_text())
            verify_mesh_contract(identity)
            verify_mesh_contract(exporter, 'reload_policy')
            if (identity.get('reload_passed') is not True or identity.get('mesh_sha256') != case['files'][MESH_FILENAME]
                    or identity.get('reload_checks_sha256') != case['files']['reload_checks.json']):
                raise ValueError('binary mesh identity differs from admitted bytes/reload evidence')
            original = json.loads((folder/'normal_orientation.json').read_text())
            reloaded = json.loads((folder/'reload_normal_orientation.json').read_text())
            if verify_normal_certificate(original) != verify_normal_certificate(reloaded):
                raise ValueError('producer/reload normal orientation factors differ')
            self.cases.append({'case': case, 'folder': folder, 'cfg': cfg, 'result': result})
        self.cfg = self.cases[0]['cfg']; self.U = self.cfg['U']; self.D = self.cfg['D']
        shared = ('velocity_order', 'pressure_order', 'geometry_order', 'diagnostic_order', 'threads')
        if any(any(c['cfg'][k] != self.cfg[k] for k in shared) for c in self.cases):
            raise ValueError('training FE/order/thread layout differs')
        if self.cfg['pressure_order'] != self.cfg['velocity_order']-1 or self.cfg['velocity_order'] < 2:
            raise ValueError('Taylor-Hood spaces required')
        if not 1 <= self.cfg['threads'] <= 16:
            raise ValueError('CPU thread cap')
        ng.SetNumThreads(self.cfg['threads'])
        self.mesh = ng.Mesh(str(self.cases[0]['folder']/MESH_FILENAME))  # Never re-Curve.
        if self.mesh.dim != 2 or self.mesh.ne > CAPS['elements']:
            raise MemoryError('unsupported dimension or element cap')
        if int(self.mesh.GetCurveOrder()) != self.cfg['geometry_order']:
            raise ValueError('binary mesh curved order differs from declared geometry')
        self.normal, self.normal_certificate = fluid_outward_normal(self.mesh, self.cfg['center'],
            order=self.cfg['diagnostic_order'], evidence_path=self.evidence_directory/'normal_orientation.json')
        factors = verify_normal_certificate(self.normal_certificate)
        for c in self.cases:
            saved = json.loads((c['folder']/'normal_orientation.json').read_text())
            if factors != verify_normal_certificate(saved):
                raise ValueError('adapter and producer normal orientation factors differ')
        self.V = ng.VectorH1(self.mesh, order=self.cfg['velocity_order'], dirichlet=ESSENTIAL)
        self.Q = ng.H1(self.mesh, order=self.cfg['pressure_order'])
        self.S = ng.H1(self.mesh, order=self.cfg['velocity_order'], dirichlet=ESSENTIAL)
        self.X = self.V*self.Q
        if self.X.ndof > CAPS['mixed_dofs']:
            raise MemoryError('mixed DOF cap before any matrix assembly')
        self.layout, self.layout_sha256 = layout_record(self.mesh, self.V, self.Q, ng)
        for c in self.cases:
            saved = json.loads((c['folder']/'fe_layout.json').read_text())
            if saved.get('sha256') != self.layout_sha256 or saved.get('layout') != self.layout:
                raise ValueError('FE DOF-order round-trip failed')
        self.free = np.array(self.layout['velocity_free'], dtype=bool)
        self.volume = self.mapped_rule(boundary=False)
        self.boundary = self.mapped_rule(boundary=True)
        self.analytic_controls = self.analytic_interface_controls()
        self.component_maps, self.component_map_evidence = self.scalar_component_maps()
        self.weights = self.volume['weights']/self.D**2
        self.roundtrip = []
        self._load_coefficients()

    def mapped_rule(self, boundary=False):
        ng = self.ng; vb = ng.BND if boundary else ng.VOL
        cap = CAPS['boundary_points'] if boundary else CAPS['volume_points']
        points = []; weights = []; labels = []; normals = []; raw_normals = []; factors = []; jacobians = []
        raw_normal = ng.specialcf.normal(2)
        for el in self.mesh.Elements(vb):
            tr = self.mesh.GetTrafo(el)
            for ip in ng.IntegrationRule(el.type, self.cfg['diagnostic_order']):
                if len(points) >= cap:
                    raise MemoryError('mapped integration point cap')
                mip = tr(ip); w = strict_scalar(ip.weight, 'reference weight')*strict_scalar(mip.measure, 'mapped measure')
                if not np.isfinite(w) or w <= 0:
                    raise ValueError('nonpositive mapped integration weight')
                points.append(np.array(mip.point, copy=True)); weights.append(w)
                if boundary:
                    label = str(el.mat)
                    labels.append(label); normals.append(np.array(self.normal(mip), copy=True))
                    raw_normals.append(np.array(raw_normal(mip), copy=True))
                    factors.append(self.normal_certificate['factors'][label])
                else:
                    jac = np.array(mip.jacobi, copy=True)
                    if np.linalg.det(jac) <= 0:
                        raise ValueError('nonpositive mapped volume Jacobian')
                    jacobians.append(jac)
        result = {'points': np.asarray(points), 'weights': np.asarray(weights)}
        if boundary:
            result.update(labels=np.asarray(labels), normals_fluid=np.asarray(normals),
                          normals_raw=np.asarray(raw_normals), orientation_factors=np.asarray(factors))
        else:
            result['jacobians'] = np.asarray(jacobians)
        return result

    def evaluate_batched(self, coefficients, vector=True, boundary=False, points=None):
        """Accepted native block evaluation with the unchanged coefficient/units API."""
        if points is not None:
            return self.evaluate(coefficients, vector, boundary, points)
        from fe_blocks import evaluated_blocks_batched
        c=np.asarray(coefficients,dtype=np.float64)
        if c.ndim==1:c=c[:,None]
        space=self.V if vector else self.Q
        if c.ndim!=2 or c.shape[0]!=space.ndof or not np.isfinite(c).all():
            raise ValueError('invalid FE coefficient matrix')
        n=len((self.boundary if boundary else self.volume)['points'])
        check_allocation((n,6 if vector else 3,c.shape[1]))
        values=np.empty((n,2,c.shape[1]) if vector else (n,c.shape[1]))
        gradients=np.empty((n,2,2,c.shape[1]) if vector else (n,2,c.shape[1]))
        covered=0
        for start,end,value,gradient in evaluated_blocks_batched(self,c,vector,boundary):
            if start!=covered or end>n:raise ValueError('native batch point coverage differs')
            values[start:end]=value;gradients[start:end]=gradient;covered=end
        if covered!=n:raise ValueError('incomplete native batch point coverage')
        return values,gradients

    def evaluate(self, coefficients, vector=True, boundary=False, points=None):
        """FE coefficients -> dimensionless values and physical-coordinate derivatives.

        Requested raw points are never nudged to the FE mesh. Boundary derivatives
        use BoundaryFromVolumeCF only for the mapped BND rule, not for normal vectors.
        """
        ng = self.ng; c = np.asarray(coefficients, dtype=np.float64)
        space = self.V if vector else self.Q
        if c.ndim == 1: c = c[:, None]
        if c.shape[0] != space.ndof or not np.isfinite(c).all():
            raise ValueError('invalid FE coefficient matrix')
        if points is not None and boundary:
            raise ValueError('raw points use volume lookup; mapped boundary must retain BND transform')
        if points is None:
            n = len(self.boundary['points'] if boundary else self.volume['points'])
        else:
            points = np.asarray(points, dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
                raise ValueError('finite 2D probe coordinates required')
            if len(points) > CAPS['volume_points']: raise MemoryError('probe cap')
            n = len(points)
        shape = (n, 2, c.shape[1]) if vector else (n, c.shape[1])
        grad_shape = (n, 2, 2, c.shape[1]) if vector else (n, 2, c.shape[1])
        check_allocation(shape); check_allocation(grad_shape)
        values = np.empty(shape); gradients = np.empty(grad_shape)
        for j in range(c.shape[1]):
            g = ng.GridFunction(space); g.vec.FV().NumPy()[:] = c[:, j]
            derivative = ng.Grad(g)
            if boundary: derivative = ng.BoundaryFromVolumeCF(derivative)
            for i, mip in enumerate(self.iter_mapped(boundary, points)):
                values[..., j][i] = (np.asarray(g(mip)) if vector else strict_scalar(g(mip), 'FE scalar point value'))
                gradients[..., j][i] = np.asarray(derivative(mip)).reshape((2, 2) if vector else (2,))
        if not np.isfinite(values).all() or not np.isfinite(gradients).all():
            raise ValueError('nonfinite reconstructed FE fields')
        return values, gradients

    def iter_mapped(self, boundary=False, points=None):
        # Evaluate while each element transformation remains alive. Do not retain
        # mapped C++ points after their transformation leaves scope.
        if points is not None:
            if not hasattr(self, '_probe_map_cache'):
                self._probe_map_cache = {}; self.probe_lookup_evidence = []
            coordinate = self.ng.CF((self.ng.x, self.ng.y))
            for point in points:
                key = tuple(float(x) for x in point)
                if key in self._probe_map_cache:
                    yield self._probe_map_cache[key]; continue
                entry = {'point': list(key), 'status': 'checking', 'point_nudged': False,
                         'reference_membership_tolerance': 64*np.finfo(float).eps,
                         'coordinate_tolerance': 1e-10*self.D}
                self.probe_lookup_evidence.append(entry)
                def finite_list(value):
                    return [float(x) if np.isfinite(x) else {'nonfinite': str(x)} for x in np.asarray(value).reshape(-1)]
                try:
                    mip = self.mesh(*point, tol=1e-12); entry['element_number'] = int(mip.nr)
                    if mip.nr < 0: raise ValueError('probe outside saved FE geometry; no nudge')
                    ref = np.asarray(mip.pnt, dtype=np.float64); entry['reference_coordinates'] = finite_list(ref)
                    if ref.shape != (3,) or not np.isfinite(ref).all(): raise ValueError('invalid native probe reference coordinates')
                    bary = np.array([ref[0], ref[1], 1-ref[0]-ref[1]])
                    entry['barycentric_coordinates'] = finite_list(bary)
                    if np.min(bary) < -64*np.finfo(float).eps or abs(ref[2]) > 64*np.finfo(float).eps:
                        raise ValueError('probe not certified inside saved FE element; no nudge')
                    mapped = np.asarray(coordinate(mip), dtype=np.float64); entry['reconstructed_point'] = finite_list(mapped)
                    if mapped.shape != (2,) or not np.isfinite(mapped).all(): raise ValueError('invalid native probe reconstruction')
                    error = float(np.linalg.norm(mapped-point)); entry['reconstruction_error'] = error
                    if error > 1e-10*self.D: raise ValueError('probe physical reconstruction differs; no nudge')
                    entry['status'] = 'passed'; self._probe_map_cache[key] = mip
                except Exception as exc:
                    entry.update(status='failed', exception=repr(exc)); raise
                finally:
                    try:
                        (self.evidence_directory/'probe_lookup.json').write_text(json.dumps({
                            'schema': 'ns-obstacle-exact-probe-lookup-v1', 'records': self.probe_lookup_evidence,
                            'point_nudging_allowed': False}, indent=2, allow_nan=False)+'\n')
                    except Exception:
                        if entry['status'] != 'failed': raise
                yield mip
            return
        ng = self.ng
        for el in self.mesh.Elements(ng.BND if boundary else ng.VOL):
            tr = self.mesh.GetTrafo(el)
            for ip in ng.IntegrationRule(el.type, self.cfg['diagnostic_order']):
                yield tr(ip)

    def control_indices(self, boundary=False):
        if not boundary:
            return set(np.linspace(0, len(self.volume['points'])-1,
                                   min(16, len(self.volume['points'])), dtype=int).tolist())
        indices = set()
        for label in np.unique(self.boundary['labels']):
            group = np.flatnonzero(self.boundary['labels'] == label)
            indices.update(int(group[j]) for j in (0, len(group)//2, len(group)-1))
        return indices

    def analytic_interface_controls(self):
        ng = self.ng; b = self.boundary
        report = {'normals': boundary_control(b['points'], b['weights'], b['normals_fluid'],
                                              b['labels'], self.cfg['center']), 'gradient': []}
        # Coordinate-map degree can exceed production velocity degree on a geometry rung.
        # Only this analytic fixture uses the sufficient degree; production spaces stay fixed.
        order = max(self.cfg['velocity_order'], self.cfg['geometry_order'])
        fixture_space = ng.VectorH1(self.mesh, order=order)
        if fixture_space.ndof > CAPS['mixed_dofs']:
            raise MemoryError('auxiliary analytic-gradient fixture DOF cap')
        for component, field in [(0, ng.CF((ng.y, 0))), (1, ng.CF((0, ng.x)))]:
            fixture = ng.GridFunction(fixture_space); fixture.Set(field)
            for boundary in (False, True):
                selected = self.control_indices(boundary); points = []; values = []; gradients = []
                derivative = ng.Grad(fixture)
                if boundary: derivative = ng.BoundaryFromVolumeCF(derivative)
                for i, mip in enumerate(self.iter_mapped(boundary)):
                    if i in selected:
                        points.append(np.array(mip.point, copy=True))
                        values.append(np.array(fixture(mip), copy=True).reshape(2))
                        gradients.append(np.array(derivative(mip), copy=True).reshape(2, 2))
                    if i >= max(selected): break
                check = gradient_control(points, values, gradients, component)
                check.update(location='boundary' if boundary else 'volume', fixture_order=order,
                             production_velocity_order=self.cfg['velocity_order'], sampled_indices=sorted(selected))
                report['gradient'].append(check)
        return report

    def scalar_component_maps(self):
        ng = self.ng
        tagged = ng.GridFunction(self.V)
        tagged.vec.FV().NumPy()[:] = np.arange(self.V.ndof, dtype=np.float64)+1
        components = tagged.components
        if len(components) != 2: raise ValueError('VectorH1 must expose two actual component views')
        maps = []; scalar_free = np.array([bool(v) for v in self.S.FreeDofs()])
        for component in components:
            tags = np.array(component.vec.FV().NumPy(), copy=True)
            ids = np.rint(tags).astype(np.int64)-1
            if tags.shape != (self.S.ndof,) or not np.array_equal(tags, ids+1):
                raise ValueError('component coefficient view does not identify scalar DOFs')
            if np.any(ids < 0) or np.any(ids >= self.V.ndof) or not np.array_equal(self.free[ids], scalar_free):
                raise ValueError('scalar/vector essential maps differ')
            for el in self.mesh.Elements(ng.VOL):
                if tuple(component.space.GetDofNrs(el)) != tuple(self.S.GetDofNrs(el)):
                    raise ValueError('scalar/component local FE DOF enumeration differs')
            maps.append(ids)
        maps = np.asarray(maps)
        if sorted(maps.ravel().tolist()) != list(range(self.V.ndof)):
            raise ValueError('component views are not a disjoint complete vector-DOF map')
        scalar = ng.GridFunction(self.S); scalar.Set(ng.x+2*ng.y)
        scalar_coeff = scalar.vec.FV().NumPy().copy(); checks = []
        for component in range(2):
            vector = ng.GridFunction(self.V); vector.vec.FV().NumPy()[:] = 0
            vector.vec.FV().NumPy()[maps[component]] = scalar_coeff
            error = 0.; selected = self.control_indices(False)
            for i, mip in enumerate(self.iter_mapped(False)):
                if i in selected:
                    expected = np.zeros(2); expected[component] = strict_scalar(scalar(mip))
                    expected_gradient = np.zeros((2, 2))
                    expected_gradient[component] = np.asarray(ng.Grad(scalar)(mip)).reshape(2)
                    error = max(error, relative_gap(vector(mip), expected),
                                relative_gap(np.asarray(ng.Grad(vector)(mip)).reshape(2, 2), expected_gradient))
                if i >= max(selected): break
            if error > 1e-10: raise ArithmeticError('component map field/gradient embedding failed')
            checks.append(error)
        return maps, {'component_maps': maps.tolist(), 'sha256': array_digest(maps),
                      'local_DOF_enumeration_checked': True, 'essential_masks_checked': True,
                      'field_gradient_embedding_max_gaps': checks, 'tolerance': 1e-10}

    def _load_coefficients(self):
        ng = self.ng; solutions = []
        for c in self.cases:
            data = load_npz_bounded(c['folder']/'full_rule.npz')
            checks = {}
            for key in ('points', 'weights', 'jacobians'):
                if key not in data:
                    raise ValueError(f'future G2 export must add {key}; G1 archive is insufficient')
                checks[key] = relative_gap(self.volume[key], data[key], self.D if key == 'points' else 1.)
            saved = ng.GridFunction(self.X); saved.Load(str(c['folder']/'solution.sol'))
            solution = data['solution_coefficients']; lift = data['stokes_coefficients']
            if solution.shape != (self.X.ndof,) or lift.shape != solution.shape:
                raise ValueError('coefficient block layout mismatch')
            checks['sol_vs_coefficients'] = relative_gap(saved.vec.FV().NumPy(), solution)
            uv, gu = self.evaluate_batched(solution[:self.V.ndof]); pp, _ = self.evaluate_batched(solution[self.V.ndof:], False)
            for key, actual, scale in [('velocity', uv[..., 0], self.U),
                ('velocity_gradient', gu[..., 0], self.U/self.D), ('pressure', pp[:, 0], self.U**2)]:
                checks[key] = relative_gap(actual, data[key], scale)
            b = json.loads((c['folder']/'boundary_rule.json').read_text())
            for key, actual in [('point', self.boundary['points']), ('weight', self.boundary['weights']),
                                ('normal', self.boundary['normals_fluid']),
                                ('raw_normal', self.boundary['normals_raw']),
                                ('normal_orientation_factor', self.boundary['orientation_factors'])]:
                checks['boundary_'+key] = relative_gap(actual, np.asarray([r[key] for r in b]))
            if [r['label'] for r in b] != self.boundary['labels'].tolist():
                raise ValueError('boundary enumeration differs on reload')
            bv, bg = self.evaluate_batched(solution[:self.V.ndof], boundary=True)
            bp, _ = self.evaluate_batched(solution[self.V.ndof:], False, boundary=True)
            for key, actual, scale in [('velocity', bv[..., 0], self.U),
                ('velocity_gradient', bg[..., 0], self.U/self.D), ('pressure', bp[:, 0], self.U**2)]:
                checks['boundary_'+key] = relative_gap(actual, [r[key] for r in b], scale)
            if max(checks.values()) > 1e-10:
                raise ArithmeticError(f'FE serialization round-trip mismatch: {checks}')
            self.roundtrip.append({'Re': c['case']['Re'], 'relative_max_gaps': checks})
            solutions.append(solution)
        sol = np.asarray(solutions)
        nv = self.V.ndof
        shared = self.admission['resolved_common_stokes_lift']; meta = shared['metadata']
        if (meta['mesh_sha256'] != self.cases[0]['case']['files'][MESH_FILENAME] or
            meta['layout_sha256'] != self.layout_sha256 or meta['U'] != self.U or meta['D'] != self.D or
            meta['velocity_ndof'] != nv or meta['pressure_ndof'] != self.Q.ndof):
            raise ValueError('common Stokes lift geometry/space/scales mismatch')
        lift_data = load_npz_bounded(shared['archive'])
        if set(lift_data) != {'velocity_coefficients', 'pressure_coefficients_per_nu'}:
            raise ValueError('unexpected common lift arrays')
        if (lift_data['velocity_coefficients'].shape != (nv,) or
            lift_data['pressure_coefficients_per_nu'].shape != (self.Q.ndof,)):
            raise ValueError('common lift coefficient shape mismatch')
        self.lift_u = lift_data['velocity_coefficients']/self.U
        self.lift_p_per_nu = lift_data['pressure_coefficients_per_nu']*self.D/self.U
        self.shared_lift_evidence = shared
        self.velocity_corrections = sol[:, :nv].T/self.U-self.lift_u[:, None]
        trace = np.max(np.abs(self.velocity_corrections[~self.free]), initial=0.)
        if trace > 1e-10: raise ValueError('velocity correction has nonzero essential FE trace coefficients')
        self.velocity_corrections[~self.free] = 0.  # Declared rounding-only restoration.
        self.trace_coefficient_roundoff_removed = float(trace)
        nu_hat = np.array([c['cfg']['nu']/(self.U*self.D) for c in self.cases])
        self.pressure_corrections = sol[:, nv:].T/self.U**2-self.lift_p_per_nu[:, None]*nu_hat

    def metric_matrices(self, space, vector):
        ng = self.ng; u, v = space.TnT()
        rules = {kind: ng.IntegrationRule(kind, self.cfg['diagnostic_order'])
                 for kind in {el.type for el in self.mesh.Elements(ng.VOL)}}
        dx = ng.dx(intrules=rules)
        mass = ng.BilinearForm(space); h1 = ng.BilinearForm(space)
        product = ng.InnerProduct(u, v)
        derivative = ng.InnerProduct(ng.Grad(u), ng.Grad(v))
        mass += product/self.D**2*dx
        h1 += (product/self.D**2+derivative)*dx
        with ng.TaskManager(): mass.Assemble(); h1.Assemble()
        return _sparse(h1, space.ndof), _sparse(mass, space.ndof), dx

    def build(self):
        """Offline training-only spaces. The returned object is NOT physical admission."""
        ng = self.ng
        import scipy
        uv, _ = self.evaluate_batched(self.velocity_corrections)
        pv, _ = self.evaluate_batched(self.pressure_corrections, False)
        constant = ng.GridFunction(self.Q); constant.Set(1.)
        one = constant.vec.FV().NumPy().copy()
        one_values, _ = self.evaluate_batched(one, False)
        if relative_gap(one_values[:, 0], np.ones(len(self.weights))) > 1e-12:
            raise ArithmeticError('constant pressure coefficient reconstruction failed')
        h1v, mv, dx = self.metric_matrices(self.V, True)
        h1q, mq, _ = self.metric_matrices(self.Q, False)
        h1s, ms, _ = self.metric_matrices(self.S, False)
        component_blocks = check_component_blocks(h1v, mv, h1s, ms, self.component_maps)
        pod = coefficient_pod_a6(self.velocity_corrections, uv.transpose(2, 0, 1),
            self.pressure_corrections, pv.T, one, self.weights, mv, mq)
        actual_u, _ = self.evaluate_batched(pod['velocity_coefficients'])
        actual_p, _ = self.evaluate_batched(pod['pressure_coefficients'], False)
        pod_velocity_gram, pod_u_check = validate_actual_pod(
            actual_u, uv.transpose(2, 0, 1), self.weights, pod['metadata']['velocity'])
        pressure_gram, pod_p_check = validate_actual_pod(
            actual_p, pv.T, self.weights, pod['metadata']['pressure'], pressure=True)
        pressure = pod['pressure_coefficients']; count = pressure.shape[1]
        ids = np.flatnonzero(self.free)
        hv = h1v[ids][:, ids]
        scalar_free = np.array([bool(v) for v in self.S.FreeDofs()])
        scalar_ids = np.flatnonzero(scalar_free)
        hs = h1s[scalar_ids][:, scalar_ids]; mss = ms[scalar_ids][:, scalar_ids]
        # Actual component block identity was certified above on these same matrices.
        # Share one bounded scalar factor with the existing scalar eigenproblem.
        factor_s, fm_s = bounded_factor(hs)
        from component_factor import ComponentFactorSolve
        component_solver = ComponentFactorSolve(factor_s.solve, self.component_maps, self.free, scalar_free)
        fm_v = dict(component_solver.evidence, backend='verified scalar H1 component factor reuse',
            additional_factorizations=0, additional_factor_seconds=0.,
            shared_factor_record='factor_velocity_scalar', matrix_sha256=sparse_digest(hv))
        rhs = np.empty((len(ids), count))
        test = self.V.TestFunction()
        for j in range(count):
            psi = ng.GridFunction(self.Q); psi.vec.FV().NumPy()[:] = pressure[:, j]
            form = ng.LinearForm(self.V); form += psi*ng.div(test)/self.D*dx
            with ng.TaskManager(): form.Assemble()
            rhs[:, j] = form.vec.FV().NumPy()[ids]
        solve_started = time.perf_counter()
        sf = component_solver.solve(rhs)
        fm_v['component_solve_seconds'] = time.perf_counter()-solve_started
        del component_solver  # Do not retain its bound SuperLU method after factor release.
        residual = np.linalg.norm(hv@sf-rhs)/max(np.linalg.norm(rhs), 1e-300)
        if residual > 1e-10: raise ArithmeticError('H1 supremizer linear residual failed')
        supremizers = np.zeros((self.V.ndof, count)); supremizers[ids] = sf
        del hv
        eigen_s, em_v = bounded_eigenmodes(hs, mss, factor_s.solve, CAPS['velocity_scalar_eigen_candidates'])
        scalar_modes = np.zeros((self.S.ndof, eigen_s.shape[1])); scalar_modes[scalar_ids] = eigen_s
        ev = expand_components(scalar_modes, self.component_maps, self.V.ndof)
        em_v.update(component_order=['(phi,0)', '(0,phi)'], expanded_candidate_count=ev.shape[1],
                    scalar_polynomial_order=self.cfg['velocity_order'], component_multiplicity_explicit=True)
        del factor_s
        factor_q, fm_q = bounded_factor(h1q)
        eq, em_q = bounded_eigenmodes(h1q, mq, factor_q.solve, CAPS['scalar_eigen_candidates'])
        del factor_q
        candidates_v = np.c_[supremizers, ev]
        candidates_q = np.c_[one, pressure, eq]
        from fe_blocks import evaluated_blocks_batched as evaluated_blocks, select_test_columns_blocks
        def candidates_in_blocks(coefficients, vector):
            for start, end, values, gradients in evaluated_blocks(self, coefficients, vector):
                yield start, end, values, self.weights[start:end], gradients*self.D if vector else None
        vm = select_test_columns_blocks(candidates_in_blocks(candidates_v, True), 64, len(self.weights), True)
        qm = select_test_columns_blocks(candidates_in_blocks(candidates_q, False), 16, len(self.weights), False)
        tests_v = candidates_v[:, vm['candidate_indices']] @ np.asarray(vm['transform'])
        tests_q = candidates_q[:, qm['candidate_indices']] @ np.asarray(qm['transform'])
        # Use the verified ACTUAL FE pressure Gram, matching psi in B and the supremizers.
        ref_beta = {str(p): reference_inf_sup(rhs[:, :p], sf[:, :p], pressure_gram[:p, :p])
                    for p in sorted(set(min(r, count) for r in (2, 4, 8)))}
        # Same dimensionless full-rule metric must reproduce the assembled metric.
        cv = tests_v.T @ (h1v @ tests_v); cq = tests_q.T @ (mq @ tests_q)
        if max(relative_gap(cv, np.eye(64)), relative_gap(cq, np.eye(16))) > 1e-9:
            raise ArithmeticError('assembled FE metric and mapped-rule test normalization disagree')
        self.coefficients = {'lift_u': self.lift_u, 'lift_p_per_nu': self.lift_p_per_nu,
            'pod_velocity': pod['velocity_coefficients'], 'pressure': pressure,
            'supremizers': supremizers, 'momentum_tests': tests_v, 'continuity_tests': tests_q}
        self.metadata = {'schema': 'ns-obstacle-fe-adapter-A6', 'physical_admission': False,
            'reference_qualification': require_qualification(self.admission),
            'split': 'training', 'Re': REYNOLDS, 'manifest_sha256': self.admission['manifest_sha256'],
            'adapter_source_sha256': adapter_digest(), 'caps': dict(CAPS), 'roundtrip': self.roundtrip,
            'admission_evidence': {k: self.admission[k] for k in ('adapter_review_sha256',
                'reference_admission_sha256', 'campaign_sha256', 'mesh_reload')},
            'source_case_files': [c['case'] for c in self.cases],
            'layout_sha256': self.layout_sha256, 'pod': pod['metadata'],
            'pod_reconstruction': {'velocity': pod_u_check, 'pressure': pod_p_check},
            'pod_velocity_L2_gram_reconstructed': pod_velocity_gram.tolist(),
            'pressure_gram_source': 'actual reconstructed FE pressure evaluations on full rule',
            'shared_stokes_lift': self.shared_lift_evidence,
            'supremizer_residual': float(residual), 'reference_inf_sup': ref_beta,
            'momentum_candidates': ['all maximum-prefix supremizers', 'scalar velocity-order H1 modes with explicit component copies'],
            'continuity_candidates': ['constant', 'all pressure modes', 'scalar Neumann Laplace eigenmodes'],
            'momentum_selection': vm, 'continuity_selection': qm,
            'eigen_velocity': em_v, 'eigen_scalar': em_q, 'factor_velocity': fm_v, 'factor_scalar': fm_q,
            'factor_velocity_scalar': fm_s, 'component_maps': self.component_map_evidence,
            'component_blocks': component_blocks, 'analytic_controls': self.analytic_controls,
            'mesh_filename': MESH_FILENAME, 'mesh_format': MESH_FORMAT,
            'normal_orientation': self.normal_certificate,
            'momentum_H1_gram': cv.tolist(), 'continuity_L2_gram': cq.tolist(),
            'pressure_L2_gram': pressure_gram.tolist(),
            'trace_coefficient_roundoff_removed': self.trace_coefficient_roundoff_removed,
            'seconds_including_load': time.perf_counter()-self.started,
            'versions': {'ngsolve': ng.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__}}
        # Retain actual operators/candidates for independent eigen, component-block,
        # supremizer and test-selection replay, without another native assembly.
        self.replay_matrices = dict(velocity_h1=h1v, velocity_mass=mv,
            pressure_h1=h1q, pressure_mass=mq, velocity_scalar_h1=h1s, velocity_scalar_mass=ms)
        self.replay_arrays = dict(velocity_scalar_modes=scalar_modes, pressure_eigenmodes=eq,
            supremizer_rhs=rhs, scalar_free_ids=scalar_ids, velocity_free_ids=ids,
            component_maps=self.component_maps)
        return self

    def export_replay(self, directory):
        from scipy.sparse import save_npz
        out = Path(directory)
        matrix_bytes = sum(a.data.nbytes+a.indices.nbytes+a.indptr.nbytes for a in self.replay_matrices.values())
        array_bytes = sum(a.nbytes for a in self.replay_arrays.values())
        projected = (matrix_bytes+array_bytes)*101//100+16*2**20
        if shutil.disk_usage(out).free < 10*2**30+projected:
            raise OSError('spectral replay would violate ten-GiB retained disk reserve')
        files = []
        for name, matrix in self.replay_matrices.items():
            filename = 'replay_metric_'+name+'.npz'
            save_npz(out/filename, matrix); files.append(filename)
        # Separate archives keep each measured P4 eigenvector/RHS payload below
        # the individual archive bound and avoid one redundant combined copy.
        for name, array in self.replay_arrays.items():
            if array.nbytes > CAPS['single_array_bytes']:
                raise MemoryError('spectral replay single-array cap')
            filename = 'replay_'+name+'.npz'
            np.savez_compressed(out/filename, values=array); files.append(filename)
        (out/'replay_build.json').write_text(json.dumps(self.metadata, indent=2, allow_nan=False)+'\n')
        files.append('replay_build.json')
        return files, dict(matrix_expanded_bytes=matrix_bytes, array_expanded_bytes=array_bytes,
            projected_output_bytes=projected, disk_reserve_bytes=10*2**30,
            scope='actual assembled operators and computed candidate coefficients; no extra solve')

    def export(self, directory, point_block_volume=False):
        """Persist arrays, FE coefficient bases and their provenance; no integration pass flag."""
        if not hasattr(self, 'coefficients'): raise RuntimeError('build required')
        export_started = time.perf_counter()
        out = Path(directory); out.mkdir(parents=True, exist_ok=False)
        for boundary, name in ((False, 'volume'), (True, 'boundary')):
            if not boundary and point_block_volume:
                from point_store import write_volume
                write_volume(self, out/'volume')
                continue
            mapped = self.boundary if boundary else self.volume
            arrays = {'points_dimensionless': mapped['points']/self.D, 'points_physical': mapped['points'],
                      'weights_dimensionless': mapped['weights']/self.D**(1 if boundary else 2)}
            if boundary:
                arrays.update(labels=mapped['labels'], normals_fluid=mapped['normals_fluid'],
                              normals_raw=mapped['normals_raw'], orientation_factors=mapped['orientation_factors'])
            else: arrays['jacobians_physical'] = mapped['jacobians']
            for key, c in self.coefficients.items():
                vector = key in ('lift_u', 'pod_velocity', 'supremizers', 'momentum_tests')
                values, gradients = self.evaluate_batched(c, vector, boundary)
                arrays[key] = values
                arrays[key+'_gradient_dimensionless'] = self.D*gradients
            np.savez_compressed(out/(name+'.npz'), **arrays)
        np.savez_compressed(out/'coefficients.npz', **self.coefficients)
        replay_files, replay_retention = self.export_replay(out)
        # One explicitly training-only archive; exact same population as all FE spaces.
        u, du = self.evaluate_batched(self.velocity_corrections)
        p, _ = self.evaluate_batched(self.pressure_corrections, False)
        from train_bank import array_hash  # Does not import JAX or run training.
        training = {'points_dimensionless': self.volume['points']/self.D, 'points_physical': self.volume['points'], 'weights_dimensionless': self.weights,
                    'velocity_corrections': u.transpose(2, 0, 1),
                    'velocity_correction_gradients_dimensionless': (du*self.D).transpose(3, 0, 1, 2),
                    'pressure_corrections': p.T, 'Re': np.asarray(REYNOLDS)}
        np.savez_compressed(out/'training.npz', **training)
        meta = dict(self.metadata)
        meta['spectral_replay_retention'] = replay_retention
        meta['training_array_sha256'] = array_hash(training['points_physical'],
            training['weights_dimensionless'], training['velocity_corrections'])
        meta['export_seconds'] = time.perf_counter()-export_started
        meta['total_preparation_seconds'] = time.perf_counter()-self.started
        meta['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        meta['scales'] = {'D': self.D, 'U': self.U, 'nu_hat': 'nu/(U*D)'}
        meta['units'] = {'primary_coordinates': 'x/D', 'volume_weights': 'dx/D**2',
            'boundary_weights': 'ds/D', 'velocity_lift': 'uL/U',
            'pressure_lift_per_nu': 'pL*D/(U*nu)', 'velocity_pressure_modes_tests': 'dimensionless',
            'all_exported_field_gradients': 'derivative with respect to x/D',
            'already_dimensionless': True, 'physical_to_dimensionless_must_not_be_applied': True,
            'physical_geometry_sidecars': ['points_physical', 'jacobians_physical'],
            'physical_points_purpose': 'raw FE lookup and frozen coordinate-network geometry API only'}
        meta['gradient_axes'] = 'point, component, derivative_coordinate, column'
        meta['boundary_normals'] = 'outward fluid; negate only obstacle normals for force_moments'
        files = ['boundary.npz', 'coefficients.npz', 'training.npz']+replay_files
        if point_block_volume:
            files.extend(str(path.relative_to(out)) for path in sorted((out/'volume').iterdir()) if path.is_file())
        else:
            files.append('volume.npz')
        meta['volume_storage'] = 'point-block-v1' if point_block_volume else 'single-archive-v1'
        meta['coefficient_files'] = {name: sha_file(out/name) for name in files}
        (out/'manifest.json').write_text(json.dumps(meta, indent=2, allow_nan=False)+'\n')
        return meta
