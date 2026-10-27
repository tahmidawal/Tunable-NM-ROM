"""Gated, bounded 2D NGSolve-to-ROM bridge. No FE operation runs on import.

This source has NOT passed a pinned-library FE integration test. A future reviewed job
must supply a genuine G2 training manifest; current G1 control files are rejected.
Only this module owns the adapter; it never imports or mutates the FOM solver module.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import resource
import time
import zipfile
import numpy as np
from fe_algebra import array_digest, coefficient_pod, reference_inf_sup
from spaces import canonicalize_eigenmodes, select_test_columns, weighted_gram

SCHEMA = 'ns-obstacle-fe-training-v1'
REYNOLDS = [15 + 1.25*j for j in range(9)]
ESSENTIAL = 'inlet|wall.*|obstacle'
FILES = ('config.json', 'mesh.vol', 'full_rule.npz', 'boundary_rule.json',
         'solution.sol', 'result.json', 'fe_layout.json', 'reload_checks.json')
CAPS = {'mixed_dofs': 100000, 'elements': 20000, 'volume_points': 100000,
        'boundary_points': 30000, 'matrix_nnz': 12000000,
        'single_array_bytes': 2**30, 'archive_bytes': 2**30,
        'factor_estimate_bytes': 8*2**30, 'process_rss_bytes': 16*2**30,
        'vector_eigen_candidates': 96, 'scalar_eigen_candidates': 32,
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
        ('fe_adapter.py', 'fe_algebra.py', 'spaces.py', 'weak.py', 'train_bank.py', 'MODEL-CONTRACT.json')}, sort_keys=True).encode()).hexdigest()


def read_training_manifest(path):
    """Check admission before opening any mesh/state archive or importing NGSolve."""
    path = Path(path)
    manifest = json.loads(path.read_text())
    if (manifest.get('schema') != SCHEMA or manifest.get('gate') != 'G2-steady' or
        manifest.get('passed') is not True or manifest.get('split') != 'training'):
        raise RuntimeError('reviewed G2-steady TRAINING manifest required; G1 is not admission')
    if manifest.get('adapter_source_sha256') != adapter_digest():
        raise RuntimeError('review must bind the exact adapter source')
    for name in ('adapter_review_sha256', 'reference_admission_sha256', 'campaign_sha256'):
        value = manifest.get(name, '')
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'missing SHA256 evidence: {name}')
    if manifest.get('mesh_reload') != 'saved_curved_verified_no_recurve':
        raise RuntimeError('curved .vol reload must be verified; automatic re-Curve is forbidden')
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
        mesh_hashes.append(case['files']['mesh.vol'])
        case['resolved_directory'] = str(folder)
    if len(set(mesh_hashes)) != 1:
        raise ValueError('training adapter requires one byte-identical admitted curved mesh')
    manifest['manifest_sha256'] = sha_file(path)
    return manifest


def load_npz_bounded(path):
    with zipfile.ZipFile(path) as z:
        if sum(x.file_size for x in z.infolist()) > CAPS['archive_bytes']:
            raise MemoryError('uncompressed FE archive exceeds bound')
    with np.load(path, allow_pickle=False) as z:
        result = {k: np.array(z[k], dtype=np.float64, copy=True) for k in z.files}
    if any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError('nonfinite FE archive')
    return result


def check_allocation(shape):
    count = int(np.prod(shape)) * 8
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
    vec, orientation = canonicalize_eigenmodes(lam, vec, vec.T @ mv)
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
                 'look_ahead_cluster_complete': True, 'look_ahead_eigenvalue': look_ahead,
                 'boundary_eigenvalue_gap': boundary_gap}


class TrainingFEAdapter:
    @classmethod
    def open(cls, manifest_path):
        admission = read_training_manifest(manifest_path)
        import ngsolve as ng  # Deliberately below all data admission checks.
        return cls(admission, ng)

    def __init__(self, admission, ng):
        self.ng = ng; self.admission = admission; self.started = time.perf_counter()
        self.cases = []
        for case in admission['cases']:
            folder = Path(case['resolved_directory'])
            cfg = json.loads((folder/'config.json').read_text())
            result = json.loads((folder/'result.json').read_text())
            reload_checks = json.loads((folder/'reload_checks.json').read_text())
            if (cfg.get('kind') != 'dfg' or cfg.get('outlet') != 'gradient' or
                cfg.get('release') != 'G2_steady_training_export' or cfg.get('precision') != 'float64'):
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
            if reload_checks.get('passed') is not True or reload_checks.get('policy') != 'saved_curved_verified_no_recurve':
                raise RuntimeError('exporter reload admission missing')
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
        self.mesh = ng.Mesh(str(self.cases[0]['folder']/'mesh.vol'))  # Never re-Curve.
        if self.mesh.dim != 2 or self.mesh.ne > CAPS['elements']:
            raise MemoryError('unsupported dimension or element cap')
        self.V = ng.VectorH1(self.mesh, order=self.cfg['velocity_order'], dirichlet=ESSENTIAL)
        self.Q = ng.H1(self.mesh, order=self.cfg['pressure_order'])
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
        self.weights = self.volume['weights']/self.D**2
        self.roundtrip = []
        self._load_coefficients()

    def mapped_rule(self, boundary=False):
        ng = self.ng; vb = ng.BND if boundary else ng.VOL
        cap = CAPS['boundary_points'] if boundary else CAPS['volume_points']
        points = []; weights = []; labels = []; normals = []; jacobians = []
        normal = ng.specialcf.normal(2)
        for el in self.mesh.Elements(vb):
            tr = self.mesh.GetTrafo(el)
            for ip in ng.IntegrationRule(el.type, self.cfg['diagnostic_order']):
                if len(points) >= cap:
                    raise MemoryError('mapped integration point cap')
                mip = tr(ip); w = float(ip.weight*mip.measure)
                if not np.isfinite(w) or w <= 0:
                    raise ValueError('nonpositive mapped integration weight')
                points.append(np.array(mip.point, copy=True)); weights.append(w)
                if boundary:
                    labels.append(str(el.mat)); normals.append(np.array(normal(mip), copy=True))
                else:
                    jac = np.array(mip.jacobi, copy=True)
                    if np.linalg.det(jac) <= 0:
                        raise ValueError('nonpositive mapped volume Jacobian')
                    jacobians.append(jac)
        result = {'points': np.asarray(points), 'weights': np.asarray(weights)}
        if boundary:
            result.update(labels=np.asarray(labels), normals_fluid=np.asarray(normals))
        else:
            result['jacobians'] = np.asarray(jacobians)
        return result

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
                values[..., j][i] = np.asarray(g(mip))
                gradients[..., j][i] = np.asarray(derivative(mip)).reshape((2, 2) if vector else (2,))
        if not np.isfinite(values).all() or not np.isfinite(gradients).all():
            raise ValueError('nonfinite reconstructed FE fields')
        return values, gradients

    def iter_mapped(self, boundary=False, points=None):
        # Evaluate while each element transformation remains alive. Do not retain
        # mapped C++ points after their transformation leaves scope.
        if points is not None:
            for point in points:
                mip = self.mesh(*point)
                if mip.nr < 0: raise ValueError('probe outside saved FE geometry; no nudge')
                yield mip
            return
        ng = self.ng
        for el in self.mesh.Elements(ng.BND if boundary else ng.VOL):
            tr = self.mesh.GetTrafo(el)
            for ip in ng.IntegrationRule(el.type, self.cfg['diagnostic_order']):
                yield tr(ip)

    def _load_coefficients(self):
        ng = self.ng; solutions = []; stokes = []
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
            uv, gu = self.evaluate(solution[:self.V.ndof]); pp, _ = self.evaluate(solution[self.V.ndof:], False)
            for key, actual, scale in [('velocity', uv[..., 0], self.U),
                ('velocity_gradient', gu[..., 0], self.U/self.D), ('pressure', pp[:, 0], self.U**2)]:
                checks[key] = relative_gap(actual, data[key], scale)
            b = json.loads((c['folder']/'boundary_rule.json').read_text())
            for key, actual in [('point', self.boundary['points']), ('weight', self.boundary['weights']),
                                ('normal', self.boundary['normals_fluid'])]:
                checks['boundary_'+key] = relative_gap(actual, np.asarray([r[key] for r in b]))
            if [r['label'] for r in b] != self.boundary['labels'].tolist():
                raise ValueError('boundary enumeration differs on reload')
            bv, bg = self.evaluate(solution[:self.V.ndof], boundary=True)
            bp, _ = self.evaluate(solution[self.V.ndof:], False, boundary=True)
            for key, actual, scale in [('velocity', bv[..., 0], self.U),
                ('velocity_gradient', bg[..., 0], self.U/self.D), ('pressure', bp[:, 0], self.U**2)]:
                checks['boundary_'+key] = relative_gap(actual, [r[key] for r in b], scale)
            if max(checks.values()) > 1e-10:
                raise ArithmeticError(f'FE serialization round-trip mismatch: {checks}')
            self.roundtrip.append({'Re': c['case']['Re'], 'relative_max_gaps': checks})
            solutions.append(solution); stokes.append(lift)
        sol = np.asarray(solutions); lifts = np.asarray(stokes)
        nv = self.V.ndof
        self.lift_u = lifts[0, :nv]/self.U
        self.lift_p_per_nu = lifts[0, nv:]*self.D/(self.U*self.cases[0]['cfg']['nu'])
        for i, c in enumerate(self.cases):
            if (relative_gap(lifts[i, :nv]/self.U, self.lift_u) > 1e-10 or
                relative_gap(lifts[i, nv:]*self.D/(self.U*c['cfg']['nu']), self.lift_p_per_nu) > 1e-10):
                raise ArithmeticError('fixed-inlet Stokes lift not viscosity-invariant/scaled')
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
        uv, _ = self.evaluate(self.velocity_corrections)
        pv, _ = self.evaluate(self.pressure_corrections, False)
        constant = ng.GridFunction(self.Q); constant.Set(1.)
        one = constant.vec.FV().NumPy().copy()
        one_values, _ = self.evaluate(one, False)
        if relative_gap(one_values[:, 0], np.ones(len(self.weights))) > 1e-12:
            raise ArithmeticError('constant pressure coefficient reconstruction failed')
        pod = coefficient_pod(self.velocity_corrections, uv.transpose(2, 0, 1),
            self.pressure_corrections, pv.T, one, self.weights)
        pressure = pod['pressure_coefficients']; count = pressure.shape[1]
        h1v, mv, dx = self.metric_matrices(self.V, True)
        h1q, mq, _ = self.metric_matrices(self.Q, False)
        ids = np.flatnonzero(self.free)
        hv = h1v[ids][:, ids]; mvv = mv[ids][:, ids]
        factor_v, fm_v = bounded_factor(hv)
        rhs = np.empty((len(ids), count))
        test = self.V.TestFunction()
        for j in range(count):
            psi = ng.GridFunction(self.Q); psi.vec.FV().NumPy()[:] = pressure[:, j]
            form = ng.LinearForm(self.V); form += psi*ng.div(test)/self.D*dx
            with ng.TaskManager(): form.Assemble()
            rhs[:, j] = form.vec.FV().NumPy()[ids]
        sf = factor_v.solve(rhs)
        residual = np.linalg.norm(hv@sf-rhs)/max(np.linalg.norm(rhs), 1e-300)
        if residual > 1e-10: raise ArithmeticError('H1 supremizer linear residual failed')
        supremizers = np.zeros((self.V.ndof, count)); supremizers[ids] = sf
        eigen_v, em_v = bounded_eigenmodes(hv, mvv, factor_v.solve, CAPS['vector_eigen_candidates'])
        ev = np.zeros((self.V.ndof, eigen_v.shape[1])); ev[ids] = eigen_v
        del factor_v
        factor_q, fm_q = bounded_factor(h1q)
        eq, em_q = bounded_eigenmodes(h1q, mq, factor_q.solve, CAPS['scalar_eigen_candidates'])
        del factor_q
        candidates_v = np.c_[supremizers, ev]
        candidates_q = np.c_[one, pressure, eq]
        vv, vg = self.evaluate(candidates_v)
        qq, _ = self.evaluate(candidates_q, False)
        _, _, vm = select_test_columns(vv, self.weights, 64, vg*self.D)
        _, _, qm = select_test_columns(qq, self.weights, 16)
        tests_v = candidates_v[:, vm['candidate_indices']] @ np.asarray(vm['transform'])
        tests_q = candidates_q[:, qm['candidate_indices']] @ np.asarray(qm['transform'])
        pressure_gram = weighted_gram(pod['pressure_values'], self.weights)
        ref_beta = {str(p): reference_inf_sup(rhs[:, :p], sf[:, :p], pressure_gram[:p, :p])
                    for p in sorted(set(min(r, count) for r in (2, 4, 8)))}
        # Same dimensionless full-rule metric must reproduce the assembled metric.
        cv = tests_v.T @ (h1v @ tests_v); cq = tests_q.T @ (mq @ tests_q)
        if max(relative_gap(cv, np.eye(64)), relative_gap(cq, np.eye(16))) > 1e-9:
            raise ArithmeticError('assembled FE metric and mapped-rule test normalization disagree')
        self.coefficients = {'lift_u': self.lift_u, 'lift_p_per_nu': self.lift_p_per_nu,
            'pod_velocity': pod['velocity_coefficients'], 'pressure': pressure,
            'supremizers': supremizers, 'momentum_tests': tests_v, 'continuity_tests': tests_q}
        self.metadata = {'schema': 'ns-obstacle-fe-adapter-v1', 'physical_admission': False,
            'split': 'training', 'Re': REYNOLDS, 'manifest_sha256': self.admission['manifest_sha256'],
            'adapter_source_sha256': adapter_digest(), 'caps': CAPS, 'roundtrip': self.roundtrip,
            'admission_evidence': {k: self.admission[k] for k in ('adapter_review_sha256',
                'reference_admission_sha256', 'campaign_sha256', 'mesh_reload')},
            'source_case_files': [c['case'] for c in self.cases],
            'layout_sha256': self.layout_sha256, 'pod': pod['metadata'],
            'supremizer_residual': float(residual), 'reference_inf_sup': ref_beta,
            'momentum_candidates': ['all maximum-prefix supremizers', 'vector H1 eigenmodes'],
            'continuity_candidates': ['constant', 'all pressure modes', 'scalar Neumann Laplace eigenmodes'],
            'momentum_selection': vm, 'continuity_selection': qm,
            'eigen_velocity': em_v, 'eigen_scalar': em_q, 'factor_velocity': fm_v, 'factor_scalar': fm_q,
            'momentum_H1_gram': cv.tolist(), 'continuity_L2_gram': cq.tolist(),
            'pressure_L2_gram': pressure_gram.tolist(),
            'trace_coefficient_roundoff_removed': self.trace_coefficient_roundoff_removed,
            'seconds_including_load': time.perf_counter()-self.started,
            'versions': {'ngsolve': ng.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__}}
        return self

    def export(self, directory):
        """Persist arrays, FE coefficient bases and their provenance; no integration pass flag."""
        if not hasattr(self, 'coefficients'): raise RuntimeError('build required')
        export_started = time.perf_counter()
        out = Path(directory); out.mkdir(parents=True, exist_ok=False)
        for boundary, name in ((False, 'volume'), (True, 'boundary')):
            mapped = self.boundary if boundary else self.volume
            arrays = {'points_physical': mapped['points'],
                      'weights_dimensionless': mapped['weights']/self.D**(1 if boundary else 2)}
            if boundary:
                arrays.update(labels=mapped['labels'], normals_fluid=mapped['normals_fluid'])
            else: arrays['jacobians_physical'] = mapped['jacobians']
            for key, c in self.coefficients.items():
                vector = key in ('lift_u', 'pod_velocity', 'supremizers', 'momentum_tests')
                values, gradients = self.evaluate(c, vector, boundary)
                arrays[key] = values
                arrays[key+'_gradient_dimensionless'] = self.D*gradients
            np.savez_compressed(out/(name+'.npz'), **arrays)
        np.savez_compressed(out/'coefficients.npz', **self.coefficients)
        # One explicitly training-only archive; exact same population as all FE spaces.
        u, du = self.evaluate(self.velocity_corrections)
        p, _ = self.evaluate(self.pressure_corrections, False)
        from train_bank import array_hash  # Does not import JAX or run training.
        training = {'points_physical': self.volume['points'], 'weights_dimensionless': self.weights,
                    'velocity_corrections': u.transpose(2, 0, 1),
                    'velocity_correction_gradients_dimensionless': (du*self.D).transpose(3, 0, 1, 2),
                    'pressure_corrections': p.T, 'Re': np.asarray(REYNOLDS)}
        np.savez_compressed(out/'training.npz', **training)
        meta = dict(self.metadata)
        meta['training_array_sha256'] = array_hash(training['points_physical'],
            training['weights_dimensionless'], training['velocity_corrections'])
        meta['export_seconds'] = time.perf_counter()-export_started
        meta['total_preparation_seconds'] = time.perf_counter()-self.started
        meta['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        meta['scales'] = {'D': self.D, 'U': self.U, 'nu_hat': 'nu/(U*D)'}
        meta['gradient_axes'] = 'point, component, derivative_coordinate, column'
        meta['boundary_normals'] = 'outward fluid; negate only obstacle normals for force_moments'
        meta['coefficient_files'] = {name: sha_file(out/name) for name in
                                    ('volume.npz', 'boundary.npz', 'coefficients.npz', 'training.npz')}
        (out/'manifest.json').write_text(json.dumps(meta, indent=2, allow_nan=False)+'\n')
        return meta
