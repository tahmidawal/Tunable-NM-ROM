"""One explicit native implementation control; not a production training loader.

Consumes one honest coarse DFG producer artifact. It does not fabricate a family,
fit a POD/bank, create training admission, or call TrainingFEAdapter.open/build.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import time

import numpy as np

from fe_adapter import (TrainingFEAdapter, CAPS, ESSENTIAL, MESH_FILENAME, MESH_FORMAT,
    RELOAD_POLICY, adapter_digest, sha_file, layout_record, load_npz_bounded,
    relative_gap, verify_mesh_contract, verify_normal_certificate,
    bounded_factor, bounded_eigenmodes, sparse_digest)
from fe_pod_a6 import check_component_blocks
from oriented_normals import fluid_outward_normal


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


class ControlView:
    """Only geometry/evaluation/metric primitives; no production admission API."""
    mapped_rule = TrainingFEAdapter.mapped_rule
    evaluate = TrainingFEAdapter.evaluate
    iter_mapped = TrainingFEAdapter.iter_mapped
    control_indices = TrainingFEAdapter.control_indices
    analytic_interface_controls = TrainingFEAdapter.analytic_interface_controls
    scalar_component_maps = TrainingFEAdapter.scalar_component_maps
    metric_matrices = TrainingFEAdapter.metric_matrices

    def __init__(self, folder, out, ng):
        self.ng = ng; self.cfg = json.loads((folder/'config.json').read_text())
        cfg = self.cfg
        if (cfg['kind'] != 'dfg' or cfg['release'] != 'G2_steady_control_candidate'
                or cfg.get('split', 'control') != 'control' or cfg['precision'] != 'float64'):
            raise ValueError('one declared DFG implementation control required')
        self.U = cfg['U']; self.D = cfg['D']
        identity = json.loads((folder/'mesh_identity.json').read_text())
        replay = json.loads((folder/'reload_checks.json').read_text())
        verify_mesh_contract(identity); verify_mesh_contract(replay)
        if (replay['passed'] is not True or identity['reload_passed'] is not True or
                identity['mesh_sha256'] != sha_file(folder/MESH_FILENAME) or
                identity['reload_checks_sha256'] != sha_file(folder/'reload_checks.json')):
            raise ValueError('producer binary replay evidence does not bind this control')
        ng.SetNumThreads(cfg['threads'])
        self.mesh = ng.Mesh(str(folder/MESH_FILENAME))
        if self.mesh.dim != 2 or self.mesh.ne > CAPS['elements']:
            raise MemoryError('native control mesh cap')
        if int(self.mesh.GetCurveOrder()) != cfg['geometry_order']:
            raise ArithmeticError('binary curve order differs')
        self.normal, self.normal_certificate = fluid_outward_normal(self.mesh, cfg['center'],
            order=cfg['diagnostic_order'], evidence_path=out/'normal_orientation.json')
        original = json.loads((folder/'normal_orientation.json').read_text())
        if verify_normal_certificate(original) != verify_normal_certificate(self.normal_certificate):
            raise ArithmeticError('control orientation factors differ from actual producer')
        self.V = ng.VectorH1(self.mesh, order=cfg['velocity_order'], dirichlet=ESSENTIAL)
        self.Q = ng.H1(self.mesh, order=cfg['pressure_order'])
        self.S = ng.H1(self.mesh, order=cfg['velocity_order'], dirichlet=ESSENTIAL)
        self.X = self.V*self.Q
        if self.X.ndof > CAPS['mixed_dofs']: raise MemoryError('native control DOF cap')
        self.layout, self.layout_sha256 = layout_record(self.mesh, self.V, self.Q, ng)
        saved = json.loads((folder/'fe_layout.json').read_text())
        if saved['layout'] != self.layout or saved['sha256'] != self.layout_sha256:
            raise ArithmeticError('native control ordered DOF layout differs')
        self.free = np.asarray(self.layout['velocity_free'], dtype=bool)
        self.volume = self.mapped_rule(); self.boundary = self.mapped_rule(boundary=True)


def replay_fields(view, folder, out):
    ng = view.ng; data = load_npz_bounded(folder/'full_rule.npz'); checks = {}
    for name, scale in [('points', view.D), ('weights', 1.), ('jacobians', 1.)]:
        checks[name] = relative_gap(view.volume[name], data[name], scale)
    solution = ng.GridFunction(view.X); solution.Load(str(folder/'solution.sol'))
    coefficients = solution.vec.FV().NumPy().copy()
    checks['solution_coefficients'] = relative_gap(coefficients, data['solution_coefficients'])
    nv = view.V.ndof
    values, gradients = view.evaluate(coefficients[:nv])
    pressure, _ = view.evaluate(coefficients[nv:], vector=False)
    checks['velocity'] = relative_gap(values[..., 0], data['velocity'], view.U)
    checks['velocity_gradient'] = relative_gap(gradients[..., 0], data['velocity_gradient'], view.U/view.D)
    checks['pressure'] = relative_gap(pressure[:, 0], data['pressure'], view.U**2)
    boundary = json.loads((folder/'boundary_rule.json').read_text())
    if view.boundary['labels'].tolist() != [r['label'] for r in boundary]:
        raise ArithmeticError('native control boundary enumeration differs')
    for actual, saved, scale in [('points', 'point', view.D), ('weights', 'weight', 1.),
            ('normals_fluid', 'normal', 1.), ('normals_raw', 'raw_normal', 1.),
            ('orientation_factors', 'normal_orientation_factor', 1.)]:
        checks['boundary_'+saved] = relative_gap(view.boundary[actual], [r[saved] for r in boundary], scale)
    bv, bg = view.evaluate(coefficients[:nv], boundary=True)
    bp, _ = view.evaluate(coefficients[nv:], vector=False, boundary=True)
    for key, actual, scale in [('velocity', bv[..., 0], view.U),
            ('velocity_gradient', bg[..., 0], view.U/view.D), ('pressure', bp[:, 0], view.U**2)]:
        checks['boundary_'+key] = relative_gap(actual, [r[key] for r in boundary], scale)
    np.savez_compressed(out/'replayed_fields.npz', velocity=values[..., 0], pressure=pressure[:, 0],
        velocity_gradient=gradients[..., 0], boundary_velocity=bv[..., 0], boundary_pressure=bp[:, 0],
        boundary_gradient=bg[..., 0], **{name: value for name, value in view.volume.items()})
    dump(out/'field_replay.json', {'threshold': 1e-10, 'checks': checks, 'passed': max(checks.values()) <= 1e-10})
    if max(checks.values()) > 1e-10: raise ArithmeticError('adapter field replay exceeds fixed threshold')
    return checks


def metric_control(view, space, vector, h1, mass):
    ng = view.ng; g = ng.GridFunction(space)
    g.Set(ng.CF((ng.x+2*ng.y, 3*ng.x-ng.y)) if vector else ng.x+2*ng.y)
    c = g.vec.FV().NumPy().copy(); values = []; gradients = []
    derivative = ng.Grad(g)
    for mip in view.iter_mapped():
        values.append(np.asarray(g(mip)).reshape(-1))
        gradients.append(np.asarray(derivative(mip)).reshape(-1))
    values = np.asarray(values); gradients = np.asarray(gradients)
    w = view.volume['weights']/view.D**2
    sampled_mass = float(w@np.sum(values*values, axis=1))
    sampled_h1 = sampled_mass + float(w@np.sum((view.D*gradients)**2, axis=1))
    assembled_mass = float(c@(mass@c)); assembled_h1 = float(c@(h1@c))
    error = max(abs(sampled_mass-assembled_mass)/max(abs(sampled_mass), 1e-300),
                abs(sampled_h1-assembled_h1)/max(abs(sampled_h1), 1e-300))
    if error > 1e-10: raise ArithmeticError('sampled/assembled dimensionless metric mismatch')
    return {'sampled_mass': sampled_mass, 'assembled_mass': assembled_mass,
            'sampled_H1': sampled_h1, 'assembled_H1': assembled_h1, 'relative_gap': error,
            'threshold': 1e-10, 'field': 'affine FE interpolation, not a training snapshot'}


def run(request_path, output):
    request_path = Path(request_path).resolve(); request = json.loads(request_path.read_text())
    if (request.get('schema') != 'ns-obstacle-rom-native-control-v1' or
            request.get('scope') != 'one_coarse_implementation_control_only' or
            request.get('adapter_source_sha256') != adapter_digest() or
            request.get('fixture_source_sha256') != sha_file(__file__)):
        raise ValueError('exact-source native control request required')
    folder = (request_path.parent/request['control_directory']).resolve()
    for name, digest in request['files'].items():
        if sha_file(folder/name) != digest: raise ValueError(f'control input differs: {name}')
    producer = json.loads((folder/'EXPORT-MANIFEST.json').read_text())
    verify_mesh_contract(producer, 'reload_policy')
    if producer['physical_admission'] is not False or producer['case']['split'] != 'control':
        raise ValueError('native fixture must use the honest unadmitted control')
    out = Path(output); out.mkdir(parents=True, exist_ok=False); started = time.perf_counter()
    report = {'passed': False, 'scope': request['scope'], 'physical_admission': False,
              'training_admission': False, 'bank_or_POD_fitted': False,
              'request_sha256': sha_file(request_path), 'adapter_source_sha256': adapter_digest(),
              'fixture_source_sha256': sha_file(__file__)}
    try:
        import ngsolve as ng
        import scipy
        from scipy.sparse import save_npz
        view = ControlView(folder, out, ng)
        report['counts'] = {'elements': view.mesh.ne, 'mixed_dofs': view.X.ndof,
                            'volume_points': len(view.volume['points']), 'boundary_points': len(view.boundary['points'])}
        report['mesh_sha256'] = sha_file(folder/MESH_FILENAME); report['layout_sha256'] = view.layout_sha256
        report['field_replay'] = replay_fields(view, folder, out)
        report['analytic_interface'] = view.analytic_interface_controls()
        maps, report['component_maps'] = view.scalar_component_maps()
        matrices = {}
        for name, space, vector in [('velocity', view.V, True), ('velocity_scalar', view.S, False), ('pressure', view.Q, False)]:
            h1, mass, _ = view.metric_matrices(space, vector)
            matrices[name] = (h1, mass)
            for kind, matrix in [('H1', h1), ('mass', mass)]: save_npz(out/f'{name}_{kind}.npz', matrix)
            report[name+'_metric'] = metric_control(view, space, vector, h1, mass)
        hv, mv = matrices['velocity']; hs, ms = matrices['velocity_scalar']
        report['component_blocks'] = check_component_blocks(hv, mv, hs, ms, maps)
        eigenarrays = {'component_maps': maps}
        for name, space, count, essential in [('velocity_scalar', view.S, CAPS['velocity_scalar_eigen_candidates'], True),
                                              ('pressure', view.Q, CAPS['scalar_eigen_candidates'], False)]:
            h1, mass = matrices[name]
            ids = np.flatnonzero(np.array([bool(v) for v in space.FreeDofs()])) if essential else np.arange(space.ndof)
            a = h1[ids][:, ids]; m = mass[ids][:, ids]
            factor, factor_info = bounded_factor(a)
            modes, info = bounded_eigenmodes(a, m, factor.solve, count)
            report[name+'_eigen'] = dict(info, factorization=factor_info,
                H1_sha256=sparse_digest(a), mass_sha256=sparse_digest(m))
            eigenarrays[name+'_free_dofs'] = ids; eigenarrays[name+'_modes'] = modes
            del factor
        np.savez_compressed(out/'eigen_controls.npz', **eigenarrays)
        report['versions'] = {'ngsolve': ng.__version__, 'scipy': scipy.__version__, 'numpy': np.__version__}
        report['passed'] = True
    except Exception as exc:
        report['failure'] = repr(exc)
        raise
    finally:
        report['seconds'] = time.perf_counter()-started
        report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        report['outputs'] = {p.name: sha_file(p) for p in sorted(out.iterdir()) if p.is_file()}
        dump(out/'result.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    arguments = parser.parse_args()
    run(arguments.request, arguments.out)
