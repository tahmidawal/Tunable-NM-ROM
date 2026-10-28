"""Real saved-FE interface control, explicitly separate from training admission.

Uses selected reviewed adapter primitives on one unadmitted producer control. It never
calls the production loader/build/export, constructs a training manifest, or fits a bank.
Native execution requires an exact-source coordinator release and a cluster allocation.
"""
from pathlib import Path
import argparse
import json
import os
import resource
import sys
import time
import numpy as np
from fe_adapter import (TrainingFEAdapter, adapter_digest, sha_file, verify_mesh_contract,
    verify_normal_certificate, layout_record, CAPS, ESSENTIAL, MESH_FILENAME,
    bounded_factor, bounded_eigenmodes, sparse_digest)
from fe_pod_a6 import check_component_blocks
from oriented_normals import fluid_outward_normal


class NativeInterface:
    """Only low-level numerical methods; no production admission or training methods."""
    mapped_rule = TrainingFEAdapter.mapped_rule
    iter_mapped = TrainingFEAdapter.iter_mapped
    evaluate = TrainingFEAdapter.evaluate
    control_indices = TrainingFEAdapter.control_indices
    analytic_interface_controls = TrainingFEAdapter.analytic_interface_controls
    scalar_component_maps = TrainingFEAdapter.scalar_component_maps
    metric_matrices = TrainingFEAdapter.metric_matrices


def run(case, release_path, output):
    started = time.perf_counter(); case = Path(case).resolve(); output = Path(output)
    release = json.loads(Path(release_path).read_text())
    if (release.get('schema') != 'ns-obstacle-native-adapter-control-release-v1'
        or release.get('adapter_source_sha256') != adapter_digest()
        or release.get('fixture_source_sha256') != sha_file(__file__)
        or release.get('physical_admission') is not False):
        raise ValueError('exact-source native interface control release required')
    producer = case/'EXPORT-MANIFEST.json'
    if sha_file(producer) != release['producer_manifest_sha256']:
        raise ValueError('released producer differs')
    manifest = json.loads(producer.read_text()); verify_mesh_contract(manifest, 'reload_policy')
    for name, expected in manifest['files'].items():
        if Path(name).name != name or sha_file(case/name) != expected:
            raise ValueError('producer file hash mismatch: '+name)
    cfg = json.loads((case/'config.json').read_text())
    if (cfg['release'] != 'G2_steady_control_candidate' or cfg['kind'] != 'dfg'
        or cfg['export_common_stokes_lift'] is not False):
        raise ValueError('one explicit unadmitted producer control required')
    replay = json.loads((case/'reload_checks.json').read_text()); verify_mesh_contract(replay)
    if replay.get('passed') is not True: raise ValueError('producer reload did not pass')
    if (sys.executable != '/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python'
        or not os.environ.get('SLURM_JOB_ID')
        or os.environ.get('JAX_ENABLE_X64') != 'true'
        or os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') != 'highest'):
        raise RuntimeError('reviewed cluster allocation with prescribed precision required')
    output.mkdir(parents=True, exist_ok=False)
    report = {'passed': False, 'physical_admission': False, 'training_released': False,
        'scope': 'single saved-FE native interface and metric/eigen control only',
        'adapter_source_sha256': adapter_digest(), 'fixture_source_sha256': sha_file(__file__),
        'producer_manifest_sha256': sha_file(producer), 'checks': {}}
    try:
        import ngsolve as ng
        ng.SetNumThreads(cfg['threads'])
        a = NativeInterface(); a.ng = ng; a.cfg = cfg; a.U = cfg['U']; a.D = cfg['D']
        a.mesh = ng.Mesh(str(case/MESH_FILENAME))
        if a.mesh.dim != 2 or a.mesh.ne > CAPS['elements'] or int(a.mesh.GetCurveOrder()) != cfg['geometry_order']:
            raise ValueError('mesh dimension/order/resource guard')
        a.normal, a.normal_certificate = fluid_outward_normal(a.mesh, cfg['center'],
            order=cfg['diagnostic_order'], evidence_path=output/'normal_orientation.json')
        if verify_normal_certificate(a.normal_certificate) != verify_normal_certificate(json.loads((case/'normal_orientation.json').read_text())):
            raise ValueError('normal factors differ')
        a.V = ng.VectorH1(a.mesh, order=cfg['velocity_order'], dirichlet=ESSENTIAL)
        a.Q = ng.H1(a.mesh, order=cfg['pressure_order'])
        a.S = ng.H1(a.mesh, order=cfg['velocity_order'], dirichlet=ESSENTIAL)
        a.X = a.V*a.Q
        if a.X.ndof > CAPS['mixed_dofs']: raise MemoryError('mixed DOF cap')
        layout, digest = layout_record(a.mesh, a.V, a.Q, ng)
        if json.loads((case/'fe_layout.json').read_text()) != {'layout': layout, 'sha256': digest}:
            raise ValueError('ordered FE layout mismatch')
        a.free = np.asarray(layout['velocity_free'], dtype=bool)
        a.volume = a.mapped_rule(); a.boundary = a.mapped_rule(True)
        a.weights = a.volume['weights']/a.D**2
        report['analytic_controls'] = a.analytic_interface_controls()
        a.component_maps, report['component_maps'] = a.scalar_component_maps()
        state = ng.GridFunction(a.X); state.Load(str(case/'solution.sol'))
        velocity, gradient = a.evaluate(state.components[0].vec.FV().NumPy().copy()/a.U)
        pressure, _ = a.evaluate(state.components[1].vec.FV().NumPy().copy()/a.U**2, False)
        with np.load(case/'full_rule.npz', allow_pickle=False) as saved:
            pairs = {'points': (a.volume['points'], saved['points'], a.D),
                'weights': (a.volume['weights'], saved['weights'], 1.),
                'jacobians': (a.volume['jacobians'], saved['jacobians'], 1.),
                'velocity': (velocity[..., 0], saved['velocity']/a.U, 1.),
                'gradient': (gradient[..., 0]*a.D, saved['velocity_gradient']*a.D/a.U, 1.),
                'pressure': (pressure[..., 0], saved['pressure']/a.U**2, 1.)}
            for key, (actual, expected, scale) in pairs.items():
                if actual.shape != expected.shape: raise ValueError('array shape mismatch')
                error = float(np.max(np.abs(actual-expected))/max(float(np.max(np.abs(expected))), scale))
                report['checks'][key] = error
                if not np.isfinite(error) or error > 1e-10: raise ArithmeticError(key+' replay failed')
        boundary = json.loads((case/'boundary_rule.json').read_text())
        if a.boundary['labels'].tolist() != [r['label'] for r in boundary]:
            raise ValueError('boundary enumeration mismatch')
        uv, duv = a.evaluate(state.components[0].vec.FV().NumPy().copy()/a.U, boundary=True)
        pv, _ = a.evaluate(state.components[1].vec.FV().NumPy().copy()/a.U**2, False, boundary=True)
        saved = lambda key: np.asarray([r[key] for r in boundary], dtype=np.float64)
        pairs = {'boundary_points': (a.boundary['points'], saved('point'), a.D),
            'boundary_weights': (a.boundary['weights'], saved('weight'), 1.),
            'boundary_raw_normals': (a.boundary['normals_raw'], saved('raw_normal'), 1.),
            'boundary_normals': (a.boundary['normals_fluid'], saved('normal'), 1.),
            'boundary_velocity': (uv[..., 0], saved('velocity')/a.U, 1.),
            'boundary_gradient': (duv[..., 0]*a.D, saved('velocity_gradient')*a.D/a.U, 1.),
            'boundary_pressure': (pv[..., 0], saved('pressure')/a.U**2, 1.)}
        for key, (actual, expected, scale) in pairs.items():
            if actual.shape != expected.shape: raise ValueError(key+' shape mismatch')
            error = float(np.max(np.abs(actual-expected))/max(float(np.max(np.abs(expected))), scale))
            report['checks'][key] = error
            if not np.isfinite(error) or error > 1e-10: raise ArithmeticError(key+' replay failed')
        expected_factors = np.asarray([a.normal_certificate['factors'][r['label']] for r in boundary])
        if not np.array_equal(a.boundary['orientation_factors'], expected_factors):
            raise ValueError('boundary factors mismatch')
        h1v, mv, _ = a.metric_matrices(a.V, True)
        h1s, ms, _ = a.metric_matrices(a.S, False)
        h1q, mq, _ = a.metric_matrices(a.Q, False)
        report['component_blocks'] = check_component_blocks(h1v, mv, h1s, ms, a.component_maps)
        ids = np.flatnonzero(np.asarray([bool(v) for v in a.S.FreeDofs()]))
        report['eigen_controls'] = {}
        for label, h, m, count in [('velocity_scalar', h1s[ids][:, ids], ms[ids][:, ids], CAPS['velocity_scalar_eigen_candidates']),
                                   ('pressure_scalar', h1q, mq, CAPS['scalar_eigen_candidates'])]:
            factor, evidence = bounded_factor(h)
            vectors, check = bounded_eigenmodes(h, m, factor.solve, count)
            np.savez_compressed(output/(label+'_eigen.npz'), coefficients=vectors)
            report['eigen_controls'][label] = {'factor': evidence, 'checks': check, 'mass_sha256': sparse_digest(m)}
            del factor
        report['counts'] = {'elements': a.mesh.ne, 'mixed_dofs': a.X.ndof,
            'volume_points': len(a.volume['points']), 'boundary_points': len(a.boundary['points'])}
        report['passed'] = True
    except BaseException as exc:
        report['failure'] = repr(exc)
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter()-started
        report['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        (output/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', required=True, type=Path)
    parser.add_argument('--release', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(); run(args.case, args.release, args.out)
