"""Unchanged-geometry diagnostic, never a PDE solve or reference admission.

The failed smoke06 did not save its obstacle mesh. This regenerates a new mesh
using the exact same geometry recipe and parameters; it does not reconstruct
the missing original normals. All normal signs are retained without correction.
"""
from __future__ import annotations
import argparse
import faulthandler
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import traceback

import numpy as np
from scalars import scalar_value, scalar_observations


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def marker(name):
    print(json.dumps({'geometry_diagnostic_phase': name}), flush=True)


def finite_array(value, shape):
    result = np.array(value, dtype=np.float64, copy=True).reshape(shape)
    if not np.isfinite(result).all():
        raise ArithmeticError('nonfinite native output')
    return result


def facet_controls(mesh, out, ng, coordinates, radial, normal, volume, policy):
    """Use a constant P0 test to select the volume-facet linear-form path.

Pinned Integral::T_Integrate uses the ordinary BND transformation for ds even
with skeleton=True. LinearForm instead dispatches to CalcFacetVector on the
adjacent volume element. No matrix, factorization or PDE solve is needed.
    """
    target = out/'volume_facet_control.json'
    record = {'status': 'started', 'method': 'constant P0 test contracted with assembled LinearForm',
              'boundary_path': 'ds(label, skeleton=True)',
              'all_element_path': 'dx(element_boundary=True)',
              'bonus_integration_order': policy['facet_bonus_order'],
              'constant_test_error': None, 'per_boundary': {}, 'all_element_boundaries': {},
              'PDE_solve_performed': False}
    dump(target, record)
    try:
        V = ng.L2(mesh, order=0, dgjumps=True)
        record['test_space_dofs'] = int(V.ndof)
        test = V.TestFunction()
        one = ng.GridFunction(V)
        one.Set(ng.CF(1))
        coeff = finite_array(one.vec.FV().NumPy(), (V.ndof,))
        constant_error = 0.
        for el in mesh.Elements(ng.VOL):
            tr = mesh.GetTrafo(el)
            ir = ng.IntegrationRule(el.type, 0)
            ip = ir[0]
            mip = tr(ip)
            constant_error = max(constant_error, abs(scalar_value(one(mip),'P0_constant_test')-1.))
        record['constant_test_error'] = constant_error
        if constant_error > 1e-12:
            raise ArithmeticError('P0 integration test is not constant one')
        terms = {'measure': ng.CF(1), 'normal_x': normal[0], 'normal_y': normal[1],
                 'position_dot_normal': ng.InnerProduct(coordinates, normal),
                 'radial_dot_normal': ng.InnerProduct(radial, normal)}
        def contract(expression, differential):
            form = ng.LinearForm(V)
            form += expression*test*differential
            form.Assemble()
            vector = finite_array(form.vec.FV().NumPy(), (V.ndof,))
            value = float(coeff@vector)
            if not math.isfinite(value):
                raise ArithmeticError('nonfinite facet integral')
            return value
        for label in ('wall_bottom', 'outlet', 'wall_top', 'inlet', 'obstacle'):
            record['per_boundary'][label] = {}
            for name, expression in terms.items():
                marker('volume_facet_'+label+'_'+name)
                ds = ng.ds(label, skeleton=True, bonus_intorder=policy['facet_bonus_order'])
                record['per_boundary'][label][name] = contract(expression, ds)
                dump(target, record)
        for name in ('normal_x', 'normal_y', 'position_dot_normal'):
            marker('all_element_boundaries_'+name)
            dx = ng.dx(element_boundary=True, bonus_intorder=policy['facet_bonus_order'])
            record['all_element_boundaries'][name] = contract(terms[name], dx)
            dump(target, record)
        surface_sum = sum(row['position_dot_normal'] for row in record['per_boundary'].values())
        record['boundary_position_dot_normal'] = surface_sum
        record['expected_twice_mapped_volume'] = 2*volume
        record['divergence_theorem_relative_defect'] = abs(surface_sum-2*volume)/(2*volume)
        record['all_element_divergence_theorem_relative_defect'] = abs(
            record['all_element_boundaries']['position_dot_normal']-2*volume)/(2*volume)
        record['status'] = 'completed_diagnostic_not_admission'
    except Exception as exc:
        record['status'] = 'failed_extra_control_original_geometry_evidence_preserved'
        record['error'] = repr(exc)
        record['traceback'] = traceback.format_exc()
    dump(target, record)
    return record


def run(policy, cfg, out, ng):
    from netgen.geom2d import SplineGeometry
    marker('unchanged_geometry_create')
    geo = SplineGeometry()
    geo.AddRectangle((cfg['xmin'], 0), (cfg['xmax'], cfg['H']),
                     bcs=('wall_bottom', 'outlet', 'wall_top', 'inlet'))
    geo.AddCircle(tuple(cfg['center']), cfg['radius'], leftdomain=0,
                  rightdomain=1, bc='obstacle', maxh=cfg['obstacle_h'])
    mesh = ng.Mesh(geo.GenerateMesh(maxh=cfg['maxh']))
    mesh.Curve(cfg['geometry_order'])
    # Save before diagnostics. This is a fresh mesh, not the missing failed mesh.
    mesh.ngmesh.Save(str(out / 'mesh.vol'))
    dump(out / 'mesh_record.json', {
        'elements': int(mesh.ne), 'vertices': int(mesh.nv),
        'curve_order': int(mesh.GetCurveOrder()), 'mesh_sha256': sha(out/'mesh.vol'),
        'geometry_recipe': 'unchanged from smoke06 manufactured case',
        'mesh_identity_with_failed_run': 'not established; failed mesh was not saved',
        'seed': cfg['seed'], 'seed_semantics': 'recorded; Netgen RNG API is not asserted'})
    if mesh.ne > policy['max_elements']:
        raise MemoryError('element cap exceeded')
    labels = ('wall_bottom', 'outlet', 'wall_top', 'inlet', 'obstacle')
    if set(mesh.GetBoundaries()) != set(labels):
        raise ValueError('unexpected boundary labels')
    c = np.array(cfg['center'], dtype=np.float64)
    normal = ng.specialcf.normal(2)
    coordinates = ng.CF((ng.x, ng.y))
    radial = coordinates - ng.CF(tuple(cfg['center']))
    normal_radial_dot = ng.InnerProduct(normal, radial)
    order = policy['sample_order']
    marker('volume_mapping')
    volume_points, volume_weights, determinants = [], [], []
    volume_centers = {}
    edge_adjacency = {}
    for el in mesh.Elements(ng.VOL):
        vertices = tuple(int(v.nr) for v in el.vertices)
        if len(vertices) != 3:
            raise ValueError('diagnostic expects the unchanged triangular mesh')
        for i in range(3):
            key = tuple(sorted((vertices[i], vertices[(i+1) % 3])))
            edge_adjacency.setdefault(key, []).append(int(el.nr))
        tr = mesh.GetTrafo(el)  # owner must survive all mapped-point accesses
        center_ir = ng.IntegrationRule(el.type, 0)
        center_ip = center_ir[0]
        center_mip = tr(center_ip)
        volume_centers[int(el.nr)] = finite_array(center_mip.point, (2,))
        ir = ng.IntegrationRule(el.type, order)
        for ip in ir:
            mip = tr(ip)
            point = finite_array(mip.point, (2,))
            jac = finite_array(mip.jacobi, (2, 2))
            weight = float(ip.weight * mip.measure)
            determinant = float(np.linalg.det(jac))
            if not math.isfinite(weight) or weight <= 0 or determinant <= 0:
                raise ArithmeticError('invalid volume map')
            volume_points.append(point)
            volume_weights.append(weight)
            determinants.append(determinant)
            if len(volume_points) > policy['max_volume_points']:
                raise MemoryError('volume point cap exceeded')
    points = np.array(volume_points)
    weights = np.array(volume_weights)
    np.savez(out/'volume_rule.npz', points=points, weights=weights,
             determinants=np.array(determinants))
    volume = float(weights.sum())
    expected_volume = (cfg['xmax']-cfg['xmin'])*cfg['H']-math.pi*cfg['radius']**2
    marker('raw_boundary_mapping_and_adjacent_fluid_witnesses')
    records = []
    raw = {label: {'measure': 0., 'normal_integral': np.zeros(2),
                   'position_dot_normal': 0., 'radial_dot_normal': 0.}
           for label in labels}
    for el in mesh.Elements(ng.BND):
        label = el.mat
        key = tuple(sorted(int(v.nr) for v in el.vertices))
        adjacent = edge_adjacency.get(key, [])
        if len(adjacent) != 1:
            raise ValueError(f'exterior boundary must have one adjacent volume: {key}: {adjacent}')
        witness = volume_centers[adjacent[0]]
        tr = mesh.GetTrafo(el)
        ir = ng.IntegrationRule(el.type, order)
        for point_number, ip in enumerate(ir):
            mip = tr(ip)
            point = finite_array(mip.point, (2,))
            n = finite_array(normal(mip), (2,))
            jac = finite_array(mip.jacobi, (2, 1))
            weight = float(ip.weight*mip.measure)
            if not math.isfinite(weight) or weight <= 0:
                raise ArithmeticError('invalid boundary weight')
            radial_np = point-c
            norm = float(np.linalg.norm(n))
            radius = float(np.linalg.norm(radial_np))
            if norm <= 0 or radius <= 0:
                raise ArithmeticError('zero diagnostic vector')
            cosine = float(n@radial_np/(norm*radius))
            offsets = []
            for fraction in policy['offset_fractions_of_D']:
                epsilon = fraction*cfg['D']
                for sign in (-1, 1):
                    q = point+sign*epsilon*n/norm
                    inside = bool(mesh.Contains(float(q[0]), float(q[1])))
                    located = mesh(float(q[0]), float(q[1])) if inside else None
                    offsets.append({'signed_distance': sign*epsilon, 'point': q.tolist(),
                        'mesh_contains': inside,
                        'located_volume_element': int(located.nr) if inside else None,
                        'analytic_fluid_contains': bool(cfg['xmin'] < q[0] < cfg['xmax']
                            and 0 < q[1] < cfg['H'] and np.linalg.norm(q-c) > cfg['radius'])})
            records.append({'element': int(el.nr), 'point_in_element': point_number,
                'label': label, 'point': point.tolist(), 'normal': n.tolist(),
                'normal_norm': norm, 'boundary_jacobian': jac.tolist(), 'weight': weight,
                'radial_cosine': cosine, 'radial_distance': radius,
                'adjacent_volume_element': adjacent[0], 'adjacent_volume_point': witness.tolist(),
                'normal_dot_toward_adjacent_volume': float(n@(witness-point)), 'offsets': offsets})
            raw[label]['measure'] += weight
            raw[label]['normal_integral'] += weight*n
            raw[label]['position_dot_normal'] += weight*float(n@point)
            raw[label]['radial_dot_normal'] += weight*float(n@radial_np)
            if len(records) > policy['max_boundary_points']:
                raise MemoryError('boundary point cap exceeded')
    # Persist unmodified native samples before any aggregate comparison.
    dump(out/'boundary_samples.json', records)
    for row in raw.values():
        row['normal_integral'] = row['normal_integral'].tolist()
    marker('ordinary_boundary_Integrate')
    integrated = {}
    for qorder in policy['integration_orders']:
        per_label = {}
        for label in labels:
            region = mesh.Boundaries(label)
            def integrate(cf):
                return scalar_value(ng.Integrate(cf, mesh, definedon=region, order=qorder),
                                    'normal_diagnostic_integral')
            per_label[label] = {
                'measure': integrate(ng.CF(1)),
                'normal_integral': [integrate(normal[j]) for j in range(2)],
                'position_dot_normal': integrate(ng.InnerProduct(coordinates, normal)),
                'radial_dot_normal': integrate(normal_radial_dot)}
        integrated[str(qorder)] = per_label
    # Persist the raw and ordinary integration paths before trying the separate
    # volume-facet API. An unsupported extra path must not erase these results.
    dump(out/'ordinary_boundary_moments.json', {'raw': raw, 'Integrate': integrated})
    marker('volume_facet_controls')
    facet_result = facet_controls(mesh, out, ng, coordinates, radial, normal, volume, policy)
    marker('summarize_without_admission')
    def totals(rows):
        full = np.sum([row['normal_integral'] for row in rows.values()], axis=0)
        outer = np.sum([rows[label]['normal_integral'] for label in labels if label != 'obstacle'], axis=0)
        divergence_sum = sum(row['position_dot_normal'] for row in rows.values())
        return {'whole_boundary_normal_integral': full.tolist(),
                'outer_rectangle_normal_integral': outer.tolist(),
                'obstacle_normal_integral': rows['obstacle']['normal_integral'],
                'boundary_position_dot_normal': divergence_sum,
                'expected_twice_mapped_volume': 2*volume,
                'divergence_theorem_relative_defect': abs(divergence_sum-2*volume)/(2*volume)}
    obstacle_rows = [row for row in records if row['label'] == 'obstacle']
    summary = {'status': 'diagnostic_completed_not_reference_admission',
        'physical_gate_passed': False, 'normal_sign_corrections_applied': False,
        'same_failed_run_mesh': False, 'PDE_solve_performed': False,
        'volume': volume, 'analytic_volume': expected_volume,
        'volume_relative_geometry_error': abs(volume-expected_volume)/expected_volume,
        'volume_points': len(points), 'boundary_points': len(records),
        'minimum_volume_point_radial_distance': float(np.linalg.norm(points-c, axis=1).min()),
        'volume_points_inside_analytic_obstacle': int(np.count_nonzero(np.linalg.norm(points-c,axis=1) < cfg['radius'])),
        'minimum_volume_jacobian_determinant': min(determinants),
        'obstacle_raw_radial_cosine_range': [min(row['radial_cosine'] for row in obstacle_rows),
                                            max(row['radial_cosine'] for row in obstacle_rows)],
        'obstacle_raw_normal_dot_toward_adjacent_volume_range':
            [min(row['normal_dot_toward_adjacent_volume'] for row in obstacle_rows),
             max(row['normal_dot_toward_adjacent_volume'] for row in obstacle_rows)],
        'raw_boundary_moments': raw, 'raw_totals': totals(raw),
        'Integrate_boundary_moments': integrated,
        'Integrate_totals': {key: totals(value) for key,value in integrated.items()},
        'volume_facet_control': facet_result,
        'raw_vs_Integrate_same_order_max_absolute_moment_difference': max(
            float(np.max(np.abs(np.asarray(raw[label][quantity])-np.asarray(integrated[str(order)][label][quantity]))))
            for label in labels for quantity in raw[label]),
        'scalar_return_contract': scalar_observations(),
        'provenance': {'source_sha256': sha(__file__), 'job_id': os.environ.get('SLURM_JOB_ID'),
            'source_commit': Path('SOURCE_COMMIT.txt').read_text().strip(),
            'hostname': platform.node(), 'ngsolve': ng.__version__, 'numpy': np.__version__,
            'precision': 'float64', 'matmul_precision': os.environ['JAX_DEFAULT_MATMUL_PRECISION']},
        'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        'threading': {'requested_before_TaskManager': cfg['threads'], 'effective_concurrency': 'unmeasured'}}
    dump(out/'result.json', summary)
    return summary


def main():
    faulthandler.enable(all_threads=True)
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    policy = json.loads(Path(args.policy).read_text())
    if sha(policy['configuration']) != policy['configuration_sha256']:
        raise ValueError('frozen geometry configuration mismatch')
    cfg = json.loads(Path(policy['configuration']).read_text())
    if os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') != 'highest' or os.environ.get('JAX_ENABLE_X64') != 'true':
        raise RuntimeError('required precision environment absent')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    dump(out/'policy.json', policy)
    dump(out/'config.json', cfg)
    try:
        import ngsolve as ng
        ng.SetNumThreads(int(cfg['threads']))
        with ng.TaskManager():
            run(policy, cfg, out, ng)
    except Exception as exc:
        dump(out/'FAILED.json', {'error': repr(exc), 'traceback': traceback.format_exc(),
            'source_sha256': sha(__file__), 'job_id': os.environ.get('SLURM_JOB_ID')})
        raise


if __name__ == '__main__':
    main()
