"""Measure unchanged pipe-minus-sphere geometry; no PDE solve or sign repair.

The failed G1 mesh was not saved. This regenerates its recipe, records the new
mesh before normal checks, and does not claim identity with the failed mesh.
"""
from __future__ import annotations
import argparse
import faulthandler
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import resource
import traceback
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def marker(name):
    print(json.dumps({'geometry_diagnostic_phase': name}), flush=True)


def array(value, shape):
    result = np.array(value, dtype=np.float64, copy=True).reshape(shape)
    if not np.isfinite(result).all():
        raise ArithmeticError('nonfinite native array')
    return result


def scalar(raw):
    value = np.asarray(raw)
    if value.size != 1 or value.dtype.kind not in 'iuf' or not np.isfinite(value).all():
        raise ArithmeticError('expected one finite real value')
    return float(value.reshape(-1)[0])


def facet_controls(mesh, out, ng, coordinates, normal, volume, labels, policy):
    """A constant P0 test selects adjacent-volume facet integration.

Plain Integrate(cf*ds(skeleton=True)) does not provide this distinction in the
pinned source. LinearForm dispatches to CalcFacetVector; no solve is performed.
    """
    target = out/'volume_facet_control.json'
    record = {'status': 'started', 'method': 'constant P0 test contracted with LinearForm',
              'boundary_path': 'ds(label,skeleton=True)', 'all_element_path': 'dx(element_boundary=True)',
              'bonus_integration_order': policy['facet_bonus_order'],
              'per_boundary': {}, 'all_element_boundaries': {}, 'PDE_solve_performed': False}
    dump(target, record)
    try:
        V = ng.L2(mesh, order=0, dgjumps=True)
        record['test_space_dofs'] = int(V.ndof)
        test = V.TestFunction()
        one = ng.GridFunction(V)
        one.Set(ng.CF(1))
        coefficients = array(one.vec.FV().NumPy(), (V.ndof,))
        error = 0.
        for element in mesh.Elements(ng.VOL):
            trafo = mesh.GetTrafo(element)
            rule = ng.IntegrationRule(element.type, 0)
            point = trafo(rule[0])
            error = max(error, abs(scalar(one(point))-1.))
        record['constant_test_error'] = error
        if error > 1e-12:
            raise ArithmeticError('P0 integration test is not constant one')
        terms = {'measure': ng.CF(1), 'normal_x': normal[0], 'normal_y': normal[1],
                 'normal_z': normal[2], 'position_dot_normal': ng.InnerProduct(coordinates, normal)}
        def contract(expression, differential):
            form = ng.LinearForm(V)
            form += expression*test*differential
            form.Assemble()
            return scalar(coefficients @ array(form.vec.FV().NumPy(), (V.ndof,)))
        for label in labels:
            record['per_boundary'][label] = {}
            for name, expression in terms.items():
                marker('volume_facet_'+label+'_'+name)
                ds = ng.ds(label, skeleton=True, bonus_intorder=policy['facet_bonus_order'])
                record['per_boundary'][label][name] = contract(expression, ds)
                dump(target, record)
        for name in ('normal_x', 'normal_y', 'normal_z', 'position_dot_normal'):
            marker('all_element_boundaries_'+name)
            dx = ng.dx(element_boundary=True, bonus_intorder=policy['facet_bonus_order'])
            record['all_element_boundaries'][name] = contract(terms[name], dx)
            dump(target, record)
        surface_sum = sum(row['position_dot_normal'] for row in record['per_boundary'].values())
        record['boundary_position_dot_normal'] = surface_sum
        record['expected_three_times_mapped_volume'] = 3*volume
        record['divergence_theorem_relative_defect'] = abs(surface_sum-3*volume)/(3*volume)
        record['all_element_divergence_theorem_relative_defect'] = abs(
            record['all_element_boundaries']['position_dot_normal']-3*volume)/(3*volume)
        record['status'] = 'completed_diagnostic_not_admission'
    except Exception as exc:
        record.update(status='failed_extra_control_raw_evidence_preserved', error=repr(exc),
                      traceback=traceback.format_exc())
    dump(target, record)
    return record


def run(policy, cfg, out, ng):
    from netgen.csg import CSGeometry, Cylinder, Plane, Pnt, Sphere, Vec
    g, m = cfg['geometry'], cfg['mesh']
    marker('unchanged_geometry_create')
    # Same CSG solids, subtraction, mesh sizes and curving as source 1c2fd8fa1.
    pipe = Cylinder(Pnt(0,0,0), Pnt(1,0,0), g['pipe_radius']).bc('wall').maxh(m['wall_maxh'])
    left = Plane(Pnt(g['x_in'],0,0), Vec(-1,0,0)).bc('inlet')
    right = Plane(Pnt(g['x_out'],0,0), Vec(1,0,0)).bc('outlet')
    sphere = Sphere(Pnt(0,0,0), g['sphere_diameter']/2).bc('sphere').maxh(m['sphere_maxh'])
    geo = CSGeometry()
    geo.Add((pipe*left*right-sphere).mat('fluid'))
    mesh = ng.Mesh(geo.GenerateMesh(maxh=m['maxh']))
    mesh.Curve(m['geometry_order'])
    mesh.ngmesh.Save(str(out/'mesh.vol'))
    dump(out/'mesh_record.json', {'elements': int(mesh.ne), 'vertices': int(mesh.nv),
        'curve_order': int(mesh.GetCurveOrder()), 'mesh_sha256': sha(out/'mesh.vol'),
        'geometry_recipe': 'unchanged from failed smoke06 geometry_sphere',
        'same_failed_run_mesh': 'not established; failed mesh was not saved',
        'curved_reload_parity': 'not tested; saved file is not a reload certificate',
        'seed': cfg['seed'], 'seed_semantics': 'NumPy seed recorded; Netgen RNG relation unverified'})
    if mesh.ne > policy['max_elements']:
        raise MemoryError('element cap exceeded')
    labels = ('inlet', 'outlet', 'wall', 'sphere')
    if set(mesh.GetBoundaries()) != set(labels):
        raise ValueError('unexpected boundary labels')
    normal = ng.specialcf.normal(3)
    coordinates = ng.CF((ng.x, ng.y, ng.z))
    order = policy['sample_order']
    marker('volume_mapping')
    volume_points, volume_weights, determinants = [], [], []
    centers, adjacency = {}, {}
    for element in mesh.Elements(ng.VOL):
        vertices = tuple(int(v.nr) for v in element.vertices)
        if len(vertices) != 4:
            raise ValueError('unchanged CSG diagnostic expects tetrahedra')
        for face in itertools.combinations(vertices, 3):
            adjacency.setdefault(tuple(sorted(face)), []).append(int(element.nr))
        trafo = mesh.GetTrafo(element)
        center_rule = ng.IntegrationRule(element.type, 0)
        center_point = trafo(center_rule[0])
        centers[int(element.nr)] = array(center_point.point, (3,))
        for ip in ng.IntegrationRule(element.type, order):
            point = trafo(ip)
            xyz = array(point.point, (3,))
            jacobian = array(point.jacobi.NumPy(), (3,3))
            determinant = float(np.linalg.det(jacobian))
            weight = float(ip.weight*point.measure)
            if not math.isfinite(weight) or determinant <= 0:
                raise ArithmeticError('invalid sampled volume map')
            volume_points.append(xyz); volume_weights.append(weight); determinants.append(determinant)
            if len(volume_points) > policy['max_volume_points']:
                raise MemoryError('volume point cap exceeded')
    points, weights = np.array(volume_points), np.array(volume_weights)
    np.savez(out/'volume_rule.npz', points=points, weights=weights, determinants=np.array(determinants))
    # Mapped samples locate fluid points; a separate high-order scalar integral
    # supplies volume for the divergence theorem on the cubic coordinate map.
    # Record any signed library weights instead of confusing them with a fitted
    # positive rule. This diagnostic cannot admit quadrature for ROM use.
    volume = scalar(ng.Integrate(ng.CF(1),mesh,order=policy['volume_integration_order']))
    if volume <= 0:
        raise ArithmeticError('nonpositive integrated fluid volume')
    expected_volume = math.pi*g['pipe_radius']**2*(g['x_out']-g['x_in'])-4*math.pi*(g['sphere_diameter']/2)**3/3
    marker('raw_boundary_mapping_and_fluid_witnesses')
    records = []
    raw = {label: {'measure': 0., 'normal_integral': np.zeros(3), 'position_dot_normal': 0.}
           for label in labels}
    def analytic_contains(q):
        return bool(g['x_in'] < q[0] < g['x_out'] and np.linalg.norm(q[1:]) < g['pipe_radius']
                    and np.linalg.norm(q) > g['sphere_diameter']/2)
    for element in mesh.Elements(ng.BND):
        label = str(element.mat)
        adjacent = adjacency.get(tuple(sorted(int(v.nr) for v in element.vertices)), [])
        if len(adjacent) != 1:
            raise ValueError('exterior triangular face must have one adjacent fluid tetrahedron')
        witness = centers[adjacent[0]]
        trafo = mesh.GetTrafo(element)
        for point_number, ip in enumerate(ng.IntegrationRule(element.type, order)):
            mapped = trafo(ip)
            point = array(mapped.point, (3,))
            n = array(normal(mapped), (3,))
            jacobian = array(mapped.jacobi.NumPy(), (3,2))
            weight = float(ip.weight*mapped.measure)
            norm = float(np.linalg.norm(n))
            if not math.isfinite(weight) or weight <= 0 or norm <= 0:
                raise ArithmeticError('invalid boundary sample')
            radial = point if label=='sphere' else np.array([0.,point[1],point[2]])
            radial_cosine = float(n@radial/(norm*np.linalg.norm(radial))) if label in ('sphere','wall') else None
            offsets = []
            for fraction in policy['offset_fractions_of_D']:
                for sign in (-1,1):
                    distance = sign*fraction*g['sphere_diameter']
                    q = point+distance*n/norm
                    inside = bool(mesh.Contains(*map(float,q)))
                    located = mesh(*map(float,q)) if inside else None
                    offsets.append({'signed_distance': distance, 'point': q.tolist(), 'mesh_contains': inside,
                        'located_volume_element': int(located.nr) if inside else None,
                        'analytic_fluid_contains': analytic_contains(q)})
            records.append({'element': int(element.nr), 'point_in_element': point_number, 'label': label,
                'point': point.tolist(), 'normal': n.tolist(), 'normal_norm': norm,
                'boundary_jacobian': jacobian.tolist(), 'weight': weight, 'radial_cosine': radial_cosine,
                'axial_normal_component': float(n[0]), 'adjacent_volume_element': adjacent[0],
                'adjacent_volume_point': witness.tolist(), 'normal_dot_toward_adjacent_volume': float(n@(witness-point)),
                'offsets': offsets})
            raw[label]['measure'] += weight
            raw[label]['normal_integral'] += weight*n
            raw[label]['position_dot_normal'] += weight*float(n@point)
            if len(records) > policy['max_boundary_points']:
                raise MemoryError('boundary point cap exceeded')
    dump(out/'boundary_samples.json', records)
    for row in raw.values():
        row['normal_integral'] = row['normal_integral'].tolist()
    marker('ordinary_BND_Integrate')
    integrated = {}
    for qorder in policy['integration_orders']:
        integrated[str(qorder)] = {}
        for label in labels:
            def integrate(cf):
                return scalar(ng.Integrate(cf, mesh, ng.BND, definedon=mesh.Boundaries(label), order=qorder))
            integrated[str(qorder)][label] = {'measure': integrate(ng.CF(1)),
                'normal_integral': [integrate(normal[j]) for j in range(3)],
                'position_dot_normal': integrate(ng.InnerProduct(coordinates,normal))}
    def totals(rows):
        total = sum(row['position_dot_normal'] for row in rows.values())
        return {'whole_boundary_normal_integral': np.sum([row['normal_integral'] for row in rows.values()],axis=0).tolist(),
            'sphere_normal_integral': rows['sphere']['normal_integral'],
            'outer_pipe_normal_integral': np.sum([rows[label]['normal_integral'] for label in labels if label!='sphere'],axis=0).tolist(),
            'boundary_position_dot_normal': total, 'expected_three_times_mapped_volume': 3*volume,
            'divergence_theorem_relative_defect': abs(total-3*volume)/(3*volume)}
    sphere_rows = [row for row in records if row['label']=='sphere']
    result = {'status': 'raw_and_ordinary_integrals_completed_no_admission',
        'normal_sign_corrections_applied': False, 'PDE_solve_performed': False, 'same_failed_run_mesh': False,
        'volume': volume, 'analytic_volume': expected_volume, 'volume_relative_geometry_error': abs(volume-expected_volume)/expected_volume,
        'volume_points': len(points), 'boundary_points': len(records),
        'mapped_volume_weight_sum': float(weights.sum()),
        'negative_sampled_volume_weight_count': int(np.count_nonzero(weights<0)),
        'volume_integration_order': policy['volume_integration_order'],
        'minimum_volume_jacobian_determinant': min(determinants),
        'volume_points_inside_analytic_sphere': int(np.count_nonzero(np.linalg.norm(points,axis=1)<g['sphere_diameter']/2)),
        'sphere_raw_radial_cosine_range': [min(row['radial_cosine'] for row in sphere_rows),max(row['radial_cosine'] for row in sphere_rows)],
        'sphere_raw_normal_dot_toward_adjacent_volume_range': [min(row['normal_dot_toward_adjacent_volume'] for row in sphere_rows),max(row['normal_dot_toward_adjacent_volume'] for row in sphere_rows)],
        'raw_boundary_moments': raw, 'raw_totals': totals(raw),
        'ordinary_BND_moments': integrated, 'ordinary_BND_totals': {key:totals(rows) for key,rows in integrated.items()},
        'raw_vs_ordinary_same_order_max_absolute_moment_difference': max(float(np.max(np.abs(np.asarray(raw[label][key])-np.asarray(integrated[str(order)][label][key])))) for label in labels for key in raw[label]),
        'provenance': {'source_sha256': sha(__file__), 'job_id': os.environ.get('SLURM_JOB_ID'),
            'source_commit': Path('SOURCE_COMMIT.txt').read_text().strip(), 'hostname': platform.node(),
            'ngsolve': ng.__version__, 'numpy': np.__version__, 'precision': 'float64'},
        'threading': {'requested_before_TaskManager':cfg['threads'],'effective_concurrency':'unmeasured'}}
    dump(out/'result.json', result)
    result['volume_facet_control'] = facet_controls(mesh,out,ng,coordinates,normal,volume,labels,policy)
    result['status'] = ('diagnostic_completed_not_reference_admission'
        if result['volume_facet_control']['status']=='completed_diagnostic_not_admission'
        else 'raw_diagnostic_completed_facet_control_failed')
    result['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    dump(out/'result.json', result)


def main():
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args = parser.parse_args()
    policy = json.loads(args.policy.read_text())
    if sha(policy['configuration']) != policy['configuration_sha256']:
        raise RuntimeError('frozen geometry configuration hash mismatch')
    if os.environ.get('JAX_ENABLE_X64')!='true' or os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')!='highest':
        raise RuntimeError('required precision environment absent')
    cfg = json.loads(Path(policy['configuration']).read_text())
    args.out.mkdir(parents=True,exist_ok=False)
    dump(args.out/'policy.json',policy); dump(args.out/'config.json',cfg)
    try:
        import ngsolve as ng
        ng.SetNumThreads(int(cfg['threads']))
        np.random.seed(cfg['seed'])
        with ng.TaskManager():
            run(policy,cfg,args.out,ng)
    except BaseException as exc:
        dump(args.out/'FAILED.json',{'error':repr(exc),'traceback':traceback.format_exc(),
            'source_sha256':sha(__file__),'job_id':os.environ.get('SLURM_JOB_ID')})
        raise


if __name__=='__main__':
    main()
