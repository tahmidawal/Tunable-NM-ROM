"""Verified fluid-outward normals for ordinary 2D boundary transformations.

NGSolve's raw BND normal follows the geometry edge orientation. Preserve that
curved unit normal and multiply it by a verified boundary-wise sign. This helper
is never used inside volume-facet integration, whose normals are already oriented
by their adjacent volume element. No analytic normal replaces a curved normal.
"""
from __future__ import annotations
import json
import hashlib
import math
from pathlib import Path

import numpy as np
from scalars import scalar_value

API_TOL = 1e-10
MIN_WITNESS_COSINE = 1e-8


def infer_label_factors(samples):
    """Infer signs from uncorrected normals and interior adjacency witnesses."""
    factors = {}
    for row in samples:
        normal = np.asarray(row['raw_normal'], dtype=np.float64)
        inward = np.asarray(row['adjacent_volume_point'], dtype=np.float64)-np.asarray(row['point'], dtype=np.float64)
        if normal.shape != (2,) or inward.shape != (2,) or not np.isfinite(normal).all() or not np.isfinite(inward).all():
            raise ValueError('invalid normal or adjacent-volume witness')
        norm = float(np.linalg.norm(normal)); distance = float(np.linalg.norm(inward))
        if abs(norm-1) > API_TOL or distance <= 0:
            raise ValueError('nonunit normal or degenerate adjacent-volume witness')
        cosine = float(normal@inward/(norm*distance))
        row['raw_normal_dot_toward_adjacent_volume'] = float(normal@inward)
        row['raw_normal_cosine_toward_adjacent_volume'] = cosine
        if abs(cosine) <= MIN_WITNESS_COSINE:
            raise ValueError('absolute adjacency witness cosine is at most 1e-8')
        factor = -1 if cosine > 0 else 1
        label = row['label']
        if label in factors and factors[label] != factor:
            raise ValueError(f'normal orientation is not constant on boundary label {label}')
        factors[label] = factor
        row['orientation_factor'] = factor
        row['fluid_outward_normal'] = (factor*normal).tolist()
    if not factors:
        raise ValueError('no boundary orientation samples')
    return factors


def witness_margins(samples):
    """Observed normalized normal-component margins, including partial failures."""
    margins = {}
    for row in samples:
        if 'raw_normal_cosine_toward_adjacent_volume' in row:
            value = abs(row['raw_normal_cosine_toward_adjacent_volume'])
            label = row['label']
            margins[label] = min(margins.get(label, value), value)
    return dict(sorted(margins.items()))


def fluid_outward_normal(mesh, center=None, order=12, evidence_path=None):
    """Return an ordinary-BND-only CF and its source-independent certificate.

Sign selection uses the unique topologically adjacent triangle's mapped interior
point. Analytic signs, closed components and a separate volume-facet LinearForm
then verify the choice. Unsupported geometry, inconsistent signs, or failed
verification stop the call; no sign is guessed from a boundary's name.
    """
    import ngsolve as ng
    samples = []
    record = {'schema': 'ns-obstacle-2d-fluid-outward-normal-v1', 'passed': False,
              'helper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'ordinary BND mapped points and ordinary BND integrals only',
              'method': 'raw curved normal times topology-inferred, independently verified per-label sign',
              'volume_facet_normals_modified': False, 'analytic_normals_substituted': False,
              'order': max(6, int(order)), 'threshold': API_TOL,
              'adjacency_witness_guard_absolute_cosine': MIN_WITNESS_COSINE,
              'adjacency_witness_guard_meaning': 'reject abs(raw_normal dot inward)/(norm(raw_normal)*norm(inward)) <= 1e-8; a near-zero numerical guard, not a mesh-quality criterion',
              'future_mesh_false_stop_risk_accepted': True,
              'samples': samples}
    def persist():
        if evidence_path is not None:
            Path(evidence_path).write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    try:
        qorder = record['order']
        raw = ng.specialcf.normal(2)
        directions = {'inlet': np.array([-1., 0.]), 'outlet': np.array([1., 0.]),
                      'wall_top': np.array([0., 1.]), 'wall_bottom': np.array([0., -1.])}
        expected_labels = set(directions) | ({'obstacle'} if center is not None else set())
        if set(mesh.GetBoundaries()) != expected_labels:
            raise ValueError('unexpected boundary labels for fixed channel geometry')
        adjacency, interior = {}, {}
        for element in mesh.Elements(ng.VOL):
            vertices = tuple(int(v.nr) for v in element.vertices)
            if len(vertices) != 3:
                raise ValueError('normal helper requires the declared triangular mesh')
            for i in range(3):
                edge = tuple(sorted((vertices[i], vertices[(i+1) % 3])))
                adjacency.setdefault(edge, []).append(int(element.nr))
            tr = mesh.GetTrafo(element)
            ir = ng.IntegrationRule(element.type, 0)
            ip = ir[0]
            mip = tr(ip)
            interior[int(element.nr)] = np.array(mip.point, dtype=np.float64, copy=True)
            if not np.isfinite(interior[int(element.nr)]).all():
                raise ValueError('invalid mapped interior point')
        for element in mesh.Elements(ng.BND):
            edge = tuple(sorted(int(v.nr) for v in element.vertices))
            neighbors = adjacency.get(edge, [])
            if len(neighbors) != 1:
                raise ValueError('boundary does not have exactly one adjacent fluid element')
            tr = mesh.GetTrafo(element)
            ir = ng.IntegrationRule(element.type, qorder)
            for index, ip in enumerate(ir):
                mip = tr(ip)
                point = np.array(mip.point, dtype=np.float64, copy=True)
                normal = np.array(raw(mip), dtype=np.float64, copy=True)
                weight = float(ip.weight*mip.measure)
                if not np.isfinite(point).all() or not math.isfinite(weight) or weight <= 0:
                    raise ValueError('invalid boundary point or weight')
                samples.append({'element': int(element.nr), 'point_in_element': index,
                    'label': str(element.mat), 'point': point.tolist(), 'weight': weight,
                    'raw_normal': normal.tolist(), 'adjacent_volume_element': neighbors[0],
                    'adjacent_volume_point': interior[neighbors[0]].tolist()})
        # Save raw data before inference or checks can fail.
        persist()
        factors = infer_label_factors(samples)
        record['factors'] = factors
        record['minimum_observed_absolute_witness_cosine_by_label'] = witness_margins(samples)
        if set(factors) != expected_labels:
            raise ValueError('missing sampled boundary label')
        factor_cf = mesh.BoundaryCF({label: ng.CF(factor) for label, factor in factors.items()})
        oriented = factor_cf*raw
        record['returned_CF_point_checks'] = {'passed': False, 'samples_checked': 0,
            'maximum_absolute_error': 0., 'maximum_raw_replay_error': 0.,
            'maximum_absolute_error_by_label': {}, 'threshold': API_TOL}
        point_checks = record['returned_CF_point_checks']
        sample_index = 0
        for element in mesh.Elements(ng.BND):
            tr = mesh.GetTrafo(element)
            ir = ng.IntegrationRule(element.type, qorder)
            for index, ip in enumerate(ir):
                mip = tr(ip)
                row = samples[sample_index]
                if row['element'] != int(element.nr) or row['point_in_element'] != index or row['label'] != str(element.mat):
                    raise ArithmeticError('certificate sample enumeration changed')
                actual = np.array(oriented(mip), dtype=np.float64, copy=True)
                raw_now = np.array(raw(mip), dtype=np.float64, copy=True)
                if actual.shape != (2,) or raw_now.shape != (2,) or not np.isfinite(actual).all() or not np.isfinite(raw_now).all():
                    raise ArithmeticError('invalid returned normal coefficient function sample')
                error = float(np.linalg.norm(actual-factors[row['label']]*raw_now))
                raw_error = float(np.linalg.norm(raw_now-np.array(row['raw_normal'])))
                row['returned_normal'] = actual.tolist()
                row['returned_normal_absolute_error'] = error
                row['returned_raw_replay_error'] = raw_error
                point_checks['samples_checked'] += 1
                point_checks['maximum_absolute_error'] = max(point_checks['maximum_absolute_error'], error)
                point_checks['maximum_raw_replay_error'] = max(point_checks['maximum_raw_replay_error'], raw_error)
                label_errors = point_checks['maximum_absolute_error_by_label']
                label_errors[row['label']] = max(label_errors.get(row['label'], 0.), error)
                if error > API_TOL or raw_error > API_TOL:
                    raise ArithmeticError('returned normal coefficient function differs from factor times raw normal')
                sample_index += 1
        if sample_index != len(samples):
            raise ArithmeticError('returned coefficient function did not cover all certificate samples')
        point_checks['passed'] = True
        record['analytic_checks'] = {}
        for label in expected_labels:
            rows = [row for row in samples if row['label'] == label]
            normals = np.array([row['returned_normal'] for row in rows])
            if label in directions:
                error = float(np.max(np.linalg.norm(normals-directions[label], axis=1)))
                record['analytic_checks'][label] = {'maximum_vector_error': error}
                if error > API_TOL:
                    raise ArithmeticError('oriented outer normal disagrees with analytic direction')
            else:
                radial = np.array([row['point'] for row in rows])-np.array(center)
                length = np.linalg.norm(radial, axis=1)
                if np.any(length <= 0):
                    raise ArithmeticError('invalid radial direction')
                dots = np.sum(normals*radial/length[:, None], axis=1)
                record['analytic_checks'][label] = {'radial_dot_range': [float(dots.min()), float(dots.max())],
                                                   'unchanged_maximum_dot_threshold': -.99}
                if dots.max() > -.99:
                    raise ArithmeticError('oriented obstacle normal fails unchanged radial sign check')
        # Use a verified constant-one test to access the actual volume-facet
        # integration path. Never put factor_cf or oriented into this path.
        V = ng.L2(mesh, order=0, dgjumps=True)
        one = ng.GridFunction(V); one.Set(ng.CF(1))
        coefficients = np.array(one.vec.FV().NumPy(), dtype=np.float64, copy=True)
        test = V.TestFunction()
        constant_error = 0.
        for element in mesh.Elements(ng.VOL):
            tr = mesh.GetTrafo(element); ir = ng.IntegrationRule(element.type, 0)
            ip = ir[0]; mip = tr(ip)
            constant_error = max(constant_error, abs(scalar_value(one(mip), 'normal_P0_constant')-1.))
        record['constant_test_error'] = constant_error
        if constant_error > API_TOL:
            raise ArithmeticError('facet integration test is not constant one')
        x = ng.CF((ng.x, ng.y))
        reference_center = (0., 0.) if center is None else tuple(center)
        radial_cf = x-ng.CF(reference_center)
        def terms(n):
            return {'normal_x': n[0], 'normal_y': n[1],
                    'position_dot_normal': ng.InnerProduct(x, n),
                    'radial_dot_normal': ng.InnerProduct(radial_cf, n)}
        ordinary_terms, facet_terms = terms(oriented), terms(raw)
        moments = {}; record['moment_checks'] = moments
        for label in sorted(expected_labels):
            rows = [row for row in samples if row['label'] == label]
            measure = sum(row['weight'] for row in rows)
            coordinate_scale = max(1., max(np.linalg.norm(row['point']) for row in rows),
                                   max(np.linalg.norm(np.array(row['point'])-reference_center) for row in rows))
            normal_sum = np.sum([row['weight']*np.array(row['fluid_outward_normal']) for row in rows], axis=0)
            entry = {'measure': measure, 'sampled_normal_integral': normal_sum.tolist(), 'moments': {}}
            moments[label] = entry
            for name in ordinary_terms:
                ordinary = scalar_value(ng.Integrate(ordinary_terms[name], mesh,
                    definedon=mesh.Boundaries(label), order=qorder), 'oriented_normal_moment')
                form = ng.LinearForm(V)
                form += facet_terms[name]*test*ng.ds(label, skeleton=True, bonus_intorder=qorder)
                form.Assemble()
                facet = float(coefficients@np.array(form.vec.FV().NumPy(), dtype=np.float64, copy=True))
                scale = measure*(1. if name.startswith('normal_') else coordinate_scale)
                error = abs(ordinary-facet)/scale
                entry['moments'][name] = {'ordinary': ordinary, 'volume_facet': facet,
                                          'scale': scale, 'relative_difference': error}
                if not math.isfinite(error) or error > API_TOL:
                    raise ArithmeticError('ordinary oriented normal does not match volume-facet moments')
        volume = scalar_value(ng.Integrate(ng.CF(1), mesh, order=qorder), 'normal_geometry_volume')
        surface_sum = sum(row['moments']['position_dot_normal']['ordinary'] for row in moments.values())
        defect = abs(surface_sum-2*volume)/(2*volume)
        record['divergence_theorem'] = {'boundary_position_dot_normal': surface_sum,
            'twice_volume': 2*volume, 'relative_defect': defect}
        if not math.isfinite(defect) or defect > API_TOL:
            raise ArithmeticError('oriented normal violates the volume divergence theorem')
        groups = {'outer_rectangle': set(directions)}
        if center is not None:
            groups['obstacle'] = {'obstacle'}
        record['closed_components'] = {}
        for name, labels in groups.items():
            total = np.sum([moments[label]['sampled_normal_integral'] for label in labels], axis=0)
            measure = sum(moments[label]['measure'] for label in labels)
            error = float(np.linalg.norm(total))/measure
            record['closed_components'][name] = {'normal_integral': total.tolist(), 'measure': measure,
                                                  'relative_defect': error}
            if not math.isfinite(error) or error > API_TOL:
                raise ArithmeticError('oriented closed boundary normal sum failed')
        record['passed'] = True
        persist()
        return oriented, record
    except Exception as exc:
        record['minimum_observed_absolute_witness_cosine_by_label'] = witness_margins(samples)
        record['error'] = repr(exc)
        persist()
        raise
