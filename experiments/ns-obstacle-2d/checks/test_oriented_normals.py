"""Replay saved geometry evidence; no NGSolve import or new FE execution."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from oriented_normals import infer_label_factors

sample_path = ROOT/'artifacts/g1-normal01/results/geometry/boundary_samples.json'
result_path = ROOT/'artifacts/g1-normal01/results/geometry/result.json'
native = json.loads(sample_path.read_text())
samples = [{'point': row['point'], 'raw_normal': row['normal'], 'label': row['label'],
            'adjacent_volume_point': row['adjacent_volume_point']} for row in native]
factors = infer_label_factors(samples)
assert factors == {'wall_bottom': 1, 'outlet': 1, 'wall_top': 1, 'inlet': 1, 'obstacle': -1}
negated = copy.deepcopy(samples)
for row in negated:
    row['raw_normal'] = (-np.array(row['raw_normal'])).tolist()
opposite = infer_label_factors(negated)
assert opposite == {label: -factor for label, factor in factors.items()}
assert all(np.array_equal(a['fluid_outward_normal'], b['fluid_outward_normal']) for a,b in zip(samples,negated))
rejections = []
for name in ('mixed_orientation', 'nonunit_normal', 'tangent_witness'):
    altered = copy.deepcopy(samples)
    if name == 'mixed_orientation':
        i = next(i for i,row in enumerate(altered) if row['label'] == 'obstacle')
        altered[i]['raw_normal'] = (-np.array(altered[i]['raw_normal'])).tolist()
    elif name == 'nonunit_normal':
        altered[0]['raw_normal'] = (2*np.array(altered[0]['raw_normal'])).tolist()
    else:
        nx,ny = altered[0]['raw_normal']
        altered[0]['adjacent_volume_point'] = (np.array(altered[0]['point'])+np.array([-ny,nx])).tolist()
    try:
        infer_label_factors(altered)
    except ValueError as exc:
        rejections.append({'name': name, 'rejected': True, 'message': str(exc)})
    else:
        raise AssertionError(name+' was not rejected')
r = json.loads(result_path.read_text())
moment_errors = []
for label,factor in factors.items():
    raw = r['raw_boundary_moments'][label]
    facet = r['volume_facet_control']['per_boundary'][label]
    for quantity in ('position_dot_normal', 'radial_dot_normal'):
        moment_errors.append(abs(factor*raw[quantity]-facet[quantity])/raw['measure'])
    moment_errors.append(float(np.linalg.norm(factor*np.array(raw['normal_integral'])-
        np.array([facet['normal_x'],facet['normal_y']])))/raw['measure'])
assert max(moment_errors) < 1e-10
print(json.dumps({'status': 'passed_saved_array_replay_only', 'native_repair_execution': False,
    'sample_source_sha256': hashlib.sha256(sample_path.read_bytes()).hexdigest(),
    'result_source_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(),
    'helper_source_sha256': hashlib.sha256((ROOT/'oriented_normals.py').read_bytes()).hexdigest(),
    'factors': factors, 'raw_sign_reversal_invariance': True, 'rejections': rejections,
    'maximum_replayed_corrected_moment_error_per_measure': max(moment_errors)}, indent=2))
