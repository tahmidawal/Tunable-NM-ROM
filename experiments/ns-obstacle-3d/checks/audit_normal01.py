"""Recompute diagnostic summaries from saved arrays; no NGSolve execution."""
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


base = Path(__file__).resolve().parents[1]
folder = base/'artifacts/g1-normal01/output'
result = json.loads((folder/'result.json').read_text())
policy = json.loads((folder/'policy.json').read_text())
mesh = json.loads((folder/'mesh_record.json').read_text())
samples = json.loads((folder/'boundary_samples.json').read_text())
with np.load(folder/'volume_rule.npz',allow_pickle=False) as data:
    points = data['points'].copy()
    weights = data['weights'].copy()
    determinants = data['determinants'].copy()
    assert all(a.dtype==np.float64 and np.isfinite(a).all() for a in (points,weights,determinants))

per_boundary = {}
for label in ('inlet','outlet','wall','sphere'):
    rows = [row for row in samples if row['label']==label]
    n = np.array([row['normal'] for row in rows])
    x = np.array([row['point'] for row in rows])
    w = np.array([row['weight'] for row in rows])
    j = np.array([row['boundary_jacobian'] for row in rows])
    cross = np.cross(j[:,:,0],j[:,:,1])
    cross_norm = np.linalg.norm(cross,axis=1)
    cross_unit = cross/cross_norm[:,None]
    ordinary = result['ordinary_BND_moments']['12'][label]
    facet = result['volume_facet_control']['per_boundary'][label]
    per_boundary[label] = {
        'saved_points':len(rows),'unique_faces':len({row['element'] for row in rows}),
        'weight_min':float(w.min()),'weight_max':float(w.max()),'nonpositive_weight_count':int(np.count_nonzero(w<=0)),
        'normal_norm_range':[float(np.linalg.norm(n,axis=1).min()),float(np.linalg.norm(n,axis=1).max())],
        'boundary_cross_product_measure_range':[float(cross_norm.min()),float(cross_norm.max())],
        'raw_normal_vs_cross_product_max_error':float(np.linalg.norm(n-cross_unit,axis=1).max()),
        'raw_normal_dot_toward_adjacent_fluid_range':[min(row['normal_dot_toward_adjacent_volume'] for row in rows),max(row['normal_dot_toward_adjacent_volume'] for row in rows)],
        'recomputed_raw_measure':float(w.sum()),'recomputed_raw_normal_integral':(w@n).tolist(),
        'recomputed_raw_position_dot_normal':float(w@np.sum(x*n,axis=1)),
        'ordinary_order12':ordinary,'native_facet_order12':facet,
        'facet_minus_ordinary_measure':facet['measure']-ordinary['measure'],
        'facet_minus_ordinary_position_dot_normal':facet['position_dot_normal']-ordinary['position_dot_normal'],
        'offset_membership_counts':{str(distance):{
            'samples':sum(1 for row in rows for item in row['offsets'] if item['signed_distance']==distance),
            'mesh_inside':sum(item['mesh_contains'] for row in rows for item in row['offsets'] if item['signed_distance']==distance),
            'analytic_inside':sum(item['analytic_fluid_contains'] for row in rows for item in row['offsets'] if item['signed_distance']==distance)
        } for distance in sorted({item['signed_distance'] for row in rows for item in row['offsets']})}}

volume = result['volume']
ordinary = result['ordinary_BND_moments']['12']
inferred_flipped_sphere_sum = sum(row['position_dot_normal'] for label,row in ordinary.items() if label!='sphere')-ordinary['sphere']['position_dot_normal']
facet = result['volume_facet_control']
summary = {
    'status':'array_audit_only_no_new_native_result','reference_admitted':False,
    'geometry_order':mesh['curve_order'],'elements':mesh['elements'],'vertices':mesh['vertices'],
    'saved_volume_samples':len(points),'saved_boundary_samples':len(samples),
    'signed_volume_determinant_range':[float(determinants.min()),float(determinants.max())],
    'nonpositive_saved_volume_determinant_count':int(np.count_nonzero(determinants<=0)),
    'sampled_volume_weight_range':[float(weights.min()),float(weights.max())],
    'nonpositive_saved_volume_weight_count':int(np.count_nonzero(weights<=0)),
    'sampled_volume_weight_sum':float(weights.sum()),'high_order_volume':volume,
    'volume_sampling_limit':'interior order4 samples only; no signed determinant at native facet or order12 volume points was saved',
    'constant_test_error':facet['constant_test_error'],
    'facet_control_status':facet['status'],'identity_acceptance':False,
    'per_boundary':per_boundary,
    'raw_totals':result['raw_totals'],'ordinary_totals':result['ordinary_BND_totals'],
    'facet_relative_divergence_defect':facet['divergence_theorem_relative_defect'],
    'all_element_relative_divergence_defect':facet['all_element_divergence_theorem_relative_defect'],
    'all_element_minus_exterior_facet_position_moment':facet['all_element_boundaries']['position_dot_normal']-facet['boundary_position_dot_normal'],
    'hypothetical_sphere_only_sign_change':{
        'status':'arithmetic illustration from saved moments; not an executed repair or accepted normal field',
        'boundary_position_dot_normal':inferred_flipped_sphere_sum,
        'relative_divergence_defect':abs(inferred_flipped_sphere_sum-3*volume)/(3*volume)},
    'ordinary_order12_vs16_max_absolute_moment_difference':max(float(np.max(np.abs(np.asarray(result['ordinary_BND_moments']['12'][label][key])-np.asarray(result['ordinary_BND_moments']['16'][label][key])))) for label in ordinary for key in ordinary[label]),
    'rule_provenance':{
        'runtime_saved_samples':{'tetrahedron_requested_order':policy['sample_order'],'points_per_tetrahedron':len(points)//mesh['elements'],'triangle_requested_order':policy['sample_order'],'points_per_triangle':sorted({sum(row['element']==element for row in samples) for element in {row['element'] for row in samples}})},
        'source_inferred_aggregate_rules':{'volume_order':policy['volume_integration_order'],'tetrahedron_points_per_element':(policy['volume_integration_order']//2+1)**3,'ordinary_triangle_orders':policy['integration_orders'],'ordinary_triangle_points_per_face':{'4':6,'12':49,'16':81},'P0_order':0,'facet_bonus_order':policy['facet_bonus_order'],'facet_actual_triangle_order':policy['facet_bonus_order'],'facet_points_per_face':(policy['facet_bonus_order']//2+1)**2},
        'qualification':'aggregate reference nodes/weights were not exported by native forms; selections follow exact pinned source and config, not direct runtime enumeration'},
    'inputs':[{'path':str(path.relative_to(base)),'sha256':sha(path)} for path in (folder/'result.json',folder/'volume_rule.npz',folder/'boundary_samples.json',folder/'mesh_record.json',folder/'policy.json')],
    'generator_sha256':sha(Path(__file__))}
(base/'checks/g1-normal01.ARRAY-AUDIT.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({key:summary[key] for key in ('signed_volume_determinant_range','nonpositive_saved_volume_determinant_count','sampled_volume_weight_range','constant_test_error','facet_relative_divergence_defect','all_element_relative_divergence_defect','hypothetical_sphere_only_sign_change')},indent=2))
lines = ['# Retained 3D geometry diagnostic measurements', '',
         'Generated by `audit_normal01.py` from the retained arrays and JSON. These are diagnostic measurements; the geometry and fluid-boundary identity gates remain failed, and no reference data are admitted.', '',
         '| Quantity | Measured value |','|---|---|']
for key in ('geometry_order','elements','vertices','saved_volume_samples','saved_boundary_samples','signed_volume_determinant_range','nonpositive_saved_volume_determinant_count','sampled_volume_weight_range','nonpositive_saved_volume_weight_count','sampled_volume_weight_sum','high_order_volume','constant_test_error','facet_relative_divergence_defect','all_element_relative_divergence_defect','all_element_minus_exterior_facet_position_moment'):
    lines.append(f'| {key} | `{summary[key]}` |')
lines += ['', 'The positive determinant range covers only the saved interior rule. Facet determinants and the higher-order volume determinants were not saved.', '',
          '| Boundary | Ordinary area | Native facet area | Ordinary position-normal moment | Native facet position-normal moment |',
          '|---|---:|---:|---:|---:|']
for label,row in per_boundary.items():
    a,b=row['ordinary_order12'],row['native_facet_order12']
    lines.append(f"| {label} | {a['measure']:.17g} | {b['measure']:.17g} | {a['position_dot_normal']:.17g} | {b['position_dot_normal']:.17g} |")
lines += ['', 'Arithmetic using the saved ordinary moments, with only the sphere contribution negated, gives a relative divergence defect of '
          f"`{summary['hypothetical_sphere_only_sign_change']['relative_divergence_defect']:.17g}`. This calculation does not construct a valid normal field, change any result, or establish a repair.", '',
          '| Boundary | Signed offset | Samples | Locator says fluid | Analytic recipe says fluid |','|---|---:|---:|---:|---:|']
for label,row in per_boundary.items():
    for distance,counts in row['offset_membership_counts'].items():
        if counts['mesh_inside']!=counts['analytic_inside']:
            lines.append(f"| {label} | {distance} | {counts['samples']} | {counts['mesh_inside']} | {counts['analytic_inside']} |")
lines += ['', 'Only disagreements are shown. The point locator is not an independent validity certificate; topological adjacency and analytic membership remain separate observations.', '',
          '## Terms', '',
          'A signed determinant measures the orientation and local volume scaling of a mapped tetrahedron. A mapped weight includes the integration-rule weight and physical scaling. A native facet is a triangle evaluated through its adjacent volume element. P0 denotes a function constant on each element. The position-normal moment is the surface integral of position dotted with the unit normal. Its divergence defect is the absolute mismatch from three times fluid volume, divided by three times fluid volume. An offset moves along the raw normal; a positive sign follows that normal. A locator asks the mesh whether a point belongs to it; analytic membership tests the prescribed pipe-minus-sphere region directly.', '']
(base/'checks/normal01-source-diagnosis/NUMBERS.generated.md').write_text('\n'.join(lines))
