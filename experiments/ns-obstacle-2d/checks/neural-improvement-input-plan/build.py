"""Training-only metadata plan; never opens a numerical validation payload."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
ROM = BASE.parents[2] / '2026-10-27-ns-obstacle-rom/experiments/ns-obstacle-rom'
PREP = ROM / 'artifacts/fine-preparation01/results/preparation/prepared'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
m = json.loads((PREP/'manifest.json').read_text())
v = json.loads((PREP/'volume/manifest.json').read_text())
expected = [15.,16.25,17.5,18.75,20.,21.25,22.5,23.75,25.]
assert m['split'] == 'training' and m['Re'] == expected
rows = []
for re in expected:
    token = format(re, 'g').replace('.', 'p')
    path = BASE / f'g2/population01/frozen/re{token}/noise.json'
    if re == 20:
        path = BASE / 'g2/population01/evidence/re20-noise.json'
    n = json.loads(path.read_text())
    assert n['split'] == 'train' and n['Re'] == re
    rows.append({'Re': re, 'noise_sha256': sha(path),
                 'conditional_delta': {k: x['delta'] for k,x in n['metrics'].items()}})
fields = ('pod_velocity','pressure','lift_u','lift_p_per_nu','supremizers')
out = {
 'schema':'ns-obstacle-training-only-neural-improvement-input-plan-v1',
 'execution_authorized':False, 'validation_accessed':False,
 'reference_qualification':m['reference_qualification'],
 'reference_admission_sha256':m['admission_evidence']['reference_admission_sha256'],
 'remote_prepared_root':'/cluster/tufts/paralab/tawal01/ns-obstacle-rom-20261027/bank-fit01/inputs/prepared',
 'prepared_manifest_sha256':sha(PREP/'manifest.json'),
 'volume_manifest_sha256':sha(PREP/'volume/manifest.json'),
 'training_manifest_sha256':m['manifest_sha256'],
 'training_Re':expected,
 'metadata_availability_check':{'required_files_present':1274,'missing':[],
     'bytes_stat':1803698087,'payload_hashes_rechecked':False,
     'note':'Remote metadata SHA and file existence/size checked; prior complete fine replay remains the raw acceptance evidence. Recheck payload hashes in released reader.'},
 'nonvolume_files':{k:m['coefficient_files'][k] for k in ('training.npz','coefficients.npz','boundary.npz')},
 'volume_fields':{k:{'columns':v['fields'][k]['columns'], 'blocks':len(v['fields'][k]['blocks']),
     'block_payload_keys':['values','gradients'], 'block_identities_bound_by':'volume_manifest_sha256'} for k in fields},
 'interface':{'reader':'PointVolumeStore.iter_blocks', 'points':v['points'],
     'block_points':v['block_points'],'values':'dimensionless', 'gradients':'d/d(x/D)',
     'velocity_shape':['point',2,'mode'],'velocity_gradient_shape':['point',2,2,'mode'],
     'pressure_shape':['point','mode'], 'physical_coordinates':'network geometry API only; chain-rule gradient scaling must match d/d(x/D)',
     'extra_scaling_for_prepared_fields':False,
     'training_snapshot_keys':['velocity_corrections','velocity_correction_gradients_dimensionless','pressure_corrections','Re'],
     'reuse':'Stream original immutable field blocks, no bulk target export or new native materialization.'},
 'reader_closure_caution':'Existing PointVolumeStore validates and iterates ALL fields, including momentum/continuity tests. Preserve the whole accepted prepared volume tree; the 1274-file stat check above covers requested targets/support only, not a reduced standalone reader closure. Any selective-field reader would need separate source review.',
 'prospective_training':{'first_prefix':4,'accuracy_ceiling':None,
     'scope':'First controlled architecture trial, not a cap on attainable accuracy or later justified rank.',
     'targets':'Actual first four stored POD velocity modes and their gradients, same signs/order.',
     'weights':'ROM owner proposes sqrt(sigma_i/sigma_0), fixed training-only per-mode value/gradient normalization; freeze exact loss before execution.',
     'pressure_lift_supremizers':'Reuse exact accepted FE fields initially, so their approximation does not confound the velocity-network test.'},
 'velocity_singular_values':m['pod']['velocity']['sample_singular_values'],
 'training_uncertainty':rows,
 'conditional_delta_ranges':{k:{'min':min(r['conditional_delta'][k] for r in rows),
     'max':max(r['conditional_delta'][k] for r in rows)} for k in rows[0]['conditional_delta']},
 'uncertainty_interpretation':'Conditional empirical h/p/solver/quadrature estimates, not rigorous truth-error bounds or an imposed accuracy stopping cap. Gradient/pointwise/individual normalized-mode error floors are not certified by these four aggregate metrics.',
 'diagnostics':[
     'Record each mode weighted full-rule value and gradient errors and Grams, and training-snapshot reconstruction using saved actual coefficients; retain raw unweighted and weighted errors.',
     'Check analytic network gradient with a bounded independent derivative control, geometry envelope and exact essential trace; preserve pressure/lift/supremizer identities.',
     'Separate native POD truncation, neural-to-POD target error, best-fit enriched-space representation error, and actual returned weak-solver error. A fit or projection is not a PDE solve.',
     'For actual solve retain coefficients, residual, full Jacobian, objective/gradient histories and original stopping criterion; stationary least squares can have nonzero residual and poor physical accuracy.',
     'Retain physical symmetric traction velocity-gradient and pressure contributions separately on the exact obstacle boundary, plus drag/transverse errors, direct-vs-tensor residual, mass/divergence and trace.',
     'Use existing full-rule accuracy/uncertainty metrics alongside training curves; no extrapolated validation claim. Preserve original failed bank and all previous diagnostic receipts.'
 ],
 'storage':'Small plan only locally; no GPU submission, data copy, payload cooling, or native evaluation. Released job must account for exact streamed working set and actual output retention.',
 'source_sha256':sha(Path(__file__))}
(HERE/'PLAN.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps({'plan_sha256':sha(HERE/'PLAN.json'),'delta_ranges':out['conditional_delta_ranges']}))
