"""Generate native producer/reload observations from retained measured artifacts."""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def summarize(root):
    case=root/'results/dfg'
    load=lambda name:json.loads((case/name).read_text())
    result=load('result.json');reload=load('reload_checks.json');trace=load('essential_trace_certificate.json')
    export=load('EXPORT-MANIFEST.json')
    for name,digest in export['files'].items():
        if sha(case/name)!=digest:raise ValueError('export artifact checksum mismatch: '+name)
    with np.load(case/'full_rule.npz',allow_pickle=False) as data:
        arrays={name:{'shape':list(data[name].shape),'dtype':str(data[name].dtype),'bytes':data[name].nbytes} for name in data.files}
        weight_sum=float(data['weights'].sum());signed_det=float(np.linalg.det(data['jacobians']).min())
    normal={}
    for name in ['normal_orientation.json','reload_normal_orientation.json']:
        cert=load(name)
        normal[name]={'sha256':sha(case/name),'passed':cert['passed'],'factors':cert['factors'],
            'returned_CF_point_checks':cert['returned_CF_point_checks'],
            'minimum_observed_absolute_witness_cosine_by_label':cert['minimum_observed_absolute_witness_cosine_by_label'],
            'divergence_theorem':cert['divergence_theorem']}
    return {'scope':'one coarse native producer plus separate fresh-process binary reload',
        'physical_admission':False,'training_released':False,
        'native_summary_passed':json.loads((root/'results/native_summary.json').read_text())['passed'],
        'source_commit':(root/'SOURCE_COMMIT.txt').read_text().strip(),
        'result_sha256':sha(case/'result.json'),'export_manifest_sha256':sha(case/'EXPORT-MANIFEST.json'),
        'all_export_manifest_hashes_verified':True,'mesh_identity':load('mesh_identity.json'),
        'mesh_bytes':(case/'mesh.vol.bin').stat().st_size,'fresh_reload_execution':reload['execution'],
        'reload_passed':reload['passed'],'reload_check_count':len(reload['checks']),
        'maximum_relative_reload_error':max(x['value'] for x in reload['checks'].values()),
        'reload_checks':reload['checks'],'prescribed_FE_trace_bound_over_U':trace['upper_bound_over_U'],
        'affine_gradient_orientation':load('affine_gradient_orientation.json'),
        'normal_certificates':normal,'full_rule_array_layout':arrays,'full_rule_weight_sum':weight_sum,
        'minimum_saved_signed_jacobian_determinant':signed_det,
        'analytic_obstacle_quadrature':result['geometry']['analytic_obstacle_quadrature'],
        'element_count':result['geometry']['element_count'],'mixed_dofs':result['geometry']['ndof'],
        'reported_producer_peak_rss_bytes':result['peak_rss_bytes'],
        'reported_reload_peak_rss_bytes':reload['peak_rss_bytes'],
        'limitations':['Fresh-process parity is measured on this newly generated coarse mesh, not the old rounded text artifact.',
            'This verifies the producer interface; coarse physical DFG errors and all G2 refinement/domain gates remain.',
            'No shared-lift, native ROM adapter or training admission is implied.',
            'Per-process RSS values are reported separately; no simultaneous or timing advantage claim.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();args.out.write_text(json.dumps(summarize(args.root),indent=2,allow_nan=False)+'\n')
