"""Read-only real Re20 interface check: no admission, population synthesis or native work."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import uncertainty2d as uncertainty
from pilot2d import array_digest

NOISE_SHA='53ef9d40df6604d4d0eb0c4c262e05648940341f3ed2b921cf2e44804ee94cf9'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def physical_digest(value):
    a=np.ascontiguousarray(value)
    h=hashlib.sha256(json.dumps({'dtype':a.dtype.str,'shape':list(a.shape)},sort_keys=True).encode())
    h.update(memoryview(a).cast('B'));return h.hexdigest()


def check(root):
    noise_path=root/'g2/population01/evidence/re20-noise.json'
    assert sha(noise_path)==NOISE_SHA
    noise=json.loads(noise_path.read_text())
    assert noise['Re']==20 and noise['split']=='train' and noise['admission'] is False
    assert set(noise['metrics'])==set(uncertainty.METRICS)
    deltas={key:uncertainty.metric_delta(noise['metrics'][key],key) for key in uncertainty.METRICS}
    common=root/'artifacts/g2-population-re20-02/results/common'
    assert sha(common/'readout.json')==noise['common_readout_sha256']
    readout=json.loads((common/'readout.json').read_text())
    original=root/readout['complete_common_rule']['source']
    identity=noise['physical_rule_identity']
    assert set(identity)=={'points_sha256','weights_sha256','hash_algorithm','original_container_sha256'}
    assert identity['hash_algorithm']=='sha256_json_dtype_shape_and_contiguous_bytes_v1'
    assert sha(original)==identity['original_container_sha256']==readout['complete_common_rule']['sha256']
    with np.load(common/'common-rule.npz',allow_pickle=False) as z, np.load(original,allow_pickle=False) as old:
        arrays={k:z[k] for k in ('points','weights')}
        for key,a in arrays.items():
            b=old[key]
            assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
            assert physical_digest(a)==identity[key+'_sha256']==noise['rule_content_sha256'][key]
        rule_sha=array_digest({'points_dimensionless':arrays['points']/.1,'weights_dimensionless':arrays['weights']/.1**2})
        assert rule_sha==noise['volume_rule_sha256'] and noise['volume_rule_hash_algorithm']==uncertainty.RULE_ALGORITHM
        point_count=len(arrays['weights'])
    candidate_path=root/'g2/population01/evidence/QUALIFICATION-CANDIDATE.json'
    candidate=json.loads(candidate_path.read_text())
    adopted=next(row for row in candidate['rows'] if row['Re']==20)['adopted_reference'];folder=root/adopted['directory']
    bindings={key:sha(folder/name) for key,name in [('export_manifest_sha256','EXPORT-MANIFEST.json'),('config_sha256','config.json'),('mesh_sha256','mesh.vol.bin'),('solution_sha256','solution.sol')]}
    bindings['layout_sha256']=json.loads((folder/'fe_layout.json').read_text())['sha256']
    assert all(adopted[k]==v for k,v in bindings.items())
    assert adopted['storage_mode']=='expanded_full_fields' and original==folder/'full_rule.npz'
    assert json.loads((folder/'config.json').read_text())['release']=='G2_steady_control_candidate'
    return {'schema':'ns-obstacle-real-re20-handoff-interface-check-v1','passed':True,'physical_admission':False,
        'scope':'actual Re20 only; metric validator and exact rule/adopted identities, not population admission',
        'full_population_loader_invoked':False,'noise_sha256':NOISE_SHA,'candidate_sha256':sha(candidate_path),
        'checker_sha256':sha(__file__),'uncertainty_source_sha256':sha(Path(uncertainty.__file__)),
        'point_count':point_count,'validated_delta':deltas,'volume_rule_sha256':rule_sha,'physical_rule_identity':identity,
        'common_rule_container_sha256':sha(common/'common-rule.npz'),'adopted_reference_verified':bindings,
        'remaining':'fourteen actual per-parameter qualifications, complete genuine admission/review and released materialization'}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--reference-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    result=check(a.reference_root.resolve());a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
