"""Metadata-only cluster registration after immutable source freeze.

Uses the exact frozen NumPy parameter generator, without importing JAX or
generating any physical field, truth trajectory or scientific error outcome.
"""
import argparse,ast,datetime,hashlib,json,os
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if args.out.exists():raise RuntimeError('registration output already exists')
    frozen=json.loads((HERE/'FROZEN_PROTOCOL.json').read_text())
    for rel,digest in frozen['files'].items():
        if sha(HERE/rel)!=digest:raise RuntimeError('source changed after freeze: '+rel)
    source=HERE/'vendor/ns3d_fom.py';tree=ast.parse(source.read_text())
    generator=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='parameters')
    env={'np':np};exec(compile(ast.Module(body=[generator],type_ignores=[]),str(source),'exec'),env)
    parameters=env['parameters'];seed=2026112301;count=32
    rows=parameters(seed,count)
    known=[{'role':'training and head-training prefix','seed':202609201,'count':512},
           {'role':'known development prefix','seed':202609202,'count':64},
           {'role':'exposed November19 cohort','seed':2026111901,'count':16}]
    overlaps=[]
    for old in known:
        values=parameters(old['seed'],old['count'])
        exact=set(map(tuple,values));rounded=set(map(tuple,np.round(values,12)))
        overlaps.append({**old,'parameter_sha256':hashlib.sha256(values.tobytes()).hexdigest(),
            'exact_overlap_count':sum(tuple(x) in exact for x in rows),
            'rounded12_overlap_count':sum(tuple(x) in rounded for x in np.round(rows,12))})
    passed=all(x['exact_overlap_count']==0 and x['rounded12_overlap_count']==0 for x in overlaps)
    result={'schema':'untouched-physical-parameter-registration-v1','seed':seed,'count':count,
        'parameters':rows.tolist(),'parameter_names':['center_x','center_y','center_z','vortex_separation','relative_strength','viscosity'],
        'parameter_sha256':hashlib.sha256(rows.tobytes()).hexdigest(),
        'frozen_protocol_sha256':sha(HERE/'FROZEN_PROTOCOL.json'),'generator_sha256':sha(source),
        'source_freeze_commit':frozen['execution_source_commit'],'registration_commit_base':os.environ.get('SOURCE_COMMIT'),
        'registered_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'known_cohort_overlap_checks':overlaps,'nonoverlap_pass':passed,
        'historical_sealed_cohorts_not_opened':[202609203,202609211,202609221],
        'physical_fields_or_truth_generated':False,'method_outcomes_inspected':False,
        'scope':'metadata-only registration on cluster; scientific runs still require GPU preflight'}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if not passed:raise RuntimeError('registered new tuples overlap known exposed/train tuples')
    print(json.dumps({'seed':seed,'count':count,'parameter_sha256':result['parameter_sha256'],'nonoverlap_pass':passed}))

if __name__=='__main__':main()
