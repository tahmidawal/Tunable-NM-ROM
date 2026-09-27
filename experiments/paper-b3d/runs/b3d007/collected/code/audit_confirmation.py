"""Independent confirmation membership and streamed-reference audit (NumPy only)."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
from audit import stencil
from audit_contract import family


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('archive');args=parser.parse_args()
    archive=Path(args.archive);out=archive/'out'
    assert json.loads((out/'confirmation-complete.json').read_text())['complete']
    members=[];maximum_initial=0.;seeds=[]
    for seed in (0,1):
        path=out/f'seed{seed}';result=json.loads((path/'result.json').read_text());cfg=result['config']
        assert result['complete'] and result['operator_complete'] and result['final_cohort_unopened']
        assert result['actual_test_modes']==642 and cfg['test_modes']==640
        rows=list(range(cfg['train_trajectories']))+cfg['validation_rows']
        expected=family(cfg['seed'],max(rows)+1);table=np.load(path/'parameters.npz')
        discrepancies={}
        for key,value in expected.items():
            value=value[rows];actual=table[key];delta=float(np.max(np.abs(actual-value)))
            # exp/log can differ by one CPU-library ulp; all direct RNG draws must match exactly.
            if key=='nu':assert np.allclose(actual,value,rtol=8*np.finfo(float).eps,atol=0)
            else:assert np.array_equal(actual,value),(seed,key,delta)
            discrepancies[key]=delta
        assert np.array_equal(table['original_rows'],rows)
        metadata=json.loads((path/'operator_metadata.json').read_text())
        assert metadata['training_rows']==list(range(cfg['train_trajectories']))
        assert metadata['observed_training_steps']==cfg['train_steps']
        assert metadata['validation_rows']==list(range(512,520))
        checkpoint=archive/('training/checkpoint.pkl' if seed==0 else 'training/seed1/checkpoint.pkl')
        ck=pickle.loads(checkpoint.read_bytes())
        assert ck['cfg']['k']==64 and ck['cfg']['r']==256 and ck['Z_tr'].shape==(4096,64)
        assert digest(checkpoint)==result['checkpoint_sha256']
        axis=np.linspace(0,1,cfg['nodes'])[1:-1]
        xyz=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3)
        mask=np.prod(4*xyz*(1-xyz),axis=1)
        for case,row in enumerate(cfg['validation_rows']):
            local=rows.index(row);value=np.zeros(len(xyz))
            for blob in range(int(table['B'][local])):
                distance=np.sum((xyz-table['c'][local,blob])**2,axis=1)
                value+=table['rho'][local,blob]*np.exp(-distance/(2*table['w'][local,blob]**2))
            value*=mask*table['A'][local]/table['s_star'][local]
            truth=np.load(path/f'reference_case{case}.npz')
            error=float(np.max(np.abs(value-truth['u0'])));assert error<2e-14
            maximum_initial=max(maximum_initial,error)
        members.append({key:table[key].copy() for key in expected})
        seeds.append(dict(seed=seed,training_cases=cfg['train_trajectories'],development_cases=len(cfg['validation_rows']),
                          checkpoint_sha256=digest(checkpoint),parameter_discrepancies=discrepancies))
    for key in members[0]:assert np.array_equal(members[0][key],members[1][key])
    assert seeds[0]['checkpoint_sha256']!=seeds[1]['checkpoint_sha256']
    profile=out/'reference-profile';record=json.loads((profile/'profile.json').read_text());references=[]
    for row in record['records']:
        arrays=np.load(profile/row['artifact']);adv,lap=stencil(arrays['worst_current'],row['nodes'])
        residual=arrays['worst_current']-arrays['worst_previous']+row['dt']*(adv-row['nu']*lap)
        defect=float(np.linalg.norm(residual)/np.linalg.norm(arrays['worst_previous']))
        assert defect<2e-9 and abs(defect-row['maximum_relative_residual'])<1e-12
        assert np.argmax(arrays['residuals'])+1==row['worst_step']
        assert len(arrays['iterations'])==row['steps'] and np.isfinite(arrays['fields']).all()
        references.append(dict(nodes=row['nodes'],dt=row['dt'],independent_worst_step_defect=defect,
                               artifact_sha256=digest(profile/row['artifact'])))
    coarse,fine,tight=[np.load(profile/row['artifact'])['fields'] for row in record['records']]
    spatial=np.linalg.norm(coarse-fine,axis=1)/np.linalg.norm(fine[0])
    temporal=np.linalg.norm(tight-fine,axis=1)/np.linalg.norm(fine[0])
    protocol=json.loads((archive/'code/final-reference-protocol.json').read_text())['physical_reference']
    result=dict(passed=True,seeds=seeds,shared_training_and_development_membership=True,
                maximum_initial_field_discrepancy=maximum_initial,streamed_references=references,
                spatial_refinement_maximum=float(spatial.max()),time_refinement_maximum=float(temporal.max()),
                spatial_budget=protocol['spatial_refinement_budget'],time_budget=protocol['time_refinement_budget'],
                spatial_gate_passed=bool(spatial.max()<=protocol['spatial_refinement_budget']),
                time_gate_passed=bool(temporal.max()<=protocol['time_refinement_budget']),
                final_cohort_unopened=True,
                scope='all training/development random membership and opened initial fields; worst recorded native residual pair per streamed reference; refinement differences on saved common nodes')
    (out/'audit-confirmation-local.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
