"""Independent NumPy membership, frozen-source and streamed-reference checks."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from audit import stencil


def family(seed,count):
    rng=np.random.default_rng(seed);rows=[]
    for _ in range(count):
        b=rng.integers(1,4);c=[];w=[];rho=[]
        for _ in range(3):c.append(rng.uniform(.2,.8,3));w.append(rng.uniform(.10,.20));rho.append(rng.uniform(.5,1.))
        amplitude=rng.uniform(.5,2.);viscosity=np.exp(rng.uniform(np.log(.01),np.log(.1)))
        rows.append(dict(B=b,c=c,w=w,rho=rho,A=amplitude,nu=viscosity))
    return {k:np.asarray([r[k] for r in rows]) for k in rows[0]}


def main():
    p=argparse.ArgumentParser();p.add_argument('out');a=p.parse_args();out=Path(a.out)
    freeze=json.loads((out/'freeze.json').read_text());record=json.loads((out/'complete.json').read_text())
    assert record['complete'] and record['freeze_sha256']==hashlib.sha256((out/'freeze.json').read_bytes()).hexdigest()
    assert record['final_cases']==freeze['final_cases'] and record['final_parameter_seed']==freeze['final_parameter_seed']
    assert record['seeds']==[0,1] and record['primary_seed_index']==freeze['primary_seed_index']==0
    expected=family(freeze['final_parameter_seed'],freeze['final_cases']);checks=[]
    for seed in (0,1):
        panel=out/f'seed{seed}';result=json.loads((panel/'result.json').read_text());tab=np.load(panel/'parameters.npz')
        discrepancies={}
        for key,value in expected.items():
            discrepancies[key]=float(np.max(np.abs(tab[key]-value)))
            # Independent CPU exp/log differs by one ulp between machines.
            # Direct random draws remain exact; the derived viscosity uses a fixed ulp allowance.
            if key=='nu':assert np.allclose(tab[key],value,rtol=8*np.finfo(float).eps,atol=0),(seed,key)
            else:assert np.array_equal(tab[key],value),(seed,key)
        assert np.array_equal(tab['parameter_rows'],np.arange(freeze['final_cases']))
        assert np.all(tab['parameter_seeds']==freeze['final_parameter_seed'])
        assert result['config']['freeze_sha256']==record['freeze_sha256'] and result['actual_test_modes']==642
        assert result['checkpoint_sha256']==freeze['checkpoint_sha256'][seed]
        for name in ('bases.npz','directions.npz'):
            assert hashlib.sha256((panel/name).read_bytes()).hexdigest()==freeze['offline_artifact_hashes'][seed][name]
        assert json.loads((out/f'replay-seed{seed}/replay-audit.json').read_text())['passed']
        n=result['config']['nodes'];axis=np.linspace(0,1,n)[1:-1]
        xyz=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),-1).reshape(-1,3);mask=np.prod(4*xyz*(1-xyz),axis=1)
        initial_maximum=0.
        for case in range(freeze['final_cases']):
            value=np.zeros(len(xyz))
            for blob in range(int(tab['B'][case])):
                distance=np.sum((xyz-tab['c'][case,blob])**2,axis=1)
                value+=tab['rho'][case,blob]*np.exp(-distance/(2*tab['w'][case,blob]**2))
            value*=mask*tab['A'][case]/tab['s_star'][case]
            reference=np.load(panel/f'reference_case{case}.npz')
            initial_maximum=max(initial_maximum,float(np.max(np.abs(value-reference['u0']))))
            assert np.array_equal(reference['fields'][0],reference['u0']) and abs(float(reference['nu'])-float(tab['nu'][case]))<1e-15
        assert initial_maximum<2e-14
        checks.append(dict(seed=seed,membership_passed=True,freeze_passed=True,replay_passed=True,
                           parameter_discrepancies=discrepancies,maximum_initial_field_discrepancy=initial_maximum))
    for key in np.load(out/'seed0/parameters.npz').files:
        assert np.array_equal(np.load(out/'seed0/parameters.npz')[key],np.load(out/'seed1/parameters.npz')[key])
    physical=out/'physical-reference';ref=json.loads((physical/'result.json').read_text());maximum=0.;count=0
    assert ref['complete'] and len(ref['comparisons'])==freeze['final_cases']
    assert sorted(v['case'] for v in ref['comparisons'])==list(range(freeze['final_cases']))
    assert ref['protocol']==freeze['physical_reference_protocol']
    pc=ref['protocol']['physical_reference']
    expected_levels={(case,level['nodes'],level['dt']) for case in range(freeze['final_cases']) for level in pc['levels']}
    expected_levels|={(case,pc['time_refinement']['nodes'],pc['time_refinement']['dt']) for case in pc['time_refinement_cases']}
    assert len(ref['records'])==len(expected_levels)
    assert {(v['case'],v['nodes'],v['dt']) for v in ref['records']}==expected_levels
    for row in ref['records']:
        data=np.load(physical/row['artifact']);adv,lap=stencil(data['worst_current'],row['nodes'])
        residual=data['worst_current']-data['worst_previous']+row['dt']*(adv-row['nu']*lap)
        relative=float(np.linalg.norm(residual)/np.linalg.norm(data['worst_previous']))
        assert relative<2e-9 and abs(relative-row['maximum_relative_residual'])<1e-12
        assert len(data['iterations'])==row['steps'] and np.isfinite(data['fields']).all()
        assert data['fields'].shape==(freeze['physical_reference_protocol']['same_grid']['steps']+1,(row['output_nodes']-2)**3)
        maximum=max(maximum,relative);count+=1
    for row in ref['comparisons']:
        case=row['case'];levels=[v for v in ref['records'] if v['case']==case and v['dt']==pc['levels'][0]['dt']]
        levels.sort(key=lambda v:v['nodes']);coarse=np.load(physical/levels[0]['artifact'])['fields'];fine=np.load(physical/levels[-1]['artifact'])['fields']
        discrepancy=np.linalg.norm(coarse-fine,axis=1)/np.linalg.norm(fine[0])
        assert np.max(np.abs(discrepancy-row['spatial_error_by_time']))<1e-12
        assert abs(discrepancy.max()-row['spatial_worst'])<1e-12
        assert row['spatial_passed']==bool(discrepancy.max()<=pc['spatial_refinement_budget'])
        same_grid=np.load(out/f'seed0/reference_case{case}.npz')['fields']
        discretization=np.linalg.norm(same_grid-fine,axis=1)/np.linalg.norm(fine[0])
        assert np.max(np.abs(discretization-row['same_grid_discrepancy_by_time']))<1e-12
        assert abs(discretization.max()-row['same_grid_discrepancy_worst'])<1e-12
        if case in pc['time_refinement_cases']:
            tight=next(v for v in ref['records'] if v['case']==case and v['dt']==pc['time_refinement']['dt'])
            fields=np.load(physical/tight['artifact'])['fields'];error=np.linalg.norm(fields-fine,axis=1)/np.linalg.norm(fine[0])
            assert np.max(np.abs(error-row['time_error_by_time']))<1e-12
            assert abs(error.max()-row['time_worst'])<1e-12
            assert row['time_passed']==bool(error.max()<=pc['time_refinement_budget'])
    assert ref['physical_reference_gate_passed']==all(v['spatial_passed'] and v.get('time_passed',True) for v in ref['comparisons'])
    answer=dict(passed=True,seeds=checks,reference_pairs_checked=count,maximum_independent_reference_residual=maximum,
        empirical_refinement_passed=ref['physical_reference_gate_passed'],
        audit_scope='all final membership and frozen hash links; independently recomputed largest-residual fine pair per reference and every saved refinement discrepancy')
    (out/'audit-contract-local.json').write_text(json.dumps(answer,indent=2)+'\n');print(json.dumps(answer,indent=2))


if __name__=='__main__':main()
