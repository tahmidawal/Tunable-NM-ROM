"""Small synthetic integration controls. Never qualify a physical reference/model."""
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
import copy
import weakref
import numpy as np
from bank import initialize
from geometry import CHANNEL
from pilot2d import build_model, galerkin_operator, save_evidence, TRAIN_RE, evaluate_case, run_accuracy_pilot, DEV_RE, VALIDATION_RE
from solver import jacobian_check
from timing import paired_cpu


def main():
    rng = np.random.default_rng(28102026)
    def fields(n):
        a = lambda *shape: rng.normal(size=shape).astype(np.float64)
        return {'points_physical': rng.uniform([.3,.05],[2.,.36],size=(n,2)),
            'weights_dimensionless': np.full(n,1/n),
            'pod_velocity': a(n,2,8), 'pod_velocity_gradient_dimensionless': a(n,2,2,8),
            'supremizers': a(n,2,8), 'supremizers_gradient_dimensionless': a(n,2,2,8),
            'pressure': a(n,8), 'lift_u': a(n,2,1),
            'lift_u_gradient_dimensionless': a(n,2,2,1), 'lift_p_per_nu': a(n,1),
            'momentum_tests': a(n,2,64), 'momentum_tests_gradient_dimensionless': a(n,2,2,64),
            'continuity_tests': a(n,16)}
    volume=fields(96); boundary=fields(16); probes=fields(2)
    boundary['labels']=np.array(['obstacle']*4+['inlet']*4+['wall']*4+['outlet']*4)
    normals=rng.normal(size=(16,2)); normals/=np.linalg.norm(normals,axis=1)[:,None]
    boundary['normals_fluid']=normals
    coefficients=rng.normal(size=(9,8))
    training={'velocity_corrections':np.einsum('niv,sv->sni',volume['pod_velocity'],coefficients),
              'pressure_corrections':np.einsum('np,sp->sn',volume['pressure'],coefficients)}
    arrays={'volume':volume,'boundary':boundary,'training':training}
    metadata={'scales':{'D':.1,'U':.2},'reference_inf_sup':{str(i):{'beta':1e-6} for i in (2,4,8)},
              'manifest_sha256':'synthetic-not-admitted'}
    bank=initialize(CHANNEL,rank=8,width=8,depth=2)
    gaps={}
    for kind in ('bank','POD'):
        model=build_model(kind,2,arrays,metadata,probes,bank)
        class MemoryBlocks:
            point_count=len(volume['points_physical'])
            manifest={'fields':{'pressure':{'columns':8},'pod_velocity':{'columns':8}}}
            def __getitem__(self,key):return volume[key]
            def iter_blocks(self):
                for start in range(0,self.point_count,17):
                    end=min(self.point_count,start+17)
                    yield start,end,{key:value[start:end] for key,value in volume.items()}
        blocked=build_model(kind,2,dict(arrays,volume=MemoryBlocks()),metadata,probes,bank)
        for name,value in vars(model.operator).items():
            if isinstance(value,np.ndarray):np.testing.assert_allclose(value,getattr(blocked.operator,name),rtol=1e-11,atol=1e-12)
        for location in model.fields:
            for name,value in model.fields[location].items():np.testing.assert_allclose(value,blocked.fields[location][name],rtol=1e-13,atol=1e-13)
        exact_fit={'passed':True,'support':list(range(96)),'weights':volume['weights_dimensionless'].tolist()}
        np.testing.assert_allclose(blocked.rule.fitted_convection(blocked.operator,exact_fit).convection,model.operator.convection,rtol=1e-12,atol=1e-12)
        state=rng.normal(size=model.operator.nv+model.operator.np)*.01
        gap=float(np.max(np.abs(np.r_[model.rule.direct(state,.05)]-model.operator.residual(state,.05,False))))
        assert gap<1e-10
        np.testing.assert_allclose(np.r_[blocked.rule.direct(state,.05)],np.r_[model.rule.direct(state,.05)],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(blocked.query(20.,'integrated_forces')['outputs']['force_coefficient'],
            model.query(20.,'integrated_forces')['outputs']['force_coefficient'],rtol=1e-9,atol=1e-10)
        assert jacobian_check(model.operator,state,.05,np.ones(len(state)))['passed']
        f=model.fields['boundary']; ids=boundary['labels']=='obstacle'
        gradient=f['lift_gradient']+f['velocity_gradient']@state[:model.operator.nv]
        pressure=.05*f['pressure_lift']+f['pressure']@state[model.operator.nv:]
        stress=.05*(gradient+gradient.swapaxes(1,2))-pressure[:,None,None]*np.eye(2)
        direct_force=2*np.einsum('n,nij,nj->i',boundary['weights_dimensionless'][ids],stress[ids],-normals[ids])
        assert np.allclose(model.force_output(state,20.),direct_force,rtol=1e-12,atol=1e-12)
        for contract in ('full_velocity_pressure_fields','fixed_probes','integrated_forces'):
            result=model.query(20.,contract)
            assert result['output_contract']==contract and result['evidence']['state'].shape==state.shape
            assert np.isfinite(result['evidence']['state']).all()
            assert result['physical_admission'] is False
        if kind=='POD':
            expected=np.linalg.lstsq(volume['pod_velocity'][...,:2].reshape(-1,2),
                training['velocity_corrections'].reshape(9,-1).T,rcond=None)[0].T
            assert np.allclose(model.interpolator.evaluate(TRAIN_RE[3])[:2],expected[3],atol=1e-12)
            reference=dict(model.reconstruct(state,20.,gradients=True), Re=20.,
                force_coefficient=model.force_output(state,20.),force_definition='physical_symmetric')
            scored=evaluate_case(model,reference,boundary)
            assert set(scored['methods'])=={'exact_tensor_weak_LS','mixed_Galerkin','coefficient_PCHIP'}
            assert all('failure' not in row for row in scored['methods'].values()),scored
            # Even identical reached states cannot qualify a fitted arm when the
            # full-rule solve failed; all numerical scores remain available.
            original_query = model.query
            def controlled_query(*args, **kwargs):
                record = copy.deepcopy(original_query(*args, **kwargs))
                operator = args[2] if len(args)>2 else kwargs.get('operator')
                record['work']['status'] = 'max_iterations' if operator is model.operator else 'stationary'
                return record
            from dataclasses import replace
            fitted = replace(model.operator)
            with patch.object(model, 'query', side_effect=controlled_query), \
                    patch('pilot2d.quadrature_state_check', return_value={'passed': True}):
                controlled = evaluate_case(model, reference, boundary, fitted=fitted)
            arm = controlled['methods']['fitted_convection_weak_LS']
            assert arm['quadrature_reached_state_check']['numerical_checks_passed']
            assert arm['quadrature_reached_state_check']['solver_statuses'] == {'full':'max_iterations','fitted':'stationary'}
            assert not arm['quadrature_reached_state_check']['passed'] and not arm['eligible_accuracy']
            np.testing.assert_allclose(galerkin_operator(blocked).convection,galerkin_operator(model).convection,rtol=1e-12,atol=1e-12)
            op=galerkin_operator(model)
            assert op.residual(state,.05).shape==state.shape
            for index,re in enumerate(TRAIN_RE):
                z=model.interpolator.evaluate(re)
                assert np.array_equal(z[2:4],np.zeros(2))
                reference=model.reconstruct(z,re)
                query=model.query(re,'full_velocity_pressure_fields',interpolate=True)
                assert np.array_equal(reference['velocity'],query['outputs']['velocity'])
        gaps[kind]=gap
    # Full source-level orchestration on explicitly unadmitted synthetic arrays.
    # Only the physical-population gate is bypassed inside this test; production
    # release/load paths are never called or given a synthetic admission manifest.
    live=[]; maximum_live=[0]
    def tracked_build(*args, **kwargs):
        assert sum(ref() is not None for ref in live)==0, 'previous full model retained'
        value=build_model(*args, **kwargs);live.append(weakref.ref(value))
        maximum_live[0]=max(maximum_live[0],sum(ref() is not None for ref in live))
        return value
    def synthetic_refs(reynolds,split):
        return [{'Re':re,'split':split,'reference_admitted':False,
                 'velocity':np.zeros((96,2)),'pressure':np.zeros(96),'velocity_gradient':np.zeros((96,2,2)),
                 'force_coefficient':[1.,0.], 'force_definition':'physical_symmetric'} for re in reynolds]
    with tempfile.TemporaryDirectory() as root:
        out=Path(root)/'pilot';opened=[]
        def validation_loader():
            assert (out/'frozen_development_choices.json').is_file()
            opened.append(True)
            return synthetic_refs(VALIDATION_RE,'validation')
        with patch('pilot2d.build_model',side_effect=tracked_build), \
                patch('pilot2d.check_population',return_value=None):
            summary=run_accuracy_pilot(arrays,metadata,probes,bank,
                synthetic_refs(DEV_RE,'development'),validation_loader,out,fit_quadrature=False)
        assert maximum_live[0]==1 and opened==[True]
        assert not summary['construction_failures'],summary['construction_failures']
        assert summary['model_memory_policy']=='one full field model at a time'
        assert summary['physical_admission'] is False
        assert len(summary['offline_rebuilds'])>=6
    # Mutation regression: actual coefficient evidence survives later reuse of buffers.
    buffer=np.zeros(1); counter=[0]
    def call():
        counter[0]+=1;buffer[0]=counter[0]
        return {'output_contract':'fixed_probes','outputs':{'u':buffer},'work':{'count':counter[0]},
                'evidence':{'state':buffer}}
    timed=paired_cpu({'A':call,'B':call},'fixed_probes',1,burn_seconds=.001)
    for row in timed['actual_invocation_outputs']:
        assert row['invocation']['evidence']['state'][0]==row['invocation']['work']['count']
        assert row['outputs']['u'][0]==row['invocation']['work']['count']
    with tempfile.TemporaryDirectory() as folder:
        info=save_evidence(folder,'timed',timed)
        assert Path(folder,'timed.json').is_file() and Path(folder,'timed.npz').is_file()
    print(json.dumps({'scope':'synthetic array integration only','physical_admission':False,
                      'direct_tensor_max_abs_gap':gaps,'same_invocation_retention':True, 'maximum_simultaneous_full_models':maximum_live[0],
                      'validation_opened_after_choices_frozen':True},indent=2))

if __name__=='__main__':main()
