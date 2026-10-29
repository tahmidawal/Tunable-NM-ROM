"""Synthetic scalar metadata selection; verification mocked, never native admission."""
from pathlib import Path
from unittest.mock import patch
import copy,json,tempfile
import numpy as np
import fom_match2d as m
from uncertainty2d import metric_delta,joint_match,METRICS,LIMITS
import uncertainty2d as uncertainty
from pilot2d import DEV_RE,OUTPUTS,save_evidence


def main():
    metric_rows={}
    for key in METRICS:
        d=LIMITS[key]/100
        raw={name:{'numerator':d,'denominator':1.,'value':d} for name in ('h_fine','p_same_geometry','p_combined','solver','quadrature')}
        row={'raw_norms':raw,'h_fine':d,'p':d,'solver':d,'quadrature':d,'delta':4*d,
            'p_definition':'max(p_same_geometry,p_combined)','evidence_sha256':{'fixture':'a'*64},
            'resolved_h':{'earlier':{'difference':4*d,'pair_noise':d/10,'interval':[3.9*d,4.1*d],'resolved':True},
                          'finest':{'difference':d,'pair_noise':d/10,'interval':[.9*d,1.1*d],'resolved':True},'decreasing':True}}
        assert np.isclose(metric_delta(row,key),4*d)
        metric_rows[key]=row
        bad=copy.deepcopy(row);bad['delta']*=2
        try:metric_delta(bad,key)
        except ValueError:pass
        else:raise AssertionError('changed conditional formula accepted')
    deltas={key:LIMITS[key]/100 for key in METRICS}
    assert not joint_match({key:0. for key in METRICS},{key:0. for key in METRICS},deltas)['matched']
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)
        uncertainty_path=root/'uncertainty-fixture.json';uncertainty_review=root/'uncertainty-fixture-review.json'
        artifact={'schema':'ns-obstacle-reference-conditional-uncertainty-v1','split':'development',
            'reference_qualification':{'synthetic':True},'reference_admission_sha256':'a'*64,'volume_rule_sha256':'b'*64,
            'volume_rule_hash_algorithm':uncertainty.RULE_ALGORITHM,'physical_rule_identity':{'points_sha256':'c'*64,
                'weights_sha256':'d'*64,'hash_algorithm':'synthetic-separate-physical-rule','original_container_sha256':'e'*64},
            'rows':[{'Re':re,'metrics':metric_rows} for re in DEV_RE]}
        def write_uncertainty():
            uncertainty_path.write_text(json.dumps(artifact));uncertainty_review.write_text(json.dumps({
                'schema':'ns-obstacle-reference-conditional-uncertainty-review-v1','passed':True,
                'uncertainty_sha256':m.sha_file(uncertainty_path),'reference_admission_sha256':'a'*64,'reference_qualification':{'synthetic':True}}))
        with patch.object(uncertainty,'require_qualification',return_value={'synthetic':True}):
            write_uncertainty();loaded=uncertainty.load(uncertainty_path,m.sha_file(uncertainty_path),uncertainty_review,m.sha_file(uncertainty_review),{'synthetic':True},'a'*64,'b'*64,'development')
            assert set(loaded)==set(DEV_RE)
            artifact['rows'][0]['Re']=20.;write_uncertainty()
            try:uncertainty.load(uncertainty_path,m.sha_file(uncertainty_path),uncertainty_review,m.sha_file(uncertainty_review),{'synthetic':True},'a'*64,'b'*64,'development')
            except ValueError:pass
            else:raise AssertionError('anchor transferred to development')
        files={key:root/(key+'.json') for key in m.INPUTS}
        for path in files.values():path.write_text('{}')
        def desc(path):return {'path':str(path),'sha256':m.sha_file(path)}
        cp={'candidates':[],'runtime_versions':{},'common_rule_points':2,'volume_rule_sha256':'a'*64}
        for name in ('config','mesh','layout','boundary','common_geometry'):
            path=root/(name+'.json');path.write_text('{}')
        cp['common_geometry']=desc(root/'common_geometry.json')
        for cid in ('cheap','accurate'):
            cp['candidates'].append({'id':cid,'field_rule_mode':'native_exact',**{k:desc(root/(k+'.json')) for k in ('config','mesh','layout','boundary')}})
        files['calibration_plan'].write_text(json.dumps(cp))
        errors={'bank_r2':dict(zip(METRICS,(.008,.015,.015,.0015))),
                'bank_r8':dict(zip(METRICS,(.002,.003,.003,.0003))),
                'bank_r4':dict.fromkeys(METRICS,0.)}
        frozen={'models':{name:{'kind':'bank','prefix':int(name[-1]),'fingerprint':'f'*64} for name in errors},'bank_gate':{'passed':True}}
        method='fitted_convection_weak_LS';summary={'records':{}}
        for name,score in errors.items():
            for re in DEV_RE:
                key=name+'_development_'+str(re)
                summary['records'][key]=save_evidence(root,key,{'methods':{method:{'eligible_accuracy':True,'scores':score}}})
        calibration={'blocks':[]}
        for cid in ('cheap','accurate'):
            e={k:v*(.8 if cid=='cheap' else .5) for k,v in errors['bank_r2' if cid=='cheap' else 'bank_r8'].items()}
            for re in DEV_RE:
                for contract in OUTPUTS:
                    block=root/(cid+str(re)+contract);block.mkdir()
                    times=[1. if cid=='cheap' else 2.]*7
                    timing=save_evidence(block,'timing',{'contract':contract,'repetitions':[{'elapsed_seconds':t} for t in times]})
                    control={'passed':True,'diagnostic_volume_gradient_relative_gap':0.,'diagnostic_boundary_velocity_relative_gap':0.,'diagnostic_boundary_gradient_relative_gap':0.,'diagnostic_boundary_pressure_relative_gap':0.}
                    scored=save_evidence(block,'accuracy',{'timing_record':timing,'readout_control':control,
                        'same_invocation_scores':[dict(e,eligible_accuracy=True) for _ in range(7)]})
                    calibration['blocks'].append({'candidate':cid,'Re':re,'contract':contract,'elapsed_seconds':times,'eligible':True,
                        'evidence':desc(block/scored['record_file'])})
        plan={'targets':[{'model':name,'method':method} for name in errors]}
        planpath=root/'plan.json';planpath.write_text('{}');release=root/'release.json';release.write_text('{}')
        fixture=(plan,files,{'synthetic':True},calibration,cp,frozen,summary,{re:deltas for re in DEV_RE})
        with patch.object(m,'verify',return_value=fixture):written=m.select(planpath,release,root/'selections')
        assert len(written)==9
        for name,winner in [('bank_r2','cheap'),('bank_r8','accurate'),('bank_r4',None)]:
            for contract in OUTPUTS:
                row=json.loads((root/'selections'/(name+'__'+method+'__'+contract+'.json')).read_text())
                assert row['selected_candidate']==winner and row['matched_accuracy']==(winner is not None)
                assert row['bank_target']['model']==name
    print('PASS exact delta formula; per-bank cheap/accurate/no-match outcomes; synthetic reviewed-scalar fixture only')


if __name__=='__main__':main()
