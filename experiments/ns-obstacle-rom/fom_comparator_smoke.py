"""Structural/mock failures only: no native query, real selection or admission."""
import copy,json,os,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from fom_comparator2d import (FrozenFOMComparator,verify_documents,load_native_modules,
                             source_digest,native_digest,sha_file,DEV_RE)
from qualification import GATE,expected_scope
from timing import paired_cpu


def fixture(root,change=None):
    write=lambda name,value:(root/name).write_text(json.dumps(value))
    qualification={'gate':GATE,'passed':True,'qualification_scope':expected_scope()}
    write('truth.json',dict(qualification,fixture_only=True))
    truth=sha_file(root/'truth.json')
    cfg={'kind':'dfg','outlet':'gradient','xmin':0.,'xmax':2.2,'H':.41,'center':[.2,.2],
         'radius':.05,'U':.2,'D':.1,'threads':1,'precision':'float64'}
    write('cfg.json',cfg)
    for name in ('mesh','layout','boundary','geometry','development'):write(name,{})
    descriptor=lambda name:{'path':name,'sha256':sha_file(root/name)}
    from fom_match2d import source_digest as matching_digest
    target={'model':'bank_r2','prefix':2,'method':'fitted_convection_weak_LS',**{k:'b'*64 for k in ('fingerprint','frozen_models_sha256','development_summary_sha256','rom_review_sha256')}}
    selection={'matching_source_sha256':matching_digest(),'bank_target':target,
        'uncertainty_sha256':'c'*64,'uncertainty_review_sha256':'d'*64,'schema':'ns-obstacle-fom-development-selection-v1','fixture_only':True,
        'selection_frozen':True,'development_only':True,'validation_used':False,'matched_accuracy':True,
        'development_Re':list(DEV_RE),'reference_admission_sha256':truth,'reference_qualification':qualification,
        'native_source_sha256':native_digest(),'output_contract':'integrated_forces',
        'common_rule_points':2,'volume_rule_sha256':'a'*64,'field_rule_mode':'certified_requested_points',
        'runtime_versions':{k:'structural-fixture' for k in ('python','ngsolve','numpy','scipy','jax')},
        **{k:descriptor(v) for k,v in [('config','cfg.json'),('mesh','mesh'),('layout','layout'),
         ('boundary','boundary'),('common_geometry','geometry'),('development_evidence','development')]}}
    if change:change(selection)
    write('selection.json',selection);selected=sha_file(root/'selection.json')
    review={'schema':'ns-obstacle-fom-selection-review-v1','fixture_only':True,'passed':True,
        'development_only':True,'matched_accuracy':True,'selection_sha256':selected,
        'reference_admission_sha256':truth,'reference_qualification':qualification}
    write('review.json',review);reviewed=sha_file(root/'review.json')
    eligibility={'fixture_only':True,'role':'frozen_fom_comparator','selection_frozen':True,
        'selection_sha256':selected,'selection_review_sha256':reviewed,'truth_reference_admission_sha256':truth}
    write('eligibility.json',eligibility)
    release={'schema':'ns-obstacle-frozen-fom-runtime-release-v1','authorized_stage':'paired_comparison',
        'fixture_only':True,'bank_target':target,'consumer_source_sha256':source_digest(),'selection_sha256':selected,
        'selection_review_sha256':reviewed,'eligibility_sha256':sha_file(root/'eligibility.json'),
        'reference_admission_sha256':truth,'reference_qualification':qualification,
        'output_contract':'integrated_forces','population':'development','reynolds':list(DEV_RE),'cores':1,'max_common_points':2,
        'reference_population_sha256':'a'*64}
    write('release.json',release)
    return [root/name for name in ('selection.json','review.json','eligibility.json','truth.json','release.json')]


def main():
    assert 'ngsolve' not in sys.modules and 'jax' not in sys.modules
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);verify_documents(*fixture(root))
        for change in (lambda x:x.update(validation_used=True),lambda x:x.update(development_only=False),
                       lambda x:x.update(development_Re=[20.]),lambda x:x.update(matched_accuracy=False),
                       lambda x:x.update(matching_source_sha256='0'*64),lambda x:x['bank_target'].update(model='bank_r4')):
            try:verify_documents(*fixture(root,change))
            except ValueError:pass
            else:raise AssertionError('unfrozen or validation-influenced selector accepted')
    modules=load_native_modules()
    assert 'ngsolve' not in sys.modules and 'jax' not in sys.modules
    digest=modules['candidate_lift'].configuration_sha256;cfg={'U':.2};config_hash=digest(cfg)
    qualification={'gate':GATE,'passed':True,'qualification_scope':expected_scope()}
    release={k:'a'*64 for k in ('selection_sha256','selection_review_sha256','eligibility_sha256',
                                'reference_admission_sha256','consumer_source_sha256','reference_population_sha256')}
    release.update(bank_target={'fixture_only':True},output_contract='integrated_forces',population='development',reynolds=[20.])
    lift=SimpleNamespace(usage_role='frozen_fom_comparator',selection_frozen=True,
        selection_sha256='a'*64,selection_review_sha256='a'*64,eligibility_sha256='a'*64,
        truth_reference_admission_sha256='a'*64,configuration_sha256=config_hash)
    class Failure(RuntimeError):pass
    class Native:
        def __init__(self):self.cfg=cfg;self.lift=lift;self.calls=0
        def query(self,reynolds,contract):
            self.calls+=1;exc=Failure('mock Newton failure')
            exc.invocation={'output_contract':contract,'outputs':None,'outputs_available':False,
                'work':{'status':'query_failed'},'query_failed':True,
                'evidence':{'state':np.array([float(self.calls)]),'history':[self.calls],'converged':False}}
            raise exc
    comparator=FrozenFOMComparator.__new__(FrozenFOMComparator);comparator.native=Native()
    comparator.release=release;comparator.qualification=qualification
    comparator.release_file_sha256='f'*64;comparator.rule_hash='b'*64
    comparator.configuration_digest=digest;comparator.expected_configuration=config_hash
    comparator.binding_digest=digest({'release':release,'qualification':qualification})
    old={k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')}
    try:
        for k in old:os.environ[k]='1'
        result=paired_cpu({'FOM':lambda:comparator.query(20.),'ROM':lambda:{'output_contract':'integrated_forces',
            'outputs':{'force_coefficient':np.zeros(2)},'work':{'status':'stationary'}}},'integrated_forces',1,burn_seconds=0)
    finally:
        for k,v in old.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
    failed=[r['invocation'] for r in result['actual_invocation_outputs'] if r['method']=='FOM']
    assert len(failed)==7 and [r['evidence']['state'][0] for r in failed]==list(range(2,9))
    assert all(r['outputs_available'] is False and r['query_failed'] and 'frozen_comparator' in r for r in failed)
    comparator.native.cfg['U']=.3
    try:comparator.query(20.)
    except ValueError:pass
    else:raise AssertionError('post-selection config mutation accepted')
    assert comparator.native.calls==8
    print('PASS structural selection gates, unchanged source imports, and seven distinct mock partial states; no native/admission/selection execution')


if __name__=='__main__':main()
