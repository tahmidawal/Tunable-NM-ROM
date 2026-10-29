"""Frozen fallback control-flow regression; all scientific inputs/solves mocked."""
from pathlib import Path
from unittest.mock import patch,Mock
from types import SimpleNamespace as NS
import json,tempfile
import numpy as np
import validation2d as v
from pilot2d import VALIDATION_RE


def main():
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);inputs={key:root/(key+'.json') for key in ('prepared','preparation_release','bank','probes','frozen_models','rom_review','population')}
        for p in inputs.values():p.write_text('{}')
        inputs['population'].write_text(json.dumps({'reference_admission_sha256':'fixture'}))
        release={'all_development_decisions_frozen':True,'no_matched_timing_or_speedup_claim':True,
            'future_validation_driven_selection_changes_forbidden':True,'model_names':['bank_r8','POD_r2'],
            'reference_population_sha256':v.sha_file(inputs['population'])}
        frozen={'models':{'bank_r8':{'kind':'bank','fingerprint':'fixture'},'POD_r2':{'kind':'POD','fingerprint':'fixture'}},
            'bank_gate':{'passed':False},'choices':{'POD_r2':{},'bank_r8':{}}}
        metadata={'reference_qualification':{'synthetic':True},'admission_evidence':{'reference_admission_sha256':'fixture'}}
        inner={'bank_sha256':v.sha_file(inputs['bank']),'probes_sha256':v.sha_file(inputs['probes'])}
        review={'reference_qualification':{'synthetic':True},'reference_admission_sha256':'fixture','frozen_models_sha256':v.sha_file(inputs['frozen_models'])}
        arrays={'volume':{'points_dimensionless':np.zeros((2,2)),'weights_dimensionless':np.ones(2)},'boundary':{}}
        built=[]
        def rebuild(record,name,*args):built.append(name);return NS(name=name),None
        def evaluate(model,reference,*args):return {'actual_mock_state':np.array([reference['Re']]),'model':model.name,'physical_admission':False}
        loader=Mock(return_value=[{'Re':r,'split':'validation'} for r in VALIDATION_RE])
        with patch.object(v,'verify_review',return_value=review),patch.object(v,'require_qualification',return_value={'synthetic':True}),\
            patch.object(v,'load_preparation',return_value=(arrays,metadata,inner)),patch.object(v,'load_frozen',return_value=frozen),\
            patch.object(v.FrozenBank,'load',return_value=object()),patch.object(v,'_read_arrays',return_value={}),\
            patch.object(v,'load_reference_population',loader),patch.object(v,'rebuild',side_effect=rebuild),patch.object(v,'evaluate_case',side_effect=evaluate):
            result=v.run(inputs,release,root/'pod-control')
            assert built==['POD_r2'] and 'bank_r8' in result['skipped_models'] and len(result['models']['POD_r2']['records'])==4
            assert result['timing_comparison_performed'] is False and result['matched_accuracy_or_speedup_claim'] is False
            loader.reset_mock();release['model_names']=['bank_r8'];built.clear()
            result=v.run(inputs,release,root/'bank-stop');assert not loader.called and not built and not result['validation_opened']
            frozen['bank_gate']['passed']=True
            result=v.run(inputs,release,root/'eligible-bank-no-match');assert built==['bank_r8'] and result['validation_opened']
            assert not result['matched_accuracy_or_speedup_claim']
    print('PASS failed bank skips validation/POD continues; eligible unmatched bank gets accuracy only;4cases retained; synthetic only')


if __name__=='__main__':main()
