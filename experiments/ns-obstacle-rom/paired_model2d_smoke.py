"""Array-only orchestration fixture; no native solve, data admission or timing claim."""
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import tempfile
import numpy as np
import paired_model2d as p
from frozen_rom2d import read_evidence
from fe_adapter import sha_file


def main():
    base=NS(nv=1,np=0);fitted=NS(nv=1,np=0);counts={}
    def query(label):
        counts[label]=counts.get(label,0)+1;n=counts[label]
        record={'output_contract':'fixed_probes','outputs':{'value':np.array([float(n)])},
            'work':{'status':'iteration_cap' if label=='exact' and n==3 else 'stationary'},
            'evidence':{'state':np.array([float(n)])},'query_failed':False}
        if label=='FOM' and n==4:
            record.update(outputs=None,outputs_available=False,query_failed=True)
            exc=RuntimeError('synthetic reconstruction failure');exc.invocation=record;raise exc
        return record
    model=NS(name='fixture',kind='bank',operator=base,stability={'passed':True},
        query=lambda re,contract,op,interp:query('exact' if op is base else 'fitted'))
    fom=NS(query=lambda re:query('FOM'),points=np.zeros((3,2)),weights=np.ones(3),native=NS(X=NS(ndof=3)))
    release={'output_contract':'fixed_probes','reference_population_sha256':'fixture','cores':1}
    with tempfile.TemporaryDirectory() as folder,patch.object(p,'score_rom',side_effect=lambda *args:{'eligible_accuracy':True}),\
        patch.object(p,'score_fom_invocation',side_effect=lambda *args:{'eligible_accuracy':True}),\
        patch.object(p,'quadrature_state_check',return_value={'passed':True}):
        out=Path(folder)
        receipt=p.run_block(model,fitted,fom,{'boundary':{}},{'reference_qualification':'synthetic only'},
                           {'frozen_models_sha256':'fixture'},release,{'Re':16.},out)
        scored=read_evidence(out/receipt['record_file'],receipt['record_sha256'])
        raw=scored['timing_record'];timing=read_evidence(out/raw['record_file'],raw['record_sha256'])
        assert len(timing['actual_invocation_outputs'])==21
        fits=[x for x in scored['same_invocation_accuracy'] if x['method']=='fitted_convection_weak_LS']
        assert len(fits)==7 and sum(x['score']['eligible_accuracy'] for x in fits)==6
        failed=[x for x in timing['actual_invocation_outputs'] if x['invocation'].get('query_failed')]
        assert len(failed)==1 and failed[0]['invocation']['outputs_available'] is False
        assert failed[0]['invocation']['evidence']['state'][0]==4
        assert (out/'BLOCK-FOOTPRINT.json').is_file()
    print('PASS all21 invocation records retained; failed FOM state and failed exact-control fitted rejection; synthetic only')


if __name__=='__main__':main()
