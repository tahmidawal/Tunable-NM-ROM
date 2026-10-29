"""Synthetic native-point interface contract, no native geometry claim."""
from pathlib import Path
from types import SimpleNamespace as NS
import json,tempfile
import numpy as np
from fe_adapter import TrainingFEAdapter


def main():
    point=np.array([.15,.2])
    for name,reference,mapped,accepted in [
        ('valid',[.2,.3,0.],point,True),('outside',[-1e-5,.3,0.],point,False),
        ('coordinate',[.2,.3,0.],point+1e-5,False),('nonfinite',[np.nan,.3,0.],point,False)]:
        with tempfile.TemporaryDirectory() as folder:
            calls=[];adapter=TrainingFEAdapter.__new__(TrainingFEAdapter);adapter.D=.1;adapter.evidence_directory=Path(folder)
            def mesh(*xy,**kwargs):
                calls.append((xy,kwargs));return NS(nr=1,pnt=np.array(reference),mapped=mapped)
            adapter.mesh=mesh;adapter.ng=NS(x=0,y=1,CF=lambda expr:lambda mip:mip.mapped)
            try:rows=list(adapter.iter_mapped(points=[point]))
            except ValueError:
                assert not accepted
            else:
                assert accepted and len(rows)==1
                list(adapter.iter_mapped(points=[point]));assert len(calls)==1
            evidence=json.loads((Path(folder)/'probe_lookup.json').read_text())
            assert evidence['records'][0]['status']==('passed' if accepted else 'failed')
            assert calls[0]==(tuple(point),{'tol':1e-12})
    print('PASS exact probe membership/reconstruction/cache; failed and nonfinite counterexamples retained; synthetic only')


if __name__=='__main__':main()
