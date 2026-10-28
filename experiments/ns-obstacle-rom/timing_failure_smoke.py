"""Synthetic callable failure retention only; no FOM/ROM scientific result."""
from pathlib import Path
import tempfile
import numpy as np
from pilot2d import paired_queries
contract='integrated_forces';count=0;scores=[]
def failed():raise RuntimeError('synthetic callable failure')
def good():
 global count
 count+=1
 return {'output_contract':contract,'outputs':{'force_coefficient':np.array([count,0.],dtype=np.float64)},
         'work':{'status':'synthetic'},'evidence':{'state':np.array([count],dtype=np.float64)}}
def score(name,record):
 scores.append(name)
 assert record['outputs']['force_coefficient'][0]==record['evidence']['state'][0]
 return {'synthetic_only':True}
with tempfile.TemporaryDirectory() as tmp:
 result=paired_queries({'FOM':failed,'synthetic':good},contract,1,score,Path(tmp))
 assert len(result['timing']['repetitions'])==14 and len(scores)==7
 failed_rows=[r for r in result['same_invocation_accuracy'] if r['method']=='FOM']
 assert len(failed_rows)==7 and all(r['score']['query_failed'] and not r['score']['eligible_accuracy'] for r in failed_rows)
 assert result['timing']['warmup_attempts'][0]['query_failed']
 assert (Path(tmp)/('paired_'+contract+'.json')).exists()
 print('PASS synthetic failed warmup and seven failed invocations retained; no fabricated state/accuracy')
