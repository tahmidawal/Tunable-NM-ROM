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
partial_state=np.zeros(1,dtype=np.float64)
def partial_failure():
 partial_state[0]+=1
 exc=RuntimeError('synthetic partial failure')
 exc.invocation={'output_contract':contract,'outputs':None,'work':{'status':'query_failed'},
                 'evidence':{'state':partial_state,'history':[{'iteration':1}],'converged':False},'state_available':True,'outputs_available':False}
 raise exc
def diagnostic_failure(name,record):raise ValueError('synthetic diagnostic failure')
with tempfile.TemporaryDirectory() as tmp:
 result=paired_queries({'FOM':partial_failure,'synthetic':good},contract,1,diagnostic_failure,Path(tmp))
 states=[r['invocation']['evidence']['state'][0] for r in result['timing']['actual_invocation_outputs'] if r['method']=='FOM']
 assert states==list(range(2,9))
 for row in result['same_invocation_accuracy']:
  assert not row['score']['eligible_accuracy']
  if row['method']=='FOM':assert row['score']['state_or_output_available']
  else:assert row['score']['diagnostic_failed'] and row['score']['actual_invocation_retained']
 assert (Path(tmp)/('paired_'+contract+'.json')).exists()
 print('PASS partial failure snapshots and diagnostic failures retain all actual timed invocations')
