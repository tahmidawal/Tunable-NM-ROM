"""Lossless repeated-buffer evidence control; no solve or physical admission."""
from pathlib import Path
import json,tempfile
import numpy as np
from pilot2d import save_evidence
x=np.arange(12,dtype=np.float64).reshape(3,4)
original={'first':x,'separate_identical_buffer':x.copy(),'alias':x,'transpose':x.T,
          'other_dtype':x.astype(np.float32),'positive_zero':np.array([0.]),
          'negative_zero':np.array([-0.]),'nan':np.array([np.nan]),'scalar':np.array(7.),'empty':np.empty((0,2))}
original['repeated_nan']=original['nan'].copy()
with tempfile.TemporaryDirectory() as tmp:
 save_evidence(tmp,'record',original)
 wrapper=json.loads((Path(tmp)/'record.json').read_text())
 with np.load(Path(tmp)/'record.npz',allow_pickle=False) as archive:
  assert len(archive.files)==8
  for key,value in original.items():
   encoded=wrapper['record'][key];actual=archive[encoded['array']]
   expected=np.array(value,copy=True,order='C')
   assert actual.shape==expected.shape and actual.dtype==expected.dtype
   assert actual.tobytes()==expected.tobytes()
 assert wrapper['record']['first']['array']==wrapper['record']['separate_identical_buffer']['array']
 assert wrapper['record']['positive_zero']['array']!=wrapper['record']['negative_zero']['array']
 print(json.dumps(wrapper['array_retention'],indent=2))
