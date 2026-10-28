"""Sub-minute unadmitted array control for bounded final bank-value evaluation."""
import argparse,json
from pathlib import Path
import numpy as np
from bank import initialize
from geometry import CHANNEL
from spaces import ordered_transform

def run():
 rng=np.random.default_rng(7721)
 x=rng.uniform([0.,0.],[2.2,.41],(937,2));x=x[CHANNEL.fluid_mask(x)]
 bank=initialize(CHANNEL);dense,_=bank.evaluate(x)
 w=rng.uniform(.1,2.,len(x));y=rng.normal(size=(9,len(x),2))
 tr,order=ordered_transform(dense,y,w);rows=[]
 for block in (7,64,4096):
  b=bank.values_blocked(x,block);other,o=ordered_transform(b,y,w)
  gap=float(np.max(abs(b-dense)));space=float(np.max(abs(b@other-dense@tr)))
  assert gap<1e-14 and space<1e-9
  rows.append({'block_points':block,'value_max_abs_gap':gap,'ordered_values_max_abs_gap':space})
 return {'synthetic_only':True,'no_training_or_admission':True,'points':len(x),'results':rows}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 a.out.write_text(json.dumps(run(),indent=2)+'\n')
