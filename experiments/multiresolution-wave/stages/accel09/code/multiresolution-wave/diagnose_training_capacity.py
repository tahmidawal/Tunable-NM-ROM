"""Training-only affine-bank capacity diagnostic, not physical rollout error."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ap=argparse.ArgumentParser();ap.add_argument('record',type=Path);ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
p=args.record/'cluster/out/pilot'
with np.load(p/'training_ladder_dirichlet.npz') as f:aa,bb,center,basis,scales,pars=f['a'],f['b'],f['center'],f['basis'],f['case_scales'],f['parameters']
with np.load(p/'fixed_encoder_training.npz') as f:stiffness=f['stiffness']
records=[]
for dim in (32,40,48,56):
    directions=basis[:,:dim];a=(aa-center)@directions@directions.T+center;b=bb@directions@directions.T
    du,dv=a-aa,b-bb
    uerr=np.linalg.norm(du,axis=-1)/scales[:,0,None]
    verr=np.linalg.norm(dv,axis=-1)/scales[:,1,None]
    energy=np.sqrt(np.sum(dv*dv,axis=-1)+pars[:,5,None]**2*np.einsum('cti,ij,ctj->ct',du,stiffness,du))/scales[:,1,None]
    records.append(dict(latent=dim,worst_field_percent=float(uerr.max()*100),worst_velocity_percent=float(verr.max()*100),
        worst_energy_percent=float(energy.max()*100),median_case_energy_percent=float(np.median(energy.max(axis=1))*100)))
output=dict(scope='Only originaltrainingcohort affineprojection insidefrozenbank; outside-bank physicalerrors omitted, no onlineevolution or heldoutclaim.',
            source_sha256={name:hashlib.sha256((p/name).read_bytes()).hexdigest() for name in ('training_ladder_dirichlet.npz','fixed_encoder_training.npz')},records=records)
args.out.write_text(json.dumps(output,indent=2)+'\n');print(json.dumps(output,indent=2))
