"""Expose the two-case spatial/time diagnostic before operator optimization."""
import argparse
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
import core as c
from run import dump,host,reference_audit

p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
cfg=json.loads(Path(a.config).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
assert jax.default_backend()=='gpu'
rows=cfg['validation_rows'];raw=c.b3.draw_param_table(cfg['seed'],max(rows)+1)
cases=cfg.get('refinement_cases',[]);indices=[rows[case] for case in cases]
tab={key:(value[indices] if isinstance(value,np.ndarray) else value) for key,value in raw.items()}
tab['m']=len(cases);tab['s_star']=c.b3.peak_on_reference_grid(tab)
n=cfg['nodes'];dt=cfg['dt'];steps=cfg['steps'];records=[]
for local,case in enumerate(cases):
    row=rows[case];nu=float(tab['nu'][local]);base=None
    for label,nn,dd,ss in [('base',n,dt,steps),('time_half',n,dt/2,steps*2),
                          ('space_fine',2*(n-1)+1,dt,steps),('space_time_fine',2*(n-1)+1,dt/2,steps*2)]:
        xyz=c.b3.grid_coords_3d(nn);idx=c.b3.interior_indices_3d(nn);u0=c.b3.blob_ic_3d(nn,tab,local,xyz)[idx]
        fields,its,rn=host(c.make_fom(nn,dd,ss)(jnp.asarray(u0),nu,1e-10,1e-11))
        defect=reference_audit(fields,nu,nn,dd);assert defect<2e-9 and np.isfinite(fields).all()
        shaped=fields.reshape((ss+1,)+(nn-2,)*3)[::int(round(dt/dd))]
        restricted=shaped if nn==n else shaped[:,1::2,1::2,1::2]
        restricted=restricted.reshape(steps+1,-1)
        if label=='base':base=fields
        artifact=f'{label}_case{case}.npz'
        np.savez_compressed(out/artifact,fields=fields,restricted=restricted,nu=nu,iterations=its,residuals=rn)
        record=dict(case=case,row=row,label=label,nodes=nn,dt=dd,steps=ss,artifact=artifact,
                    numpy_max_relative_residual=defect,**c.metrics(base,restricted))
        records.append(record);dump(out/'result.json',dict(backend=jax.default_backend(),complete=False,records=records))
        print('EARLY REFERENCE',case,label,record['worst_evolved'],flush=True)
dump(out/'result.json',dict(backend=jax.default_backend(),complete=True,records=records,
                          scope='Two opened development cases; empirical refinement discrepancy, not a certified continuum error bound'))
