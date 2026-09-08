"""Actual displacement snapshots from the audited larger-head wave run."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('input_directory',type=Path)
    ap.add_argument('--out-directory',type=Path,required=True)
    args=ap.parse_args(); native=args.input_directory; out=args.out_directory
    raw=native/'result.json'; data=json.loads(raw.read_text()); cfg=data['config']
    assert data['complete'] and not data['final_test_opened']
    n=max(cfg['meshes']); case=cfg['validation_indices'][-1]
    method=f"new_mlp32_seed{cfg['training']['optimizer_seeds'][0]}"
    dt=max(cfg['nonlinear_dts']); frames=np.rint(np.linspace(0,cfg['query_observations']-1,7)).astype(int)
    provenance=dict(source_result_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),job_id=data['provenance']['job_id'],
        source_commit=data['provenance']['source_commit'],solver_intervals=n,display_intervals=cfg['comparison_intervals'],
        case=case,method=method,dt=dt,frames=frames.tolist(),sources={},figures=[],
        selection='Same last declared development case used in earlier still plots; first predeclared optimizer seed and largest repeated timestep. Both seeds and all cases remain in numeric tables.',
        limitation='Displacement only. Fixed color scales can hide late absorbing relative errors; velocity and energy-state metrics are retained in the report.')
    for bc in cfg['boundaries']:
        ref=native/f'reference_{bc}_{case}.npz'
        selected=[next(r for r in data['invocations'] if (r['boundary'],r['intervals'],r['case'],r['method'],r['setting'],r['repetition'])==(bc,n,case,name,step,0))
            for name,step in [('frozen_mlp16_seed691200',cfg['primary_dt']),(method,dt)]]
        files=[ref]+[native/(r['invocation_id']+'.npz') for r in selected]
        fields=[]
        for p in files:
            with np.load(p) as f:a=f['u']
            if bc=='dirichlet':a=np.pad(a,((0,0),(1,1),(1,1)))
            assert a.shape==(cfg['query_observations'],cfg['comparison_intervals']+1,cfg['comparison_intervals']+1)
            assert np.isfinite(a).all()
            fields.append(a);provenance['sources'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        truth,old,new=fields;error=np.abs(new-truth)
        amplitude=max(float(np.max(np.abs(a))) for a in fields); error_max=float(np.max(error))
        fig,axes=plt.subplots(4,len(frames),figsize=(16,9),sharex=True,sharey=True,layout='constrained')
        pad=.5/cfg['comparison_intervals']
        for col,frame in enumerate(frames):
            for row,values in enumerate([truth,old,new,error]):
                kwargs=dict(cmap='RdBu_r',vmin=-amplitude,vmax=amplitude) if row<3 else dict(cmap='magma',vmin=0,vmax=error_max)
                artist=axes[row,col].imshow(values[frame].T,origin='lower',extent=(-pad,1+pad,-pad,1+pad),interpolation='nearest',**kwargs)
                axes[row,col].set(xlim=(0,1),ylim=(0,1),xticks=[0,.5,1],yticks=[0,.5,1])
                if row==0:axes[row,col].set_title(f't = {frame*cfg["observation_dt"]:g}');field_artist=artist
                if row==3:axes[row,col].set_xlabel('x');error_artist=artist
        labels=['Reference u','Earlier MLP16 u',f'MLP32 u\nseed {cfg["training"]["optimizer_seeds"][0]}','|MLP32 − reference|']
        for row,label in enumerate(labels):axes[row,0].set_ylabel(label+'\ny')
        fig.colorbar(field_artist,ax=axes[:3,:],shrink=.7,label='Displacement; fixed scale')
        fig.colorbar(error_artist,ax=axes[3,:],shrink=.8,label='Absolute error; fixed scale')
        title='Reflective' if bc=='dirichlet' else 'Absorbing'
        fig.suptitle(f'{title} wave evolving — development case {case}\n'
            f'Solve: {n} intervals; display: {cfg["comparison_intervals"]}; MLP16 dt={cfg["primary_dt"]:g}, MLP32 dt={dt:g}\n'
            'Displacement snapshots do not replace late-time velocity and energy-error checks.',fontsize=12)
        stem=out/f'{bc}-wave-k32-evolution-case{case}'
        for suffix in ['.png','.pdf']:
            p=stem.with_suffix(suffix);fig.savefig(p,dpi=180);provenance['figures'].append(dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        plt.close(fig)
    (out/'wave-k32-evolution-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__=='__main__':main()
