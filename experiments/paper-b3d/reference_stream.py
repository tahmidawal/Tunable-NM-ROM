"""Stream refined Burgers references while retaining coarse samples and audit pairs."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import core as c
from run import burn,dump
from audit import stencil


def stream(tab,case,nodes,dt,output_nodes=33,output_dt=.005,final_time=.25,retain_endpoints=False):
    coords=c.b3.grid_coords_3d(nodes);interior=c.b3.interior_indices_3d(nodes)
    initial=c.b3.blob_ic_3d(nodes,tab,case,coords)[interior];nu=float(tab['nu'][case])
    factor=(nodes-1)//(output_nodes-1);stride=int(round(output_dt/dt));steps=int(round(final_time/dt))
    assert factor*(output_nodes-1)==nodes-1 and abs(stride*dt-output_dt)<1e-12 and abs(steps*dt-final_time)<1e-12
    def restriction(u):return np.asarray(u.reshape((nodes-2,)*3)[factor-1::factor,factor-1::factor,factor-1::factor]).ravel()
    fn=c.make_fom(nodes,dt,1);u=jnp.asarray(initial)
    before=time.perf_counter();jax.block_until_ready(fn(u,nu,1e-10,1e-11));compile_seconds=time.perf_counter()-before
    burn();before=time.perf_counter();fields=[restriction(u)];iterations=[];residuals=[];worst=-1.;pair=None
    for step in range(1,steps+1):
        previous=u;answer=fn(u,nu,1e-10,1e-11);jax.block_until_ready(answer)
        u=answer[0][1];it=int(answer[1][0]);rn=float(answer[2][0])
        assert np.isfinite(rn) and rn<2e-9
        iterations.append(it);residuals.append(rn)
        if rn>worst:worst=rn;pair=(np.asarray(previous),np.asarray(u));worst_step=step
        if step%stride==0:fields.append(restriction(u))
    seconds=time.perf_counter()-before
    adv,lap=stencil(pair[1],nodes);independent=float(np.linalg.norm(pair[1]-pair[0]+dt*(adv-nu*lap))/np.linalg.norm(pair[0]))
    assert abs(independent-worst)<1e-12 and independent<2e-9
    arrays=dict(fields=np.asarray(fields),iterations=np.asarray(iterations),residuals=np.asarray(residuals),
        worst_previous=pair[0],worst_current=pair[1])
    if retain_endpoints:arrays.update(native_initial=initial,native_final=np.asarray(u))
    info=dict(nodes=nodes,dt=dt,steps=steps,output_nodes=output_nodes,output_dt=output_dt,nu=nu,seconds=seconds,
        compile_seconds=compile_seconds,worst_step=worst_step,maximum_relative_residual=worst,numpy_worst_step_residual=independent,
        audit_scope='independent NumPy residual at largest recorded step residual; every iteration/residual retained',
        output_contract='exact nodal restriction at every original output time; full fine trajectory not retained')
    return arrays,info


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();cfg=json.loads(Path(a.config).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    assert jax.default_backend()=='gpu'
    row=cfg['validation_rows'][0];raw=c.b3.draw_param_table(cfg['seed'],row+1)
    tab={key:(value[row:row+1] if isinstance(value,np.ndarray) else value) for key,value in raw.items()};tab['m']=1
    tab['s_star']=c.b3.peak_on_reference_grid(tab)
    records=[]
    for nodes,dt in [(65,.0025),(129,.0025),(129,.00125)]:
        arrays,info=stream(tab,0,nodes,dt,retain_endpoints=True)
        name=f'n{nodes}_dt{dt:g}.npz';np.savez_compressed(out/name,**arrays)
        records.append(dict(row=row,parameter_seed=cfg['seed'],artifact=name,**info))
        estimate=32*sum(v['seconds'] for v in records if v['dt']==.0025)+4*sum(v['seconds'] for v in records if v['dt']==.00125)
        dump(out/'profile.json',dict(final_cohort_unopened=True,development_case_profile=True,records=records,
            projected_final_stream_seconds=estimate,estimate_scope='linear extrapolation of one opened development case; excludes final panel, compilation, compression, audit and collection'))
        print('REFERENCE STREAM PROFILE',records[-1],'PROJECTED FINAL SOLVER SECONDS',estimate,flush=True)


if __name__=='__main__':main()
