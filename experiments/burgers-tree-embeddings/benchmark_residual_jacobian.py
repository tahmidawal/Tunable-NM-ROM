"""Paired retained-state derivative/step profiling; not a trajectory accuracy claim."""
import argparse
from common import *
from neural import stable as S,residual_tree as R,residual_weak as W
from neural.build_data import resolve
from training_io import read_checkpoint


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True);ap.add_argument('--model-job',default='b8childprimary')
    ap.add_argument('--trajectory-job',default='b8treepde');ap.add_argument('--repetitions',type=int,default=5)
    a=ap.parse_args();assert a.repetitions>=3
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);ns=Path(os.environ['STUDY_NAMESPACE'])
    result=dict(meta=preflight(),status='preparing',diagnostic=True,full_trajectory_run=False,final_test_opened=False)
    mp=resolve(ns/a.model_job/'output/nonlinear_tree.pkl',ns);mr=json.loads((mp.parent/'result.json').read_text())
    assert digest(mp)==mr['selected_sha256']
    tp=resolve(ns/a.trajectory_job/'output/result.json',ns);tr=json.loads(tp.read_text())
    assert tr['status']=='complete_fixed_tree_diagnostic' and tr['model_sha256']==digest(mp)
    assert (tr['root_k'],tr['child_k'],tr['nodes'],tr['L'],tr['Q'],tr['M'])==(32,16,4,16384,128,512)
    assert tr['dt']==.0005 and tr['theta']==.5 and tr['iterations']==60
    cp=tp.parent/tr['trajectory_codes_file'];assert digest(cp)==tr['trajectory_codes_sha256']
    trajectory=np.load(cp);assert trajectory['codes'].shape==(501,96)
    params=jax.tree.map(jnp.asarray,read_checkpoint(mp)['params']);cells=jnp.asarray(trajectory['cells'])
    cache,assembly=timer(lambda:W.prepare(params,cells,16384,128,512))
    dt=jnp.float64(.0005);nu=jnp.float64(.005);station=jnp.float64(1e-8);update=jnp.float64(1e-10)
    assert tr['station_tolerance']==float(station) and tr['update_tolerance']==float(update)
    mass,adv,previous,residual=W.make_operators(16384,128)
    forward_root=jax.jit(jax.jacfwd(S.apply,argnums=1));reverse_root=jax.jit(S.code_jacobian)
    kernels={
        'root_jacobian':dict(forward=forward_root,reverse=reverse_root),
        'residual_jacobian':{mode:jax.jit(W.make_jacobians(16384,128,root_jacobian=mode)[-1]) for mode in ['forward','reverse']},
        'step':{mode:W.make_step(16384,128,root_jacobian=mode) for mode in ['forward','reverse']}}
    result.update(model_job=a.model_job,model_sha256=digest(mp),trajectory_job=a.trajectory_job,
        trajectory_record_sha256=digest(tp),trajectory_codes_sha256=digest(cp),cache_assembly_seconds=assembly,
        equations=dict(L=16384,Q=128,M=512,dt=.0005,nu=.005,theta=.5,iterations=60,
                       station_tolerance=1e-8,update_tolerance=1e-10),repetitions=a.repetitions,
        conditioning=burn_gpu(),compilation_seconds={},states=[],
        contract='Alternating paired execution, synchronized each call; compilation/assembly separate. No full PDE trajectory or requested4096 output. No runtime superiority claim from microbenchmark.')
    save(out/'result.json',result)
    compiled={}
    for state_index in [0,250,499]:
        z=jnp.asarray(trajectory['codes'][state_index]);old=previous(params,z,cache,dt,nu)
        norm=jnp.maximum(jnp.linalg.norm(W.field(params,z,cache))/128,1e-15)
        args={'root_jacobian':(params['root'],z[:32],cache['xy']),
              'residual_jacobian':(params,z,old,cache,dt,nu,norm),
              'step':(params,z,cache,dt,nu,station,update)}
        row=dict(index=state_index,time=float(trajectory['times'][state_index]),timings={},parity={},conditioning={})
        for name in kernels:
            outputs={};measurements={mode:[] for mode in ['forward','reverse']}
            for mode in ['forward','reverse']:
                if (name,mode) not in compiled:
                    begin=time.perf_counter();compiled[name,mode]=kernels[name][mode].lower(*args[name]).compile()
                    result['compilation_seconds'][name+'_'+mode]=time.perf_counter()-begin
                # Both paths execute and synchronize before timing either path.
                outputs[mode]=compiled[name,mode](*args[name]);jax.block_until_ready(outputs[mode])
            row['conditioning'][name]=burn_gpu()  # After compilation/warmups, immediately before paired timings.
            for repeat in range(a.repetitions):
                order=['forward','reverse'] if repeat%2==0 else ['reverse','forward']
                for mode in order:
                    outputs[mode],elapsed=timer(lambda:compiled[name,mode](*args[name]))
                    measurements[mode].append(elapsed)
            row['timings'][name]=dict(seconds=measurements,
                median_forward=float(np.median(measurements['forward'])),median_reverse=float(np.median(measurements['reverse'])),
                median_ratio_reverse_over_forward=float(np.median(measurements['reverse'])/np.median(measurements['forward'])))
            if name!='step':
                f,r=outputs['forward'],outputs['reverse'];diff=float(jnp.max(abs(f-r)))
                relative=float(jnp.linalg.norm(f-r)/jnp.maximum(jnp.linalg.norm(f),1e-30))
                passed=diff<=1e-9 and relative<=1e-10
                row['parity'][name]=dict(max_absolute=diff,relative=relative,passed=passed)
            else:
                (zf,sf),(zr,sr)=outputs['forward'],outputs['reverse']
                uf,ur=W.field(params,zf,cache),W.field(params,zr,cache)
                field_change=float(jnp.linalg.norm(uf-ur)/jnp.linalg.norm(uf))
                rf=residual(params,zf,old,cache,dt,nu,norm);rr=residual(params,zr,old,cache,dt,nu,norm)
                residual_change=float(jnp.linalg.norm(rf-rr))
                same_status=bool(jnp.array_equal(sf[3:],sr[3:]))
                passed=field_change<=1e-10 and residual_change<=1e-10 and same_status and bool(sf[-1]) and bool(sr[-1])
                row['parity']['step']=dict(relative_field_change=field_change,residual_vector_change=residual_change,
                    forward_status=np.asarray(sf),reverse_status=np.asarray(sr),same_iteration_acceptance_convergence=same_status,passed=passed)
        result['states'].append(row);save(out/'result.json',result);print('profile_state',state_index,row,flush=True)
    passed=all(r['passed'] for row in result['states'] for r in row['parity'].values())
    faster=all(row['timings']['step']['median_ratio_reverse_over_forward']<1 for row in result['states'])
    result.update(status='complete_profile',parity_passed=passed,all_three_step_medians_faster=faster,
        eligible_for_separately_registered_full_trajectory=passed and faster)
    save(out/'result.json',result)

if __name__=='__main__':main()
