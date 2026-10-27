"""Explicit-argument adapters around frozen kernels for the same-allocation frontier."""
from pathlib import Path
import time
import numpy as np
import jax
import jax.numpy as jnp
import c3common as C3
import s3common as S
import qn3
import fomrun as FOM
import psjax
import ps_phases
import fs3
from work_counts import fixed_sweep,quasi_newton

HERE=Path(__file__).resolve().parent
def memory():
    m=jax.devices()[0].memory_stats()
    return dict(current_bytes=m.get('bytes_in_use'),whole_job_cumulative_peak_bytes=m.get('peak_bytes_in_use'),
                interpretation='JAX device allocation counters; peak is cumulative for this entire job, not standalone method memory')

def prepared_field(cx,j):return jnp.asarray(C3.C.initial_interior(65,cx.tab,j))

def bank(cx,cfg,shared):
    name=cfg['setting'];bname='M2' if name=='M2_512' else 'W1024';r=cfg['r']
    if bname not in cx.banks:
        before=memory();t=time.perf_counter();cx.banks[bname]=C3.load_bank(bname)
        cx.tables[bname]=C3.Tables(65,cx.banks[bname],[512] if bname=='M2' else [256,512,768,1024])
        shared[bname]=dict(seconds=time.perf_counter()-t,memory_before=before,memory_after=memory(),bank_sha256=cx.banks[bname]['sha256'])
    before=memory();t=time.perf_counter();tb=cx.tables[bname]
    if name=='W512k8':tb.M_of[512]=4097
    rules=S.load_rules(list(dict.fromkeys([cfg['rule']]+cfg.get('successors',[]))))
    st=S.Setting(dict(name=name,bank=bname,Rp=r,a=3 if bname=='M2' else 6),cx.banks,cx.tables,rules)
    assert st.M==cfg['M'],(name,st.M,cfg['M'])
    d=st.data(cfg['rule']);sp=(cfg['solver'],cfg['scheme'],cfg['dt'],cfg['work_policy'])
    q=S.make_query(sp,st)
    def run(u,nu):return q(u[None],jnp.asarray([nu]),d)
    def phase_init(u):return q.project(u[None],d)
    def phase_evolve(initial,nu):return q.evolve(initial[0],jnp.asarray([nu]),initial[1],d)
    def phase_decode(evolved):return q.decode(evolved['W_all'][::q.keep],d)
    def state(o):return np.asarray(o['W_all'][:,0])
    def output(o):return o['fields'][0]
    def weights(o):return np.asarray(o['W'][:,0])
    def valid(o):return S.run_valid(sp,o,0)
    def work(o):
        rec=S.step_record(sp,o,0)
        if sp[0]=='qn':
            return dict(quasi_newton(np.asarray(o['q'])[:,0],np.asarray(o['n_bt'])[:,0],np.asarray(o['reason'])[:,0],batch_size=1),
                nonlinear_iterations=int(np.asarray(o['q']).sum()),jacobian_contractions=int(np.asarray(o['n_exec']).sum()),raw_step_records=rec)
        a=fs3.a_effective(cfg['scheme'],cfg['dt'],.01,st.a,q.steps)
        return dict(fixed_sweep(np.asarray(o['it'])[:,0],a,cfg['work_policy'],cfg['scheme']),raw_step_records=rec)
    effective=dict(r=r,M=st.M,m=len(rules[cfg['rule']][1]),trust=st.trust,trust_fraction=.05,
                   LMM_startup_steps=int(fs3.SCHEMES[cfg['scheme']][5]))
    if cfg['solver']=='qn':effective.update(q_max=12,g_in=5e-4,slow_threshold=6,refresh=cfg['work_policy'],FS_a_eff=None)
    else:effective.update(K=cfg['work_policy'],gtol=1e-3,lm_budget=50,tol_ad_multiplier=1e-12,
                          FS_base_a=st.a,FS_a_eff=fs3.a_effective(cfg['scheme'],cfg['dt'],.01,st.a,q.steps))
    return dict(id=cfg['id'],kind='bank',config=cfg,run=run,prepare=lambda j:prepared_field(cx,j),
        output=output,weights=weights,state=state,valid=valid,work=work,st=st,data=d,query=q,spec=sp,
        certifier=qn3.make_certify(cfg['scheme'],cfg['dt'],q.steps),adaptive_queries={},
        effective_parameters=effective,
        phase_init=phase_init,phase_evolve=phase_evolve,phase_decode=phase_decode,
        setup_seconds=time.perf_counter()-t,setup_memory_before=before,setup_memory_after=memory(),shared_setup_keys=[bname],
        input_contract=dict(shape=[63**3],points='interior of65-node mesh, x slowest',residency='device'))

def pod(cx,cfg,subjects,shared):
    s=next(s for s in subjects if s['name']==cfg['subject_name'])
    cfg=dict(cfg,scheme=s['scheme'],dt=s['dt'],steps=s['steps'],r=s['r'],production_gtol=float(s['sch']['gtol']),
             production_tolf=float(s['sch']['tolf']),solver='common',max_steps=s['steps'],hyper_reduced=False)
    def run(u,nu):return s['query'](u,nu,s['data'],s['sch'])
    def work(o):
        it=int(np.asarray(o['stats']['it'])[:s['steps']].sum())
        return dict(nonlinear_iterations=it,residual_jacobian_pairs=s['steps']+it,residual_evaluations=4*s['steps']+it,
                    history_F_evaluations=s['steps'],count_kind='logical calls, not GPU kernels',
                    raw_step_records={k:np.asarray(v)[:s['steps']].tolist() for k,v in o['stats'].items()})
    return dict(id=cfg['id'],kind='POD-full-residual',config=cfg,run=run,prepare=lambda j:prepared_field(cx,j),
        output=lambda o:o['fields'],weights=lambda o:np.asarray(o['W']),state=lambda o:np.asarray(o['C']),
        valid=lambda o:bool(np.isfinite(np.asarray(o['C'])).all()),work=work,pod=s,
        phase_init=lambda u:s['initialization_query'](u,s['data']),
        phase_evolve=lambda initial,nu:s['evolution_query'](initial[0],nu,initial[1],s['data'],s['sch']),
        phase_decode=lambda evolved:s['decode_query'](evolved[0],s['data']),
        setup_seconds=0.,setup_memory_before=memory(),setup_memory_after=memory(),shared_setup_keys=['POD'],
        active_metadata=dict(s['metadata'],name=s['name'],scheme=s['scheme'],dt=s['dt'],steps=s['steps'],
                             production_gtol=float(s['sch']['gtol']),production_tolf=float(s['sch']['tolf'])),
        effective_parameters=dict(r=s['r'],trust=s['metadata']['trust_radius'],production_gtol=float(s['sch']['gtol']),
                                  production_tolf=float(s['sch']['tolf']),max_steps=s['steps'],hyper_reduced=False),
        input_contract=dict(shape=[63**3],points='interior of65-node mesh, x slowest',residency='device'))

def fom(cx,cfg,foms):
    sp=tuple(cfg['spec']);before=memory();t=time.perf_counter()
    if sp[0]=='fd':foms.solver(sp[2],FOM.FAM[sp[1]][0])
    def run(u,nu):
        F,info=foms.run(sp,u,nu)
        return dict(fields=F,info=info)
    def work(o):
        if sp[0]=='ps':return dict(time_steps=int(round(.25/sp[2])),nonlinear_evaluations=4*int(round(.25/sp[2])),count_kind='logical ETDRK4 nonlinear calls')
        info=o['info'];k=int(info['stages'])
        return dict(time_stages=k,nonlinear_iterations=int(np.asarray(info['nwt'])[:k].sum()),
                    residual_evaluations=None,JVP_evaluations=None,linear_iterations=None,
                    missing_counter_reason='unchanged BiCGStab kernel does not expose Krylov/JVP call counts',
                    raw_stage_residuals=np.asarray(info['res'])[:k].tolist(),raw_stage_Newton=np.asarray(info['nwt'])[:k].tolist())
    prep=(lambda j:psjax.ic(cx.tab,j,sp[1])) if sp[0]=='ps' else (lambda j:prepared_field(cx,j))
    if sp[0]=='ps':
        ev,dec,Sm=ps_phases.make(sp[1],sp[2])
        phase_evolve=lambda u,nu:ev(u,nu)
        phase_decode=lambda B:dec(B,Sm)
    else:
        adv,scheme=FOM.FAM[sp[1]];ntol,ltol=FOM.TOLS[sp[4]]
        phase_evolve=lambda u,nu:foms.solver(sp[2],adv)(u,nu,sp[3],scheme,ntol,ltol)
        phase_decode=lambda o:foms.lat[sp[2]](o['out'])
    return dict(id=cfg['id'],kind='PS' if sp[0]=='ps' else 'FD-central',config=cfg,run=run,prepare=prep,
        output=lambda o:o['fields'],weights=lambda o:None,state=lambda o:None,
        valid=lambda o:FOM.FOMs.converged(sp,o['info'],o['fields'])[0],work=work,foms=foms,spec=sp,
        phase_init=None,phase_evolve=phase_evolve,phase_decode=phase_decode,
        initialization_status='not applicable: supplied native field/coefficients are already the dynamic state',
        effective_parameters=(dict(N=sp[1],dt=sp[2],ETDRK4_contour_points=32,nonlinear_oversampling_intervals=3*sp[1]//2)
                              if sp[0]=='ps' else dict(intervals=sp[2],dt=sp[3],tolerance_level=sp[4],ntol=FOM.TOLS[sp[4]][0],ltol=FOM.TOLS[sp[4]][1])),
        setup_seconds=time.perf_counter()-t,setup_memory_before=before,setup_memory_after=memory(),shared_setup_keys=[],
        input_contract=dict(shape=[sp[1]-1]*3 if sp[0]=='ps' else [63**3],
                            points='continuous sine coefficients' if sp[0]=='ps' else 'interior of65-node mesh, x slowest',residency='device'))
