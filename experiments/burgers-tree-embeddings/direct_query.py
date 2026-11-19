"""Complete fixed-tree query boundaries for a later registered cost diagnostic.

Construction is reusable setup. The returned query starts with the supplied
initial-field callable and ends with actual device output fields. No reference
solution or saved development embedding is an input to this implementation.
"""
from common import *
import data, local_polynomial as LP, discrete_weak as W, packed_fields as PF
import cell_catalogue as CC, compact_fields as CF
from fixed_tree_pilot import visibility


def local_work_shape(cache,cells,kernel):
    q=cache[kernel]
    if kernel=='packed':
        entries=int(q['X'].shape[0]*q['X'].shape[1]*q['Y'].shape[1]*len(cells))
        return dict(local_field_entries=entries,unused_slot_entries=0,
            interpretation='full-domain local arithmetic; support zeros remain charged')
    entries=CF.patch_entries(q);unused=0;slots=[]
    for d,b in enumerate(q['buckets'],1):
        count=b['X'].shape[1];active=sum(c[0]==d for c in cells)
        per=int(b['X'].shape[0]*b['X'].shape[2]*b['Y'].shape[2])
        unused+=(count-active)*per;slots.append(dict(depth=d,allocated=count,active=active))
    return dict(local_field_entries=entries,unused_slot_entries=unused,slots=slots,
        interpretation='compact patch arithmetic including unused slots and boundary padding; not a FLOP count')


class FixedQuery:
    def __init__(self,basis,rotation,L,Q,tests,dt,*,catalogue=None,nodes=52,
                 depth=4,output_points=4096,reuse_normal=False):
        self.basis,self.rotation=basis,rotation
        self.L,self.Q,self.tests,self.dt=L,Q,tests,dt
        self.catalogue,self.nodes,self.depth=catalogue,nodes,depth
        self.output_points,self.reuse_normal=output_points,reuse_normal
        self.field_kernel=catalogue.get('field_kernel','packed') if catalogue is not None else 'packed'
        field=PF if self.field_kernel=='packed' else CF
        assert np.isclose(round(.05/dt)*dt,.05) and L%Q==0
        self.x,self.w=data.grid(128,True)
        self.axis=jnp.arange(1,Q)/Q
        self.xy=jnp.stack(jnp.meshgrid(self.axis,self.axis,indexing='ij'),axis=-1)
        self.fit=W.make_initial_fit(field.center)
        self.step,_=W.make_solver(field.center,L,theta=.5,advection_tests=field.tested_advection(L,Q),
                                  reuse_linear_normal=reuse_normal)
        self.output=jax.jit(field.center)
        if catalogue is None:
            weak,_=W.prepare(basis,[],[],rotation,L,tests,jnp.eye(len(basis)))
            self.cache={key:weak[key] for key in ['A','indices','lam','gram']}
            self.cache['packed']=PF.prepare_stencil(basis,[],[],rotation,L,Q)
            axis=np.linspace(0,1,output_points)[None]
            self.output_cache={'packed':PF.prepare(basis,[],[],rotation,axis,axis)}
            self.visibility=visibility(self.cache,tests)
            assert self.visibility['minimum']>=1e-6
            if reuse_normal:self.cache['linear_normal']=self.cache['A'].T@self.cache['A']
            jax.block_until_ready((self.cache,self.output_cache))
        else:
            assert (catalogue['L'],catalogue['Q'],catalogue['tests'],catalogue['output_points'])==(L,Q,tests,output_points)
            assert catalogue['root']==len(basis) and catalogue['rank']==rotation.shape[1]

    def __call__(self,initial,outputs=1,components=True):
        assert outputs in (1,5)
        start=time.perf_counter() if components else 0.
        if self.catalogue is None:
            ca,output_cache=self.cache,self.output_cache
            cells=[];vis=self.visibility
        else:
            cells=LP.callable_tree(self.x,self.w,initial,self.nodes,self.depth,'swept')
            selected=CC.select(self.catalogue,cells)
            ca,output_cache=selected['cache'],selected['output_cache']
            vis=visibility(ca,self.tests)
            if vis['minimum']<1e-6:
                return dict(complete=False,failure='insufficient tested visibility',
                            total_query_seconds=time.perf_counter()-start if components else None,visibility=vis)
            if self.reuse_normal:
                ca=dict(ca,linear_normal=ca['A'].T@ca['A']);jax.block_until_ready(ca['linear_normal'])
        setup=time.perf_counter()-start if components else None
        tfit=time.perf_counter() if components else 0.;u0=initial(self.xy)
        c,fit_error,linear_residual=self.fit(u0,ca);jax.block_until_ready((c,fit_error,linear_residual))
        # The fixed initial norm is evaluated once; its value is the same as
        # evaluating it again inside every step's runaway guard.
        initial_norm=float(jnp.linalg.norm(u0)/self.Q)
        cold=time.perf_counter()-tfit if components else None
        states=[c];statuses=[];complete=True;failure=None
        tsolve=time.perf_counter() if components else 0.
        for it in range(1,round(.25/self.dt)+1):
            c,status=self.step(c,ca,self.dt,data.NU,1e-8,1e-10)
            jax.block_until_ready((c,status));s=np.asarray(status);statuses.append(s)
            converged=np.all(np.isfinite(s)) and s[1]<=1e-8 and (s[2]<=1e-10 or s[1]<=1e-10)
            finite=bool(jnp.all(jnp.isfinite(c)))
            bounded=float(jnp.sqrt(jnp.maximum(c@ca['gram']@c,0)))<=10*initial_norm
            if not (converged and finite and bounded):
                complete=False;failure='returned step convergence or finite-field guard';break
            if it%round(.05/self.dt)==0:states.append(c)
        solving=time.perf_counter()-tsolve if components else None
        fields=()
        tout=time.perf_counter() if components else 0.
        if complete:
            chosen=states[-1:] if outputs==1 else states[1:]
            fields=tuple(self.output(v,output_cache) for v in chosen)
            jax.block_until_ready(fields)
        reconstruction=time.perf_counter()-tout if components else None
        total=time.perf_counter()-start if components else None
        # Host transfers of the requested fields, physical scoring and storage
        # are deliberately outside this device-output contract. A caller must
        # time/report any requested host transfer separately.
        return dict(complete=complete,failure=failure,fields=fields,coefficients=jnp.stack(states),
            cells=cells,step_status=np.asarray(statuses),visibility=vis,initial_fit_error=float(fit_error),
            initial_linear_residual=float(linear_residual),active_variables=len(c),
            setup_seconds=setup,initial_fit_seconds=cold,solving_seconds=solving,
            requested_reconstruction_seconds=reconstruction,total_query_seconds=total,
            requested_field_count=outputs,rejected_trials=0,fallback_invocations=0,
            field_kernel=self.field_kernel,
            field_work_shape=dict(stencil=local_work_shape(ca,cells,self.field_kernel),
                per_requested_output=local_work_shape(output_cache,cells,self.field_kernel)),
            output_contract='actual device fields; host transfer reported separately',
            component_instrumentation=components,
            reusable_setup_excluded=True,query_specific_compilation_included=True)
