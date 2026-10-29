"""Reusable DFG Newton query, extracted from fom.py; native validation pending.

Construction, immutable mesh/lift loading and output evaluator preparation are offline.
The caller must run under the reviewed NGSolve TaskManager/thread allocation. Production
use additionally requires exact-source release and measured equivalence to the producer.
"""
from __future__ import annotations
import resource
import hashlib
from pathlib import Path
import time
import numpy as np
from shared_lift import CommonStokesLift
from reference_export import layout_record

OUTPUT_CONTRACTS=('full_velocity_pressure_fields','fixed_probes','integrated_forces')

class FOMQueryFailure(RuntimeError):
    def __init__(self,message,invocation):
        super().__init__(message);self.invocation=invocation


class SteadyFOMQuery:
    def __init__(self, ng, mesh_path, cfg, common_lift, output_evaluator_factory):
        started=time.perf_counter()
        if cfg['kind']!='dfg' or cfg['outlet']!='gradient':
            raise ValueError('registered unforced DFG gradient-outlet family only')
        if not isinstance(common_lift,CommonStokesLift):
            raise TypeError('reviewed cached common lift required')
        if getattr(common_lift,'usage_role',None) not in ('G2_truth_admitted','FD1_fixed_finite_channel_truth_admitted','native_equivalence_control','development_calibration_only','frozen_fom_comparator'):
            raise ValueError('explicit validated common-lift role required')
        if common_lift.usage_role=='FD1_fixed_finite_channel_truth_admitted':
            from reference_scope import validate_geometry
            validate_geometry(cfg)
        if hasattr(common_lift,'configuration_sha256'):
            from candidate_lift import configuration_sha256
            if common_lift.configuration_sha256!=configuration_sha256(cfg):raise ValueError('candidate lift configuration binding changed')
        if not callable(output_evaluator_factory):
            raise TypeError('reviewed same-mesh output evaluator factory required')
        if cfg['pressure_order']!=cfg['velocity_order']-1 or cfg['velocity_order']<2:
            raise ValueError('Taylor-Hood pair required')
        if not 1<=cfg['newton_max']<=30:raise ValueError('Newton cap')
        mesh_path=Path(mesh_path)
        with mesh_path.open('rb') as stream:mesh_digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if mesh_digest!=common_lift.metadata['mesh_sha256']:
            raise ValueError('query binary mesh differs from admitted common lift')
        mesh=ng.Mesh(str(mesh_path))  # Exact binary curve coefficients, never re-Curve.
        self.ng,self.cfg,self.lift=ng,dict(cfg),common_lift
        self.mesh=mesh
        self.V=ng.VectorH1(mesh,order=cfg['velocity_order'],dirichlet='inlet|wall.*|obstacle')
        self.Q=ng.H1(mesh,order=cfg['pressure_order']);self.X=self.V*self.Q
        _,digest=layout_record(mesh,self.V,self.Q,ng)
        if digest!=common_lift.metadata['layout_sha256']:
            raise ValueError('common lift and query layout differ')
        if mesh.ne>cfg['max_elements'] or self.X.ndof>cfg['max_dofs']:
            raise MemoryError('query space exceeds reviewed capacity')
        U,D=cfg['U'],cfg['D']
        if U!=common_lift.metadata['U'] or D!=common_lift.metadata['D']:
            raise ValueError('common lift scaling differs')
        self.evaluators=dict(output_evaluator_factory(mesh,self.V,self.Q,self.X,dict(cfg)))
        if set(self.evaluators)!=set(OUTPUT_CONTRACTS):
            raise ValueError('all three reviewed output evaluators required')
        self.g=ng.GridFunction(self.X)
        (u,p),(v,q)=self.X.TnT()
        self.nu=ng.Parameter(cfg['nu'])
        self.form=ng.BilinearForm(self.X)
        self.form+=(self.nu*ng.InnerProduct(ng.Grad(u),ng.Grad(v))+(ng.Grad(u)*u)*v-p*ng.div(v)-q*ng.div(u))*ng.dx(bonus_intorder=cfg['assembly_bonus_order'])
        self.free=np.array([bool(x) for x in self.X.FreeDofs()],dtype=bool)
        self.scale=np.r_[np.full(self.V.ndof,U**2*D),np.full(self.Q.ndof,U*D)]
        self.offline_seconds=time.perf_counter()-started

    def diagnostics_from_evidence(self,evidence):
        """Untimed evaluation of a retained invocation state; never re-solve."""
        coefficients=np.asarray(evidence['state']);Re=float(evidence['Re'])
        nu=self.cfg['U']*self.cfg['D']/Re
        if (coefficients.dtype!=np.dtype('float64') or coefficients.shape!=(self.X.ndof,)
            or not np.isfinite(coefficients).all() or not 15<=Re<=25 or evidence['nu']!=nu):
            raise ValueError('invalid same-invocation diagnostic evidence')
        if hasattr(self.lift,'allows_Re') and not self.lift.allows_Re(Re):raise ValueError('diagnostic parameter outside reviewed candidate role')
        function=getattr(self.evaluators['full_velocity_pressure_fields'],'diagnostics',None)
        if not callable(function):raise ValueError('reviewed untimed state diagnostics unavailable')
        retained=self.ng.GridFunction(self.X);retained.vec.FV().NumPy()[:]=coefficients
        return function(*retained.components,nu)

    @staticmethod
    def _array(vector):return vector.FV().NumPy()

    def query(self, Re, output_contract):
        context={'started':time.perf_counter(),'state_available':False,'converged':False,
                 'history':[],'memory_checks':[],'work':{},'outputs_available':False}
        try:return self._query_impl(Re,output_contract,context)
        except Exception as exc:
            stopped=time.perf_counter();solve=context.get('solve_started');output=context.get('output_started')
            evidence={'Re':Re,'nu':context.get('nu'),'converged':context['converged'],
                      'history':context['history'],'memory_checks':context['memory_checks'],
                      'state_available':context['state_available'],'initialization':'cached_common_Stokes_lift',
                      'lift_role':self.lift.usage_role,'reference_qualification':getattr(self.lift,'reference_qualification',None),'reference_admission_sha256':getattr(self.lift,'reference_admission_sha256',getattr(self.lift,'truth_reference_admission_sha256',None)),'pressure_gauge':'natural outlet, unshifted'}
            if context['state_available']:evidence['state']=self._array(self.g.vec).copy()
            work=context['work'];work['status']='query_failed_after_root_convergence' if context['converged'] else 'query_failed'
            invocation={'output_contract':output_contract,'outputs':context.get('outputs'),'outputs_available':context['outputs_available'],
                        'work':work,'evidence':evidence,'query_failed':True,'exception':repr(exc),'physical_admission':False,
                        'peak_host_rss_bytes':int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)*1024,
                        'peak_host_rss_scope':'process lifetime high-water at invocation failure; includes offline setup and prior invocations',
                        'component_seconds':{'initialization':(solve if solve is not None else stopped)-context['started'],
                            'coefficient_solve':0. if solve is None else (output if output is not None else stopped)-solve,
                            'output_reconstruction':0. if output is None else context.get('output_seconds',stopped-output),
                            'total_query':time.perf_counter()-context['started']},
                        'scope':'failed actual invocation retained; no eligibility or fabricated outputs'}
            raise FOMQueryFailure(str(exc),invocation) from exc

    def _query_impl(self, Re, output_contract, context):
        query_started=context['started']
        if not np.isfinite(Re) or not 15<=Re<=25 or output_contract not in OUTPUT_CONTRACTS:
            raise ValueError('outside registered Reynolds family/output contract')
        if hasattr(self.lift,'allows_Re') and not self.lift.allows_Re(Re):raise ValueError('query parameter outside reviewed candidate role')
        cfg=self.cfg;nu=cfg['U']*cfg['D']/Re;context['nu']=nu;self.nu.Set(nu)
        # Every invocation starts from the identical cached lift. No warm state leaks.
        self._array(self.g.vec)[:]=self.lift.state_for_nu(nu)
        context['state_available']=True
        work={'residual_evaluations':0,'Jacobian_assemblies':0,'factorizations':0,'linear_solves':0,
              'Newton_updates':0,'output_evaluations':0}
        histories=context['history'];memory=context['memory_checks'];context['work']=work
        def norm(vector):return float(np.linalg.norm((self._array(vector)/self.scale)[self.free]))
        def residual():
            r=self.g.vec.CreateVector();self.form.Apply(self.g.vec,r)
            work['residual_evaluations']+=1
            return r
        solve_started=time.perf_counter();context['solve_started']=solve_started
        initialization_seconds=solve_started-query_started
        initial=norm(residual());denominator=max(initial,cfg['residual_floor'])
        converged=False
        for iteration in range(cfg['newton_max']+1):
            r=residual();value=norm(r)
            row={'iteration':iteration,'scaled_residual_norm':value,'residual_denominator':denominator,
                 'relative_residual':value/denominator};histories.append(row)
            if value/denominator<=cfg['nonlinear_relative_tolerance']:
                converged=True;context['converged']=True;break
            if iteration==cfg['newton_max']:break
            self.form.AssembleLinearization(self.g.vec);work['Jacobian_assemblies']+=1
            nnz=int(self.form.mat.nze)
            estimated=nnz*16*cfg['factor_fill_estimate']+self.X.ndof*8*32
            rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            memory.append({'nnz':nnz,'estimated_factor_bytes':estimated,'current_maxrss_bytes':rss})
            if nnz>cfg['max_nnz'] or estimated+rss>cfg['memory_stop_gib']*2**30:
                raise MemoryError('query pre-factorization guard')
            inverse=self.form.mat.Inverse(self.X.FreeDofs(),inverse=cfg['inverse_backend']);work['factorizations']+=1
            delta=self.g.vec.CreateVector();delta.data=inverse*r;work['linear_solves']+=1
            linear=r.CreateVector();linear.data=self.form.mat*delta-r
            row['linear_relative_residual']=norm(linear)/max(value,1e-30)
            if row['linear_relative_residual']>cfg['linear_relative_tolerance']:
                raise ArithmeticError('Newton linear residual failed')
            base=self._array(self.g.vec).copy();accepted=False
            for damping in cfg['damping_sequence']:
                self._array(self.g.vec)[:]=base-damping*self._array(delta)
                candidate=norm(residual())
                if candidate<=(1-cfg['armijo']*damping)*value:
                    accepted=True;row['damping']=damping;break
            del inverse
            if not accepted:
                self._array(self.g.vec)[:]=base
                raise ArithmeticError('no declared Newton damping accepted')
            work['Newton_updates']+=1
        if not converged:raise ArithmeticError('Newton iteration cap; no accepted solution')
        solve_seconds=time.perf_counter()-solve_started
        # Evaluator cost is included in the query; diagnostic scoring is external.
        output_started=time.perf_counter();context['output_started']=output_started
        outputs=self.evaluators[output_contract](*self.g.components,nu)
        output_seconds=time.perf_counter()-output_started
        context.update(outputs=outputs,outputs_available=True,output_seconds=output_seconds)
        work['output_evaluations']+=1
        work['status']='root_converged'
        retained_state=self._array(self.g.vec).copy()
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        component_seconds={'initialization':initialization_seconds,'coefficient_solve':solve_seconds,
            'output_reconstruction':output_seconds,'total_query':time.perf_counter()-query_started}
        return {'output_contract':output_contract,'outputs':outputs,'outputs_available':True,'query_failed':False,'work':work,'component_seconds':component_seconds,
                'peak_host_rss_bytes':peak_rss,'peak_host_rss_scope':'process lifetime high-water at invocation end; includes offline setup and prior invocations',
                'evidence':{'state':retained_state,'Re':float(Re),'nu':nu,
                    'converged':converged,'history':histories,'memory_checks':memory,
                    'initialization':'cached_common_Stokes_lift','lift_role':self.lift.usage_role,'reference_qualification':getattr(self.lift,'reference_qualification',None),'reference_admission_sha256':getattr(self.lift,'reference_admission_sha256',getattr(self.lift,'truth_reference_admission_sha256',None)),'pressure_gauge':'natural outlet, unshifted'},
                'physical_admission':False,'scope':'native equivalence and paired runtime validation pending'}
