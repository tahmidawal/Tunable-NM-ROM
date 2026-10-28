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

class SteadyFOMQuery:
    def __init__(self, ng, mesh_path, cfg, common_lift, output_evaluators):
        started=time.perf_counter()
        if cfg['kind']!='dfg' or cfg['outlet']!='gradient':
            raise ValueError('registered unforced DFG gradient-outlet family only')
        if not isinstance(common_lift,CommonStokesLift):
            raise TypeError('genuinely admitted cached common lift required')
        if set(output_evaluators)!=set(OUTPUT_CONTRACTS):
            raise ValueError('all three reviewed output evaluators required')
        if cfg['pressure_order']!=cfg['velocity_order']-1 or cfg['velocity_order']<2:
            raise ValueError('Taylor-Hood pair required')
        if not 1<=cfg['newton_max']<=30:raise ValueError('Newton cap')
        mesh_path=Path(mesh_path)
        with mesh_path.open('rb') as stream:mesh_digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if mesh_digest!=common_lift.metadata['mesh_sha256']:
            raise ValueError('query binary mesh differs from admitted common lift')
        mesh=ng.Mesh(str(mesh_path))  # Exact binary curve coefficients, never re-Curve.
        self.ng,self.cfg,self.lift=ng,dict(cfg),common_lift
        self.evaluators=dict(output_evaluators)
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
        self.g=ng.GridFunction(self.X)
        (u,p),(v,q)=self.X.TnT()
        self.nu=ng.Parameter(cfg['nu'])
        self.form=ng.BilinearForm(self.X)
        self.form+=(self.nu*ng.InnerProduct(ng.Grad(u),ng.Grad(v))+(ng.Grad(u)*u)*v-p*ng.div(v)-q*ng.div(u))*ng.dx(bonus_intorder=cfg['assembly_bonus_order'])
        self.free=np.array([bool(x) for x in self.X.FreeDofs()],dtype=bool)
        self.scale=np.r_[np.full(self.V.ndof,U**2*D),np.full(self.Q.ndof,U*D)]
        self.offline_seconds=time.perf_counter()-started

    @staticmethod
    def _array(vector):return vector.FV().NumPy()

    def query(self, Re, output_contract):
        if not np.isfinite(Re) or not 15<=Re<=25 or output_contract not in OUTPUT_CONTRACTS:
            raise ValueError('outside registered Reynolds family/output contract')
        cfg=self.cfg;nu=cfg['U']*cfg['D']/Re;self.nu.Set(nu)
        # Every invocation starts from the identical cached lift. No warm state leaks.
        self._array(self.g.vec)[:]=self.lift.state_for_nu(nu)
        work={'residual_evaluations':0,'Jacobian_assemblies':0,'factorizations':0,'linear_solves':0,
              'Newton_updates':0,'output_evaluations':0}
        histories=[];memory=[]
        def norm(vector):return float(np.linalg.norm((self._array(vector)/self.scale)[self.free]))
        def residual():
            r=self.g.vec.CreateVector();self.form.Apply(self.g.vec,r)
            work['residual_evaluations']+=1
            return r
        initial=norm(residual());denominator=max(initial,cfg['residual_floor'])
        converged=False
        for iteration in range(cfg['newton_max']+1):
            r=residual();value=norm(r)
            row={'iteration':iteration,'scaled_residual_norm':value,'residual_denominator':denominator,
                 'relative_residual':value/denominator};histories.append(row)
            if value/denominator<=cfg['nonlinear_relative_tolerance']:
                converged=True;break
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
        # Evaluator cost is included in the query; diagnostic scoring is external.
        outputs=self.evaluators[output_contract](*self.g.components)
        work['output_evaluations']+=1
        return {'output_contract':output_contract,'outputs':outputs,'work':work,
                'evidence':{'state':self._array(self.g.vec).copy(),'Re':float(Re),'nu':nu,
                    'converged':converged,'history':histories,'memory_checks':memory,
                    'initialization':'cached_common_Stokes_lift','pressure_gauge':'natural outlet, unshifted'},
                'physical_admission':False,'scope':'native equivalence and paired runtime validation pending'}
