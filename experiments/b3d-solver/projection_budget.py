"""Offline E0 geometry control on the common scoring lattice; excluded from query timings."""
from pathlib import Path
import hashlib
import numpy as np
import jax
import jax.numpy as jnp
import c3common as C3

def ah(a):return hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()

class ProjectionBudget:
    def __init__(self,cx,result):
        self.cx,self.result=cx,result;self.cache={};self.kernels={};self.metrics={}
        result['projection_budgets']={}

    def measure(self,sub,j,W,actual_errors):
        if sub['kind'] not in ('bank','POD-full-residual'):return None
        cx=self.cx;r=W.shape[1]
        span=('bank_'+sub['st'].bk['sha256'][:16]+'_'+str(r)) if sub['kind']=='bank' else 'POD512_full_residual'
        if span not in self.kernels:
            if sub['kind']=='bank':
                @jax.jit
                def project(R,d):
                    C=jax.vmap(lambda u:C3.TB.project(u,d)[0])(R)
                    residual=R-C3.TB.decode(C,d)
                    orth=jax.vmap(lambda u:C3.TB.project(u,d)[0])(residual)
                    return C,residual,orth
                self.metrics[span]=np.asarray(sub['st'].lean['L'])
                lean=sub['st'].lean
                self.kernels[span]=lambda R:project(R,lean)
            else:
                Phi=sub['pod']['data']['Phi'];G=jax.block_until_ready(Phi.T@Phi);L=jnp.linalg.cholesky(G)
                @jax.jit
                def project(R,Phi,L):
                    C=jax.scipy.linalg.cho_solve((L,True),Phi.T@R.T).T
                    residual=R-C@Phi.T
                    orth=jax.scipy.linalg.cho_solve((L,True),Phi.T@residual.T).T
                    return C,residual,orth
                self.metrics[span]=np.asarray(L)
                self.kernels[span]=lambda R:project(R,Phi,L)
            self.result['projection_budgets'][span]=dict(r=r,scoring_lattice=dict(nodes_per_axis=65,interior_per_axis=63),cases={},
                interpretation='Best least-squares field fit of approved E0 ST outputs. Orthogonal geometry only; not a causal split of interacting time/space/solver errors.')
        if (span,j) not in self.cache:
            R,meta,status=cx.ref(j);n0=float(meta['judge']['n0'])
            C,residual,orth=jax.block_until_ready(self.kernels[span](jnp.asarray(R)))
            C,residual,orth=map(np.asarray,(C,residual,orth));L=self.metrics[span]
            floor=np.linalg.norm(residual,axis=1)/n0
            orthnorm=np.linalg.norm(orth@L,axis=1)/n0;assert orthnorm.max()<=1e-10
            folder=cx.out/'data/projections'/span;folder.mkdir(parents=True,exist_ok=True);path=folder/f'c{j:03d}.npz'
            arrays=dict(best_coefficients=C,residual_norms=np.linalg.norm(residual,axis=1),initial_reference_norm=n0,orthogonal_projection_coefficients=orth)
            full_residual=bool(sub['kind']=='bank' and r==1024 and j in cx.cfg['phase_cases'])
            if full_residual:arrays['residual_fields']=residual
            np.savez_compressed(path,**arrays)
            record=dict(reference=f'3d/val64/ST/c{j:03d}',reference_field_sha256=ah(R),normalisation=n0,
                floor_e_t=floor.tolist(),floor_max=float(floor[1:].max()),orthogonality_relative=orthnorm.tolist(),full_residual_fields_saved=full_residual,
                raw=dict(path=str(path.relative_to(cx.out)),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            self.result['projection_budgets'][span]['cases'][str(j)]=record
            self.cache[(span,j)]=(C,floor,n0)
        C,floor,n0=self.cache[(span,j)];evolve=np.linalg.norm((W-C)@self.metrics[span],axis=1)/n0
        actual=np.asarray(actual_errors);defect=actual**2-floor**2-evolve**2
        assert np.max(np.abs(defect))<=1e-8*max(float(np.max(actual**2)),1e-16)+1e-12
        return dict(span_id=span,reference_projection_floor_e_t=floor.tolist(),evolution_in_span_distance_e_t=evolve.tolist(),
                    pythagorean_squared_error_defect=defect.tolist(),scope='offline E0 scoring-lattice diagnostic; excluded from every online query timer')
