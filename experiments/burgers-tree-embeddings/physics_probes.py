"""Truth-free independent integration probes; thresholds are not calibrated.

These probes measure a scaled, tested fine-upwind CN defect. They are not an
error certificate. Local tests cover a fixed geometry universe and are shared
across candidate trees. No reference data is an input.
"""
from common import *
import cell_catalogue as CC,packed_fields as PF,compact_fields as CF,discrete_weak as W


class PhysicsProbes:
    def __init__(self,catalogue,solve_quadrature,solve_tests):
        assert catalogue['Q']!=solve_quadrature and catalogue['tests']>solve_tests
        self.catalogue=catalogue;self.Q=catalogue['Q'];self.L=catalogue['L']
        self.root=catalogue['root'];self.rank=catalogue['rank']
        self.field=CF if catalogue['field_kernel']=='compact' else PF
        adv=self.field.tested_advection(self.L,self.Q)
        @jax.jit
        def residual(current,cache,previous,previous_cache,dt,nu):
            # Exact discrete eigen-diffusion; only integration differs from the
            # solve rule. Previous fields may use a retained smaller tree.
            old=previous_cache['A']@previous
            new=cache['A']@current
            old_adv=adv(previous,previous_cache);new_adv=adv(current,cache)
            norm=jnp.sqrt(jnp.maximum(previous@previous_cache['gram']@previous,1e-30))
            denom=1+.5*dt*nu*cache['lam']
            return (new*denom-old*(1-.5*dt*nu*cache['lam'])+.5*dt*(old_adv+new_adv))/(denom*norm)
        self.residual=residual
        # Normalize the fixed local test vectors, independently of a candidate's
        # active columns. Overlapping tests are not treated as independent errors.
        all_A=catalogue['arrays']['cache']['A'][:,self.root:]
        norms=jnp.sqrt(jnp.sum(all_A**2,axis=0))
        assert bool(jnp.all(norms>1e-12))
        self.local_tests=all_A/norms
        @jax.jit
        def local_scores(r,tests):
            values=(tests.T@r).reshape(-1,self.rank)
            return jnp.sqrt(jnp.mean(values**2,axis=1))
        self.local_scores=local_scores
        axis=jnp.arange(1,self.Q)/self.Q
        self.xy=jnp.stack(jnp.meshgrid(axis,axis,indexing='ij'),axis=-1)

    def select(self,cells,rank=None):
        return CC.select(self.catalogue,cells,rank=rank)['cache']

    def summarize(self,r,global_score=None):
        local=self.local_scores(r,self.local_tests)
        spectral=jnp.linalg.norm(r)
        jax.block_until_ready((spectral,local))
        return dict(global_score=float(spectral if global_score is None else global_score),
            tested_score=float(spectral),local_scores=np.asarray(local),
            cells=self.catalogue['cells'].tolist(),quadrature_intervals=self.Q,
            tests=self.catalogue['tests'],finite=bool(jnp.all(jnp.isfinite(r))),
            normalization='previous computed field norm; initial known-field norm at t=0',
            truth_used=False,accuracy_certificate=False)

    def initial(self,current,cache,initial):
        known=initial(self.xy);error=self.field.center(current,cache)-known
        norm=jnp.maximum(jnp.linalg.norm(known)/self.Q,1e-30)
        idx=cache['indices'];r=(W.dst2(error)/self.Q)[idx[0],idx[1]]/norm
        return self.summarize(r,jnp.linalg.norm(error)/(self.Q*norm))

    def step(self,current,cache,previous,previous_cache,dt,nu):
        return self.summarize(self.residual(current,cache,previous,previous_cache,dt,nu))
