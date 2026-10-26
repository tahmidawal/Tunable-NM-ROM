"""Tiny consistency controls for full and weighted POD residuals and the new admission metric."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import replay as X

jax, jnp, H, R, B = X.jax, X.jnp, X.H, X.R, X.B


def main():
    assert jax.default_backend() == 'gpu' and jax.config.jax_enable_x64
    rng = np.random.default_rng(20261026)
    L, rank = 5, 4
    phi = np.linalg.qr(rng.normal(size=((L-1)**3, rank)))[0]
    Phi = jnp.asarray(phi)
    full = dict(Phi=Phi, LPhi=H.lap_cols(Phi, 3, L), DPhi=H.dsum_cols(Phi, 3, L))
    data, _ = B.ecsw_data(Phi, np.arange(len(phi)), np.ones(len(phi)), 3, L)
    data = dict(data, Phi=Phi)
    sf, fj = R.make_arm('none', 3, L)
    se, ej = R.make_arm('ecsw', 3, L)
    c, cn, cm = [jnp.asarray(rng.normal(size=rank)*.1) for _ in range(3)]
    co, dt, nu = jnp.asarray(H.SCHEMES['BDF2'][:5]), jnp.float64(1/60), jnp.float64(.025)
    rf, jf = fj(c, sf(cn, cm, nu, co, dt, full), nu, co, dt, full)
    re, je = ej(c, se(cn, cm, nu, co, dt, data), nu, co, dt, data)
    residual = float(np.max(np.abs(rf-re)))
    jacobian = float(np.max(np.abs(jf-je)))
    direction = jnp.asarray(rng.normal(size=rank))
    eps = 1e-6
    def objective(z):
        rr, _ = fj(z, sf(cn, cm, nu, co, dt, full), nu, co, dt, full)
        return .5*jnp.dot(rr, rr)
    fd = float((objective(c+eps*direction)-objective(c-eps*direction))/(2*eps))
    exact = float(direction @ (jf.T@rf))
    gradient = abs(fd-exact)/max(1.,abs(exact))
    C = jnp.stack([cn,c,c*.8,c*.6])
    sch=R.schedule(3,'BDF2',1/60,'prod')
    m=X.metric_summary(jax.block_until_ready(X.make_metrics(L,rank,3)(C,nu,full,data,sch)))
    equivalence=max(m['delta_g'])
    bad=dict(data,sw=data['sw']*jnp.sqrt(2.))
    mb=X.metric_summary(jax.block_until_ready(X.make_metrics(L,rank,3)(C,nu,full,bad,sch)))
    corrupt_detected=max(mb['delta_g'])>.001
    zeros=X.quotient(np.array([0.,1.,np.nan]),np.array([0.,0.,1.]))
    zero_handling=bool(zeros[0]==0 and np.isinf(zeros[1:]).all())
    schedule=bool(int(sch['steps'])==15 and int(sch['nstart'])==1 and
                  np.array_equal(np.flatnonzero(np.asarray(sch['slot'])>=0),np.arange(0,16,3)))
    out=dict(backend=jax.default_backend(),precision='f64/highest',residual_maxabs=residual,
             jacobian_maxabs=jacobian,gradient_directional_error=gradient,
             all_nodes_delta_g_max=equivalence,weight_scale_control_fired=corrupt_detected,
             zero_handling=zero_handling,schedule=schedule)
    out['pass']=bool(residual<1e-12 and jacobian<1e-12 and gradient<1e-9 and equivalence<1e-12
                     and corrupt_detected and zero_handling and schedule)
    print(json.dumps(out,indent=2),flush=True)
    assert out['pass']
    p=Path(__file__).parent/'checks/controls.json'
    p.write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':main()
