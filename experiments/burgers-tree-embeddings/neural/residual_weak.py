"""Cached child features in the unchanged fine-stencil weak PDE equations.

The cache belongs to fixed feature weights, cells and integration coordinates.
Rebuild it whenever any of these change. L is the physics stencil resolution;
Q is integration resolution only. All feature arrays remain runtime arguments.
"""
from common import *
from neural import weak as W, residual_tree as R
from discrete_weak import dst2


def prepare(params, cells, L, Q, M):
    cache = W.prepare(L, Q, M)
    B = jax.vmap(lambda xy: R.feature_matrix(params['feature'], cells, xy))(cache['stencil'])
    return dict(cache, B=B[0], B_stencil=B)


def field(params, z, cache):
    return R.cached(dict(params, B=cache['B']), z, cache['xy'])


def field_jacobian(params, z, cache, root_jacobian='forward'):
    assert root_jacobian in ('forward','reverse')
    evaluator=R.cached_jacobian if root_jacobian=='forward' else R.cached_jacobian_reverse
    return evaluator(dict(params, B=cache['B']), z, cache['xy'])


def _evaluators(L, Q, theta, root_jacobian='forward'):
    assert root_jacobian in ('forward','reverse')
    block_jac=R.cached_jacobian if root_jacobian=='forward' else R.cached_jacobian_reverse
    assert theta in (.5, 1.)
    def project(u, cache):
        b = dst2(u.reshape(Q-1, Q-1))/Q
        return b[cache['indices'][0], cache['indices'][1]]
    def project_jac(J, cache):
        return jax.vmap(project, in_axes=(1, None), out_axes=1)(J, cache)
    def stencil(params, z, cache):
        return jax.vmap(lambda xy, B: R.cached(dict(params, B=B), z, xy))(
            cache['stencil'], cache['B_stencil'])
    def stencil_jac(params, z, cache):
        return jax.vmap(lambda xy, B: block_jac(dict(params, B=B), z, xy))(
            cache['stencil'], cache['B_stencil'])
    def mass(params, z, cache):
        return project(field(params, z, cache), cache)
    def mass_jac(params, z, cache):
        return project_jac(field_jacobian(params, z, cache, root_jacobian), cache)
    def advection(params, z, cache):
        c, xp, xm, yp, ym = stencil(params, z, cache)
        diff = jnp.where(c>0, c-xm, xp-c)+jnp.where(c>0, c-ym, yp-c)
        return project(c*L*diff, cache)
    def advection_jac(params, z, cache):
        c, xp, xm, yp, ym = stencil(params, z, cache)
        Jc, Jxp, Jxm, Jyp, Jym = stencil_jac(params, z, cache)
        diff = jnp.where(c>0, c-xm, xp-c)+jnp.where(c>0, c-ym, yp-c)
        Jdiff = (jnp.where((c>0)[:, None], Jc-Jxm, Jxp-Jc)
                 +jnp.where((c>0)[:, None], Jc-Jym, Jyp-Jc))
        return project_jac(L*(Jc*diff[:, None]+c[:, None]*Jdiff), cache)
    def previous(params, z, cache, dt, nu):
        return (mass(params, z, cache)*(1-(1-theta)*dt*nu*cache['lam'])
                -(1-theta)*dt*advection(params, z, cache))
    def residual(params, z, old, cache, dt, nu, norm):
        denom = 1+theta*dt*nu*cache['lam']
        return (mass(params, z, cache)+(theta*dt*advection(params, z, cache)-old)/denom)/norm
    def residual_jac(params, z, old, cache, dt, nu, norm):
        denom = 1+theta*dt*nu*cache['lam']
        return (mass_jac(params, z, cache)+theta*dt*advection_jac(params, z, cache)/denom[:, None])/norm
    return mass, advection, previous, residual, mass_jac, advection_jac, residual_jac


def make_operators(L, Q, theta=.5):
    """Same four-callable interface as neural.weak.make_operators."""
    return _evaluators(L, Q, theta)[:4]


def make_jacobians(L, Q, theta=.5, root_jacobian='forward'):
    """Return explicit mass, advection and residual code Jacobians."""
    return _evaluators(L, Q, theta, root_jacobian)[4:]


def make_step(L, Q, iterations=60, theta=.5, root_jacobian='forward'):
    """LM arithmetic and stopping criteria match neural.weak.make_step."""
    _, _, previous, residual, _, _, jac = _evaluators(L, Q, theta, root_jacobian)
    @jax.jit
    def step(params, z0, cache, dt, nu, station_tolerance=1e-8, update_tolerance=1e-10):
        u0 = field(params, z0, cache)
        norm = jnp.maximum(jnp.linalg.norm(u0)/Q, 1e-15)
        old = previous(params, z0, cache, dt, nu)
        fun = lambda z: residual(params, z, old, cache, dt, nu, norm)
        physical_field = lambda z: field(params, z, cache)/(Q*norm)
        def body(_, state):
            def advance(state):
                z, loss, damp, accepted, done, update, used = state
                r = fun(z); J = jac(params, z, old, cache, dt, nu, norm)
                g = J.T@r; H = J.T@J
                physical = field_jacobian(params, z, cache, root_jacobian)/(Q*norm)
                diag = jnp.maximum(jnp.sum(physical**2, axis=0), 1e-14)
                scale = jax.lax.rsqrt(diag)
                delta = scale*jnp.linalg.solve(H*scale[:, None]*scale[None, :]
                    +damp*jnp.eye(len(z)), -scale*g)
                update = jnp.linalg.norm(physical_field(z+delta)-physical_field(z))
                rr = fun(z+delta); newloss = rr@rr
                predicted = -2*g@delta-delta@H@delta
                difference = rr-r
                reduction = -2*r@difference-difference@difference
                good = jnp.isfinite(newloss)&(reduction>0)&(predicted>0)
                ratio = reduction/jnp.maximum(predicted, 1e-30)
                station = jnp.max(jnp.abs(g)*scale)
                done = ((station<=station_tolerance)
                    &((update<=update_tolerance)|(station<=station_tolerance*.01)))
                damp = jnp.where(good, jnp.where(ratio>.75, damp/3,
                    jnp.where(ratio<.25, damp*2, damp)), damp*5)
                return (jnp.where(good, z+delta, z), jnp.where(good, newloss, loss),
                    jnp.clip(damp, 1e-10, 1e12), accepted+good.astype(jnp.int32), done, update, used+1)
            return jax.lax.cond(state[4], lambda s: s, advance, state)
        r = fun(z0)
        initial = (z0, r@r, jnp.float64(1e-3), jnp.int32(0), jnp.bool_(False), jnp.float64(0), jnp.int32(0))
        z, loss, damp, accepted, done, update, used = jax.lax.fori_loop(0, iterations, body, initial)
        r = fun(z); J = jac(params, z, old, cache, dt, nu, norm)
        P = field_jacobian(params, z, cache, root_jacobian)/(Q*norm)
        station = jnp.max(jnp.abs(J.T@r)/jnp.sqrt(jnp.maximum(jnp.sum(P**2, axis=0), 1e-14)))
        converged = ((station<=station_tolerance)
            &((update<=update_tolerance)|(station<=station_tolerance*.01)))
        return z, jnp.array([jnp.linalg.norm(r), station, update, used, accepted, converged])
    return step
