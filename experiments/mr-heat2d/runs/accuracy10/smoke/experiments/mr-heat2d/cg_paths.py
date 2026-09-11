"""Compiled full-grid CG heat solvers with matched output and timing contracts."""
import heat_core as hc
import jax
import jax.numpy as jnp
import numpy as np


def cg_solve(operator, rhs, initial, tolerance, budget):
    """Unpreconditioned CG recurrence, with iteration and true-residual evidence.

    Same recurrence/relative stopping rule as JAX CG with identity preconditioner.
    The final explicit residual check is charged to the full-order query.
    """
    norm2 = jnp.sum(rhs*rhs)
    threshold = tolerance**2*norm2
    residual = rhs-operator(initial)
    gamma = jnp.sum(residual*residual)
    state = initial, residual, gamma, residual, jnp.int32(0)
    def condition(state):
        return (state[2] > threshold) & (state[4] < budget) & jnp.isfinite(state[2])
    def step(state):
        x, r, gamma, p, count = state
        ap = operator(p)
        alpha = gamma/jnp.sum(p*ap)
        xnew = x+alpha*p
        rnew = r-alpha*ap
        gnew = jnp.sum(rnew*rnew)
        return xnew, rnew, gnew, rnew+(gnew/gamma)*p, count+1
    x, _, gamma, _, count = jax.lax.while_loop(condition, step, state)
    scale = jnp.maximum(jnp.sqrt(norm2), 1e-300)
    true_relative = jnp.linalg.norm(rhs-operator(x))/scale
    recurrent_relative = jnp.sqrt(gamma)/scale
    converged = jnp.isfinite(true_relative) & (true_relative <= tolerance*(1+1e-6)+1e-12)
    return x, jnp.array([count, recurrent_relative, true_relative, converged.astype(jnp.float64)])


def build(n, times, specification):
    dt = specification['dt']; theta = specification['theta']
    stride = int(round((times[1]-times[0])/dt))
    assert np.allclose(np.arange(len(times))*stride*dt, times)
    @jax.jit
    def query(u0, diffusivity):
        def operator(u): return u+theta*dt*diffusivity*hc.negative_laplacian(u,n)
        def inner(u, _):
            rhs = u-(1-theta)*dt*diffusivity*hc.negative_laplacian(u,n)
            u_new, info = cg_solve(operator,rhs,u,specification['tolerance'],specification['max_iterations'])
            return u_new, info
        def outer(u, _):
            u_new, infos = jax.lax.scan(inner,u,None,length=stride)
            return u_new, (u_new,infos)
        _, (later,infos) = jax.lax.scan(outer,u0,None,length=len(times)-1)
        return jnp.concatenate((u0[None],later)), infos.reshape(-1,4)
    return query


def verify():
    """Independent dense finite-difference solves for a nonsingle-mode fixture."""
    import scipy.linalg
    from jax.scipy.sparse.linalg import cg
    n = 8; nu = .02; times = [0.,.1,.2,.3,.4,.5]
    rng = np.random.default_rng(790718)
    u0 = rng.normal(size=(n-1,n-1))
    axis = np.diag(np.full(n-1,2.))-np.diag(np.ones(n-2),1)-np.diag(np.ones(n-2),-1)
    lap = n*n*(np.kron(axis,np.eye(n-1))+np.kron(np.eye(n-1),axis))
    errors = {}
    for name,dt,theta in [('cn',.025,.5),('be50',.01,1.)]:
        spec = dict(dt=dt,theta=theta,tolerance=1e-12,max_iterations=1000)
        fields,info = build(n,times,spec)(jnp.asarray(u0),nu)
        left = np.eye((n-1)**2)+theta*dt*nu*lap
        right = np.eye((n-1)**2)-(1-theta)*dt*nu*lap
        factor = scipy.linalg.lu_factor(left); u = u0.reshape(-1); truth = [u.copy()]
        stride = round(.1/dt)
        for i in range(round(.5/dt)):
            u = scipy.linalg.lu_solve(factor,right@u)
            if (i+1)%stride == 0: truth.append(u.copy())
        error = float(np.linalg.norm(np.asarray(fields).reshape(len(times),-1)-truth)/np.linalg.norm(truth))
        assert error < 1e-10 and np.all(np.asarray(info)[:,3]==1), (name,error,info)
        # Verify the instrumented recurrence against the installed JAX CG.
        operator = lambda x: x+theta*dt*nu*hc.negative_laplacian(x,n)
        rhs = jnp.asarray((right@u0.reshape(-1)).reshape(n-1,n-1))
        a,_ = jax.jit(lambda b,x: cg_solve(operator,b,x,1e-12,1000))(rhs,jnp.asarray(u0))
        b,_ = jax.jit(lambda b,x: cg(operator,b,x0=x,tol=1e-12,maxiter=1000))(rhs,jnp.asarray(u0))
        parity = float(np.linalg.norm(np.asarray(a-b))/np.linalg.norm(np.asarray(b)))
        assert parity < 1e-10
        errors[name] = dict(dense_time_step_relative_error=error,jax_cg_field_parity=parity)
    return dict(cg_independent_fixture=errors)
