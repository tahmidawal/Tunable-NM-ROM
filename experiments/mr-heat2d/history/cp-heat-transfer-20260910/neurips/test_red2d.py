"""Equivalence test: dense-Jacobian LM rollout vs exact reduced-Gram LM rollout.

Runs LOCALLY on random weights (no checkpoints), so the algebra is validated
before any cluster time is spent.  Both paths must produce the same JtJ, Jtr,
the same Gauss-Newton counts, and the same final (z, s).

The claim being tested
----------------------
h2opt's residual is

    r(z,s) = s * T(kap) @ ( h(z) @ Vm + bias*m ) - u_prev,   T = I + DT*kap*K

with h(z) the rank-dim output of a small MLP and Vm = mask (*) V the CP basis.
Because r is affine in h and T is affine in kap, every entry of J^T J and J^T r
is a polynomial in kap with coefficients that are rank x rank / rank / scalar
contractions of (Vm, K Vm, m, K m).  Those are mesh-sized to BUILD (offline,
once per cell) and mesh-free to USE.
"""
import os
import numpy as np
import jax
import jax.numpy as jnp

X64 = os.environ.get('X64', '1') == '1'
jax.config.update('jax_enable_x64', X64)

L = 1.0
DT = 5e-3
AMP_EPS = 1e-6


def make_ops(N):
    dx = L / (N - 1)

    def K_op(u_flat):
        u = u_flat.reshape((N, N))
        out = jnp.zeros_like(u)
        out = out.at[1:-1, 1:-1].set(
            (4 * u[1:-1, 1:-1] - u[0:-2, 1:-1] - u[2:, 1:-1]
             - u[1:-1, 0:-2] - u[1:-1, 2:]) / dx ** 2)
        out = out.at[0, :].set(u[0, :])
        out = out.at[-1, :].set(u[-1, :])
        out = out.at[:, 0].set(u[:, 0])
        out = out.at[:, -1].set(u[:, -1])
        return out.flatten()

    m = jnp.ones((N, N))
    m = m.at[0, :].set(0.).at[-1, :].set(0.).at[:, 0].set(0.).at[:, -1].set(0.)
    return K_op, m.flatten()


def rand_weights(key, k, rank, hid, N):
    ks = jax.random.split(key, 8)
    sc = 0.3
    return dict(
        W1=sc * jax.random.normal(ks[0], (k, hid)) / np.sqrt(k),
        b1=jnp.zeros(hid),
        W2=sc * jax.random.normal(ks[1], (hid, hid)) / np.sqrt(hid),
        b2=jnp.zeros(hid),
        Wr=sc * jax.random.normal(ks[2], (hid, rank)) / np.sqrt(hid),
        br=jnp.zeros(rank),
        Wd=sc * jax.random.normal(ks[3], (k, rank)) / np.sqrt(k),
        bd=jnp.zeros(rank),
        W_x=0.05 * jax.random.normal(ks[4], (rank, N)),
        W_y=0.05 * jax.random.normal(ks[5], (rank, N)),
        bias=jnp.array(0.01),
    )


def swish(x):
    return x * jax.nn.sigmoid(x)


def make_h(p):
    """z -> h  (the rank-dim feature vector).  Mirrors LinearCPDecoder."""
    def h_of_z(z):
        a = swish(z @ p['W1'] + p['b1'])
        a = swish(a @ p['W2'] + p['b2'])
        a = a @ p['Wr'] + p['br']
        return z @ p['Wd'] + p['bd'] + a
    return h_of_z


# ----------------------------------------------------------------- dense path
def make_dense(p, N, K_op, m, k, IT, TOL):
    h_of_z = make_h(p)
    EYE = jnp.eye(k + 1)

    def decode_norm(z):
        h = h_of_z(z)
        return jnp.einsum('r,ri,rj->ij', h, p['W_x'], p['W_y']).flatten() \
            + p['bias']

    def cdn(z):
        return m * decode_norm(z)          # u_g == 0 for the 2D cells

    def _residual(zs, u_prev, kap):
        z_, s_ = zs[:-1], zs[-1]
        u_n = cdn(z_)
        return s_ * (u_n + DT * kap * K_op(u_n)) - u_prev

    def _gn_step(z_init, scale_init, u_prev, kap, tap=None):
        zs0 = jnp.concatenate([z_init, jnp.asarray([scale_init])])
        mu0 = 1e-4

        def _body(carry):
            zs, mu, _, itr = carry
            J = jax.jacfwd(lambda q: _residual(q, u_prev, kap))(zs)
            r = _residual(zs, u_prev, kap)
            Jtr = J.T @ r
            JtJ = J.T @ J
            if tap is not None:
                tap.append((np.asarray(JtJ), np.asarray(Jtr)))
            diag_scale = jnp.mean(jnp.diag(JtJ)) + 1e-8
            A = JtJ + mu * diag_scale * EYE
            d = jnp.linalg.solve(A, -Jtr)
            zs_new = zs + d
            r_new = _residual(zs_new, u_prev, kap)
            accept = jnp.linalg.norm(r_new) < jnp.linalg.norm(r)
            zs_next = jnp.where(accept, zs_new, zs)
            zs_next = zs_next.at[-1].set(jnp.maximum(zs_next[-1], 1e-8))
            mu_next = jnp.where(accept, jnp.maximum(mu * 0.3, 1e-7),
                                jnp.minimum(mu * 10.0, 1e2))
            return zs_next, mu_next, jnp.linalg.norm(Jtr), itr + 1

        def _cond(carry):
            _, _, jtr_norm, itr = carry
            return (jtr_norm > TOL) & (itr < IT)

        carry = (zs0, mu0, jnp.inf, 0)
        while _cond(carry):                # python loop so we can tap
            carry = _body(carry)
        zs_f, _, _, itr_f = carry
        return zs_f[:-1], zs_f[-1], itr_f

    def rollout(z0, s0, u0, kap, n_steps, tap=None):
        z_, s_, u_prev = z0, s0, u0
        counts = []
        for _ in range(n_steps):
            z_, s_, ni = _gn_step(z_, s_, u_prev, kap, tap)
            u_prev = s_ * cdn(z_)
            counts.append(int(ni))
            tap = None                     # only tap the first step
        return z_, s_, counts, u_prev

    return rollout, cdn


# --------------------------------------------------------------- reduced path
def build_reduced(p, N, K_op, m):
    """Offline, once per cell.  All returned objects are rank x rank / rank /
    scalar -- independent of N except through the build."""
    rank = p['W_x'].shape[0]
    V = jnp.einsum('ri,rj->rij', p['W_x'], p['W_y']).reshape(rank, N * N)
    Y = V * m[None, :]                     # mask (*) V
    Yt = jax.vmap(K_op)(Y)                 # row r = K (mask (*) V_r)
    Km = K_op(m)
    return dict(
        A0=Y @ Y.T, A1=Y @ Yt.T, A2=Yt @ Yt.T,
        v0=Y @ m, v1=Y @ Km, v2=Yt @ m, v3=Yt @ Km,
        s0=m @ m, s1=m @ Km, s2=Km @ Km,
        bias=p['bias'], rank=rank,
        # kept only so the test can project u0; the real harness does the same
        Y=Y, Yt=Yt, m=m, Km=Km)


def make_reduced(p, R, k, IT, TOL):
    h_of_z = make_h(p)
    EYE = jnp.eye(k + 1)
    A0, A1, A2 = R['A0'], R['A1'], R['A2']
    v0, v1, v2, v3 = R['v0'], R['v1'], R['v2'], R['v3']
    s0, s1, s2 = R['s0'], R['s1'], R['s2']
    bias = R['bias']

    def kap_terms(kap):
        t = DT * kap
        Gw = A0 + t * (A1 + A1.T) + t * t * A2          # W W^T
        wg = v0 + t * (v1 + v2) + t * t * v3            # W g
        gg = s0 + 2 * t * s1 + t * t * s2               # g^T g
        return Gw, wg, gg, t

    def _gn_step(z_init, s_init, proj, kap, tap=None):
        """proj = (a, b, c0, c1, e) = (Y u_prev, Yt u_prev, m.u_prev,
        Km.u_prev, u_prev.u_prev)"""
        Gw, wg, gg, t = kap_terms(kap)
        a, b, c0, c1, e = proj
        Wup = a + t * b                                  # W u_prev   (rank)
        gup = c0 + t * c1                                # g^T u_prev (scalar)

        def normal_eqs(zs):
            z_, s_ = zs[:-1], zs[-1]
            h, Jh = jax.jvp, None
            h = h_of_z(z_)
            Jh = jax.jacfwd(h_of_z)(z_)                  # (rank, k)
            Gwh = Gw @ h
            qq = h @ Gwh + 2 * bias * (h @ wg) + bias * bias * gg
            Wq = Gwh + bias * wg                         # W q   (rank)
            qup = h @ Wup + bias * gup                   # q^T u_prev
            # J^T J
            top = Jh.T @ (Gw @ Jh)                       # (k, k) / s^2
            JtJ = jnp.zeros((k + 1, k + 1))
            JtJ = JtJ.at[:k, :k].set(s_ * s_ * top)
            cross = s_ * (Jh.T @ Wq)
            JtJ = JtJ.at[:k, k].set(cross).at[k, :k].set(cross)
            JtJ = JtJ.at[k, k].set(qq)
            # J^T r
            Jtr = jnp.zeros(k + 1)
            Jtr = Jtr.at[:k].set(s_ * (Jh.T @ (s_ * Wq - Wup)))
            Jtr = Jtr.at[k].set(s_ * qq - qup)
            r2 = s_ * s_ * qq - 2 * s_ * qup + e         # ||r||^2
            return JtJ, Jtr, r2

        zs0 = jnp.concatenate([z_init, jnp.asarray([s_init])])
        mu0 = 1e-4

        def _body(carry):
            zs, mu, _, itr = carry
            JtJ, Jtr, r2 = normal_eqs(zs)
            if tap is not None:
                tap.append((np.asarray(JtJ), np.asarray(Jtr)))
            diag_scale = jnp.mean(jnp.diag(JtJ)) + 1e-8
            A = JtJ + mu * diag_scale * EYE
            d = jnp.linalg.solve(A, -Jtr)
            zs_new = zs + d
            _, _, r2_new = normal_eqs(zs_new)
            accept = r2_new < r2
            zs_next = jnp.where(accept, zs_new, zs)
            zs_next = zs_next.at[-1].set(jnp.maximum(zs_next[-1], 1e-8))
            mu_next = jnp.where(accept, jnp.maximum(mu * 0.3, 1e-7),
                                jnp.minimum(mu * 10.0, 1e2))
            return zs_next, mu_next, jnp.linalg.norm(Jtr), itr + 1

        def _cond(carry):
            _, _, jtr_norm, itr = carry
            return (jtr_norm > TOL) & (itr < IT)

        carry = (zs0, mu0, jnp.inf, 0)
        while _cond(carry):
            carry = _body(carry)
        zs_f, _, _, itr_f = carry
        return zs_f[:-1], zs_f[-1], itr_f

    def advance_proj(z_, s_):
        """u_prev_new = s * (Y^T h + bias*m) -- project it without touching the
        grid."""
        h = h_of_z(z_)
        return (s_ * (A0 @ h + bias * v0),
                s_ * (A1.T @ h + bias * v2),
                s_ * (h @ v0 + bias * s0),
                s_ * (h @ v1 + bias * s1),
                s_ * s_ * (h @ (A0 @ h) + 2 * bias * (h @ v0)
                           + bias * bias * s0))

    def rollout(z0, s0, u0, kap, n_steps, tap=None):
        # one-time projection of the initial condition (O(rank * n), online)
        proj = (R['Y'] @ u0, R['Yt'] @ u0, R['m'] @ u0, R['Km'] @ u0, u0 @ u0)
        z_, s_ = z0, s0
        counts = []
        for _ in range(n_steps):
            z_, s_, ni = _gn_step(z_, s_, proj, kap, tap)
            proj = advance_proj(z_, s_)
            counts.append(int(ni))
            tap = None
        return z_, s_, counts

    return rollout


def main():
    N, k, rank, hid = 24, 6, 12, 16
    n_steps, IT, TOL = 8, 20, 1e-6
    K_op, m = make_ops(N)

    worst_jtj = worst_jtr = worst_z = worst_s = 0.0
    bad_counts = 0
    ncfg = 0
    for seed in range(12):
        key = jax.random.PRNGKey(seed)
        p = rand_weights(key, k, rank, hid, N)
        kap = float(0.05 + 0.4 * (seed % 5))
        z0 = 0.5 * jax.random.normal(jax.random.PRNGKey(1000 + seed), (k,))
        s0 = 1.0 + 0.1 * seed
        u0 = m * jax.random.normal(jax.random.PRNGKey(2000 + seed), (N * N,))

        dense_roll, cdn = make_dense(p, N, K_op, m, k, IT, TOL)
        R = build_reduced(p, N, K_op, m)
        red_roll = make_reduced(p, R, k, IT, TOL)

        tap_d, tap_r = [], []
        zd, sd, cd, _ = dense_roll(z0, s0, u0, kap, n_steps, tap=tap_d)
        zr, sr, cr = red_roll(z0, s0, u0, kap, n_steps, tap=tap_r)

        assert len(tap_d) == len(tap_r), (len(tap_d), len(tap_r))
        for (Jd, gd), (Jr, gr) in zip(tap_d, tap_r):
            worst_jtj = max(worst_jtj,
                            np.abs(Jd - Jr).max() / (np.abs(Jd).max() + 1e-30))
            worst_jtr = max(worst_jtr,
                            np.abs(gd - gr).max() / (np.abs(gd).max() + 1e-30))
        worst_z = max(worst_z, float(jnp.abs(zd - zr).max()
                                     / (jnp.abs(zd).max() + 1e-30)))
        worst_s = max(worst_s, float(abs(sd - sr) / (abs(sd) + 1e-30)))
        if cd != cr:
            bad_counts += 1
            print(f'  seed {seed}: GN counts differ {cd} vs {cr}')
        ncfg += 1

    print(f'configs                       : {ncfg}')
    print(f'max rel dev in J^T J          : {worst_jtj:.3e}')
    print(f'max rel dev in J^T r          : {worst_jtr:.3e}')
    print(f'max rel dev in final z        : {worst_z:.3e}')
    print(f'max rel dev in final s        : {worst_s:.3e}')
    print(f'configs with differing GN cnt : {bad_counts}/{ncfg}')


if __name__ == '__main__':
    main()
