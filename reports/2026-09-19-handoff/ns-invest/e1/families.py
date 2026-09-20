"""families.py -- the candidate 2D NS families of experiment E1 (64^2 pre-screen).

Every family: draw(seed, count) -> params (count, P) drawn COLUMN BY COLUMN (the params_draw
convention: the first n rows of a count=512 draw are the training set, so the set extends
consistently); initial(N, p) -> (omega_0 (N,N), nu, uv (2,), fp (3,)).
Training params = draw(20260917, 512)[:NTR], held-out = draw(20260918, 64)[:NDEV] for every family
(the lane's seeds, so family A / E reproduce the lane's cohorts bit for bit).

  A  control   the lane's frozen 12-amplitude band-limited family (ns2d_fom.params_draw / initial)
  B  dipole    two opposite-sign Gaussian vortices (core sigma = 0.08, circulations +-Gamma) at
               separation d, oriented to travel along angle theta, on a uniform background flow of
               speed U_b and angle phi; the dipole self-propels at ~Gamma/(2 pi d) in direction theta
  C  kolmo     Kolmogorov flow, forcing wavenumber KF=4 in the vorticity equation with amplitude
               A = nu (2 pi KF)^3 (laminar velocity amplitude 1, Re = 1/nu), stripes drifting with a
               Galilean frame of speed U_d, angle alpha; IC = laminar profile + a velocity
               perturbation eps [cos(2 pi x + chi1) x-hat-mode + cos(2 pi (x+y) + chi2)]
  D  kh        periodic double shear layer u = tanh((y-1/4)/delta), tanh((3/4-y)/delta), thickness
               delta, perturbed by v = eps1 sin(2 pi x + chi1) + eps2 sin(4 pi x + chi2)
  E  threemode the lane's 3-mode family (DESIGN A9 / ns303: params_draw(nmodes=3))
nu is log-uniform in [1e-3, 1e-2] (Re 100-1000 at unit velocity) for all families."""
import numpy as np
import fom_ext
F = fom_ext.F
PI = np.pi
NU_LO, NU_HI = 1e-3, 1e-2
SIGMA_B = 0.08
KF = fom_ext.KF

# name: (T, dt, nsteps, out_every, description of the parameter columns)
META = {
    'A': dict(T=1.0, dt=2e-3, nsteps=500, out_every=20, forced=False,
              cols=['a1..a6', 'b1..b6', 'nu'], P=13),
    'B': dict(T=1.5, dt=2e-3, nsteps=750, out_every=30, forced=False,
              cols=['Gamma~U[0.5,1.5]', 'd~U[0.15,0.25]', 'theta~U[0,2pi)', 'U_b~U[0,0.5]', 'phi~U[0,2pi)', 'nu'], P=6),
    'C': dict(T=1.0, dt=2e-3, nsteps=500, out_every=20, forced=True,
              cols=['U_d~U[0.5,1.5]', 'alpha~U[0,2pi)', 'eps~U[0.2,0.6]', 'chi1~U[0,2pi)', 'chi2~U[0,2pi)', 'nu'], P=6),
    'D': dict(T=1.0, dt=2e-3, nsteps=500, out_every=20, forced=False,
              cols=['delta~U[0.03,0.06]', 'eps1~logU[0.02,0.1]', 'chi1~U[0,2pi)', 'r=eps2/eps1~U[0,0.5]', 'chi2~U[0,2pi)', 'nu'], P=6),
    'E': dict(T=1.0, dt=2e-3, nsteps=500, out_every=20, forced=False,
              cols=['a1..a3', 'b1..b3', 'nu'], P=7),
    # ---- E1 variant (run 3): the three levers on B.  B2 = sharper cores (sigma 0.06), narrower Re
    # (nu in [1e-3, 3e-3], Re 330-1000), NO background flow (4 params).  B3 = the same sigma and Re
    # range but WITH the background flow (6 params): the control arm that isolates the parameter-count lever.
    'B2': dict(T=1.5, dt=2e-3, nsteps=750, out_every=30, forced=False, sigma=0.06, nu_hi=3e-3,
               cols=['Gamma~U[0.5,1.5]', 'd~U[0.15,0.25]', 'theta~U[0,2pi)', 'nu~logU[1e-3,3e-3]'], P=4),
    'B3': dict(T=1.5, dt=2e-3, nsteps=750, out_every=30, forced=False, sigma=0.06, nu_hi=3e-3,
               cols=['Gamma~U[0.5,1.5]', 'd~U[0.15,0.25]', 'theta~U[0,2pi)', 'U_b~U[0,0.5]', 'phi~U[0,2pi)', 'nu~logU[1e-3,3e-3]'], P=6),
}


def _nu(r, count, hi=NU_HI):
    return np.exp(r.uniform(np.log(NU_LO), np.log(hi), count))


def draw(fam, seed, count):
    if fam == 'A':
        return F.params_draw(seed, count, 6)
    if fam == 'E':
        return F.params_draw(seed, count, 3)
    r = np.random.default_rng(seed)
    if fam in ('B', 'B3'):
        cols = [r.uniform(0.5, 1.5, count), r.uniform(0.15, 0.25, count), r.uniform(0, 2 * PI, count),
                r.uniform(0.0, 0.5, count), r.uniform(0, 2 * PI, count)]
    elif fam == 'B2':
        cols = [r.uniform(0.5, 1.5, count), r.uniform(0.15, 0.25, count), r.uniform(0, 2 * PI, count)]
    elif fam == 'C':
        cols = [r.uniform(0.5, 1.5, count), r.uniform(0, 2 * PI, count), r.uniform(0.2, 0.6, count),
                r.uniform(0, 2 * PI, count), r.uniform(0, 2 * PI, count)]
    elif fam == 'D':
        cols = [r.uniform(0.03, 0.06, count), np.exp(r.uniform(np.log(0.02), np.log(0.1), count)),
                r.uniform(0, 2 * PI, count), r.uniform(0.0, 0.5, count), r.uniform(0, 2 * PI, count)]
    else:
        raise ValueError(fam)
    cols.append(_nu(r, count, META[fam].get('nu_hi', NU_HI)))
    return np.stack(cols, 1)


def _gauss_torus(X, Y, x0, y0, sigma):
    dx = (X - x0 + 0.5) % 1.0 - 0.5
    dy = (Y - y0 + 0.5) % 1.0 - 0.5
    return np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))


def initial(fam, N, p):
    """-> omega_0 (N, N) float64, nu, uv (2,), fp (3,) = [A_forcing, phase, V_drift]"""
    X, Y = F.coords(N)
    p = np.asarray(p, float)
    uv, fp = np.zeros(2), np.zeros(3)
    if fam in ('A', 'E'):
        return F.initial(N, p), float(p[-1]), uv, fp
    if fam in ('B', 'B2', 'B3'):
        if fam == 'B2':
            G, d, th, nu = p; Ub, ph = 0.0, 0.0
        else:
            G, d, th, Ub, ph, nu = p
        sig = META[fam].get('sigma', SIGMA_B)
        n = np.array([-np.sin(th), np.cos(th)])
        x1, x2 = 0.5 + 0.5 * d * n, 0.5 - 0.5 * d * n
        g1 = _gauss_torus(X, Y, x1[0], x1[1], sig)
        g2 = _gauss_torus(X, Y, x2[0], x2[1], sig)
        h2 = 1.0 / (N * N)
        w = G * (g1 / (g1.sum() * h2) - g2 / (g2.sum() * h2))    # discrete circulations exactly +-Gamma
        uv = Ub * np.array([np.cos(ph), np.sin(ph)])
        return w - w.mean(), float(nu), uv, fp
    if fam == 'C':
        Ud, al, eps, c1, c2, nu = p
        uv = Ud * np.array([np.cos(al), np.sin(al)])
        A = nu * (2 * PI * KF) ** 3
        w = (2 * PI * KF) * np.cos(2 * PI * KF * Y) \
            + 2 * PI * eps * (np.cos(2 * PI * X + c1) + np.sqrt(2.0) * np.cos(2 * PI * (X + Y) + c2))
        fp = np.array([A, 0.0, uv[1]])
        return w - w.mean(), float(nu), uv, fp
    if fam == 'D':
        de, e1, c1, rr, c2, nu = p
        e2 = rr * e1
        lower = Y < 0.5
        uy = np.where(lower, 1.0 / np.cosh((Y - 0.25) / de) ** 2 / de, -1.0 / np.cosh((0.75 - Y) / de) ** 2 / de)
        vx = 2 * PI * e1 * np.cos(2 * PI * X + c1) + 4 * PI * e2 * np.cos(4 * PI * X + c2)
        w = vx - uy
        return w - w.mean(), float(nu), uv, fp
    raise ValueError(fam)
