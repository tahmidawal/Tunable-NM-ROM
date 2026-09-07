"""Scientific FOM verification. Run only in a GPU batch job, before any training."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
import jax
import jax.numpy as jnp
from fresh_fom import Grid, integrate, integrate_balance, energy, localized_initial, bump, provenance, damping_ratio
from test_fresh_fom import independent_matrices


def norm(q, grid):
    return float(np.sqrt(np.sum(grid.mass()*np.asarray(q)**2)))


def orders(rows, key):
    return [float(np.log2(rows[i][key]/rows[i+1][key])) for i in range(len(rows)-1)]


def propagate(grid, u, v, c, end, dt_max, balance=False, forcing=None):
    steps = int(np.ceil(end/dt_max))
    dt = end/steps
    fn = integrate_balance if balance else integrate
    kwargs = {} if balance else {"forcing": forcing}
    result = fn(jnp.asarray(u), jnp.asarray(v), c, dt, grid=grid, steps=steps, stride=steps, **kwargs)
    return tuple(np.asarray(a[-1]) for a in result), dt


def oblique_exact(t, grid, c):
    xy = jnp.asarray(grid.coordinates())
    kx, ky = 1.7*np.pi, .8*np.pi
    omega = c*np.sqrt(kx*kx+ky*ky)
    phase = kx*xy[..., 0]+ky*xy[..., 1]-omega*t
    return jnp.cos(phase), omega*jnp.sin(phase)


def oblique_forcing(t, grid, c):
    xy = jnp.asarray(grid.coordinates())
    kx, ky = 1.7*np.pi, .8*np.pi
    omega = c*jnp.sqrt(kx*kx+ky*ky)
    s = jnp.sin(kx*xy[..., 0]+ky*xy[..., 1]-omega*t)
    f = jnp.zeros(grid.shape, dtype=jnp.float64)
    f = f.at[0, :].add(c*(omega+c*kx)*s[0, :]*2/grid.h)
    f = f.at[-1, :].add(c*(omega-c*kx)*s[-1, :]*2/grid.h)
    f = f.at[:, 0].add(c*(omega+c*ky)*s[:, 0]*2/grid.h)
    f = f.at[:, -1].add(c*(omega-c*ky)*s[:, -1]*2/grid.h)
    return f


def mode_space(kind):
    rows = []
    for n in (16, 32, 64, 128):
        c, end = 1.07, .31
        if kind == "standing":
            grid = Grid(n)
            xy = grid.coordinates()
            mode = np.sin(np.pi*xy[..., 0])*np.sin(2*np.pi*xy[..., 1])
            omega = c*np.pi*np.sqrt(5)
            u0, v0 = .6*mode, .8*omega*mode
            ue = (.6*np.cos(omega*end)+.8*np.sin(omega*end))*mode
            ve = omega*(-.6*np.sin(omega*end)+.8*np.cos(omega*end))*mode
            forcing = None
        elif kind == "periodic":
            grid = Grid(n, "periodic", "periodic")
            xy = grid.coordinates()
            omega = c*2*np.pi*np.sqrt(2)
            phase = 2*np.pi*(xy[..., 0]+xy[..., 1])
            u0, v0 = np.cos(phase), omega*np.sin(phase)
            ue, ve = np.cos(phase-omega*end), omega*np.sin(phase-omega*end)
            forcing = None
        else:
            grid = Grid(n, "absorbing", "absorbing")
            u0, v0 = map(np.asarray, oblique_exact(0., grid, c))
            ue, ve = map(np.asarray, oblique_exact(end, grid, c))
            omega = c*np.pi*np.sqrt(1.7**2+.8**2)
            forcing = oblique_forcing
        (u, v), dt = propagate(grid, u0, v0, c, end, .08*grid.h/c, forcing=forcing)
        err = np.sqrt(omega**2*norm(u-ue, grid)**2+norm(v-ve, grid)**2)
        denom = np.sqrt(omega**2*norm(u0, grid)**2+norm(v0, grid)**2)
        rows.append({"n": n, "dt": dt, "u_error": norm(u-ue, grid), "v_error": norm(v-ve, grid), "joint_relative_error": float(err/denom)})
    return {"rows": rows, "joint_orders": orders(rows, "joint_relative_error")}


def time_and_stability():
    grid, c, end = Grid(5, "absorbing", "absorbing"), 1.1, .12
    m, k, damp = independent_matrices(grid, c)
    count = len(m)
    a = np.block([[np.zeros_like(m), np.eye(count)], [-np.linalg.solve(m, k), -np.linalg.solve(m, damp)]])
    rng = np.random.default_rng(690699)
    u0, v0 = rng.normal(size=(2, *grid.shape))
    # Deliberately concentrate large entries at corners and faces.
    u0[0, 0], v0[-1, -1], v0[0, 2] = 4., -6., 5.
    y0 = np.r_[u0.ravel(), v0.ravel()]
    ref = expm(end*a)@y0
    rows = []
    for steps in (6, 12, 24, 48):
        (u, v, q), dt = propagate(grid, u0, v0, c, end, end/steps, balance=True)
        err = np.linalg.norm(np.r_[u.ravel(), v.ravel()]-ref)/np.linalg.norm(ref)
        e0 = float(energy(jnp.asarray(u0), jnp.asarray(v0), grid, c))
        balance = (float(energy(jnp.asarray(u), jnp.asarray(v), grid, c))+float(q)-e0)/e0
        rows.append({"steps": steps, "dt": dt, "joint_relative_error": float(err), "energy_balance_relative": float(balance)})
    eig = np.linalg.eigvals(a)
    dt = .2*grid.h/c
    zz = dt*eig
    ampl = 1+zz+zz**2/2+zz**3/6+zz**4/24
    return {"rows": rows, "joint_orders": orders(rows, "joint_relative_error"), "rk4_amplification_max_at_cfl_0_2": float(np.max(abs(ampl))), "eigenvalue_real_max": float(np.max(eig.real))}


def plane_pulse(kind, meshes=(32, 64, 128)):
    rows = []
    c, end = 1., .9
    def f(x):
        return bump((x-.35)/.22)
    fp = jax.vmap(jax.grad(f))
    for n in meshes:
        grid = Grid(n, "dirichlet" if kind == "reflective" else "absorbing", "periodic")
        x = jnp.asarray(grid.axis(grid.bx))
        u0, v0 = f(x), -c*fp(x)
        xe = x-c*end
        ue, ve = f(xe), -c*fp(xe)
        if kind == "reflective":
            image_x = 2-x-c*end
            ue, ve = ue-f(image_x), ve+c*fp(image_x)
        tile = lambda z: np.broadcast_to(np.asarray(z)[:, None], grid.shape).copy()
        (u, v), dt = propagate(grid, tile(u0), tile(v0), c, end, .08*grid.h/c)
        denom = np.sqrt(norm(tile(v0), grid)**2 + (c/.22)**2*norm(tile(u0), grid)**2)
        err = np.sqrt(norm(v-tile(ve), grid)**2+(c/.22)**2*norm(u-tile(ue), grid)**2)/denom
        e0 = float(energy(jnp.asarray(tile(u0)), jnp.asarray(tile(v0)), grid, c))
        rows.append({"n": n, "dt": dt, "joint_relative_error": float(err), "u_error": norm(u-tile(ue), grid), "v_error": norm(v-tile(ve), grid), "remaining_energy_fraction": float(energy(jnp.asarray(u), jnp.asarray(v), grid, c))/e0,
                     "displacement_min": float(u.min()), "displacement_max": float(u.max())})
    return {"rows": rows, "joint_orders": orders(rows, "joint_relative_error")}


def fft_open_reference(n, parameters, time, length=4):
    """Independent continuum Fourier propagation in a 4x4 periodic box.

    Image separation exceeds propagation reach through t=2.4 at c<=1.15.
    The desired unit-square samples have offset 1.5 in the enlarged box.
    """
    size = length*n
    offset = (length-1)/2
    x = np.arange(size)/n-offset
    xx, yy = np.meshgrid(x, x, indexing="ij")
    cx, cy, sx, sy, amp, c, vx, vy = parameters[:8]
    def bn(r):
        out = np.zeros_like(r)
        inside = abs(r) < 1
        out[inside] = np.exp(1-1/(1-r[inside]**2))
        return out
    def dbn(r):
        out = np.zeros_like(r)
        inside = abs(r) < 1
        out[inside] = bn(r[inside])*(-2*r[inside]/(1-r[inside]**2)**2)
        return out
    rx, ry = (xx-cx)/sx, (yy-cy)/sy
    u0 = amp*bn(rx)*bn(ry)
    v0 = -c*amp*(vx*dbn(rx)*bn(ry)/sx+vy*bn(rx)*dbn(ry)/sy)
    if len(parameters) == 10:
        sigx, sigy = parameters[8:]
        factor = np.exp(-.5*(((xx-cx)/sigx)**2+((yy-cy)/sigy)**2))
        v0 = factor*(v0+c*u0*(vx*(xx-cx)/sigx**2+vy*(yy-cy)/sigy**2))
        u0 = factor*u0
    freq = 2*np.pi*np.fft.fftfreq(size, d=1/n)
    omega = c*np.sqrt(freq[:, None]**2+freq[None, :]**2)
    co, si = np.cos(omega*time), np.sin(omega*time)
    sinc = np.full_like(omega, time)
    np.divide(si, omega, out=sinc, where=omega != 0)
    uh, vh = np.fft.fft2(u0), np.fft.fft2(v0)
    u = np.fft.ifft2(co*uh+sinc*vh).real
    v = np.fft.ifft2(-omega*si*uh+co*vh).real
    begin = int(offset*n)
    return u[begin:begin+n+1, begin:begin+n+1], v[begin:begin+n+1, begin:begin+n+1]


def actual_family(out):
    rows, arrays = [], {}
    # Predetermined minimum/maximum width controls, independent of model outcomes.
    parameters = [( .5, .49, .27, .27, 1., 1.15, .5, -.5), (.48, .51, .34, .34, .9, .85, 0., 0.)]
    for bc in ("dirichlet", "absorbing"):
        for pi, p in enumerate(parameters):
            solutions = {}
            for n in (32, 64, 128):
                grid = Grid(n, bc, bc)
                u0, v0 = localized_initial(grid, p)
                # Four fixed observations, with a common dt dividing .05 exactly.
                stride = int(np.ceil(.05/(.12*grid.h/p[5])))
                dt = .05/stride
                us, vs, qs = integrate_balance(u0, v0, p[5], dt, grid=grid, steps=48*stride, stride=stride)
                us, vs, qs = map(np.asarray, (us, vs, qs))
                e = np.asarray(energy(jnp.asarray(us), jnp.asarray(vs), grid, p[5]))
                solutions[n] = (grid, us, vs, e)
                key = f"{bc}_{pi}_{n}"
                arrays[key+"_u"], arrays[key+"_v"] = us[[0, 7, 24, 48]], vs[[0, 7, 24, 48]]
                arrays[key+"_energy"], arrays[key+"_outflux"] = e, qs
                inv = np.sum(grid.mass()*(vs+np.asarray(damping_ratio(grid, p[5]))*us), axis=(-2, -1)) if bc == "absorbing" else None
                rows.append({"bc": bc, "parameter_index": pi, "parameters": list(p), "n": n, "dt": dt, "initial_energy": float(e[0]), "max_balance_relative": float(np.max(abs(e+qs-e[0]))/e[0]),
                             "invariant_drift": None if inv is None else float(np.max(abs(inv-inv[0]))), "final_energy_fraction": float(e[-1]/e[0])})
            for coarse, fine in ((32, 64), (64, 128)):
                g, u, v, e = solutions[coarse]
                _, uf, vf, _ = solutions[fine]
                # Dirichlet interior coarse nodes are fine indices 1,3,...; absorber 0,2,... .
                sl = slice(1, -1, 2) if bc == "dirichlet" else slice(None, None, 2)
                ur, vr = uf[:, sl, sl], vf[:, sl, sl]
                du, dv = jnp.asarray(u-ur), jnp.asarray(v-vr)
                state_errors = np.sqrt(np.maximum(0., np.asarray(energy(du, dv, g, p[5])))/e[0])
                displacement_scale = max(norm(u[0], g), 1e-12)
                uerr = np.sqrt(np.sum(g.mass()*(u-ur)**2, axis=(-2,-1)))/displacement_scale
                rows.append({"bc": bc, "parameter_index": pi, "coarse_n": coarse, "fine_n": fine, "max_energy_state_difference": float(np.max(state_errors)), "final_energy_state_difference": float(state_errors[-1]), "max_scaled_displacement_difference": float(np.max(uerr)), "state_difference_array": state_errors.tolist(), "displacement_difference_array": uerr.tolist()})
            if bc == "absorbing":
                g, u, v, e = solutions[128]
                for ti in (7, 24, 48):
                    ur, vr = fft_open_reference(128, p, ti*.05)
                    error = float(np.sqrt(max(0., float(energy(jnp.asarray(u[ti]-ur), jnp.asarray(v[ti]-vr), g, p[5])))/e[0]))
                    rows.append({"bc": bc, "parameter_index": pi, "n": 128, "time": ti*.05, "open_fft_energy_state_difference": error, "absorber_mean": float(np.sum(g.mass()*u[ti])), "open_fft_mean": float(np.sum(g.mass()*ur)), "interpretation": "absorber-model plus numerical error, not a discretization gate"})
    np.savez_compressed(out/"reference_arrays.npz", **arrays)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    meta = provenance()
    if meta["jax_backend"] != "gpu" or not meta["x64"] or meta["matmul_precision"] != "highest":
        raise RuntimeError("GPU/f64/highest required")
    result = {"provenance": meta, "reference_seed": 690699}
    for key, fn in [("standing", lambda: mode_space("standing")), ("periodic", lambda: mode_space("periodic")), ("manufactured_absorber", lambda: mode_space("manufactured")), ("temporal", time_and_stability), ("plane_reflection", lambda: plane_pulse("reflective")), ("plane_absorption", lambda: plane_pulse("absorbing")), ("actual_family", lambda: actual_family(out))]:
        print("verification", key, flush=True)
        result[key] = fn()
        (out/"result.json").write_text(json.dumps(result, indent=2)+"\n")
    gates = {"standing_space": min(result["standing"]["joint_orders"][-2:]) >= 1.7,
             "periodic_space": min(result["periodic"]["joint_orders"][-2:]) >= 1.7,
             "manufactured_space": min(result["manufactured_absorber"]["joint_orders"][-2:]) >= 1.7,
             "temporal": min(result["temporal"]["joint_orders"][-2:]) >= 3.5,
             "rk4_stability": result["temporal"]["rk4_amplification_max_at_cfl_0_2"] <= 1+1e-12,
             "plane_reflection_converges": result["plane_reflection"]["joint_orders"][-1] >= 1.5,
             "plane_absorption_converges": result["plane_absorption"]["joint_orders"][-1] >= 1.5}
    result["gates"], result["passed"] = gates, all(gates.values())
    (out/"result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"gates": gates, "passed": result["passed"]}), flush=True)
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
