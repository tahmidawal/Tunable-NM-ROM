"""ns2d_phase1.py -- Phase 1 driver: FOM gates and the hashed dataset (DESIGN.md).

Every gate is written as a NUMBER with its threshold and a `passed` flag into
output/result.json; nothing is asserted silently.  Fields needed by the NumPy-only audit
(audit_phase1.py) are saved as .npz beside the JSON.  The run is complete only when every
gate has been evaluated; `complete=false` with `smoke=true` on any local run.

    NS=64,128,256  MESH_NS=64,128,256,512  DT=2e-3  T=1.0  OUT_EVERY=20  TRAIN=512 DEV=64 SEALED=64
    INDEP_N=64 INDEP_STEPS=100  SMOKE=0  OUT=output
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp                                              # noqa: E402

import ns2d_fom as F                                                 # noqa: E402
import ns2d_indep as I                                               # noqa: E402

E = os.environ.get
NS = [int(v) for v in E('NS', '64,128,256').split(',')]
MESH_NS = [int(v) for v in E('MESH_NS', '64,128,256,512').split(',')]
DT = float(E('DT', '2e-3'))
T = float(E('T', '1.0'))
OUT_EVERY = int(E('OUT_EVERY', '20'))
N_TRAIN, N_DEV, N_SEALED = int(E('TRAIN', '512')), int(E('DEV', '64')), int(E('SEALED', '64'))
SEEDS = dict(train=int(E('SEED_TRAIN', '20260917')), dev=int(E('SEED_DEV', '20260918')),
             sealed=int(E('SEED_SEALED', '20260919')))
INDEP_N = int(E('INDEP_N', '64'))
INDEP_STEPS = int(E('INDEP_STEPS', '100'))
SMOKE = int(E('SMOKE', '0'))
OUT = Path(E('OUT', 'output'))
NTOL, LTOL = float(E('NTOL', '1e-11')), float(E('LTOL', '1e-9'))
TIMED8 = int(E('TIMED8', '8'))
NSTEPS = int(round(T / DT))
assert abs(NSTEPS * DT - T) < 1e-12 and NSTEPS % OUT_EVERY == 0


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def host(x):
    return jax.tree_util.tree_map(np.asarray, x)


def log(*a):
    print(*a, flush=True)


def dump(report):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


def gate(report, name, passed, **kw):
    kw['passed'] = bool(passed)
    report['gates'][name] = kw
    log(f'GATE {name}: {"PASS" if passed else "FAIL"} ' +
        ' '.join(f'{k}={v:.3e}' if isinstance(v, float) else f'{k}={v}'
                 for k, v in kw.items() if k != 'passed' and not isinstance(v, (list, dict))))


def git_commit():
    c = os.environ.get('SOURCE_COMMIT')
    if c:
        return c
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:                                                 # pragma: no cover
        return 'unknown'


# ------------------------------------------------------------------- gates ----

def gate_lap(report, N, rng):
    w = rng.standard_normal((N, N))
    w -= w.mean()
    psi = F.poisson(jnp.asarray(w), N)
    r = F.rel(-F.laplacian(psi, N), w)
    gate(report, f'F-LAP_N{N}', r <= 1e-12, N=N, rel=r, tol=1e-12)


def gate_jac(report, N, rng):
    """Conservation identities of J_A (normalised), order 2 vs the analytic J of a smooth
    pair, and the centred-difference negative control."""
    X, Y = F.coords(N)
    psi = np.sin(2 * np.pi * X) * np.cos(4 * np.pi * Y) + 0.3 * np.cos(2 * np.pi * (X + Y))
    z = np.cos(6 * np.pi * X) * np.sin(2 * np.pi * Y) + 0.5 * np.sin(4 * np.pi * X)
    # analytic J = psi_x z_y - psi_y z_x
    px = 2 * np.pi * np.cos(2 * np.pi * X) * np.cos(4 * np.pi * Y) - 0.6 * np.pi * np.sin(2 * np.pi * (X + Y))
    py = -4 * np.pi * np.sin(2 * np.pi * X) * np.sin(4 * np.pi * Y) - 0.6 * np.pi * np.sin(2 * np.pi * (X + Y))
    zx = -6 * np.pi * np.sin(6 * np.pi * X) * np.sin(2 * np.pi * Y) + 2 * np.pi * np.cos(4 * np.pi * X)
    zy = 2 * np.pi * np.cos(6 * np.pi * X) * np.cos(2 * np.pi * Y)
    Jex = px * zy - py * zx
    JA = np.asarray(F.arakawa(jnp.asarray(psi), jnp.asarray(z), N))
    JC = np.asarray(F.jac_centred(jnp.asarray(psi), jnp.asarray(z), N))
    den1 = np.sum(np.abs(JA))
    denz = np.sum(np.abs(z * JA))
    denp = np.sum(np.abs(psi * JA))
    ids = dict(sum_J=abs(JA.sum()) / den1, sum_zJ=abs((z * JA).sum()) / denz,
               sum_psiJ=abs((psi * JA).sum()) / denp)
    ctrl = abs((z * JC).sum()) / np.sum(np.abs(z * JC))
    anti = F.rel(np.asarray(F.arakawa(jnp.asarray(z), jnp.asarray(psi), N)), -JA)
    err = F.rel(JA, Jex)
    worst = max(ids.values())
    return dict(N=N, **ids, worst_identity=worst, antisymmetry_rel=anti, analytic_rel=err,
                centred_control_zJ=ctrl)


def gate_tg(report, N, nu=1e-2):
    w0, lam_h, lam_c = F.taylor_green(N)
    run, _ = F.make_fom(N, DT, NSTEPS, OUT_EVERY)
    states, it, rn = host(run(jnp.asarray(w0), nu, 1e-13, 1e-11))
    ex = F.tg_discrete(w0, lam_h, nu, DT, NSTEPS)
    e_exact = F.rel(states[-1], ex)
    e_semi = F.rel(states[-1], w0 * np.exp(-nu * lam_h * T))
    e_cont = F.rel(states[-1], w0 * np.exp(-nu * lam_c * T))
    pred_cont = abs(np.exp(-nu * T * lam_h) - np.exp(-nu * T * lam_c)) / np.exp(-nu * T * lam_c)
    # negative controls: wrong lambda (4/3) and backward Euler, against the SAME closed form
    ex_wrong = F.tg_discrete(w0, lam_h * 4.0 / 3.0, nu, DT, NSTEPS)
    e_wrong = F.rel(states[-1], ex_wrong)
    run_be, _ = F.make_fom(N, DT, NSTEPS, OUT_EVERY, scheme='be')
    st_be, _, _ = host(run_be(jnp.asarray(w0), nu, 1e-13, 1e-11))
    e_be = F.rel(st_be[-1], ex)
    np.savez_compressed(OUT / f'tg_N{N}.npz', final=states[-1], w0=w0, lam_h=lam_h, nu=nu,
                        dt=DT, nsteps=NSTEPS, be_final=st_be[-1])
    return dict(N=N, exact_rel=e_exact, semi_rel=e_semi, cont_rel=e_cont, cont_pred=pred_cont,
                cont_pred_dev=abs(e_cont - pred_cont) / pred_cont, wrong_lambda_rel=e_wrong,
                be_rel=e_be, newton_max_it=int(it.max()), newton_worst_rn=float(rn.max()))


def gate_mms(report, N, nu, dt, nsteps, jac_sign=1.0):
    omega_fn, f_fn = F.mms_on_grid(N, nu)
    run, _ = F.make_fom(N, dt, nsteps, nsteps, f_fn=f_fn, jac_sign=jac_sign)
    w0 = np.asarray(omega_fn(0.0))
    states, it, rn = host(run(jnp.asarray(w0), nu, 1e-13, 1e-11))
    ex = np.asarray(omega_fn(dt * nsteps))
    return F.rel(states[-1], ex), states[-1], ex, int(it.max()), float(rn.max())


def gate_budget(report, N, phys):
    """Discrete enstrophy/energy identities per step, and exact conservation at nu=0."""
    w0 = F.initial(N, phys)
    nu = float(phys[12])
    nst = 100 if not SMOKE else 20
    run, _ = F.make_fom(N, DT, nst, 1)
    states, it, rn = host(run(jnp.asarray(w0), nu, 1e-13, 1e-11))
    Z = np.array([float(F.enstrophy(jnp.asarray(s), N)) for s in states])
    Eng = np.array([float(F.energy(jnp.asarray(s), N)) for s in states])
    worst_z, worst_e = 0.0, 0.0
    for n in range(nst):
        wm = 0.5 * (states[n] + states[n + 1])
        lap = np.asarray(F.laplacian(jnp.asarray(wm), N))
        psim = np.asarray(F.poisson(jnp.asarray(wm), N))
        dZ = DT * nu * np.sum(wm * lap) / (N * N)
        dE = DT * nu * np.sum(psim * lap) / (N * N)
        worst_z = max(worst_z, abs((Z[n + 1] - Z[n]) - dZ) / abs(dZ))
        worst_e = max(worst_e, abs((Eng[n + 1] - Eng[n]) - dE) / abs(dE))
    # nu = 0: exact conservation, with backward Euler as the negative control
    st0, _, _ = host(run(jnp.asarray(w0), 0.0, 1e-13, 1e-11))
    Z0 = np.array([float(F.enstrophy(jnp.asarray(s), N)) for s in st0])
    E0 = np.array([float(F.energy(jnp.asarray(s), N)) for s in st0])
    cons_z = float(np.max(np.abs(Z0 - Z0[0])) / Z0[0])
    cons_e = float(np.max(np.abs(E0 - E0[0])) / E0[0])
    run_be, _ = F.make_fom(N, DT, nst, 1, scheme='be')
    stb, _, _ = host(run_be(jnp.asarray(w0), 0.0, 1e-13, 1e-11))
    Zb = np.array([float(F.enstrophy(jnp.asarray(s), N)) for s in stb])
    ctrl = float(np.max(np.abs(Zb - Zb[0])) / Zb[0])
    np.savez_compressed(OUT / f'budget_N{N}.npz', states=states, nu=nu, dt=DT, states_nu0=st0,
                        states_be_nu0=stb)
    return dict(N=N, nu=nu, steps=nst, identity_enstrophy=worst_z, identity_energy=worst_e,
                conservation_enstrophy=cons_z, conservation_energy=cons_e,
                be_control_enstrophy=ctrl, newton_max_it=int(it.max()),
                newton_worst_rn=float(rn.max()))


def gate_mesh(report, phys, nu, tag):
    """Mesh refinement of the family: error at t=T against the finest mesh (restricted)."""
    finals = {}
    for N in MESH_NS:
        w0 = F.initial(N, phys)
        run, _ = F.make_fom(N, DT, NSTEPS, NSTEPS)
        st, it, rn = host(run(jnp.asarray(w0), nu, NTOL, LTOL))
        finals[N] = st[-1]
        log(f'   mesh {tag} N={N} newton max {int(it.max())} worst rn {float(rn.max()):.2e}')
    Nf = MESH_NS[-1]
    errs = []
    for N in MESH_NS[:-1]:
        s = Nf // N
        errs.append(F.rel(finals[N], finals[Nf][::s, ::s]))
    orders = F.observed_order(errs)
    np.savez_compressed(OUT / f'mesh_{tag}.npz', **{f'N{N}': v for N, v in finals.items()},
                        nu=nu)
    return dict(tag=tag, nu=nu, meshes=MESH_NS, errors_vs_finest=errs, orders=orders)


def gate_indep(report, phys):
    N = INDEP_N
    nu = float(phys[12])
    w0 = F.initial(N, phys)
    run, _ = F.make_fom(N, DT, INDEP_STEPS, 1)
    t0 = time.time()
    sj, it, rn = host(run(jnp.asarray(w0), nu, 1e-13, 1e-11))
    tj = time.time() - t0
    t0 = time.time()
    sn, itn, rnn = I.run_np(w0, nu, DT, N, INDEP_STEPS, ntol=1e-13)
    tn = time.time() - t0
    per = [F.rel(sj[k], sn[k]) for k in range(INDEP_STEPS + 1)]
    np.savez_compressed(OUT / 'indep.npz', jax=sj, numpy=sn, nu=nu, dt=DT)
    return dict(N=N, steps=INDEP_STEPS, nu=nu, worst_rel=float(max(per)), per_step_rel=per,
                jax_seconds=tj, numpy_seconds=tn, jax_newton_max=int(it.max()),
                numpy_newton_max=int(itn.max()), numpy_worst_rn=float(rnn.max()))


def generate(report, N, name, phys_all):
    run, _ = F.make_fom(N, DT, NSTEPS, OUT_EVERY)
    nout = NSTEPS // OUT_EVERY + 1
    U = np.empty((len(phys_all), nout, N * N))
    worst_rn, max_it, cfl = 0.0, 0, 0.0
    t0 = time.time()
    for i, phys in enumerate(phys_all):
        w0 = F.initial(N, phys)
        st, it, rn = host(run(jnp.asarray(w0), float(phys[12]), NTOL, LTOL))
        assert np.isfinite(st).all()
        U[i] = st.reshape(nout, -1)
        worst_rn = max(worst_rn, float(rn.max()))
        max_it = max(max_it, int(it.max()))
        c, _ = F.cfl_of(st, N, DT)
        cfl = max(cfl, c)
    sec = time.time() - t0
    info = dict(N=N, cohort=name, trajectories=len(phys_all), states=nout, dt=DT, T=T,
                ntol=NTOL, ltol=LTOL, worst_rel_residual=worst_rn, newton_max_it=max_it,
                max_cfl=cfl, seconds=sec, sha256=sha(U), physical_sha256=sha(phys_all))
    log(f'   data N={N} {name}: {len(phys_all)} traj, {sec:.0f}s, cfl {cfl:.2f}, '
        f'newton max {max_it}, rn {worst_rn:.1e}, sha {info["sha256"][:12]}')
    return U, info


def main():
    t_begin = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    backend = jax.default_backend()
    report = dict(lane='ns2d', phase=1, commit=git_commit(), job_id=os.environ.get('SLURM_JOB_ID'),
                  backend=backend, x64=bool(jax.config.jax_enable_x64),
                  matmul_precision=os.environ.get('JAX_DEFAULT_MATMUL_PRECISION'),
                  jax_version=jax.__version__, host=platform.node(),
                  device=str(jax.devices()[0]), smoke=bool(SMOKE),
                  config=dict(NS=NS, MESH_NS=MESH_NS, DT=DT, T=T, OUT_EVERY=OUT_EVERY,
                              NSTEPS=NSTEPS, N_TRAIN=N_TRAIN, N_DEV=N_DEV, N_SEALED=N_SEALED,
                              SEEDS=SEEDS, NTOL=NTOL, LTOL=LTOL, INDEP_N=INDEP_N,
                              INDEP_STEPS=INDEP_STEPS, MODES=F.MODES.tolist(),
                              NU_LO=F.NU_LO, NU_HI=F.NU_HI),
                  gates={}, data={}, complete=False)
    dump(report)
    gate(report, 'S0', backend == 'gpu' and jax.config.jax_enable_x64
         and os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest' or bool(SMOKE),
         backend=backend, precision=str(os.environ.get('JAX_DEFAULT_MATMUL_PRECISION')))
    rng = np.random.default_rng(1)

    phys = {k: F.params_draw(SEEDS[k], n) for k, n in
            (('train', N_TRAIN), ('dev', N_DEV), ('sealed', N_SEALED))}
    for a in phys:
        for b in phys:
            if a < b:
                assert not any(np.allclose(x, y) for x in phys[a] for y in phys[b]), (a, b)
    report['data']['physical_sha256'] = {k: sha(v) for k, v in phys.items()}
    report['data']['nu_range_dev'] = [float(phys['dev'][:, 12].min()), float(phys['dev'][:, 12].max())]

    # ---- operator gates at every N
    for N in NS:
        gate_lap(report, N, rng)
        j = gate_jac(report, N, rng)
        gate(report, f'F-JAC_N{N}', j['worst_identity'] <= 1e-13 and j['antisymmetry_rel'] <= 1e-13
             and j['centred_control_zJ'] >= 1e-3, **j)
    # analytic-J order over the NS ladder
    errs = [report['gates'][f'F-JAC_N{N}']['analytic_rel'] for N in NS]
    orders = F.observed_order(errs) if len(NS) > 1 else []
    gate(report, 'F-JAC-ORDER', all(abs(o - 2) <= 0.1 for o in orders) if orders else True,
         errors=errs, orders=orders)
    dump(report)

    # ---- Taylor-Green
    for N in NS:
        t = gate_tg(report, N)
        gate(report, f'F-TG_N{N}', t['exact_rel'] <= 1e-10 and t['semi_rel'] <= 1e-6
             and t['wrong_lambda_rel'] >= 1e-4 and t['be_rel'] >= 1e-4
             and t['cont_pred_dev'] <= 0.02, **t)
        dump(report)
    if len(NS) > 1:
        e = [report['gates'][f'F-TG_N{N}']['cont_rel'] for N in NS]
        o = F.observed_order(e)
        gate(report, 'F-TG-ORDER', all(abs(x - 2) <= 0.05 for x in o), errors=e, orders=o)

    # ---- MMS: spatial order at fixed small dt, temporal order at N=NS[1] (or NS[0])
    nu_mms = 1e-2
    dt_s = DT / 4 if not SMOKE else DT
    nst_s = int(round(0.2 / dt_s))
    sp_err = []
    for N in NS:
        e, fin, ex, it, rn = gate_mms(report, N, nu_mms, dt_s, nst_s)
        sp_err.append(e)
        np.savez_compressed(OUT / f'mms_space_N{N}.npz', final=fin, exact=ex, dt=dt_s, nsteps=nst_s)
        log(f'   mms N={N} dt={dt_s:.1e} err {e:.3e} newton max {it} rn {rn:.1e}')
    sp_ord = F.observed_order(sp_err) if len(NS) > 1 else []
    Nt = NS[min(1, len(NS) - 1)]
    tm_err = []
    for k in range(3):
        d = DT / 2 ** k
        e, fin, ex, it, rn = gate_mms(report, Nt, nu_mms, d, int(round(0.2 / d)))
        # subtract the spatial error using the finest-dt run as the reference below
        tm_err.append((d, e, fin))
    # temporal order from differences between successive dt runs (spatial error cancels)
    fins = [x[2] for x in tm_err]
    dtt = [F.rel(fins[0], fins[2]), F.rel(fins[1], fins[2])]
    tm_ord = float(np.log(dtt[0] / dtt[1]) / np.log(2.0)) if dtt[1] > 0 else float('nan')
    # the (dt - dt/2) vs (dt/2 - dt/4) ratio is 4 for order 2 given 3 levels:
    d01, d12 = F.rel(fins[0], fins[1]), F.rel(fins[1], fins[2])
    tm_ord3 = float(np.log(d01 / d12) / np.log(2.0))
    ctrl, _, _, _, _ = gate_mms(report, NS[0], nu_mms, dt_s, nst_s, jac_sign=-1.0)
    gate(report, 'F-MMS', (all(abs(o - 2) <= 0.05 for o in sp_ord) if sp_ord else True)
         and abs(tm_ord3 - 2) <= 0.1 and ctrl >= 1e-2,
         spatial_errors=sp_err, spatial_orders=sp_ord, temporal_N=Nt,
         temporal_dts=[x[0] for x in tm_err], temporal_errors=[x[1] for x in tm_err],
         temporal_diffs=[d01, d12], temporal_order=tm_ord3, temporal_order_vs_finest=tm_ord,
         flipped_sign_control=ctrl)
    dump(report)

    # ---- budgets
    for N in NS:
        b = gate_budget(report, N, phys['dev'][0])
        gate(report, f'F-BUDGET_N{N}', b['identity_enstrophy'] <= 1e-10 and b['identity_energy'] <= 1e-10
             and b['conservation_enstrophy'] <= 1e-9 and b['conservation_energy'] <= 1e-9
             and b['be_control_enstrophy'] >= 1e-4, **b)
        dump(report)

    # ---- mesh refinement of the family: dev case with nu closest to 1e-2 and to 1e-3
    if len(MESH_NS) >= 3:
        for tag, target in (('re100', 1e-2), ('re1000', 1e-3)):
            i = int(np.argmin(np.abs(np.log(phys['dev'][:, 12]) - np.log(target))))
            p = phys['dev'][i].copy()
            p[12] = target
            m = gate_mesh(report, p, target, tag)
            ok = all(abs(o - 2) <= 0.1 for o in m['orders']) if tag == 're100' \
                else abs(m['orders'][-1] - 2) <= 0.15
            gate(report, f'F-MESH_{tag}', ok, dev_index=i, **m)
            dump(report)

    # ---- independent implementation
    d = gate_indep(report, phys['dev'][1])
    gate(report, 'F-INDEP', d['worst_rel'] <= 1e-10,
         **{k: v for k, v in d.items() if k != 'per_step_rel'})
    report['gates']['F-INDEP']['per_step_rel'] = d['per_step_rel']
    dump(report)

    # ---- the dataset: hashes at every N; the 8 timed dev cases saved at every N
    for N in NS:
        for name in ('train', 'dev', 'sealed'):
            U, info = generate(report, N, name, phys[name])
            report['data'][f'{name}_N{N}'] = info
            ok = info['worst_rel_residual'] <= NTOL and info['max_cfl'] <= 2.0
            gate(report, f'F-DATA_{name}_N{N}', ok, worst_rel_residual=info['worst_rel_residual'],
                 max_cfl=info['max_cfl'], newton_max_it=info['newton_max_it'])
            if name == 'dev':
                np.savez_compressed(OUT / f'dev{TIMED8}_N{N}.npz', U=U[:TIMED8],
                                    physical=phys['dev'][:TIMED8], dt=DT, out_every=OUT_EVERY)
            del U
            dump(report)

    report['all_passed'] = all(g['passed'] for g in report['gates'].values())
    report['complete'] = not SMOKE
    report['seconds'] = time.time() - t_begin
    dump(report)
    log(f'PHASE1 done in {report["seconds"]:.0f}s; all_passed={report["all_passed"]}; '
        f'failed={[k for k, g in report["gates"].items() if not g["passed"]]}')


if __name__ == '__main__':
    main()
