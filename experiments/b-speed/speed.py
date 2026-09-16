"""b-speed: profile the frozen Burgers EQ query and measure the optimisation ladder.

One job, one GPU, one process. Every ROM arm and every full-order control is interleaved
in a randomized order; the incumbent is `arms.make_query` imported unmodified, so the
reference is the retained solver rather than a re-implementation of it. Nothing here
changes a budget, a tolerance, an initializer or a quadrature rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import engines as e
import iterative_paths as ip
import arms as A
import fast as F
import ladders


# ------------------------------------------------------------------ utils ---

def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def host(x):
    return jax.tree_util.tree_map(np.asarray, x)


def dump(p, x):
    Path(p).write_text(json.dumps(x, indent=2, allow_nan=False) + '\n')


def rel(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def worst_rel(a, b):
    """Per-output-time relative deviation, the parity statistic."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(max(rel(a[t], b[t]) for t in range(b.shape[0])))


def timed(fn, reps, burn):
    """Median of `reps` device-synchronised repetitions, all retained."""
    out = []
    value = None
    for _ in range(reps):
        e.burn(burn)
        t = time.perf_counter()
        value = fn()
        jax.block_until_ready(value)
        out.append(time.perf_counter() - t)
    return float(np.median(out)), out, value


def census(compiled):
    text = compiled.as_text()
    return dict(fusions=text.count('fusion('), custom_calls=text.count('custom-call('),
                whiles=text.count(' while('), conditionals=text.count(' conditional('),
                dots=text.count(' dot('), bytes_text=len(text))


def analyse(fn, *args):
    """FLOPs, bytes and the kernel census of one COMPILED program."""
    try:
        low = jax.jit(fn).lower(*args)
        comp = low.compile()
        ca = comp.cost_analysis()
        if isinstance(ca, (list, tuple)):
            ca = ca[0] if ca else {}
        out = {k: float(v) for k, v in dict(ca or {}).items()
               if isinstance(v, (int, float))}
        out.update(census(comp))
        return out
    except Exception as exc:                              # pragma: no cover - diagnostic
        return dict(error=f'{type(exc).__name__}: {exc}')


# -------------------------------------------------------------- micro loop ---

def micro(body, state, consts, T, reps, burn):
    """us per in-loop iteration from a two-point slope, which removes the fixed cost.

    Everything the body reads is a jit ARGUMENT: closing over the frozen operators would
    let XLA embed them as compile-time constants and specialise the loop, which is not
    what the real program does."""
    fn = jax.jit(lambda s, cs, n: jax.lax.fori_loop(0, n, lambda i, q: body(q, cs), s),
                 static_argnums=2)
    jax.block_until_ready(fn(state, consts, 4))
    lo, lo_all, _ = timed(lambda: fn(state, consts, T), reps, burn)
    hi, hi_all, _ = timed(lambda: fn(state, consts, 2 * T), reps, burn)
    return dict(per_iteration_us=(hi - lo) / T * 1e6, low_seconds=lo, high_seconds=hi,
                iterations_low=T, iterations_high=2 * T,
                low_repetitions=lo_all, high_repetitions=hi_all)


# ------------------------------------------------------------------- build ---

def build_mesh(params, Zold, K, R, L, cfg):
    bank = A.CoordBank(params, K, R)
    head = A.neural_head(params)
    M = int(cfg['test_multiplier'] * K)
    m = int(cfg['quadrature_multiplier'] * M)
    t0 = time.perf_counter()
    data, info = A.build_operators(bank, L, M, 'eq', Zcoef=Zold, m=m, eq_seed=cfg['eq_seed'],
                                   candidate_cap=cfg['candidate_cap'],
                                   fit_states=cfg['fit_states'], head=head)
    stride = max(1, len(Zold) // cfg['decoder_code_subsample'])
    cold, cinfo = A.build_cold(bank, head, np.asarray(Zold[::stride]), cfg['cold_axis_points'])
    trust = .01 * float(np.max(np.linalg.norm(Zold - Zold.mean(0), axis=1)))
    info.update(cold=cinfo, trust_radius=trust, M=M, m_points=m,
                total_setup_seconds=time.perf_counter() - t0,
                code_stride=int(stride), codes=int(len(Zold[::stride])))
    return data, cold, trust, M, m, info


# ----------------------------------------------------------------- profile ---

def profile(params, K, L, dt, m, trust, data, cold, u0, nu, cfg, strict):
    """S0: where the incumbent query's time goes, from device-synchronised sub-timers,
    in-loop microbenchmarks and the XLA cost analysis of the compiled artefacts."""
    reps, burn = cfg['profile_reps'], cfg['burn_seconds']
    out = dict(intervals=L)
    parts = F.make_incumbent_parts(params, K, L, dt, trust, **strict)
    init = jax.jit(parts['initialize'])
    evol = jax.jit(parts['evolve'])
    deco = jax.jit(parts['decode'])

    z, scale = jax.block_until_ready(init(u0, data, cold))
    internal = jax.block_until_ready(evol(z, nu, scale, data))
    Z = internal[::parts['stride']]
    jax.block_until_ready(deco(Z, data))

    phases = {}
    phases['initial_fit'] = timed(lambda: init(u0, data, cold), reps, burn)[:2]
    phases['evolution'] = timed(lambda: evol(z, nu, scale, data), reps, burn)[:2]
    phases['decode_six_fields'] = timed(lambda: deco(Z, data), reps, burn)[:2]
    fields = jax.block_until_ready(deco(Z, data))
    phases['host_transfer'] = timed(lambda: np.asarray(fields), reps, burn)[:2]
    whole = A.make_query(A.neural_head(params), K, L, dt, trust, 'eq', linear='gj', **strict)
    jax.block_until_ready(whole(u0, nu, data, cold))
    phases['whole_query'] = timed(lambda: whole(u0, nu, data, cold), reps, burn)[:2]
    out['phases'] = {k: dict(median_seconds=v[0], repetitions=v[1]) for k, v in phases.items()}
    out['phase_note'] = ('each phase is its own completed device computation, separately '
                         'compiled; none is obtained by subtracting another and their sum '
                         'is not the whole query')

    # marginal LM iteration cost, from unconditional fixed budgets (diagnostic only)
    rows = []
    for budget in cfg['budget_probe']:
        fb = F.make_fixed_budget_evolve(params, K, L, dt, trust, budget)
        jax.block_until_ready(fb(z, nu, scale, data))
        med, allr, _ = timed(lambda: fb(z, nu, scale, data), reps, burn)
        rows.append(dict(budget=budget, median_seconds=med, repetitions=allr))
        del fb
    b = np.array([r['budget'] for r in rows], dtype=float)
    t = np.array([r['median_seconds'] for r in rows], dtype=float)
    Amat = np.stack((np.ones_like(b), b), 1)
    coef, *_ = np.linalg.lstsq(Amat, t, rcond=None)
    pred = Amat @ coef
    out['budget_probe'] = dict(
        rows=rows, steps=int(round(.25 / dt)),
        fixed_seconds_per_step=float(coef[0] / round(.25 / dt)),
        marginal_seconds_per_iteration=float(coef[1] / round(.25 / dt)),
        r_squared=float(1 - np.sum((t - pred) ** 2) / max(np.sum((t - t.mean()) ** 2), 1e-300)),
        note='unconditional fixed-budget LM; capping iterations changes the answer and this '
             'is never a production arm')

    # in-loop component microbenchmarks
    T = cfg['micro_iterations']
    TL = cfg['micro_iterations_light']
    o0 = F.opts()
    ol = F.opts(lean=1)
    setup0, kernel0, assemble0, project0, rescale0 = F.make_weak(params, L, dt, o0, m)
    setupl, kernell, assemblel, projectl, rescalel = F.make_weak(params, L, dt, ol, m)
    tabl = F.build_tables(params, data, cold, ol)
    con0 = setup0(nu, data)
    conl = setupl(nu, data)
    p0 = project0(z, data, {})
    pl = rescalel(projectl(z, data, tabl), conl)

    def res0(q, cs):
        return assemble0(*kernel0(q, cs['data'], {}), cs['p0'], cs['con0'], cs['data'])

    def resl(q, cs):
        return assemblel(*kernell(q, cs['data'], cs['tabl']), cs['pl'], cs['conl'], cs['data'])

    def rj_jacfwd(q, cs):
        f = lambda w: res0(w, cs)
        return jax.jacfwd(f)(q).T @ f(q)

    def rj_linearize(q, cs):
        r, jvp = jax.linearize(lambda w: res0(w, cs), q)
        return jax.vmap(jvp, out_axes=1)(jnp.eye(K)).T @ r

    consts = dict(data=data, tabl=tabl, p0=p0, pl=pl, con0=con0, conl=conl)
    J0 = jax.jacfwd(lambda q: res0(q, consts))(z)
    H0 = J0.T @ J0
    H0 = jax.block_until_ready(H0 + 1e-6 * jnp.diag(jnp.diag(H0) + 1e-30))
    g0 = jax.block_until_ready(J0.T @ res0(z, consts))
    vec = jnp.asarray(np.linspace(1., 2., 64))

    def chain(n):
        def body(sv, cs):
            for _ in range(n):
                sv = jnp.sin(sv) * 1.0000001
            return sv
        return body

    mb = {}
    mb['loop_control_scalar'] = micro(lambda sv, cs: sv + 1.0, jnp.asarray(1.0), consts,
                                      TL, reps, burn)
    for n in (1, 4, 16):
        mb[f'elementwise_chain_{n}'] = micro(chain(n), vec, consts, TL, reps, burn)
    mb['head_h_of_z'] = micro(
        lambda q, cs: q + 1e-30 * A.sc.head(params, q)[:K], z, consts, T, reps, burn)
    mb['kernel_G5h_and_Ah'] = micro(
        lambda q, cs: q + 1e-30 * kernel0(q, cs['data'], {})[1][:K], z, consts, T, reps, burn)
    mb['residual'] = micro(lambda q, cs: q + 1e-30 * res0(q, cs)[:K], z, consts, T, reps, burn)
    mb['residual_and_jacfwd'] = micro(
        lambda q, cs: q + 1e-30 * rj_jacfwd(q, cs), z, consts, T, reps, burn)
    mb['residual_and_linearize'] = micro(
        lambda q, cs: q + 1e-30 * rj_linearize(q, cs), z, consts, T, reps, burn)
    mb['lean_kernel'] = micro(
        lambda q, cs: q + 1e-30 * kernell(q, cs['data'], cs['tabl'])[1][:K], z, consts,
        T, reps, burn)
    mb['lean_residual'] = micro(lambda q, cs: q + 1e-30 * resl(q, cs)[:K], z, consts, T, reps, burn)
    solves = dict(H0=H0)
    mb['gj_solve_16'] = micro(lambda sv, cs: sv + 1e-30 * e.gj_solve(cs['H0'], sv), g0,
                              solves, TL, reps, burn)
    for bs in (2, 4, 8):
        mb[f'gj_block{bs}_16'] = micro(
            lambda sv, cs, bs=bs: sv + 1e-30 * F.gj_block(cs['H0'], sv, bs), g0, solves,
            TL, reps, burn)
    out['microbenchmarks'] = mb

    # compiled-artefact census and cost analysis
    cost = {}
    cost['whole_query_incumbent'] = analyse(whole, u0, nu, data, cold)
    cost['initial_fit'] = analyse(parts['initialize'], u0, data, cold)
    cost['evolution'] = analyse(parts['evolve'], z, nu, scale, data)
    cost['decode_six_fields'] = analyse(parts['decode'], Z, data)
    cost['residual'] = analyse(lambda q, cs: res0(q, cs), z, consts)
    cost['residual_and_jacfwd'] = analyse(
        lambda q, cs: (res0(q, cs), jax.jacfwd(lambda w: res0(w, cs))(q)), z, consts)
    cost['residual_and_linearize'] = analyse(
        lambda q, cs: (lambda r, jvp: (r, jax.vmap(jvp, out_axes=1)(jnp.eye(K))))(
            *jax.linearize(lambda w: res0(w, cs), q)), z, consts)
    cost['lean_residual'] = analyse(lambda q, cs: resl(q, cs), z, consts)
    cost['gj_solve_16'] = analyse(lambda H, g: e.gj_solve(H, g), H0, g0)
    for bs in (2, 4, 8):
        cost[f'gj_block{bs}_16'] = analyse(
            lambda H, g, bs=bs: F.gj_block(H, g, bs), H0, g0)
    out['cost_analysis'] = cost
    del parts, init, evol, deco, whole, tabl
    jax.clear_caches()
    return out


# -------------------------------------------------------------------- main ---

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--reference', default='')
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    assert jax.default_backend() == 'gpu', jax.default_backend()
    assert jax.config.jax_enable_x64
    assert os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()

    ck = pickle.load(open(a.checkpoint, 'rb'))
    params = jax.tree_util.tree_map(jnp.asarray, host(ck['params']))
    Zold = np.asarray(ck['Z_tr'])
    K = int(Zold.shape[1])
    R = int(np.asarray(ck['params']['h_lin']).shape[1])
    dt = cfg['dt']
    strict = cfg['strict']

    report = dict(
        config=cfg, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
        backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, x64=True,
        matmul_precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'], jax_version=jax.__version__,
        checkpoint_sha256=sha_file(a.checkpoint), K=K, R=R,
        spatial_bank_frozen=True, network_weights_frozen=True, final_cohort_unopened=True,
        output_times=[0, .05, .1, .15, .2, .25],
        timing_contract=('supplied dense initial field on GPU to six dense GPU output '
                         'fields; same-invocation host transfers also measured; identical '
                         'for every arm; the incumbent is arms.make_query imported unmodified'),
        arm_declarations={k: dict(opts=v['opts'], declared_class=v['cls'], note=v['note'])
                          for k, v in ladders.ALL.items()},
        mesh_setup=[], parity=[], profile=[], invocations=[], throughput=[],
        declared_subjects=[], gates={}, verification=ip.verify(), complete=False)
    save = lambda: dump(out / 'result.json', report)
    save()

    physical = np.concatenate((e.params_draw(cfg['eval_seed'], cfg['eval_cases']),
                               e.params_draw(cfg['eval_fresh_seed'], cfg['eval_fresh_cases'])))
    report['physical_cases'] = physical.tolist()
    report['cohort_roles'] = (['opened development'] * cfg['eval_cases']
                              + ['fresh development'] * cfg['eval_fresh_cases'])
    save()

    order_rng = np.random.default_rng(cfg['order_seed'])
    names = list(cfg['arms'])
    for n in names:
        assert n in ladders.ARMS, n

    for L in cfg['meshes']:
        print('MESH', L, flush=True)
        data, cold, trust, M, m, minfo = build_mesh(params, Zold, K, R, L, cfg)
        minfo['intervals'] = L
        report['mesh_setup'].append(minfo)
        save()
        print('SETUP', L, round(minfo['total_setup_seconds'], 1), flush=True)

        host_inputs = [e.initial(L, ph) for ph in physical]
        inputs = [jax.device_put(x) for x in host_inputs]
        jax.block_until_ready(inputs)
        nus = [float(ph[4]) for ph in physical]

        incumbent = A.make_query(A.neural_head(params), K, L, dt, trust, 'eq',
                                 linear='gj', **strict)
        base = []
        for c in range(len(physical)):
            base.append(host(incumbent(inputs[c], nus[c], data, cold)))
        print('INCUMBENT', L, [int(sum(b[1])) for b in base], flush=True)

        # gate: the in-job incumbent against the retained abl01 fields
        if a.reference and L == cfg.get('reference_mesh'):
            devs = []
            for c in range(min(6, len(base))):
                f = Path(a.reference) / f'L{L}_a_neural_eq_case{c}_rep0.npz'
                if f.exists():
                    ref = np.load(f)['fields']
                    devs.append(dict(case=c, field_relative=worst_rel(base[c][0], ref),
                                     bitwise=bool(np.array_equal(np.asarray(base[c][0]), ref))))
            report['gates']['incumbent_vs_abl01'] = dict(
                cases=devs, worst=max([d['field_relative'] for d in devs], default=None),
                bar=1e-12, passed=bool(devs) and max(d['field_relative'] for d in devs) <= 1e-12)
            print('GATE abl01', report['gates']['incumbent_vs_abl01']['worst'], flush=True)
            save()

        # ------------------------------------------------------------- arms
        built = [dict(name='incumbent', query=incumbent, tab=None, opts=None,
                      cls='reference')]
        tables = {}

        def table_for(o):
            key = (bool(o['lean']), o['decode'].startswith('lean'), bool(o['nodot']))
            if key not in tables:
                tables[key] = F.build_tables(params, data, cold, o)
            return tables[key]

        for name in names:
            o = ladders.ARMS[name]
            t0 = time.perf_counter()
            tab = table_for(o)
            q = F.make_query(params, K, L, dt, m, trust, o, **strict)
            built.append(dict(name=name, query=q, tab=tab, opts=o,
                              cls=ladders.ALL[name]['cls'],
                              table_bytes=int(sum(x.nbytes for x in jax.tree_util.tree_leaves(tab))),
                              table_seconds=time.perf_counter() - t0))
            report['declared_subjects'].append(dict(intervals=L, name=name, method='rom',
                                                    opts=o, cls=ladders.ALL[name]['cls']))
        print('ARMS BUILT', L, len(built), flush=True)

        # parity, measured once per case outside every timer
        for b in built[1:]:
            row = dict(intervals=L, arm=b['name'], declared_class=b['cls'], cases=[])
            for c in range(len(physical)):
                v = host(b['query'](inputs[c], nus[c], data, cold, b['tab']))
                ref = base[c]
                row['cases'].append(dict(
                    case=c,
                    field_relative=worst_rel(v[0], ref[0]),
                    latent_relative=rel(v[7], ref[7]),
                    iterations_identical=bool(np.array_equal(v[1], ref[1])),
                    ic_iterations_identical=bool(int(v[5]) == int(ref[5])),
                    reasons_identical=bool(np.array_equal(v[3], ref[3])),
                    ic_reason_identical=bool(int(v[6]) == int(ref[6])),
                    field_bitwise=bool(np.array_equal(np.asarray(v[0], dtype=np.float64),
                                                      np.asarray(ref[0], dtype=np.float64)))))
            cs = row['cases']
            row.update(worst_field_relative=max(x['field_relative'] for x in cs),
                       worst_latent_relative=max(x['latent_relative'] for x in cs),
                       iterations_identical=all(x['iterations_identical'] and
                                                x['ic_iterations_identical'] for x in cs),
                       reasons_identical=all(x['reasons_identical'] and
                                             x['ic_reason_identical'] for x in cs),
                       bitwise=all(x['field_bitwise'] for x in cs))
            row['parity'] = bool(row['worst_field_relative'] <= cfg['parity_bar']
                                 and row['iterations_identical'] and row['reasons_identical'])
            report['parity'].append(row)
            print('PARITY', L, b['name'], f"{row['worst_field_relative']:.3e}",
                  row['parity'], flush=True)
            save()

        # ------------------------------------------------------------ profile
        if cfg.get('profile_mesh') == L:
            report['profile'].append(profile(params, K, L, dt, m, trust, data, cold,
                                             inputs[0], nus[0], cfg, strict))
            print('PROFILE done', round(time.perf_counter() - begin, 1), flush=True)
            save()

        # --------------------------------------------------------------- FOM
        foms = {}
        for fs in cfg['fom_settings']:
            key = fs['dt']
            if key not in foms:
                foms[key] = ip.make_fom(L, key, 'fft')
            report['declared_subjects'].append(dict(intervals=L, method='fom', **fs))

        subjects = [dict(kind='rom', name=b['name'], index=i) for i, b in enumerate(built)]
        subjects += [dict(kind='fom', name=fs['name'], setting=fs) for fs in cfg['fom_settings']]

        def invoke(sub, u, case):
            if sub['kind'] == 'rom':
                b = built[sub['index']]
                if b['name'] == 'incumbent':
                    return b['query'](u, nus[case], data, cold)
                return b['query'](u, nus[case], data, cold, b['tab'])
            fs = sub['setting']
            fn, pre = foms[fs['dt']]
            return fn(u, nus[case], fs['ntol'], fs['ltol'], *pre)

        t = time.perf_counter()
        for sub in subjects:
            jax.block_until_ready(invoke(sub, inputs[0], 0))
        print('WARMUP', L, round(time.perf_counter() - t, 1), flush=True)
        report.setdefault('compile_warmup', []).append(
            dict(intervals=L, seconds=time.perf_counter() - t, subjects=len(subjects)))
        save()

        truth = {}
        for fs in cfg['fom_settings']:
            if fs['name'] == cfg['accuracy_reference']:
                fn, pre = foms[fs['dt']]
                for c in range(len(physical)):
                    truth[c] = np.asarray(host(fn(inputs[c], nus[c], fs['ntol'], fs['ltol'], *pre))[0])

        artifacts = {}
        for rep in range(cfg['reps']):
            for case in range(len(physical)):
                for i in order_rng.permutation(len(subjects)):
                    sub = subjects[int(i)]
                    e.burn(cfg['burn_seconds'])
                    ht = time.perf_counter()
                    u = jax.device_put(np.array(host_inputs[case], copy=True))
                    jax.block_until_ready(u)
                    gt = time.perf_counter()
                    value = invoke(sub, u, case)
                    jax.block_until_ready(value)
                    gs = time.perf_counter() - gt
                    f = np.asarray(value[0])
                    hs = time.perf_counter() - ht
                    v = host(value)
                    assert np.isfinite(f).all()
                    h = sha_array(f)
                    key = (sub['name'], case, h)
                    if key not in artifacts:
                        fn_ = f"L{L}_{sub['name']}_case{case}_rep{rep}.npz"
                        extra = dict(internal_latents=v[7]) if sub['kind'] == 'rom' else {}
                        # The archived field is the exact nested-node restriction to
                        # `archive_max_intervals` when the mesh is finer, so a fine-mesh
                        # job does not park tens of GB on a 91%-full share; case 0 keeps
                        # the full-resolution field for every subject. The in-job parity
                        # statistic above is always computed on the FULL field.
                        cap = int(cfg.get('archive_max_intervals', 0)) or L
                        st = max(1, L // cap) if (L > cap and case not in
                                                  cfg.get('archive_full_cases', [0])) else 1
                        np.savez_compressed(out / fn_, fields=f[:, ::st, ::st],
                                            archive_stride=np.int64(st), **extra)
                        artifacts[key] = fn_
                        artifacts[('stride',) + key] = st
                    row = dict(intervals=L, case=case, cohort=report['cohort_roles'][case],
                               rep=rep, kind=sub['kind'], name=sub['name'], gpu_seconds=gs,
                               host_seconds=hs, output_bytes=int(f.nbytes), field_sha256=h,
                               artifact=artifacts[key], dtype=str(f.dtype),
                               archive_stride=int(artifacts.get(('stride',) + key, 1)),
                               iterations=v[1].tolist(), residuals=v[2].tolist(), finite=True)
                    if case in truth:
                        row['same_grid'] = e.errors(f.astype(np.float64), truth[case], L)
                    if sub['kind'] == 'rom':
                        reasons = v[3].tolist()
                        row.update(stop_reasons=reasons, ic_iterations=int(v[5]),
                                   ic_reason=int(v[6]), step_stationarity=v[8].tolist(),
                                   ic_stationarity=float(v[9]), ic_residual=float(v[10]),
                                   budget_exits=int(sum(1 for r in reasons if r == 0)),
                                   rejected_exits=int(sum(1 for r in reasons if r == 3)),
                                   completed=bool(all(r in (1, 2, 4) for r in reasons)
                                                  and int(v[6]) in (1, 2, 4)))
                    else:
                        row['nonlinear_converged'] = bool(
                            np.max(v[2]) <= sub['setting']['ntol'] * (1 + 1e-9))
                    report['invocations'].append(row)
            print('TIMED', L, rep, round(time.perf_counter() - begin, 1), flush=True)
            save()

        # ---------------------------------------------------------- throughput
        for name in cfg['throughput_arms']:
            o = ladders.ARMS[name]
            tab = table_for(o)
            q = F.make_parts(params, K, L, dt, m, trust, o, **strict)['query']
            batched = jax.jit(lambda U, NU, dd, cc, tt:
                              jax.vmap(lambda u, nu: q(u, nu, dd, cc, tt))(U, NU))
            U = jnp.stack([jnp.asarray(x) for x in inputs])
            NU = jnp.asarray(nus)
            jax.block_until_ready(batched(U, NU, data, cold, tab))
            med, allr, value = timed(lambda: batched(U, NU, data, cold, tab),
                                     cfg['reps'], cfg['burn_seconds'])
            fields = np.asarray(value[0])
            devs = [worst_rel(fields[c], base[c][0]) for c in range(len(physical))]
            report['throughput'].append(dict(
                intervals=L, arm=name, batch=int(len(physical)), median_seconds=med,
                repetitions=allr, per_query_seconds=med / len(physical),
                worst_field_relative_vs_incumbent=max(devs), per_case=devs,
                label='THROUGHPUT, not latency: a batched while_loop pays each batch its '
                      'worst-case iteration count, and the full-order controls were NOT '
                      'batched for comparison'))
            print('THROUGHPUT', L, name, round(med / len(physical) * 1e3, 3), flush=True)
            del batched, q
            save()

        del built, foms, data, cold, inputs, host_inputs, tables
        jax.clear_caches()

    report['checkpoint_sha256_after'] = sha_file(a.checkpoint)
    report['gates'].update(
        checkpoint_unchanged=report['checkpoint_sha256'] == report['checkpoint_sha256_after'],
        backend_gpu=report['backend'] == 'gpu', x64=True,
        precision_highest=report['matmul_precision'] == 'highest')
    report['elapsed_seconds'] = time.perf_counter() - begin
    report['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('B-SPEED COMPLETE', flush=True)


if __name__ == '__main__':
    main()
