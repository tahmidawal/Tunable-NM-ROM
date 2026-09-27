"""burgers3d-retry panel job (DESIGN.md): one mesh, one allocation, lean tables (tables.py).

mode 'panel'   : cohort references (kept on the host), every arm once per case (errors, iteration records, saved
                 restricted fields), the FOM grid once per case, then the timed phases A1 (ROM) - B (FOM) - A2 (ROM).
mode 'certify' : every arm on the certification populations; rho on every accepted state k = 1..50 (or 1..25).
Same record layout, gates and timing protocol as burgers3d-span/panel.py; output/result.json, output/fields/*.npz.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'burgers3d-span'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import tables as TB  # noqa: E402
from common import block  # noqa: E402


def load_model(model_dir, heads):
    b = pickle.loads((Path(model_dir) / 'bank.pkl').read_bytes())
    tj = lambda t: jax.tree_util.tree_map(jnp.asarray, t)
    out = dict(bank=tj(b['params']), T=np.asarray(b['rotation']), spread=b['coefficient_rms_spread'],
               sha={'bank.pkl': C.sha_file(Path(model_dir) / 'bank.pkl')}, heads={})
    for tag in heads:
        f = Path(model_dir) / f'head_{tag}.pkl'
        h = pickle.loads(f.read_bytes())
        z = np.asarray(h['codes'])
        H = np.asarray(h['library_H']) if 'library_H' in h else np.asarray(jax.jit(C.head)(tj(h['params']), jnp.asarray(z)))
        out['heads'][tag] = dict(params=tj(h['params']), Z=z, H=H,
                                 code_spread=float(np.sqrt(np.mean(np.sum((z - z.mean(0)) ** 2, axis=1)))))
        out['sha'][f.name] = C.sha_file(f)
    return out


def arm_name(a, gtol0):
    g = a.get('gtol', gtol0)
    base = f"span_R{a['Rp']}" if a['kind'] == 'span' else f"head{a['head']}_R{a['Rp']}"
    return base + f"_{a['solver']}_dt{a['dt']:g}" + ('_t32' if a.get('t32') else '') + ('' if g == gtol0 else f'_g{g:g}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    out = Path(a.out)
    (out / 'fields').mkdir(parents=True, exist_ok=True)
    if not cfg.get('local_smoke'):
        assert jax.default_backend() == 'gpu', jax.default_backend()
    assert jax.config.jax_enable_x64 and os.environ['JAX_DEFAULT_MATMUL_PRECISION'] == 'highest'
    print(f'jax_backend={jax.default_backend()} x64=True precision=highest', flush=True)
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)
    try:
        smi = subprocess.check_output(['nvidia-smi', '-L'], text=True).strip().splitlines()
    except Exception:  # noqa: BLE001
        smi = []
    n = int(cfg['mesh'])
    arms = {}
    for s in cfg['arms']:
        s = dict(s, gtol=s.get('gtol', cfg['gtol']))
        arms[arm_name(s, cfg['gtol'])] = dict(s, name=arm_name(s, cfg['gtol']), rule='tensor')
    heads_needed = sorted({str(x['head']) for x in arms.values() if x['kind'] == 'head'})
    model = load_model(a.model, heads_needed)
    rep = dict(config=cfg, mode=cfg['mode'], mesh=n, commit=os.environ.get('SOURCE_COMMIT'),
               job_id=os.environ.get('SLURM_JOB_ID'), backend=jax.default_backend(),
               gpu=jax.devices()[0].device_kind, nvidia_smi=smi, x64=True,
               precision=os.environ['JAX_DEFAULT_MATMUL_PRECISION'], jax_version=jax.__version__,
               model_sha256=model['sha'], gates={}, arms={}, fom={}, timing={}, certificates={}, complete=False,
               timing_contract=('GPU query: supplied dense initial interior field resident on the GPU -> six dense GPU '
                                'output fields, block_until_ready on the whole output; compilation excluded'))
    save = lambda: C.dump(out / 'result.json', C.clean(rep))
    save()
    log = lambda s: print(f'[{el()}s] {s}', flush=True)
    for k, v in model['sha'].items():
        want = cfg.get('expected_model_sha256', {}).get(k)
        assert want is None or want == v, (k, v, want)

    # ------------------------------------------------------------ tables + gates
    R = int(model['T'].shape[1])
    ladder = sorted(set(x['Rp'] for x in arms.values()) | {R})
    order, lam_all = C.mode_order(n)
    M_of = lambda r: C.complete_M(n, 4 * r, order, lam_all)
    t_tab = time.perf_counter()
    tb = TB.build_tables(n, model['bank'], model['T'], ladder, M_of, log=log)
    rep['table_seconds'] = time.perf_counter() - t_tab
    rep['gates']['gram_condition'] = tb['gram_cond']
    rng = np.random.default_rng(20260923)
    kx = tb['kxyz']
    pick = rng.choice(tb['M_max'], size=min(6, tb['M_max']), replace=False)
    nodes = np.stack(np.meshgrid(*([np.arange(1, n - 1)] * 3), indexing='ij'), -1).reshape(-1, 3)
    cols = np.sort(rng.choice(R, size=4, replace=False))
    e_rows = np.zeros((4, R))
    e_rows[np.arange(4), cols] = 1.
    Gc = np.asarray(TB.decode(jnp.asarray(e_rows), tb))                         # (4, N) bank columns
    worst = 0.
    for kk in pick:                                                              # explicit Phi, one mode at a time
        ph = C.phi_explicit(n, kx[kk:kk + 1], nodes)[:, 0]
        worst = max(worst, float(np.abs(Gc @ ph - np.asarray(tb['A'])[kk, cols]).max() / np.abs(Gc @ ph).max()))
    rep['gates']['A_dst_vs_explicit'] = worst
    del nodes, ph
    Lm = tb['L']
    rec = Lm @ Lm.T
    gm = jnp.asarray(Gc) @ jnp.asarray(Gc).T
    rep['gates']['cholesky_reconstruction'] = float(jnp.abs(rec[np.ix_(cols, cols)] - gm).max() / jnp.abs(gm).max())
    c = rng.normal(size=R) / np.sqrt(R)
    u = TB.decode(jnp.asarray(c)[None], tb)[0]
    idx = tuple(jnp.asarray(kx[:, j]) for j in range(3))
    direct = np.asarray(C.phiT(C.backward_adv(u, n)[None], n, idx)[0])
    via = np.asarray(0.5 * jnp.einsum('mij,i,j->m', tb['Tsym'], jnp.asarray(c), jnp.asarray(c)))
    rep['gates']['tensor_vs_direct'] = float(np.linalg.norm(via - direct) / np.linalg.norm(direct))
    del Gc, u
    log(f"table gates {rep['gates']}")
    assert rep['gates']['A_dst_vs_explicit'] < 1e-12 and rep['gates']['cholesky_reconstruction'] < 1e-12, rep['gates']
    assert rep['gates']['tensor_vs_direct'] < 1e-10, rep['gates']
    save()

    # ------------------------------------------------------------ arms
    data, queries, coefs, hps, cache = {}, {}, {}, {}, {}
    for name, s in arms.items():
        M = M_of(s['Rp']) if s['kind'] == 'span' else M_of(s['K'])
        s['M'] = int(M)
        hk = model['heads'].get(str(s.get('head'))) if s['kind'] == 'head' else None
        key = (s['kind'], s['Rp'], M, s.get('head'), bool(s.get('t32')))
        if key not in cache:
            cache[key] = TB.arm_data(tb, s['Rp'], M, t32=bool(s.get('t32')), head=hk)
        data[name] = cache[key]
        trust = cfg['trust_fraction'] * (model['spread'][str(s['Rp'])] if s['kind'] == 'span' else hk['code_spread'])
        s['trust'] = float(trust)
        assert M >= (4 * s['Rp'] if s['kind'] == 'span' else 4 * s['K']) and M <= tb['M_max'], (name, M)
        if s['solver'] == 'fsc':
            assert s['kind'] == 'span'
            queries[name], coefs[name] = TB.make_fsc(n, s['Rp'], M, dt=s['dt'], gtol=s['gtol'], trust=trust)
        else:
            assert s['solver'] == 'fsh' and s['kind'] == 'head' and not s.get('t32')
            queries[name], coefs[name] = TB.make_fsh(n, s['Rp'], M, s['K'], dt=s['dt'], gtol=s['gtol'], trust=trust)
        hps[name] = hk['params'] if hk is not None else {}
        rep['arms'][name] = dict(spec=s)
    del tb['Tsym']
    save()

    def run(name, u0, nu):
        return queries[name](u0, nu, data[name], hps[name])

    # ================================================================ certify
    if cfg['mode'] == 'certify':
        bar = cfg['rho_bar']
        for name, s in arms.items():
            cert = dict(draws=[], bar=bar)
            rho_fn = TB.make_rho(n, s['M'], kx)
            for di, (seed, count) in enumerate(cfg['cert_draws']):
                tab = C.table(seed, count)
                rows = []
                for j in range(count):
                    u0 = jnp.asarray(C.initial_interior(n, tab, j))
                    o = run(name, u0, float(tab['nu'][j]))
                    Cs = jax.vmap(lambda w: coefs[name](w, hps[name]))(o[1][1:])
                    rho, umin = map(np.asarray, rho_fn(Cs, data[name]))
                    rmax = float(rho.max()) if np.isfinite(rho).all() else float('inf')
                    rows.append(dict(row=j, rho=rho.tolist(), rho_max=rmax, umin=float(umin.min()),
                                     reasons=np.asarray(o[3]).tolist(), iterations=np.asarray(o[2]).tolist(),
                                     finite=bool(np.isfinite(np.asarray(o[0])).all())))
                    if j == 0 or rmax >= max(r['rho_max'] for r in rows):
                        np.savez(out / 'fields' / f'cert_{name}_d{di}.npz', coefs=np.asarray(Cs), rho=rho,
                                 seed=seed, row=j, arg=int(np.argmax(rho)))
                rm = max(r['rho_max'] for r in rows)
                cert['draws'].append(dict(seed=seed, count=count, table_sha256=tab['sha256'], rho_max=rm,
                                          passed=bool(rm <= bar), rows=rows, umin=min(r['umin'] for r in rows)))
                log(f'CERT {name} draw {di} seed {seed}: rho_max {rm:.4g} umin {cert["draws"][-1]["umin"]:.3g}')
            cert['confirmed'] = all(d['passed'] for d in cert['draws'])
            cert['rho_max_draws'] = [d['rho_max'] for d in cert['draws']]
            rep['certificates'][name] = cert
            save()
        rep['complete'] = True
        rep['seconds'] = el()
        save()
        print('PANEL COMPLETE', flush=True)
        return

    # ================================================================ panel
    tab = C.table(cfg['cohort_seed'], cfg['cohort_count'])
    rep['cohort'] = dict(seed=cfg['cohort_seed'], count=cfg['cohort_count'], table_sha256=tab['sha256'])
    want = cfg.get('expected_cohort_sha256')
    assert want is None or want == tab['sha256'], 'cohort differs'
    cases = list(range(cfg['cohort_count']))
    U0 = [block(jnp.asarray(C.initial_interior(n, tab, j))) for j in cases]
    NU = [float(tab['nu'][j]) for j in cases]
    ref_fom = C.make_fom(n, C.DT, cfg['ref_ntol'], cfg['ref_ltol'])
    REF, refrec = [], []
    for j in cases:
        f, it, rn = ref_fom(U0[j], NU[j])
        REF.append(np.asarray(f))                                                # host copy (large meshes)
        refrec.append(dict(max_newton=int(jnp.max(it)), max_rel_residual=float(jnp.max(rn)),
                           finite=bool(np.isfinite(REF[-1]).all())))
        del f
    rep['reference'] = dict(ntol=cfg['ref_ntol'], ltol=cfg['ref_ltol'], cases=refrec)
    rep['gates']['reference_residual'] = max(r['max_rel_residual'] for r in refrec)
    assert rep['gates']['reference_residual'] < 1e-9 and all(r['finite'] for r in refrec), refrec
    log(f'references done, worst residual {rep["gates"]["reference_residual"]:.2e}')
    ridx_np = C.restrict_index(n, cfg.get('audit_lattice', 16))
    ridx = jnp.asarray(ridx_np)
    errf = jax.jit(lambda f, r: jnp.linalg.norm(f - r, axis=1) / jnp.linalg.norm(r[0]))
    errr = jax.jit(lambda f, r, i: jnp.linalg.norm(f[:, i] - r[:, i], axis=1) / jnp.linalg.norm(r[0][i]))
    np.savez(out / 'fields' / 'reference_restricted.npz', **{f'c{j}': REF[j][:, ridx_np] for j in cases},
             u0=np.stack([np.asarray(U0[j][ridx]) for j in cases]))
    np.savez(out / 'fields' / 'reference_case0_full.npz', f=REF[0])
    audit_arms = set(cfg.get('audit_arms', []))
    save()

    def error_of(f, j):
        r = jnp.asarray(REF[j])
        e, er = np.asarray(errf(f, r)), np.asarray(errr(f, r, ridx))
        del r
        return e, er

    quick_restricted = {}
    for name, s in arms.items():
        recs = []
        t_c = time.perf_counter()
        for j in cases:
            o = run(name, U0[j], NU[j])
            block(o)
            if j == 0:
                compile_s = time.perf_counter() - t_c
            e, er = error_of(o[0], j)
            fr = np.asarray(o[0][:, ridx])
            quick_restricted[(name, j)] = fr
            recs.append(dict(case=j, err=e.tolist(), err_restricted=er.tolist(), worst_evolved=float(e[1:].max()),
                             iterations=np.asarray(o[2]).tolist(), reasons=np.asarray(o[3]).tolist(),
                             gradients=np.asarray(o[4]).tolist(), residuals=np.asarray(o[5]).tolist(),
                             rejected=np.asarray(o[6]).tolist(), initial_fit=np.asarray(o[7]).tolist(),
                             finite=bool(np.isfinite(np.asarray(o[0])).all())))
            np.savez(out / 'fields' / f'{name}_c{j}.npz', f=fr, internal=np.asarray(o[1]))
            if j == 0 and name in audit_arms:
                np.savez(out / 'fields' / f'{name}_c0_full.npz', f=np.asarray(o[0]))
            del o
        its = np.array([r['iterations'] for r in recs])
        rs = np.array([r['reasons'] for r in recs])
        rep['arms'][name].update(quick=recs, compile_and_first_s=compile_s,
                                 worst_evolved=max(r['worst_evolved'] for r in recs),
                                 median_worst_evolved=float(np.median([r['worst_evolved'] for r in recs])),
                                 lm_iterations_per_query_median=float(np.median(its.sum(1))),
                                 reason_counts={str(k): int((rs == k).sum()) for k in range(5)},
                                 all_finite=all(r['finite'] for r in recs))
        log(f"ARM {name}: worst evolved {rep['arms'][name]['worst_evolved']:.4%} "
            f"median {rep['arms'][name]['median_worst_evolved']:.4%} "
            f"its/query {rep['arms'][name]['lm_iterations_per_query_median']:.0f} "
            f"reasons {rep['arms'][name]['reason_counts']}")
        save()

    foms = {}
    for dt, nt, lt in cfg['fom_grid']:
        name = f'fom_dt{dt:g}_nt{nt:g}_lt{lt:g}'
        foms[name] = C.make_fom(n, dt, nt, lt)
        recs = []
        for j in cases:
            f, it, rn = foms[name](U0[j], NU[j])
            block(f)
            e, er = error_of(f, j)
            quick_restricted[(name, j)] = np.asarray(f[:, ridx])
            recs.append(dict(case=j, err=e.tolist(), err_restricted=er.tolist(), worst_evolved=float(e[1:].max()),
                             newton=np.asarray(it).tolist(), max_rel_residual=float(jnp.max(rn)),
                             hit_newton_cap=bool(int(jnp.max(it)) >= C.MAX_NEWTON),
                             finite=bool(np.isfinite(np.asarray(f)).all())))
            np.savez(out / 'fields' / f'{name}_c{j}.npz', f=quick_restricted[(name, j)])
            if j == 0 and name in audit_arms:
                np.savez(out / 'fields' / f'{name}_c0_full.npz', f=np.asarray(f))
            del f
        rep['fom'][name] = dict(dt=dt, ntol=nt, ltol=lt, quick=recs, worst_evolved=max(r['worst_evolved'] for r in recs),
                                median_worst_evolved=float(np.median([r['worst_evolved'] for r in recs])),
                                all_finite=all(r['finite'] for r in recs))
        log(f"FOM {name}: worst evolved {rep['fom'][name]['worst_evolved']:.4%}")
        save()
    del REF

    # -------------------------------------------------- timing A-B-A
    reps = cfg['reps']
    prng = np.random.default_rng(cfg.get('timing_seed', 923777))
    x = jnp.ones((2048, 2048))
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 2.0:
        block(x @ x)
    inv = []

    def timed(kind, name, j, phase):
        t = time.perf_counter()
        o = run(name, U0[j], NU[j]) if kind == 'rom' else foms[name](U0[j], NU[j])
        block(o)
        dt_ = time.perf_counter() - t
        d = float(jnp.max(jnp.abs(o[0][:, ridx] - jnp.asarray(quick_restricted[(name, j)]))))
        inv.append(dict(kind=kind, name=name, case=j, phase=phase, seconds=dt_, max_diff_vs_quick=d))
        if dt_ >= 0.5:
            time.sleep(min(1.0, dt_))

    for phase, kind, names in (('A1', 'rom', list(arms)), ('B', 'fom', list(foms)), ('A2', 'rom', list(arms))):
        order_ = [(nm, j) for nm in names for j in cases for _ in range(reps)]
        prng.shuffle(order_)
        for nm, j in order_:
            timed(kind, nm, j, phase)
        log(f'timing phase {phase} done ({len(order_)} invocations)')
    rep['timing']['invocations'] = inv
    tim = {}
    for nm in list(arms) + list(foms):
        rows = [r for r in inv if r['name'] == nm]
        tim[nm] = dict(median_ms=1e3 * float(np.median([r['seconds'] for r in rows])),
                       n=len(rows), max_diff_vs_quick=max(r['max_diff_vs_quick'] for r in rows))
        if nm in arms:
            a1 = [r['seconds'] for r in rows if r['phase'] == 'A1']
            a2 = [r['seconds'] for r in rows if r['phase'] == 'A2']
            tim[nm]['drift_A2_over_A1'] = float(np.median(a2) / np.median(a1))
    rep['timing']['summary'] = tim
    rom_inv = [r for r in inv if r['kind'] == 'rom']
    med = {}
    for r in rom_inv:
        med.setdefault((r['name'], r['case']), []).append(r['seconds'])
    med = {k: np.median(v) for k, v in med.items()}
    after_long, after_short = [], []
    for ph in ('A1', 'A2'):
        seq = [r for r in rom_inv if r['phase'] == ph]
        for prev, cur in zip(seq[:-1], seq[1:]):
            v = cur['seconds'] / med[(cur['name'], cur['case'])]
            (after_long if tim[prev['name']]['median_ms'] >= 4 * tim[cur['name']]['median_ms'] else after_short).append(v)
    drift = [v['drift_A2_over_A1'] for k, v in tim.items() if 'drift_A2_over_A1' in v]
    rep['gates']['timing_drift_worst'] = float(max(max(drift), 1 / min(drift)))
    rep['gates']['timing_drift_pass'] = bool(rep['gates']['timing_drift_worst'] <= 1.10)
    nb = float(np.mean(after_long) / np.mean(after_short)) if after_long and after_short else 1.0
    rep['gates']['timing_neighbour_ratio'] = nb
    fom_inv = [r for r in inv if r['kind'] == 'fom']
    fmed = {}
    for r in fom_inv:
        fmed.setdefault((r['name'], r['case']), []).append(r['seconds'])
    fmed = {k: np.median(v) for k, v in fmed.items()}
    fl, fs = [], []
    for prev, cur in zip(fom_inv[:-1], fom_inv[1:]):
        v = cur['seconds'] / fmed[(cur['name'], cur['case'])]
        (fl if tim[prev['name']]['median_ms'] >= 4 * tim[cur['name']]['median_ms'] else fs).append(v)
    nbf = float(np.mean(fl) / np.mean(fs)) if fl and fs else 1.0
    rep['gates']['timing_neighbour_ratio_fom'] = nbf
    rep['gates']['timing_neighbour_pass'] = bool(nb <= 1.10 and nbf <= 1.10)
    rep['gates']['deterministic_outputs'] = max(r['max_diff_vs_quick'] for r in inv)
    log(f"timing gates drift {rep['gates']['timing_drift_worst']:.3f} neighbour {nb:.3f} / {nbf:.3f} "
        f"determinism {rep['gates']['deterministic_outputs']:.2e}")
    for nm, v in tim.items():
        log(f'TIME {nm}: {v["median_ms"]:.2f} ms')
    rep['complete'] = True
    rep['seconds'] = el()
    save()
    print('PANEL COMPLETE', flush=True)


if __name__ == '__main__':
    main()
