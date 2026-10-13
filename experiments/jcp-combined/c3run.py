"""jcp-combined 3D driver (DESIGN 3.2, 3.3, 8; A1.0, A1.2, A1.6, A1.7, A2.4, A2.5, A2.11, A3.4). Modes:

  esc   (`c3desc`) J4 escalation and rule-selection data. Per setting: Gauss rules of the escalation (W768/W1024: 48^3,
        56^3, 64^3; others 56^3, 64^3) under the two fixed-sweep selection solver arms (BE dt0 K=1; LSPG-CN 2.5 dt0 K=1)
        and, for W768/W1024, the two adaptive diagnostic arms (LSPG-BE dt0 tight, LSPG-CN 2.5 dt0 tight). In-job decision
        (A2.4): converged rule = first of 56^3, 64^3 whose distance to the next finer rule is <= conv_bar under both
        selection arms on every case with every run verified; Gauss 80^3 is run (all arms of the setting) only if 56^3
        fails. Operator-level check on common states. Then the ladder (lattice + Gauss + control lat256) under both
        selection arms; an A-B-A panel of every ladder rule's two selection arms (selection itself is offline).
  grid  (`c3grid`) per setting with its frozen deployed rule: the vendor fixed-sweep BE (G-FS1), fixed-sweep BE and
        CN/CN-R/BDF2 (K = 1, 2) on the step ladder, adaptive LSPG probes (tight and tighter) of every fixed-sweep scheme
        and step, adaptive GAL-CN, anchors GAL-BDF2 at dt0/8 and dt0/16 (tight, tighter); per-setting A-B-A panels and
        one common panel (A2.11).
Saved: W (6, R') of every run per case (W_<mode>.npz, keys setting|rule|spec), w0 per setting, per-step records, refined
errors per time on the 63^3 lattice (in-job; score.py recomputes), result.json.

    python c3run.py --config configs/<attempt>.json --out <dir> [--dry-run]
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import c3common as C3
from c3common import C, TB, OM, DT0, jax, jnp, sha


def grid_specs(g):
    sp = [('vendor',)]
    for f in g['dt_factors']:
        dt = DT0 * f
        sp.append(('fs', 'BE', dt, 1))
        for sc in g['fs_schemes']:
            for K in g['Ks']:
                sp.append(('fs', sc, dt, K))
        for sc in ['BE'] + g['fs_schemes']:
            for lev in ('tight', 'tighter'):
                sp.append(('ad', 'LSPG', sc, dt, lev))
    for f in g['gal_factors']:
        sp.append(('ad', 'GAL', 'CN', DT0 * f, 'tight'))
    for f in g['anchor_factors']:
        for lev in ('tight', 'tighter'):
            sp.append(('ad', 'GAL', 'BDF2', DT0 * f, lev))
    return sp


SEL = [('fs', 'BE', DT0, 1), ('fs', 'CN', 2.5 * DT0, 1)]
DIAG = [('ad', 'LSPG', 'BE', DT0, 'tight'), ('ad', 'LSPG', 'CN', 2.5 * DT0, 'tight')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    mode = cfg['mode']
    if a.dry_run:
        inv = {}
        for s in cfg['settings']:
            if mode == 'grid':
                sp = grid_specs(cfg['grid'])
                inv[s['name']] = dict(specs=len(sp), steps=sum(25 if x[0] == 'vendor' else int(round(.25 / x[2] if x[0] == 'fs' else .25 / x[3])) for x in sp))
            else:
                inv[s['name']] = dict(esc_rules=s['esc_rules'], ladder=s['ladder'])
        print(json.dumps(inv, indent=1))
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    C3.check_env(cfg)
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)
    n = int(cfg['mesh'])
    rep = dict(config=cfg, mode=mode, commit=C3.os.environ.get('SOURCE_COMMIT'), job_id=C3.os.environ.get('SLURM_JOB_ID'),
               backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, nvidia_smi=C3.gpu_info(),
               jax_version=jax.__version__, mesh=n, offmesh_sha256=C3.sha_file(OM.__file__),
               fs3_sha256=C3.sha_file(C3.HERE / 'fs3.py'), settings={}, rows=[], timing={}, complete=False)
    save = lambda: C.dump(out / 'result.json', C.clean(rep))
    tab, cases, REF = C3.load_cohort(cfg, rep)
    rules = C3.load_rules()
    i65 = jnp.asarray(C3.lattice65_index(n))
    U0 = {j: jax.block_until_ready(jnp.asarray(C.initial_interior(n, tab, j))) for j in cases}
    NU = {j: float(tab['nu'][j]) for j in cases}
    N0 = {j: float(jnp.linalg.norm(U0[j])) for j in cases}
    N0R = {j: float(np.linalg.norm(REF[j][0])) for j in cases}
    for j in cases:
        assert float(np.abs(np.asarray(U0[j])[np.asarray(i65)] - REF[j][0]).max()) <= 1e-12 * max(1., np.abs(REF[j][0]).max())
    tables, banks = {}, {}
    for bname in sorted({s['bank'] for s in cfg['settings']}):
        banks[bname] = C3.load_bank(bname)
        tables[bname] = C3.Tables(n, banks[bname], sorted({int(s['Rp']) for s in cfg['settings'] if s['bank'] == bname}),
                                  log=lambda s_: print(s_, flush=True))
    print('SETUP', el(), flush=True)
    Wstore, w0store = {}, {}
    live = {}                     # setting -> (Solvers, {rule: data}) kept for timing panels
    acc_sha = {}

    def rollouts(sname, sol, rule, data, specs, Lt, store=True):
        """Run specs on every case; record rows; return {speckey: (cases,6,R') array} and verified flags."""
        res, ver = {}, {}
        for spec in specs:
            sk = C3.spec_key(spec)
            arr = np.full((len(cases), 6, sol.Rp), np.nan)
            vs = []
            t_s = time.perf_counter()
            for ci, j in enumerate(cases):
                t1 = time.perf_counter()
                o, W, w0 = sol.run(spec, U0[j], NU[j], data)
                jax.block_until_ready(o)
                secs = time.perf_counter() - t1
                W = np.asarray(W)
                arr[ci] = W
                w0 = np.asarray(w0)
                if sname not in w0store:
                    w0store[sname] = np.full((len(cases), sol.Rp), np.nan)
                if np.isnan(w0store[sname][ci]).all():
                    w0store[sname][ci] = w0
                assert np.array_equal(w0, w0store[sname][ci]), 'initial coefficients differ across runs'
                v = C3.verified(spec, o)
                vs.append(v)
                F = W @ np.asarray(live_G65[sname])
                e = np.linalg.norm(F - REF[j], axis=1) / N0R[j]
                rep['rows'].append(dict(setting=sname, rule=rule, spec=sk, case=j, verified=v, seconds_first=secs,
                                        e_ref_per_time=e.tolist(), e_ref=float(e[1:].max()), steps=C3.step_record(spec, o)))
                if ci < cfg['timing']['cases']:
                    acc_sha[(sname, rule, sk, ci)] = sha(W)
                del o
            res[sk], ver[sk] = arr, vs
            if store:
                Wstore[f'{sname}|{rule}|{sk}'] = arr
            print('RUN', sname, rule, sk, f'{time.perf_counter() - t_s:.1f}s', 'verified', sum(vs), '/', len(vs), el(), flush=True)
        return res, ver

    def dist(A_, B_, Lt):
        """worst over cases of max over evolved times of ||G (a - b)|| / ||u0|| (Gram-Cholesky field metric, C1)."""
        # Gram G G^T = L L^T  =>  ||G^T c|| = ||L^T c||; for row vectors c^T L = (A - B) @ L = (A - B) @ Lt.T (as C1 w3d)
        d = np.linalg.norm(np.einsum('cki,ji->ckj', A_ - B_, Lt), axis=2) / np.array([N0[j] for j in cases])[:, None]
        return float(d[:, 1:].max()), d[:, 1:].max(1).tolist()

    tc = cfg['timing']
    tcases = cases[:tc['cases']]

    def time_panel(label, sname, sol, A, Bs, data_of, sol_of=None, ncases=None, reps=None):
        """A-B-A panel. A, Bs: (rule, spec) (or (setting, rule, spec) with sol_of for the common panel)."""
        def unpack(x):
            if sol_of is None:
                return sname, sol, x[0], x[1]
            return x[0], sol_of(x[0]), x[1], x[2]

        def one(x, ci):
            sn, so, rule, spec = unpack(x)
            j = tcases[ci]
            t_, h_ = _timed(so, spec, U0[j], NU[j], data_of(sn, rule) if sol_of else data_of(rule), tc['burn'])
            return t_, h_, acc_sha.get((sn, rule, C3.spec_key(spec), ci))
        name = lambda x: '|'.join([unpack(x)[0], unpack(x)[2], C3.spec_key(unpack(x)[3])])
        nc = ncases or len(tcases)
        nr = reps or tc['reps']
        for x in [A] + Bs:                      # warm every subject on every timing case
            for ci in range(nc):
                one(x, ci)
        rng = np.random.default_rng(int(tc['seed']) + len(rep['timing']))
        inv = []
        for rr in range(nr):
            order_ = [(b, ci) for b in range(len(Bs)) for ci in range(nc)]
            rng.shuffle(order_)
            for b, ci in order_:
                ta1, h1, e1 = one(A, ci)
                tb, hb, eb = one(Bs[b], ci)
                ta2, h2, _ = one(A, ci)
                inv.append(dict(B=name(Bs[b]), case=tcases[ci], rep=rr, tA1=ta1, tB=tb, tA2=ta2, ratio=tb / (.5 * (ta1 + ta2)),
                                drift=ta2 / ta1, A_sha=[h1, h2], A_expected=e1, B_sha=hb, B_expected=eb))
        rep['timing'][label] = dict(A=name(A), subjects=[name(x) for x in Bs], invocations=inv, cases=tcases[:nc],
                                    reps=nr, burn=tc['burn'], gpu=C3.gpu_info())
        save()
        print('TIMING', label, len(inv), el(), flush=True)

    def common_panel(base, subj):
        A = (base, live[base][1], ('fs', 'BE', DT0, 1))
        Bs = [x for x in subj if x != A]
        time_panel('common', None, None, A, Bs, lambda sn, rule: live[sn][2], sol_of=lambda sn: live[sn][0],
                   ncases=tc['common_cases'], reps=tc['common_reps'])

    live_G65 = {}
    X65 = np.stack(np.meshgrid(*(np.arange(1, 64) / 64.,) * 3, indexing='ij'), -1).reshape(-1, 3)
    for s in cfg['settings']:
        sname, bname, Rp, a_ = s['name'], s['bank'], int(s['Rp']), int(s['a'])
        bk, tb = banks[bname], tables[bname]
        trust = float(cfg['trust_fraction'] * bk['spread'][str(Rp)])
        sol = C3.Solvers(n, Rp, trust, cfg['gtol'], a_)
        lean = tb.lean(Rp)
        Lt = np.asarray(lean['L']).T
        live_G65[sname] = TB.feature_rows(bk['params'], bk['T'][:, :Rp], X65)
        S_rep = rep['settings'][sname] = dict(bank=bname, Rp=Rp, M=tb.M_of[Rp], trust=trust, a=a_)
        datas = {}

        def data_of(rule):
            if rule not in datas:
                X, w = rules[rule]
                datas[rule] = tb.rule_data(Rp, X, w)
            return datas[rule]

        if mode == 'esc':
            esc = {}
            specs = SEL + (DIAG if s.get('diagnostic') else [])
            for rule in s['esc_rules']:
                esc[rule] = rollouts(sname, sol, rule, data_of(rule), specs, Lt)
            dd = {}

            def pair(r1, r2):
                out_ = {}
                for spec in specs:
                    sk = C3.spec_key(spec)
                    worst, per = dist(esc[r1][0][sk], esc[r2][0][sk], Lt)
                    ok = all(esc[r1][1][sk]) and all(esc[r2][1][sk])
                    out_[sk] = dict(worst=worst, per_case=per, all_verified=ok,
                                    pass_bar=bool(ok and worst <= cfg['conv_bar']))
                return out_
            rr = s['esc_rules']
            for i in range(len(rr) - 1):
                dd[f'{rr[i]}|{rr[i + 1]}'] = pair(rr[i], rr[i + 1])
            sel_ok = lambda pr: all(pr[C3.spec_key(x)]['pass_bar'] for x in SEL)
            conv = None
            if 'gl56|gl64' in dd and sel_ok(dd['gl56|gl64']):
                conv = 'gl56'
            else:
                esc['gl80'] = rollouts(sname, sol, 'gl80', data_of('gl80'), specs, Lt)
                dd['gl64|gl80'] = pair('gl64', 'gl80')
                if sel_ok(dd['gl64|gl80']):
                    conv = 'gl64'
            S_rep['escalation'] = dict(pairs=dd, converged=conv, bar=cfg['conv_bar'],
                                       status='pass' if conv else 'unresolved')
            # operator-level check on common states (W768/W1024): reached states of the finest adaptive BE run
            if s.get('diagnostic'):
                fin = 'gl80' if 'gl80' in esc else 'gl64'
                Cs = esc[fin][0][C3.spec_key(DIAG[0])].reshape(-1, Rp)
                Cs = jnp.asarray(Cs[np.isfinite(Cs).all(1)])
                Nf = OM.adv_offmesh_batch(data_of(fin), Cs)
                opc = {}
                for rule in esc:
                    if rule == fin:
                        continue
                    Nr = OM.adv_offmesh_batch(data_of(rule), Cs)
                    den = jnp.linalg.norm(Nf, axis=1)
                    rel = jnp.where(den > 0, jnp.linalg.norm(Nr - Nf, axis=1) / jnp.maximum(den, 1e-300), jnp.nan)
                    opc[f'{rule}|{fin}'] = dict(worst=float(jnp.nanmax(rel)), median=float(jnp.nanmedian(rel)),
                                                states=int(Cs.shape[0]))
                S_rep['operator_check'] = opc
            save()
            if conv is not None:
                ladder = s['ladder']
                for rule in ladder:
                    if rule in esc:
                        continue
                    rollouts(sname, sol, rule, data_of(rule), SEL, Lt)
                # A-B-A panel of every ladder rule's two selection arms (A = conv rule's fixed BE dt0)
                panel_subjects = [(rule, spec) for rule in ladder + [conv] for spec in SEL]
                time_panel(f'family|{sname}', sname, sol, (conv, SEL[0]), panel_subjects, data_of)
            for rule in list(datas):
                datas.pop(rule)
            gc.collect()
        else:
            rule = s['rule']
            specs = grid_specs(cfg['grid'])
            res, ver = rollouts(sname, sol, rule, data_of(rule), specs, Lt)
            # G-FS1 (A2.1): fs BE dt0 K=1 vs vendor
            fv, vv = res[C3.spec_key(('fs', 'BE', DT0, 1))], res['vendor|BE|0.01|K1']
            rel = float(np.nanmax(np.abs(fv - vv)) / np.nanmax(np.abs(vv)))
            S_rep['G_FS1'] = dict(coef_max_rel=rel, exact_coefficients=bool(rel <= 1e-12))
            S_rep['G_FS1']['steps_equal'] = all(
                ra['steps']['reason'] == rb['steps']['reason'] and ra['steps']['it'] == rb['steps']['it']
                for ra, rb in zip([r for r in rep['rows'] if r['setting'] == sname and r['spec'] == 'fs|BE|0.01|K1'],
                                  [r for r in rep['rows'] if r['setting'] == sname and r['spec'] == 'vendor|BE|0.01|K1']))
            live[sname] = (sol, rule, data_of(rule))
            fs_c = [x for x in specs if x[0] == 'fs' and x != ('fs', 'BE', DT0, 1)] + [('vendor',)]
            time_panel(f'setting|{sname}', sname, sol, (rule, ('fs', 'BE', DT0, 1)), [(rule, x) for x in fs_c],
                       lambda r_: live[sname][2])
        save()
        np.savez_compressed(out / f'W_{mode}.npz', **Wstore, **{f'w0|{k}': v for k, v in w0store.items()})

    if mode == 'grid':
        base = cfg['common_baseline'] if cfg['common_baseline'] in live else cfg['common_baseline_fallback']
        subj = []
        for sname, (sol, rule, data) in live.items():
            for x in grid_specs(cfg['grid']):
                if x[0] == 'fs' and x[2] >= DT0 - 1e-12:
                    subj.append((sname, rule, x))
        common_panel(base, subj)
    rep['elapsed_seconds'] = el()
    rep['complete'] = True
    save()
    np.savez_compressed(out / f'W_{mode}.npz', **Wstore, **{f'w0|{k}': v for k, v in w0store.items()})
    (out / 'COMPLETE').write_text('complete\n')
    print('C3RUN COMPLETE', flush=True)


# timing helpers are defined at module level but need the driver state; they are bound in main() via closures below
def _timed(sol, spec, u0, nu, data, burn_s):
    C3.burn(burn_s)
    t1 = time.perf_counter()
    o, W, _ = sol.run(spec, u0, nu, data)
    jax.block_until_ready(o)
    return time.perf_counter() - t1, sha(np.asarray(W))


if __name__ == '__main__':
    main()
