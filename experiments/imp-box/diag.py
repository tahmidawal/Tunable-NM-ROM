"""imp-box diagnostics in stage `time` (DESIGN A1.4, A1.9, A2.7, A3.3): query-cost components per finalist, the bank's
equal-cost size r_x (evaluation and query contracts) with projection floors, and the finer-lattice (1/512) check at
A3/A4. Writes diag_<arm>.json.
"""
from __future__ import annotations

import json
import time

import numpy as np
import jax
import jax.numpy as jnp

import finalize as FZ
import geom as G
import ladders as LS
import podsine as PS
import rom as RM
import score as SC
import spec as SP
import cell as CL
import refgate as RG

STATS = ('median', 'worst')


TRAW = {}                      # raw timing arrays per label (persisted in diag_<arm>.json; code audit 1)


def tmed(f, x, reps=20, burn=5, label=None, args=()):
    for _ in range(burn):
        jax.block_until_ready(f(x, *args))
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        jax.block_until_ready(f(x, *args))
        ts.append(time.perf_counter() - t)
    if label is not None:
        TRAW[label] = ts
    return float(np.median(ts))


def tcases(f, D, reps=5, label=None):
    return float(np.mean([tmed(f, jnp.asarray(d), reps=reps, burn=3, label=None if label is None else f'{label}|{i}')
                          for i, d in enumerate(D)]))


def context_of(sel, name, key):
    """Selection context of one configuration: every (tau, stat) entry it is a finalist of, with status and labels."""
    return [dict(tau=t, stat=st_, status=rec['status'], boundary_labels=rec.get('boundary_labels', []),
                 construction_failures=rec.get('construction_failures', 0))
            for (nm, t, st_), rec in sel.items() if nm == name and any(f['key'] == key for f in rec['finalists'])]


def finals_of(sel, name):
    out = {}
    for (nm, tau, stat), rec in sel.items():
        if nm == name:
            for f in rec['finalists']:
                out[f['key']] = f['cfg']
    return out


# ----------------------------------------------------------------------------------------- components ---------

def comps_reduced(c, arm, cfg):
    D4 = c.Dv[:4]
    r, n = cfg['r'], cfg['nsteps']
    if arm.name == 'ecsw':
        sp = arm.subs[(r, cfg['kappa'])]
    else:
        sp = LS.space_of(arm.sps, cfg['q'])
    Pi, wi = jnp.asarray(sp.Pi[:, :r]), jnp.asarray(sp.wi)
    Xi = jnp.asarray(sp.meta['ic_points'])
    Mi = jnp.asarray(np.linalg.inv(sp.M[:r, :r]))
    centred = arm.name == 'trans'

    def ic(d):
        dd = d.at[0].set(0.5).at[1].set(0.5) if centred else d
        return Mi @ (Pi.T @ (wi * G.ic('B', dd, Xi[:, 0], Xi[:, 1], jnp)))
    lb = 'comp|' + '|'.join(f'{k}={cfg[k]}' for k in sorted(cfg))           # full configuration key (code audit 3)
    out = dict(ic=tcases(jax.jit(ic), D4, label=lb + '|ic'))
    if cfg['st'] == 'gal_etdrk4':
        A = np.linalg.solve(sp.M[:r, :r], sp.K[:r, :r])
        lam = jnp.asarray(np.linalg.eigvals(A))
        dt = G.T_END / n
        out['setup'] = tcases(jax.jit(lambda d: RM.etdrk4_coefs(-d[4] * lam, dt)), D4, label=lb + '|setup')
    else:
        out['setup'] = 0.0
    if arm.name == 'ecsw':
        qz = RM.make_query(sp, r, 'gal_etdrk4', n, form='point', out='z')
    else:
        qz = arm.build_on(cfg, sp, True)
    out['query_no_decode'] = tcases(qz, D4, label=lb + '|nodecode')
    Z, _ = qz(jnp.asarray(c.Dv[0]))
    if centred:                                       # the shift is a dynamic argument (no constant folding)
        dec = PS.make_shift_decoder(arm.Phi[:r])
        dl = jnp.asarray([c.Dv[0][0] - 0.5, c.Dv[0][1] - 0.5])
        out['decode'] = tmed(jax.jit(lambda Z_, dl_: jax.vmap(lambda z: dec(z, dl_))(Z_)), Z, args=(dl,),
                             label=lb + '|decode')
    else:
        Pl = jnp.asarray(sp.Pl[:, :r])
        out['decode'] = tmed(jax.jit(lambda Z_: Z_ @ Pl.T), Z, label=lb + '|decode')
    out['fused'] = tcases(arm.build(cfg), D4, label=lb + '|fused')
    out['evolve_derived'] = out['query_no_decode'] - out['ic'] - out['setup']
    out['sum_vs_fused'] = (out['query_no_decode'] + out['decode']) / out['fused']
    return out


def comps_fom(c, cfg):
    D4 = c.Dv[:4]
    N = cfg['N']
    tab = SP.ICTable(N)
    lam = SP.lam_grid(N)
    lb = 'comp|' + '|'.join(f'{k}={cfg[k]}' for k in sorted(cfg))
    out = dict(ic=tcases(jax.jit(tab.coefs), D4, label=lb + '|ic'))
    if cfg['fam'] == 'spec':
        dt = G.T_END / cfg['nsteps']
        out['setup'] = tcases(jax.jit(lambda d: SP._coefs(lam, d[4], dt)), D4, label=lb + '|setup')
        qc = SP.make_fom(N, cfg['nsteps'], out='coef', xform=cfg.get('xform', 'fft'))
    else:
        dts = [0.05 / 2 ** j for j in range(SP.AD_J + 2)]
        out['setup'] = tcases(jax.jit(lambda d: [SP._coefs(lam, d[4], dt) for dt in dts]), D4, label=lb + '|setup')
        qc = SP.make_fom_ad(N, cfg['tol'], out='coef', xform=cfg.get('xform', 'fft'))
    out['query_no_decode'] = tcases(qc, D4, label=lb + '|nodecode')
    B, _ = qc(jnp.asarray(c.Dv[0]))
    syn = jax.jit(jax.vmap(SP.synth_fn(N)))
    out['decode'] = tmed(syn, B, label=lb + '|decode')
    out['fused'] = tcases(LS.build_spec(cfg), D4, label=lb + '|fused')
    out['evolve_derived'] = out['query_no_decode'] - out['ic'] - out['setup']
    out['sum_vs_fused'] = (out['query_no_decode'] + out['decode']) / out['fused']
    return out


# -------------------------------------------------------------------------------------------- finer lattice ----

def finer(c, arm, cfg, R512, D8):
    if arm.name in ('spec', 'specad'):
        xf = cfg.get('xform', 'fft')
        q = SP.make_fom(cfg['N'], cfg['nsteps'], lat=512, xform=xf) if cfg['fam'] == 'spec' else \
            SP.make_fom_ad(cfg['N'], cfg['tol'], lat=512, xform=xf)
    else:
        r = cfg['r']
        if arm.name == 'ecsw':
            sp = arm.subs[(r, cfg['kappa'])]
        else:
            sp = LS.space_of(arm.sps, cfg['q'])
        x5 = np.arange(0, 513) / 512
        if arm.name == 'trans':
            q = PS.make_trans_query(sp, arm.Phi, r, cfg['nsteps'], form=LS.form_of(cfg['q']), lat=512)
        else:
            if arm.name.startswith('bank'):
                import bank as BK
                b = BK.load(arm.ctx.out / 'bank.pkl')
                pj = jax.tree_util.tree_map(jnp.asarray, b['params'])
                pts = G.lattice('B', 512)[0]
                Pl = np.concatenate([np.asarray(BK.features('B', pj, jnp.asarray(pts[s:s + 16384])) @ jnp.asarray(b['T']))
                                     for s in range(0, len(pts), 16384)])
            else:
                Pl, _ = PS.eval_modes(arm.Phi[:sp.R], x5, deriv=False)
            sp5 = RM.Space(sp.P, sp.S, sp.w, sp.M, sp.K, sp.Pi, sp.wi, Pl, name=sp.name + '-lat512', meta=sp.meta)
            q = RM.make_query(sp5, r, cfg['st'], cfg['nsteps'], form=LS.form_of(cfg['q']) if arm.name != 'ecsw'
                              else 'point', tol=cfg.get('tol', 'prod'))
    O, _ = FZ.run_unbatched(q, D8)
    e = SC.err(O, R512)
    return dict(worst=float(np.max(e)), median=float(np.median(e)), e=e.tolist())


def refcoef(c, D8):
    """Reference coefficients of val[0:8] (eligible) at each case's reference size, (1, 6, n, n) per case."""
    rj = json.loads((c.out / 'ref.json').read_text())
    elig = np.load(c.out / 'refs_val.npz')['eligible']
    Ns = [n for n, e in zip(rj['splits']['val']['N_case'], elig) if e][:len(D8)]
    return [np.asarray(SP.make_fom(N, SP.ref_cfl_steps(N), out='coef')(jnp.asarray(d))[0])[None] for d, N in zip(D8, Ns)]


def ref512(c, D8):
    rj = json.loads((c.out / 'ref.json').read_text())
    Ns = rj['splits']['val']['N_case']
    elig = np.load(c.out / 'refs_val.npz')['eligible']
    Ns = [n for n, e in zip(Ns, elig) if e][:len(D8)]
    out = []
    for d, N in zip(D8, Ns):
        q = SP.make_fom(N, SP.ref_cfl_steps(N), lat=512)
        out.append(np.asarray(q(jnp.asarray(d))[0]))
    return np.stack(out)


# ------------------------------------------------------------------------------------------------- r_x ---------

def r_times(c, arm, sel, T, out):
    """A2.7 / A3.3 / A5.4: r_x^eval with floors (bank stop; selectable forms only, rank-specific IC projection) and
    r_x^query (diagnostic), per (tau, stat)."""
    sps = arm.sps
    sp12 = sps[12]
    R = sp12.R
    rl = [r for r in LS.R_LADDER if r <= R]
    d0 = c.Dv[0]
    Xi = sp12.meta['ic_points']
    u0 = G.ic('B', d0, Xi[:, 0], Xi[:, 1], np)
    tb, forms, cands = {}, {}, {}
    for r in rl:
        z = jnp.asarray(np.linalg.solve(sp12.M[:r, :r], sp12.Pi[:, :r].T @ (sp12.wi * u0)))    # rank-r projection
        cand = {}
        for Mt, tag, st in ((r, 'gal', 'gal_etdrk4'), (min(4 * r, R), 'lspg', 'lspg_cnr')):
            T_ = RM.tensor(jnp.asarray(sp12.P[:, :Mt]), jnp.asarray(sp12.P[:, :r]), jnp.asarray(sp12.S[:, :r]),
                           jnp.asarray(sp12.w))
            cand[f'tensor-{tag}'] = tmed(jax.jit(lambda z_, T_=T_: jnp.einsum('kij,i,j->k', T_, z_, z_)), z,
                                         label=f'rx|r{r}|tensor-{tag}')
            del T_
            for q in LS.RULES:
                s_ = sps[q]
                if LS.flop_skip(s_, q, r, st):
                    continue
                P, S, Th, w = (jnp.asarray(x) for x in (s_.P[:, :r], s_.S[:, :r], s_.P[:, :Mt], s_.w))
                cand[f'point{q}-{tag}'] = tmed(jax.jit(lambda z_, P=P, S=S, Th=Th, w=w: Th.T @ (w * (P @ z_) * (S @ z_))),
                                               z, label=f'rx|r{r}|point{q}-{tag}')
        k = min(cand, key=cand.get)
        tb[r], forms[r], cands[r] = cand[k], k, cand
    floors = {r: [RM.floor_err(sp12, r, c.Uv[i]) for i in range(len(c.Uv))] for r in rl}
    zero = [float(np.max(np.linalg.norm(c.Uv[i][1:], axis=1)) / np.linalg.norm(c.Uv[i][0])) for i in range(len(c.Uv))]
    # r_x^query: cheapest valid bank ladder row per r (both steppers), re-screened on val[0:4] (same statistic as FOM)
    rows = [r_ for r_ in c.L.done.values() if r_['cfg'].get('fam') == 'bank' and r_.get('stats_valid')
            and r_.get('screen') is not None]
    tq = {}
    import timing as TM
    for r in rl:                          # the 3 cheapest valid rows per r (ladder screen) re-screened on val[0:4] (A7)
        rr = sorted([x for x in rows if x['cfg']['r'] == r], key=lambda x: (x['screen'], x['key']))[:3]
        for x in rr:
            if x['key'] not in c.cache_scr:
                c.cache_scr[x['key']] = TM.rescreen(arm.build(x['cfg']), c.Dv[:4])
        if rr:
            best = min(rr, key=lambda x: (c.cache_scr[x['key']], x['key']))
            tq[r] = (c.cache_scr[best['key']], best['key'])
    res = {}
    for tau in LS.TAUS:
        for stat in STATS:
            fa = [(fn, f) for fn in ('spec', 'specad') for f in sel[(fn, tau, stat)]['finalists']]
            if not fa:
                res[f'{tau}|{stat}'] = dict(status='no FOM finalist', verdict='missing')
                continue
            fbest = min(fa, key=lambda x: (x[1]['rescreen'], x[1]['key']))
            N = fbest[1]['cfg']['N']
            b0 = SP.ICTable(N).coefs(jnp.asarray(c.Dv[0]))
            xf = fbest[1]['cfg'].get('xform', 'fft')
            tF = tmed(jax.jit(lambda b: SP.nl_fun(N, xform=xf)(b)), b0, label=f'rx|fom|N{N}|{xf}|{tau}|{stat}')
            aff = [r for r in rl if tb[r] <= tF]
            ent = dict(fom_key=fbest[1]['key'], N=N, t_F=tF, comparator_rule='smallest val[0:4] rescreen time (A5.4)',
                       fom_selection={fn: dict(status=sel[(fn, tau, stat)]['status'],
                                               boundary_labels=sel[(fn, tau, stat)].get('boundary_labels', []))
                                      for fn in ('spec', 'specad')})
            if aff:
                rx = max(aff)
                fl = np.array(floors[rx])
                v = float(np.median(fl)) if stat == 'median' else float(np.max(fl))
                state = 'above' if v - c.U > tau else ('below' if v + c.U <= tau else 'unresolved')
                ent.update(r_x=rx, form=forms[rx], t_b=tb[rx], floor=v, verdict=state)
            else:
                ent.update(r_x=None, verdict='no affordable tested rank')
            tFq = fbest[1]['rescreen']
            affq = [r for r in tq if tq[r][0] <= tFq]
            ent['query'] = dict(t_F=tFq, r_x=max(affq) if affq else None,
                                row=tq[max(affq)][1] if affq else None,
                                note='validity only (stats_valid); numerical qualification not required (A2.7)')
            res[f'{tau}|{stat}'] = ent
    return dict(t_b=tb, forms=forms, candidates={str(k): v for k, v in cands.items()},
                floors={str(k): v for k, v in floors.items()}, zero_field=zero, entries=res,
                query_times={str(k): v for k, v in tq.items()})


# ----------------------------------------------------------------------------------------------- drivers -------

def population(c):
    """Claim labels (A3.1): eligible counts and training-data qualification, attached to every diagnostic file."""
    dd = json.loads((c.out / 'data.json').read_text())
    n = json.loads((c.out / 'cohorts.json').read_text())['n']
    return dict(val=f'{len(c.Dv)} of {n["val"]}', dev=f'{len(c.Dd)} of {n["dev"]}',
                val_eligible=len(c.Dv), val_total=n['val'], dev_eligible=len(c.Dd), dev_total=n['dev'],
                training_data_qualified={k: v['gate']['passed'] for k, v in dd.items()
                                         if isinstance(v, dict) and 'gate' in v})


def arm_diag(c, arm, sel, T, out):
    rec = dict(components={}, finer=None, population=population(c))
    fins = finals_of(sel, arm.name)
    rec['selection_context'] = {k: context_of(sel, arm.name, k) for k in fins}
    for k, cfg in fins.items():
        try:
            rec['components'][k] = comps_reduced(c, arm, cfg)
        except Exception as ex:                                          # noqa: BLE001 (recorded)
            rec['components'][k] = dict(error=repr(ex)[:300])
    if arm.name == 'bank_lspg':
        rec['r_x'] = r_times(c, arm, sel, T, out)
    if arm.name == 'trans':
        bd = {}
        for k, cfg in fins.items():
            sp = LS.space_of(arm.sps, cfg['q'])
            qz = arm.build_on(cfg, sp, True)
            r = cfg['r']
            worst = 0.0
            for i, d in enumerate(c.Dv[:8]):
                Z, _ = qz(jnp.asarray(d))
                worst = max(worst, PS.boundary_defect(arm.Phi[:r], np.asarray(Z), (d[0] - 0.5, d[1] - 0.5),
                                                      float(np.max(np.abs(c.Uv[i][0])))))
            bd[k] = worst
        rec['boundary_defect'] = bd
    rec['timing_raw'] = dict(TRAW)
    TRAW.clear()
    if c.a.lev in ('A3', 'A4') and fins:
        D8 = c.Dv[:8]
        R5 = ref512(c, D8)
        rec['finer'] = {k: finer(c, arm, cfg, R5, D8) for k, cfg in fins.items()}
    (out / f'diag_{arm.name}.json').write_text(json.dumps(rec, indent=1, default=float))


def fom_diag(c, Darm, sel, out):
    rec = dict(components={}, finer=None, population=population(c))
    allf = {}
    rec['selection_context'] = {}
    for name in ('spec', 'specad'):
        for k, cfg in finals_of(sel, name).items():
            allf[k] = (name, cfg)
            rec['selection_context'][k] = context_of(sel, name, k)
            rec['components'][k] = comps_fom(c, cfg)
    if c.a.lev in ('A3', 'A4') and allf:
        D8 = c.Dv[:8]
        R5 = ref512(c, D8)
        rec['finer'] = {k: finer(c, Darm[nm], cfg, R5, D8) for k, (nm, cfg) in allf.items()}
        Bref = refcoef(c, D8)
        rec['coef_space'] = {}
        for k, (nm, cfg) in allf.items():
            xf = cfg.get('xform', 'fft')
            q = SP.make_fom(cfg['N'], cfg['nsteps'], out='coef', xform=xf) if cfg['fam'] == 'spec' else \
                SP.make_fom_ad(cfg['N'], cfg['tol'], out='coef', xform=xf)
            B, _ = FZ.run_unbatched(q, D8)
            e = [float(RG.coef_diff(B[i:i + 1], Bref[i])[0]) for i in range(len(D8))]
            rec['coef_space'][k] = dict(worst=max(e), median=float(np.median(e)), e=e)
    rec['timing_raw'] = dict(TRAW)
    TRAW.clear()
    (out / 'diag_fom.json').write_text(json.dumps(rec, indent=1, default=float))
