"""jcp-budget 3D job driver (DESIGN 4, 5, A1.2, A2.1, A2.2): B2 corners (Q, T, S full factorial), the tight-end form
panel (LSPG/GAL x rule x time), the ideal end u_star and its U_star replays, converged-rule check replays, B7 long-horizon
rollouts with certification, and the 3D controls. Built on speed3d's job context (s3ctx.Ctx, horizon-aware copy).

    python b3run.py --config configs/<attempt>.json --out <dir>

Corner -> (rule role, spec) for a member 'setting|rule|kind|scheme|dt|K-or-k':
  0    (q,    deployed spec)          Q   (q_inf, deployed spec)
  T    (q,    same kind at BDF2, dt0/8)          S   (q, ('ad', scheme, dt, 'tight'))
  and their unions; the form panel adds ('adg', ., ., 'tight') at {q, q_inf} x {(scheme, dt), (BDF2, dt0/8)}.
Outputs: result.json (rows with per-run validity / certification), W_b3.npz (every coefficient array, keyed
'<tag>|<spec>'), GLAT_<setting>.npy (the lattice basis (R', 63^3) used in-job, for the offline analysis), COMPLETE.
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

import c3common as C3
import s3common as S
from s3ctx import Ctx

DT0 = S.DT0


def cert(spc):
    """A3.5: fixed-sweep / QN runs are certified per step (speed3d gamma); adaptive runs carry their own stopping test."""
    return spc[0] in ('fs', 'fsp', 'qn')


def member_specs(key, qinf):
    sname, rule, *spec = key.split('|')
    sp = S.parse_key('|'.join(spec))
    kind, scheme, dt = sp[0], sp[1], sp[2]
    spT = (kind, 'BDF2', DT0 / 8) + tuple(sp[3:])
    adS, adTS = ('ad', scheme, dt, 'tight'), ('ad', 'BDF2', DT0 / 8, 'tight')
    corners = {'0': (rule, sp), 'Q': (qinf, sp), 'T': (rule, spT), 'S': (rule, adS), 'QT': (qinf, spT), 'QS': (qinf, adS),
               'TS': (rule, adTS), 'QTS': (qinf, adTS)}
    form = {'F|S': (rule, ('adg', scheme, dt, 'tight')), 'F|QS': (qinf, ('adg', scheme, dt, 'tight')),
            'F|TS': (rule, ('adg', 'BDF2', DT0 / 8, 'tight')), 'F|QTS': (qinf, ('adg', 'BDF2', DT0 / 8, 'tight'))}
    return sname, rule, sp, corners, form


def run_member_b2(cx, m, b, ids, chk, rep):
    qinf, qcheck = b['qinf'], b['qcheck']
    sname, rule, sp, corners, form = member_specs(m['key'], qinf)
    st = cx.ST[sname]
    res = dict(key=m['key'], corners={}, form={}, check={})
    for name, (rl, spc) in list(corners.items()) + list(form.items()):
        data = st.data_gal(rl) if spc[0] == 'adg' else None
        r = cx.rollouts(sname, rl, [spc], ids, data=data, tag=f"{m['key']}#{name}", certify=cert(spc))[S.spec_key(spc)]
        (res['form'] if name.startswith('F|') else res['corners'])[name] = dict(
            spec=S.spec_key(spc), rule=rl, valid=[bool(v) for v in r['valid']], nonstationary=[int(x) for x in r['nonst']],
            steps=[int(x) for x in r['steps']])
    for name, (rl, spc) in list(corners.items()) + list(form.items()):
        if rl != qinf:
            continue
        data = st.data_gal(qcheck) if spc[0] == 'adg' else None
        r = cx.rollouts(sname, qcheck, [spc], chk, data=data, tag=f"{m['key']}#{name}#check", certify=cert(spc))[S.spec_key(spc)]
        res['check'][name] = dict(spec=S.spec_key(spc), rule=qcheck, valid=[bool(v) for v in r['valid']],
                                  nonstationary=[int(x) for x in r['nonst']], steps=[int(x) for x in r['steps']])
    rep['b2'][m['key']] = res
    cx.save()


def run_star(cx, sname, b, ids, chk, rep):
    st = cx.ST[sname]
    out = {}
    specs = {'star': ('adg', 'BDF2', DT0 / 8, 'tight'), 'star16': ('adg', 'BDF2', DT0 / 16, 'tight'),
             'starx': ('adg', 'BDF2', DT0 / 8, 'tighter'), 'star16x': ('adg', 'BDF2', DT0 / 16, 'tighter')}
    for nm, spc in specs.items():
        r = cx.rollouts(sname, b['qinf'], [spc], ids, data=st.data_gal(b['qinf']), tag=f'{sname}#{nm}', certify=False)
        rc = cx.rollouts(sname, b['qcheck'], [spc], chk, data=st.data_gal(b['qcheck']), tag=f'{sname}#{nm}#check',
                         certify=False)
        out[nm] = dict(spec=S.spec_key(spc), valid=[bool(v) for v in r[S.spec_key(spc)]['valid']],
                       check_valid=[bool(v) for v in rc[S.spec_key(spc)]['valid']])
    rep['b2'][f'{sname}#star'] = out
    cx.save()


def run_member_b7(cx, m, b, ids, rep):
    sname, rule, sp, corners, _ = member_specs(m['key'], m['qp'])
    st = cx.ST[sname]
    key = S.spec_key(sp)
    tight, tighter = ('ad', sp[1], sp[2], 'tight'), ('ad', sp[1], sp[2], 'tighter')
    q2 = [j for j in b.get('q2_cases', range(8)) if j in ids]
    qpp = [j for j in b.get('qpp_cases', range(4)) if j in ids]
    dev = [j for j in b.get('tighter_cases', range(6)) if j in ids]
    T = f"T{S.HORIZON[0]:g}"
    tag = lambda nm: f"{m['key']}#B7{T}#{nm}"
    r = cx.rollouts(sname, rule, [sp], ids, tag=tag('prod'), certify=True)[key]
    rt = cx.rollouts(sname, rule, [tight], ids, tag=tag('tight'), certify=False)[S.spec_key(tight)]
    rtt = cx.rollouts(sname, rule, [tighter], dev, tag=tag('tighter'), certify=False)[S.spec_key(tighter)]
    rq = cx.rollouts(sname, m['qp'], [sp], q2, tag=tag('qp'), certify=False)[key]
    rqq = cx.rollouts(sname, m['qpp'], [sp], qpp, tag=tag('qpp'), certify=False)[key]
    d, dq = st.data(rule), st.data(m['qp'])
    rho = []
    for ci, j in enumerate(q2):
        W = r['W'][ids.index(j)]
        row = []
        for k in range(W.shape[0]):
            a_ = np.asarray(C3.nl3(jnp.asarray(W[k]), d))
            b_ = np.asarray(C3.nl3(jnp.asarray(W[k]), dq))
            nb = float(np.linalg.norm(b_))
            row.append(float(np.linalg.norm(a_ - b_) / nb) if nb > 0 else (float('nan') if np.linalg.norm(a_) == 0 else float('inf')))
        rho.append(row)
    rep['b7'][m['key']] = dict(horizon=S.HORIZON[0], valid=[bool(v) for v in r['valid']], nonstationary=int(sum(r['nonst'])),
                               steps=int(sum(r['steps'])), tight_valid=[bool(v) for v in rt['valid']],
                               tighter_valid=[bool(v) for v in rtt['valid']], qp_valid=[bool(v) for v in rq['valid']],
                               qpp_valid=[bool(v) for v in rqq['valid']], rho=rho, q2_cases=q2, qpp_cases=qpp,
                               tighter_cases=dev, cases=list(ids))
    cx.save()


def run_controls(cx, c, rep):
    ids = list(c.get('cases', range(4)))
    res = {}
    # C-Q: W256 fs CN 0.01 K1 with the deployed rule replaced by lat256, vs the converged rule
    spq = ('fs', 'CN', 0.01, 1)
    a = cx.rollouts('W256', c['under_resolved'], [spq], ids, tag='CTRL#Q#ctrl', certify=False)
    b = cx.rollouts('W256', c['qinf'], [spq], ids, tag='CTRL#Q#qinf', certify=False)
    res['C_Q'] = dict(valid=[bool(x) for x in a[S.spec_key(spq)]['valid']])
    # C-T: W256 gl24 fs BE 0.01 K1 vs fs BDF2 dt0/8 K1
    spb, spt = ('fs', 'BE', 0.01, 1), ('fs', 'BDF2', DT0 / 8, 1)
    cx.rollouts('W256', 'gl24', [spb, spt], ids, tag='CTRL#T', certify=False)
    # C-S / G-Q: W256 lat8192 qn BDF2 0.025 k4 with mutation q0 (no iterations), certified, vs ad tight
    sp0 = ('qn', 'BDF2', 0.025, 4, 'q0')
    r0 = cx.rollouts('W256', 'lat8192', [sp0], ids, tag='CTRL#S#q0', certify=True)[S.spec_key(sp0)]
    cx.rollouts('W256', 'lat8192', [('ad', 'BDF2', 0.025, 'tight')], ids, tag='CTRL#S#tight', certify=False)
    res['G_Q'] = dict(nonstationary=int(sum(r0['nonst'])), steps=int(sum(r0['steps'])),
                      fired=bool(sum(r0['nonst']) > .01 * max(1, sum(r0['steps']))))
    rep['controls'] = res
    cx.save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    S.HORIZON[0] = .25
    cx = Ctx(cfg, a.out)
    cx.rep.update(b2={}, b7={}, controls={})
    ids = list(cx.cases)
    if 'b2' in cfg:
        b = cfg['b2']
        chk = [j for j in b.get('check_cases', range(4)) if j in ids]
        for sname in sorted({m['key'].split('|')[0] for m in cfg['members']}):
            run_star(cx, sname, b, ids, chk, cx.rep)
        for m in cfg['members']:
            run_member_b2(cx, m, b, ids, chk, cx.rep)
    if 'controls' in cfg:
        run_controls(cx, dict(cfg['controls'], qinf=cfg['b2']['qinf']), cx.rep)
    if 'b7' in cfg:
        S.HORIZON[0] = float(cfg['b7']['horizon'])
        cx.queries.clear()
        cx.certs.clear()
        for m in cfg['members']:
            run_member_b7(cx, m, cfg['b7'], ids, cx.rep)
    cx.dump()
    for sname in sorted({m['key'].split('|')[0] for m in cfg['members']}):      # lattice basis for the offline analysis
        np.save(Path(a.out) / f'GLAT_{sname}.npy', np.asarray(cx.Gl[sname]))
    cx.rep['complete'] = True
    cx.rep['elapsed_seconds'] = cx.el()
    cx.save()
    (Path(a.out) / 'COMPLETE').write_text('complete\n')
    print('B3RUN COMPLETE', cx.el(), flush=True)


if __name__ == '__main__':
    main()
