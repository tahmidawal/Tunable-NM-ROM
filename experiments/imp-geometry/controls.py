"""imp-geometry must-fail controls and gates (DESIGN 8, A1.9). Each control records the clean value, the faulted value
and whether the fault was detected ("fires"). Writes <out>/controls.json.

  python controls.py --out <celldir> --dom L --fam wide --lev S1 [--smoke] [--geometry-only]

Geometry-only controls (no cell artefacts): MF-SEM, MF-FD, MF-MG, MF-QUAD, MF-ENV, MF-INTERP, score fixtures.
Cell controls (need the cell's ref / data / bank / ladder artefacts): MF-TENS, MF-Q1, MF-Q2, MF-MATCH (real), MF-REF (real).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import geom as G      # noqa: E402
import fd             # noqa: E402
import sem            # noqa: E402
import score as SC    # noqa: E402
import rom as RM      # noqa: E402


def geometry_controls(dom, smoke):
    out = {}
    # MF-SEM: curved elements converge spectrally (>= 4 decades from p=4 to p=12); chordal plateaus (<= 2 decades)
    if dom != 'B':
        ps = (4, 8, 12)
        ec = [sem.mms_helmholtz(dom, p, 2)[0] for p in ps]
        rec = dict(curved=ec, curved_decades=float(np.log10(ec[0] / ec[-1])))
        if dom == 'H':
            eh = [sem.mms_helmholtz(dom, p, 2, chordal=True)[0] for p in ps]
            rec.update(chordal=eh, chordal_decades=float(np.log10(eh[0] / eh[-1])))
            rec['fires'] = bool(rec['curved_decades'] >= 4 and rec['chordal_decades'] <= 2)
        else:
            rec['fires'] = bool(rec['curved_decades'] >= 4)
        out['MF-SEM'] = rec
    # MF-FD: observed order of the manufactured Helmholtz solution; staircase on H fails (< 1.5)
    ns = (64, 128) if smoke else (64, 128, 256)
    def orders(stair):
        e = [fd.mms_helmholtz(dom, n, stair, tol=1e-10)[0] for n in ns]
        return e, np.log2(np.array(e[:-1]) / np.array(e[1:])).tolist()
    e, p = orders(False)
    rec = dict(sw_err=e, sw_order=p)
    ok = all(1.8 <= x <= 2.2 for x in p)
    if dom == 'H':
        e2, p2 = orders(True)
        rec.update(stair_err=e2, stair_order=p2)
        ok = ok and min(p2) < 1.5
    rec['fires'] = bool(ok)
    out['MF-FD'] = rec
    # MF-MG: contraction per V-cycle on the time-step operator (alpha = 1.5/dt, dt = 1.25e-4, nu = 0.04) at n = 256;
    # identity smoother must not reach the tolerance in 50 cycles
    n = 128 if smoke else 256
    mg = fd.MG(dom, n)
    g = mg.levels[0]
    _, f = fd.mms_fields(dom, g.X, g.Y, 1.5 / 1.25e-4, 0.04)
    fj = jnp.asarray(f * g.mask_np)
    u, it, ok, rr = mg.solve(1.5 / 1.25e-4, 0.04, fj, jnp.zeros_like(fj), 1e-10, 50)
    mg0 = fd.MG(dom, n, smoother=False)
    u0, it0, ok0, rr0 = mg0.solve(1.5 / 1.25e-4, 0.04, fj, jnp.zeros_like(fj), 1e-10, 50)
    contr = float(10 ** (np.log10(float(rr)) / max(int(it), 1)))
    out['MF-MG'] = dict(cycles=int(it), converged=bool(ok), contraction=contr, noop_converged=bool(ok0),
                        noop_ratio=float(rr0), fires=bool(ok and contr <= 0.35 and not ok0))
    # MF-QUAD: composite rule area exact; chordal rule on H wrong
    X, w = G.composite_rule(dom, 4, 8)
    rec = dict(area_relerr=float(abs(w.sum() - G.area(dom)) / G.area(dom)),
               all_inside=bool(np.all(G.inside(dom, X[:, 0], X[:, 1]))))
    ok = rec['area_relerr'] <= 1e-12 and rec['all_inside']
    if dom == 'H':
        Xc, wc = G.composite_rule(dom, 4, 8, chordal=True)
        rec['chordal_area_relerr'] = float(abs(wc.sum() - G.area(dom)) / G.area(dom))
        ok = ok and rec['chordal_area_relerr'] > 1e-6
    rec['fires'] = bool(ok)
    out['MF-QUAD'] = rec
    # MF-ENV: envelope trace on the boundary; the box envelope on L / H has a non-zero trace there
    B = G.boundary_samples(dom)
    tr = float(np.max(np.abs(G.envelope_np(dom, B[:, 0], B[:, 1]))))
    rec = dict(trace=tr)
    ok = tr <= 1e-12
    if dom != 'B':
        trb = float(np.max(np.abs(G.envelope_np(dom, B[:, 0], B[:, 1], box_only=True))))
        rec['box_envelope_trace'] = trb
        ok = ok and trb > 1e-3
    # IC compatibility: chi vanishes on the boundary
    rec['ic_chi_trace'] = float(np.max(np.abs(G.chi(dom, B[:, 0], B[:, 1]))))
    rec['fires'] = bool(ok and rec['ic_chi_trace'] <= 1e-12)
    out['MF-ENV'] = rec
    # MF-INTERP: SEM lattice interpolation of a degree-p polynomial (in reference coordinates on affine elements)
    S = sem.SEMSpace(dom, 6, 2)
    pts = G.lattice(dom)[0]
    elem, Lx, Ly = S.lattice_interp(pts)
    fpoly = lambda x, y: (x ** 3) * y ** 2 - 0.3 * x * y ** 3 + 0.7
    ug = fpoly(S.Xg[:, 0], S.Xg[:, 1])
    vi = np.einsum('pi,pij,pj->p', Lx, ug[S.conn[elem]], Ly)
    curved = np.array([S.mesh.elems[e][0].curved for e in elem])
    err = float(np.max(np.abs(vi - fpoly(pts[:, 0], pts[:, 1]))[~curved]))
    bad = (elem + 1) % S.E
    vb = np.einsum('pi,pij,pj->p', Lx, ug[S.conn[bad]], Ly)
    errb = float(np.max(np.abs(vb - fpoly(pts[:, 0], pts[:, 1]))))
    out['MF-INTERP'] = dict(err_affine=err, err_wrong_element=errb, fires=bool(err <= 1e-12 and errb > 1e-3))
    out['score_fixtures'] = SC.selftest()
    out['score_fixtures']['fires'] = all(out['score_fixtures'].values())
    out['G-COH'] = dict(fires=bool(G.check_cohorts()))
    return out


def cell_controls(a, celldir):
    import run_cell as RC
    out = {}
    D = RC.cohorts(a.fam, a.lev, a.smoke)
    rj = json.loads((celldir / 'ref.json').read_text())
    out['MF-REF-real'] = dict(fires=bool(rj['MF_REF_real']))
    Uv = RC.load_refs(celldir, 'val')
    Dv = D['val'][:2]
    sp = RC.pod_space(a, celldir)
    # MF-TENS: tensor = point form (r in {32, 128} capped at R); the largest-contribution slice zeroed must differ
    rec = {}
    for r in [x for x in (32, 128) if x <= sp.R]:
        qt = RM.make_query(sp, r, 'gal_etdrk4', 40, form='tensor')
        qp = RM.make_query(sp, r, 'gal_etdrk4', 40, form='point')
        Ot = np.stack([np.asarray(qt(jnp.asarray(d))[0]) for d in Dv])
        Op = np.stack([np.asarray(qp(jnp.asarray(d))[0]) for d in Dv])
        # largest measured slice contribution on a reached state
        qz = RM.make_query(sp, r, 'gal_etdrk4', 40, form='point', out='z')
        z = np.asarray(qz(jnp.asarray(Dv[0]))[0])[2]
        T = np.asarray(RM.tensor(jnp.asarray(sp.P[:, :r]), jnp.asarray(sp.P[:, :r]), jnp.asarray(sp.S[:, :r]),
                                 jnp.asarray(sp.w)))
        contrib = np.abs(np.einsum('kij,i,j->k', T, z, z))
        k = int(np.argmax(contrib))
        qf = RM.make_query(sp, r, 'gal_etdrk4', 40, form='tensor', zero_slice=k)
        Of = np.stack([np.asarray(qf(jnp.asarray(d))[0]) for d in Dv])
        # independent fault in point form: the k-th test function removed from the nonlinear term
        qpf = RM.make_query(sp, r, 'gal_etdrk4', 40, form='point', zero_test=k)
        Opf = np.stack([np.asarray(qpf(jnp.asarray(d))[0]) for d in Dv])
        rel = float(np.max(np.abs(Ot - Op)) / np.max(np.abs(Op)))
        relf = float(np.max(np.abs(Of - Ot)) / np.max(np.abs(Ot)))
        agree = float(np.max(np.abs(Of - Opf)) / np.max(np.abs(Ot)))
        rec[f'r{r}'] = dict(tensor_vs_point=rel, zeroed_slice=k, zeroed_change=relf, tensor_vs_point_fault=agree,
                            fires=bool(rel <= 1e-9 and relf >= 1e-6 and agree <= 1e-9))
    rec['fires'] = all(v['fires'] for v in rec.values() if isinstance(v, dict))
    out['MF-TENS'] = rec
    # MF-Q1 (A3.6): the perturbed iterate's gamma is computed independently (NumPy, dense Jacobian by finite
    # differences of an independent residual); the in-job certificate must flag every step whose independent gamma
    # exceeds 1e-3, and at least one such step must exist
    r = min(32, sp.R)
    nt = 40
    qc = RM.make_query(sp, r, 'lspg_cnr', nt, form='tensor', cert=True)
    qcp = RM.make_query(sp, r, 'lspg_cnr', nt, form='tensor', cert=True, perturb_cert=1e-3)
    qz = RM.make_query(sp, r, 'lspg_cnr', nt, form='tensor', n_out=nt, out='z')
    s0 = np.asarray(qc(jnp.asarray(Dv[0]))[1])
    s1 = np.asarray(qcp(jnp.asarray(Dv[0]))[1])
    Zs = np.asarray(qz(jnp.asarray(Dv[0]))[0])          # states after every step
    Mt = min(4 * r, sp.R)
    P, Sx, Th, w = sp.P[:, :r], sp.S[:, :r], sp.P[:, :Mt], sp.w
    Mm, Kk = sp.M[:Mt, :r], sp.K[:Mt, :r]
    nu = float(Dv[0][4])
    dt = G.T_END / nt
    Nf = lambda z: Th.T @ (w * (P @ z) * (Sx @ z))
    def res(z, zn, th):
        return Mm @ (z - zn) / dt + th * (nu * Kk @ z + Nf(z)) + (1 - th) * (nu * Kk @ zn + Nf(zn))
    n_viol = 0
    for i in range(nt):
        zn, z = Zs[i], Zs[i + 1]
        th = 1.0 if i < 2 else 0.5
        zc = z + 1e-3 * np.linalg.norm(z) * np.cos(np.arange(r) * 1.7 + 0.3) / np.sqrt(r)
        rv = res(zc, zn, th)
        J = np.stack([(res(zc + 1e-7 * e, zn, th) - res(zc - 1e-7 * e, zn, th)) / 2e-7 for e in np.eye(r)], 1)
        g = np.linalg.norm(J.T @ rv) / (np.linalg.norm(J) * np.linalg.norm(rv) + 1e-300)
        n_viol += int(g > 1e-3)
    out['MF-Q1'] = dict(clean_nonstationary=float(s0[0]), clean_max_gamma=float(s0[1]), faulted_flagged=float(s1[0]),
                        independent_violations=n_viol, steps=nt,
                        fires=bool(s0[0] / s0[2] <= 0.01 and n_viol >= 1 and int(s1[0]) == n_viol))
    # MF-Q2: reference rule with the elements carrying the largest share of ||N|| zeroed must exceed the rho bar
    if (celldir / 'bank.pkl').exists():
        import bank as BK
        b = BK.load(celldir / 'bank.pkl')
        s16 = BK.bank_space(a.dom, b['params'], b['T'], b['k_q'], 8)
        qz = RM.make_query(s16, r, 'lspg_cnr', 40, form='point', out='z')
        Z = np.asarray(qz(jnp.asarray(Dv[0]))[0])
        z = Z[2]
        P, S_ = s16.P[:, :r], s16.S[:, :r]
        Mt = min(4 * r, s16.R)
        contrib = np.abs(s16.w * (P @ z) * (S_ @ z))
        nel = len(G.Mesh(a.dom, b['k_q']).elems)
        per_el = contrib.reshape(nel, -1).sum(1)
        top = np.argsort(per_el)[::-1][:max(1, nel // 20)]
        w2 = s16.w.reshape(nel, -1).copy()
        w2[top] = 0
        w2 = w2.reshape(-1)
        N1 = s16.P[:, :Mt].T @ (s16.w * (P @ z) * (S_ @ z))
        N2 = s16.P[:, :Mt].T @ (w2 * (P @ z) * (S_ @ z))
        rho = float(np.linalg.norm(N2 - N1) / np.linalg.norm(N1))
        import finalize as FZ
        out['MF-Q2'] = dict(rho_faulted=rho, n_zeroed=int(len(top)), violation_established=bool(rho > 0.116),
                            checker_flags=not FZ.rho_ok(rho), fires=bool(rho > 0.116 and not FZ.rho_ok(rho)))
    # MF-MATCH (real, deterministic fault): the best-resolved valid FD ladder output on val, perturbed by exactly 2 tau
    # (relative, at every scored time), must be scored 'misses' at tau = 0.01; unperturbed it is scored as recorded
    rows = [json.loads(l) for l in (celldir / 'ladder.jsonl').read_text().splitlines()]
    fdr = [x for x in rows if x['cfg'].get('fam') == 'fdmg' and x.get('valid')]
    if fdr:
        x = min(fdr, key=lambda x: x['worst'])
        q = RC.build_fdmg(a.dom)(x['cfg'])
        O = np.stack([np.asarray(q(jnp.asarray(d))[0]) for d in D['val'][:len(Uv)]])
        e0 = SC.err(O, Uv)
        rng = np.random.default_rng(5)
        Of = O.copy()
        for j in range(len(Of)):
            for k in range(1, 6):
                dv = rng.standard_normal(Of.shape[-1])
                Of[j, k] = Uv[j, k] + dv / np.linalg.norm(dv) * 0.02 * np.linalg.norm(Uv[j, 0])
        ef = SC.err(Of, Uv)
        # independent recomputation of the faulted worst error (plain loops)
        ef_ind = max(max(np.linalg.norm(Of[j, k] - Uv[j, k]) for k in range(1, 6)) / np.linalg.norm(Uv[j, 0])
                     for j in range(len(Of)))
        Uall = rj['allowance']['U']
        out['MF-MATCH-real'] = dict(key=x['key'], worst_clean=float(e0.max()), recorded=x['worst'],
                                    worst_faulted=float(ef.max()), U=Uall,
                                    verdict_faulted=SC.three_way(float(ef.max()), Uall, 0.01),
                                    violation_established=bool(ef_ind - Uall > 0.01),
                                    fires=bool(ef_ind - Uall > 0.01 and SC.three_way(float(ef.max()), Uall, 0.01) == 'misses'
                                               and abs(float(e0.max()) - x['worst']) <= 1e-6 * max(x['worst'], 1e-12)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--dom', required=True)
    ap.add_argument('--fam', default='wide')
    ap.add_argument('--lev', default='S1')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--geometry-only', action='store_true')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    res = dict(geometry=geometry_controls(a.dom, a.smoke))
    if not a.geometry_only:
        res['cell'] = cell_controls(a, out)
    allc = {**res['geometry'], **res.get('cell', {})}
    res['all_fire'] = all(v.get('fires', False) for v in allc.values() if isinstance(v, dict))
    res['not_firing'] = [k for k, v in allc.items() if isinstance(v, dict) and not v.get('fires', False)]
    (out / f'controls_{a.dom}.json').write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(dict(all_fire=res['all_fire'], not_firing=res['not_firing'])), flush=True)


if __name__ == '__main__':
    main()
