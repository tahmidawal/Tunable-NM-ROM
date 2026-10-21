"""A4 F15 hybrid (sine-enriched) floors with controls (DESIGN §2 A4). CPU, NumPy only.

Bases: H = span(G_512 + S_512); S_1024; B_512; replication span(G_384 + S_512); must-fail control H_hi =
span(G_512 + sines of Laplacian rank 40000..40511); containment identity f_H <= min(f_B512, f_S512) + 1e-12.
Stop rule (ideation F15, verbatim): stop if the hybrid's C1 worst floor is not >= 2x below the sine control and below
the bank-512 floor. Writes checks/a4_hybrid.json.
"""
from __future__ import annotations

import json

import numpy as np

import fcommon as F


def qr_span(*blocks):
    Q, R = np.linalg.qr(np.concatenate(blocks, 1))
    d = np.abs(np.diag(R))
    return Q[:, d > d.max() * 1e-12]


def floors_t(u, Q, n0):
    R = u.reshape(6, -1)[1:].T
    return np.linalg.norm(R - Q @ (Q.T @ R), axis=0) / n0      # per time t1..t5


def rule(fl, cohort='C1'):
    w = {b: max(v[b] for c, v in fl.items() if c.split('|')[0] == cohort) for b in ('H', 'S1024', 'B512', 'Hhi', 'H0')}
    cont = lambda h: (w[h] <= 0.5 * w['S1024']) and (w[h] < w['B512'] - 1e-12)
    return w, dict(continue_hybrid=bool(cont('H')), continue_hhi_empirical=bool(cont('Hhi')), continue_null=bool(cont('H0')))


def main():
    G512 = F.G(512)
    S512 = F.sine_cols(np.arange(512))
    bases = dict(B512=F.orth(G512), S512=S512, S1024=F.sine_cols(np.arange(1024)),
                 H=qr_span(G512, S512), H384=qr_span(F.G(384), S512),
                 Hhi=qr_span(G512, F.sine_cols(np.arange(40000, 40512))),
                 H0=F.orth(np.concatenate([G512, G512], 1)))           # guaranteed null enrichment (span = B512)
    dims = {k: int(v.shape[1]) for k, v in bases.items()}
    fl, cont_viol = {}, 0.
    for coh, c in F.all_cases(F.IN_RANGE + F.REGIMES):
        u, m = F.load_ref(coh, c)
        if not m['eligible']:
            continue
        ft = {b: floors_t(u, Q, m['n0']) for b, Q in bases.items()}
        cont_viol = max(cont_viol, float(np.max(ft['H'] - np.minimum(ft['B512'], ft['S512']))))
        fl[f'{coh}|{c}'] = {b: float(v.max()) for b, v in ft.items()}
    worst, dec = rule(fl)
    summ = {}
    for coh in F.IN_RANGE + F.REGIMES:
        R = [v for c, v in fl.items() if c.split('|')[0] == coh]
        summ[coh] = dict(n=len(R), **{f'{b}_median': float(np.median([x[b] for x in R])) for b in bases},
                         **{f'{b}_worst': float(np.max([x[b] for x in R])) for b in bases})
    ctrl = dict(containment_max_violation=cont_viol, containment_ok=cont_viol <= 1e-12,
                null_must_be_stopped=not dec['continue_null'], hhi_empirical_stopped=not dec['continue_hhi_empirical'])
    ctrl['controls_pass'] = bool(ctrl['containment_ok'] and ctrl['null_must_be_stopped'])
    verdict = ('INCONCLUSIVE (control failed)' if not ctrl['controls_pass'] else
               ('F15 continues' if dec['continue_hybrid'] else 'STOP F15'))
    out = dict(design='DESIGN.md §2 A4', provenance=F.provenance(), dims=dims, C1_worst=worst, decision=dec,
               controls=ctrl, verdict=verdict, summary=summ, rows=fl)
    F.write_json('a4_hybrid', out)
    print(json.dumps(dict(dims=dims, C1_worst=worst, controls=ctrl, verdict=verdict), indent=1))
    for coh, s in summ.items():
        print(coh, s['n'], {b: (round(100 * s[f'{b}_median'], 4), round(100 * s[f'{b}_worst'], 3)) for b in bases})


if __name__ == '__main__':
    main()
