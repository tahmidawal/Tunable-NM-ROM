"""A3 F7 truncation screen (DESIGN §2 A3). CPU, NumPy only.

Decodes every saved jcp-regimes ROM coefficient output (R_*.npz, 9 configurations) on the 257^2 lattice, truncates the
decoded fields to the first K Laplacian sines, scores both against the E0 / E0-style ST references.
Stop rule (ideation F7, verbatim): stop if the worst error rises by more than 0.1 percentage points at K = 9216.
Controls: K = 65025 identity (|rise| <= 1e-12 every case), K = 16 must trip the rule; e(v) reproduces analysis.json.
Writes checks/a3_trunc.json.
"""
from __future__ import annotations

import json

import numpy as np

import fcommon as F

KS = (16, 2304, 4096, 9216, 65025)
PRIMARY = 'R384|LSPG_CNR_1'


def main():
    A = json.loads((F.REG / 'checks/analysis.json').read_text())['configs']
    refs = {}
    for coh, c in F.all_cases(F.IN_RANGE + F.REGIMES):
        u, m = F.load_ref(coh, c)
        if m['eligible']:
            refs[f'{coh}|{c}'] = (u, m['n0'])
    configs = {}
    for p in sorted(F.ROMDIR.glob('R_*.npz')):
        Z = np.load(p, allow_pickle=True)
        key = p.stem[2:].replace('_', '|', 1)
        r = Z['W'].shape[-1]
        G = F.G(r)
        rows, invalid = {}, []
        for i, cid in enumerate(Z['case_ids']):
            cid = str(cid)
            if cid not in refs:
                continue
            if not bool(Z['valid'][i]):
                invalid.append(cid)
                continue
            u, n0 = refs[cid]
            v = (Z['W'][i] @ G.T).reshape(6, 257, 257)
            row = dict(e=F.err(v, u, n0), lane_e=A[key]['per_case'][cid]['e'] if cid in A[key]['per_case'] else None)
            for K in KS:
                row[f'e{K}'] = F.err(F.sine_truncate(v, K), u, n0)
            rows[cid] = row
        def scope(cohs):
            R = {c: x for c, x in rows.items() if c.split('|')[0] in cohs}
            w = max(x['e'] for x in R.values())
            return dict(n=len(R), worst=w,
                        **{f'dworst_{K}': max(x[f'e{K}'] for x in R.values()) - w for K in KS},
                        **{f'maxrise_{K}': max(x[f'e{K}'] - x['e'] for x in R.values()) for K in KS})
        sc = dict(in_range=scope(F.IN_RANGE), all_regimes=scope(F.REGIMES), everything=scope(F.IN_RANGE + F.REGIMES))
        repro = max(abs(x['e'] - x['lane_e']) for x in rows.values() if x['lane_e'] is not None)
        ident = max(abs(x['e65025'] - x['e']) for x in rows.values())
        configs[key] = dict(r=int(r), n_invalid=len(invalid), invalid=invalid, scopes=sc, repro_max_abs=repro,
                            identity_max_abs=ident,
                            stop_K9216_in_range=bool(sc['in_range']['dworst_9216'] > 1e-3),
                            stop_K9216_all=bool(sc['everything']['dworst_9216'] > 1e-3),
                            pass_K={str(K): bool(sc['in_range'][f'dworst_{K}'] <= 1e-3) for K in KS},
                            rows=rows)
        print(key, {s: (round(100 * v['worst'], 4), {K: round(100 * v[f'dworst_{K}'], 5) for K in KS}) for s, v in sc.items()},
              'repro', repro, 'ident', ident, flush=True)
    # constructed fixture (Amendment 1): u* = u_ref + h, h in Laplacian ranks 9217..20000, ||h(t_k)|| = 0.005 ||u(t0)||
    rng = np.random.default_rng(20261021)
    fx = {}
    for cid, (u, n0) in refs.items():
        if cid.split('|')[0] not in F.IN_RANGE:
            continue
        hc = np.zeros((6, F.NI * F.NI))
        hc[1:, F.LAP_ORDER[9216:20000]] = rng.standard_normal((5, 20000 - 9216))
        hf = np.zeros((6, 257, 257))
        hf[1:, 1:-1, 1:-1] = F.idstn(hc[1:].reshape(5, F.NI, F.NI), type=1, norm='ortho', axes=(1, 2))
        hf[1:] *= (0.005 * n0 / np.linalg.norm(hf[1:].reshape(5, -1), axis=1))[:, None, None]
        us = u + hf
        fx[cid] = (F.err(us, us, n0), F.err(F.sine_truncate(us, 9216), us, n0))
    fixture_dworst = max(b for a, b in fx.values()) - max(a for a, b in fx.values())
    ctrl = dict(fixture_dworst_9216=fixture_dworst, fixture_trips=bool(fixture_dworst > 1e-3),
                identity_all=all(c['identity_max_abs'] <= 1e-12 for c in configs.values()),
                repro_all=all(c['repro_max_abs'] <= 1e-9 for c in configs.values()))
    ctrl['controls_pass'] = bool(ctrl['fixture_trips'] and ctrl['identity_all'] and ctrl['repro_all'])
    stress = {k: c['scopes']['in_range']['dworst_16'] for k, c in configs.items()}   # K = 16 empirical stress (reported)
    pc = configs[PRIMARY]
    verdict = ('INCONCLUSIVE (control failed)' if not ctrl['controls_pass'] else
               ('STOP F7 (worst error rises > 0.1 pp at K = 9216)' if pc['stop_K9216_in_range'] else 'F7 screen passes'))
    out = dict(design='DESIGN.md §2 A3', provenance=F.provenance(), controls=ctrl, primary=PRIMARY, verdict=verdict, k16_stress_dworst_in_range=stress,
               verdict_all_regimes=('STOP' if pc['stop_K9216_all'] else 'pass'),
               timing_gate='deferred (GPU, not in this lane brief)', configs=configs)
    F.write_json('a3_trunc', out)
    print(json.dumps(dict(controls=ctrl, verdict=verdict, k16_stress_dworst_in_range=stress, all=out['verdict_all_regimes']), indent=1))


if __name__ == '__main__':
    main()
