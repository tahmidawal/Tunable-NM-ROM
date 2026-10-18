"""Offline (CPU) recomputation of the Q2-3D evidence of one q3run archive (DESIGN 2.4; code audit 1 item 9).

From the saved coefficients (W_q2.npz: '<setting>|<rule>|<spec>' arrays and their 'cases|...' ids) it recomputes, without
the job's context object: the lattice distance q vs q+ on every case and q+ vs q++ on the qpp cases (decode with the
bank's lean lattice rows built here on CPU), and rho on cases 0 and 1 (all six outputs) from rule tables built here.
Writes checks/q3offline-<attempt>.json. analyze.label3d requires agreement with the in-job values.

    JAX_PLATFORMS=cpu python q3offline.py <attempt> [<attempt> ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp

import analyze as AN
import c3common as C3
import front3d as F3
import s3common as S

HERE = Path(__file__).resolve().parent
RHO_CASES = (0, 1)


def run(att):
    d = AN.out_dir(att, 'q3run')
    res = json.loads((d / 'result.json').read_text())
    z = np.load(d / 'W_q2.npz')
    tab = C3.C.table(923801, 64)
    members = dict(res['members'])
    if res.get('control'):
        members['control|' + res['control']['key']] = res['control']
    by_bank = {}
    for m in members.values():
        st = F3.SETTINGS[m['setting']]
        by_bank.setdefault(st['bank'], set()).add(int(st['Rp']))
    rules = S.load_rules()
    out = {}
    for bname, rps in by_bank.items():
        bank = C3.load_bank(bname)
        tb = C3.Tables(65, bank, sorted(rps), log=lambda s_: None)
        for key, m in members.items():
            st = F3.SETTINGS[m['setting']]
            if st['bank'] != bname:
                continue
            Rp = int(st['Rp'])
            G = np.concatenate([np.asarray(g) for g in tb.lean(Rp)['GTb']], 0)            # (R', 63^3)
            k_q, k_p = f"{m['setting']}|{m['rule']}|{m['spec']}", f"{m['setting']}|{m['qp']}|{m['spec']}"
            Wq, Wp = z[k_q], z[k_p]
            cq, cp = [int(x) for x in z[f'cases|{k_q}']], [int(x) for x in z[f'cases|{k_p}']]
            assert cq == m['cases'] == cp, ('population', key)
            n0 = {j: float(np.linalg.norm(C3.C.initial_interior(65, tab, j))) for j in cq}
            dist = lambda A_, B_, j: float(np.max(np.linalg.norm((np.asarray(A_) - np.asarray(B_)) @ G, axis=1)[1:]) / n0[j])
            dq = [dist(Wq[i], Wp[i], j) for i, j in enumerate(cq)]
            rec = dict(dist_qp=dq, population=cq)
            if m.get('qpp'):
                k_pp = f"{m['setting']}|{m['qpp']}|{m['spec']}"
                cpp = [int(x) for x in z[f'cases|{k_pp}']]
                assert cpp == m['qpp_cases'], ('qpp population', key)
                rec['dist_qp_qpp'] = [dist(Wp[cp.index(j)], z[k_pp][i], j) for i, j in enumerate(cpp)]
            dq_ = tb.rule_data(Rp, *rules[m['rule']])
            dp_ = tb.rule_data(Rp, *rules[m['qp']])
            rho = []
            for j in RHO_CASES:
                i = cq.index(j)
                for k in range(6):
                    c = jnp.asarray(Wq[i, k])
                    a, b = np.asarray(C3.nl3(c, dq_)), np.asarray(C3.nl3(c, dp_))
                    rho.append(float(np.linalg.norm(a - b) / np.linalg.norm(b)) if np.linalg.norm(b) > 0 else float('nan'))
            rec['rho_cases01'] = rho
            out[key] = rec
            del dq_, dp_
            print(att, key, max(dq), flush=True)
    (HERE / 'checks' / f'q3offline-{att}.json').write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    for a in sys.argv[1:]:
        run(a)
