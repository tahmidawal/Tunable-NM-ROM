"""jcp-regimes: offline (local CPU, via jaxrun) re-execution of the Q-wrong-mesh control (code audit 2, fix 6): a rule
table built on one mesh must be refused by a bank on another mesh, and a perturbed eigenvalue table must be refused by
the metadata check. Same code path as qacc.run_controls, at small meshes (256 bank, 128 table) and R128.

    JAX_PLATFORMS=cpu jaxrun python ctrl_wrongmesh.py <out.json>
"""
import json
import sys

import p1common as P


def main():
    bank = P.Bank(256, [0, 128])
    small = P.Bank(128, [0, 128])
    st_ok = P.Setting(bank, 'own', 128, 'fib4181', 'frozen')
    sts = P.Setting(small, 'wrongmesh', 128, 'fib4181', 'frozen')
    res = {}
    try:
        st_ok.check('main', bank)
        res['own_table_accepted'] = True
    except ValueError as e:
        res['own_table_accepted'] = str(e)
    try:
        sts.check('main', bank)
        res['wrong_mesh_refused'] = False
    except ValueError as e:
        res['wrong_mesh_refused'] = str(e)
    saved = st_ok.data[('main', 'LSPG')]
    st_ok.data[('main', 'LSPG')] = dict(saved, lam=saved['lam'] * (1 + 1e-6))
    try:
        st_ok.check('main', bank)
        res['perturbed_table_refused'] = False
    except ValueError as e:
        res['perturbed_table_refused'] = str(e)
    res['passed'] = bool(res['own_table_accepted'] is True and res['wrong_mesh_refused'] and res['perturbed_table_refused'])
    open(sys.argv[1], 'w').write(json.dumps(res, indent=1) + '\n')
    print('WRONGMESH', res)


if __name__ == '__main__':
    main()
