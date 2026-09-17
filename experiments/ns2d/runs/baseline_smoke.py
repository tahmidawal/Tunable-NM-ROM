"""Rule 7: reproduce an audited parent-lane number to <=1e-9 before any submit.
Recompute the Stokes S-FOM manufactured-solution velocity/pressure errors at N=32 and N=64
with the certified stk2d_common FOM and compare to the archived JSON rows."""
import json, sys
import numpy as np
sys.path.insert(0, '.')
import stk2d_common as stk
d = json.load(open('runs/stk2d/stk2d_fom_gates_nu1_M64.json'))
rows = d['gates']['S_FOM']['rows']
print('archived row keys:', list(rows[0].keys()))
worst = 0.0
for row in rows:
    N = int(row['N'])
    if N > 64:
        continue
    g = stk.MacGrid(N)
    mf = stk.manufactured(g)
    u, p, info = stk.solve_stokes(g, mf['f'], nu=1.0)
    eu = stk.mass_rel(g, u, mf['u'])
    ep = stk.mass_rel(g, p, mf['p'] - mf['p'].mean())
    ku = [k for k in row if 'u' in k.lower() and 'err' in k.lower()]
    kp = [k for k in row if 'p' in k.lower() and 'err' in k.lower()]
    au, ap = row[ku[0]], row[kp[0]]
    du, dp = abs(eu - au) / au, abs(ep - ap) / ap
    worst = max(worst, du, dp)
    print(f'N={N}: u_err={eu:.12e} archived({ku[0]})={au:.12e} rel_dev={du:.2e} | '
          f'p_err={ep:.12e} archived({kp[0]})={ap:.12e} rel_dev={dp:.2e}')
print('WORST_REL_DEV', f'{worst:.3e}', 'PASS' if worst <= 1e-9 else 'FAIL')
