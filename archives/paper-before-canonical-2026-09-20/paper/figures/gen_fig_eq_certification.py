"""Figure: NNLS fit residual against held-out rho_max over every fitted quadrature rule
(b-eqtop summary.json, table 'rules'), with the primary and tight bars drawn.  The point
of the figure: the fit residual does not predict the held-out error.  PROVISIONAL: one
draw per rule; the replication job in summary.json['pending'] is pending.

Usage:  /home/tahmid/Dev/.venv/bin/python paper/figures/gen_fig_eq_certification.py
"""
import hashlib, json
from collections import defaultdict
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC = Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-17-b-eqtop/experiments/b-eqtop/reports/summary.json')
OUT = Path(__file__).with_name('fig_eq_certification')
PAL = {0: '#3b6ea5', 16: '#4f8a3d', 32: '#7a5ba6', 64: '#d17f2a', 128: '#b5443c', 256: '#2b2b2b'}
BAR, TIGHT = 0.116, 0.06   # pre-registered bars (b-eqtop DESIGN.md)

d = json.loads(SRC.read_text())
R = defaultdict(dict)
for r in d['rows']:
    if r['table'] == 'rules':
        R[(r['q'], r['arm'], r['m'], r['population'], r['job_id'])][r['metric']] = r['value']
pts = [{'q': k[0], 'arm': k[1], 'm': k[2], 'population': k[3], 'job_id': k[4], 'fit': v['relative_fit'], 'rho_max': v['rho_max'],
        'static': k[1] == 'static'} for k, v in R.items() if v.get('relative_fit') is not None and v.get('rho_max') is not None]
plt.rcParams.update({'font.size': 7})
fig, ax = plt.subplots(figsize=(3.6, 2.6))
for p in pts:
    ax.plot(p['fit'], p['rho_max'], 's' if p['static'] else 'o', color=PAL[p['q']], ms=4, mfc=PAL[p['q']] if not p['static'] else 'white', mew=0.9, alpha=0.9)
ax.axhline(BAR, color='#2b2b2b', lw=0.8, ls='--'); ax.text(ax.get_xlim()[0] if False else 1.2e-5, BAR * 1.08, f'primary bar {BAR}', fontsize=6)
ax.axhline(TIGHT, color='#6b6b6b', lw=0.6, ls=':'); ax.text(1.2e-5, TIGHT * 1.08, f'tight bar {TIGHT}', fontsize=6, color='#6b6b6b')
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('NNLS relative fit residual on the fitting set'); ax.set_ylabel(r'held-out $\rho_{\max}$ on reachable states')
from matplotlib.lines import Line2D
h = [Line2D([], [], marker='o', ls='', color=PAL[q], label=f'$q={q}$') for q in sorted(PAL)] + [Line2D([], [], marker='s', ls='', color='#2b2b2b', mfc='white', label='static-snapshot fit')]
ax.legend(handles=h, fontsize=5.5, frameon=False, ncol=2, loc='lower left')
ax.grid(True, color='#e6e6e6', lw=0.5); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(OUT.with_suffix('.pdf')); fig.savefig(OUT.with_suffix('.png'), dpi=220)
OUT.with_suffix('.json').write_text(json.dumps({'source': str(SRC), 'sha256': hashlib.sha256(SRC.read_bytes()).hexdigest(), 'bars': {'primary': BAR, 'tight': TIGHT},
                                               'status': d.get('status'), 'points': pts}, indent=1))
n_low_fit_fail = sum(1 for p in pts if p['fit'] < 1e-3 and p['rho_max'] > BAR)
print('wrote', OUT.with_suffix('.pdf'), len(pts), 'rules;', n_low_fit_fail, 'rules fit below 1e-3 yet fail the primary bar')
