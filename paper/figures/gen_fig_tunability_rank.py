"""Figure 2: worst error against speedup over one named full-order setting per series, at 4096^2 (log-log).

Reads ONLY paper/tables/headline-provenance.json ("tunability", written by gen_headline.py from hash-pinned
snapshots): one frozen model and one allocation per series; each series divides by ONE full-order setting,
chosen by the paper's rule for its most accurate rung.  Rank rungs are joined in order of (q, M); the tolerance
setting is a separate marker at the same error; rungs whose quadrature rule failed its held-out check are grey.
"""
import hashlib, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
raw = (HERE.parent / 'tables/headline-provenance.json').read_bytes(); prov = json.loads(raw)
INK = '#2b2b2b'; MUTED = '#8a8a8a'
COL = {'Poisson': ('#2a78d6', 'o'), 'Heat': ('#eb6834', 'P'), 'Burgers': ('#1baf7a', '^')}
plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.edgecolor': '#6b6b6b', 'axes.labelcolor': INK,
                     'xtick.color': '#6b6b6b', 'ytick.color': '#6b6b6b', 'axes.linewidth': 0.6})
fig, ax = plt.subplots(figsize=(6.6, 1.85))
ax.axvline(1, color='#6b6b6b', lw=0.8, ls=':'); ax.text(1.05, 7.0, 'FOM parity', fontsize=6.5, color='#6b6b6b')
out = {}
for s in prov['tunability']:
    fam = s['series'].split()[0]; c, mk = COL[fam]; hollow = s['style'] == 'hollow'
    for r in s['rungs']:
        assert abs(s['fom']['ms'] / r['ms'] - r['speedup']) < 1e-9          # every ratio reproduces from ms
    rank = sorted([r for r in s['rungs'] if r['kind'] == 'rank'], key=lambda r: (r['q'], r.get('M') or 0))
    ok = [r for r in rank if r['certified']]
    ax.plot([r['speedup'] for r in ok], [r['err'] for r in ok], color=c, lw=1.2, ls='--' if hollow else '-', zorder=2)
    for r in rank:
        cc = c if r['certified'] else MUTED
        ax.plot(r['speedup'], r['err'], marker=mk, ms=5.5, mfc='white' if (hollow or not r['certified']) else cc, mec=cc, mew=1.1, ls='none', zorder=3)
        if not r['certified'] and not hollow:
            ax.annotate('rule not certified', (r['speedup'], r['err']), xytext=(-4, -9), textcoords='offset points', fontsize=6, color=MUTED, ha='right')
    for r in [r for r in s['rungs'] if r['kind'] == 'tolerance']:
        ax.plot(r['speedup'], r['err'], marker='*', ms=7, mfc='white' if hollow else c, mec=c, mew=1.0, ls='none', zorder=4)
    if not hollow:
        for r in ok:
            lab = f"$q={r['q']}$" + (f", $M={r['M']}$" if fam == 'Burgers' and r['q'] == 256 else '')
            left = fam == 'Poisson' or (fam == 'Burgers' and r['q'] == 256)
            ax.annotate(lab, (r['speedup'], r['err']), xytext=(-6, -2) if left else (5, 2), ha='right' if left else 'left',
                        textcoords='offset points', fontsize=6, color=INK)
    out[s['series']] = dict(fom=s['fom'], rungs=s['rungs'])
ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlim(0.8, 400); ax.set_xlabel('speedup over one named full-order setting (4096$^2$)')
ax.set_ylabel('worst error (%)'); ax.grid(True, which='major', color='#e6e6e6', lw=0.5); ax.set_axisbelow(True)
for sp_ in ('top', 'right'): ax.spines[sp_].set_visible(False)
h = [Line2D([], [], color=COL[f][0], marker=COL[f][1], lw=1.2, label=l) for f, l in
     (('Burgers', 'Burgers, development'), ('Poisson', 'Poisson'), ('Heat', 'Heat (wide bank, sealed)'))]
h += [Line2D([], [], color=COL['Burgers'][0], marker='^', mfc='white', lw=1.2, ls='--', label='Burgers, held-out 64'),
      Line2D([], [], color=INK, marker='*', mfc=INK, ls='none', label='looser stopping tolerance')]
ax.legend(handles=h, frameon=False, fontsize=6, loc='lower left', bbox_to_anchor=(0.36, 0.0), ncol=2, columnspacing=1.0, handlelength=2.2)
fig.tight_layout(pad=0.4)
for ext in ('pdf', 'png'):
    fig.savefig(HERE / f'fig_tunability_rank.{ext}', dpi=300)
(HERE / 'fig_tunability_rank.json').write_text(json.dumps(dict(evidence_sha256=hashlib.sha256(raw).hexdigest(), series=out), indent=2) + '\n')
print('fig_tunability_rank:', sum(len(v['rungs']) for v in out.values()), 'points')
