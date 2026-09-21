"""Speedup over the named full-order solver against mesh resolution (log-log).

Reads ONLY ``paper/tables/headline-provenance.json`` (written by gen_headline.py
from hash-pinned snapshots), so the figure and the headline table cannot
disagree.  A new mesh point appears here as soon as its audited summary is
listed in ``paper/headline-intake.json`` and gen_headline.py --refresh is run;
nothing in this file names a mesh or a value.

One colour + marker per problem; solid line / filled marker = fast setting,
dashed line / hollow marker = accurate setting; left panel 2D, right panel 3D.
Writes fig_speedup_resolution.{pdf,png,json}.

Usage (CPU only): /home/tahmid/Dev/.venv/bin/python paper/figures/gen_fig_speedup_resolution.py
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
# dataviz validator: passes light mode; aqua/magenta sit in the 6-8 CVD band, so every
# series also carries its own marker shape and a direct label (secondary encoding).
STYLE = {'Poisson': ('#2a78d6', 'o'), 'Poisson (dev. sources)': ('#2a78d6', 'v'), 'Poisson, L-shape': ('#e87ba4', 'D'), 'Heat': ('#eb6834', 's'), 'Heat (wide bank)': ('#eb6834', 'P'), 'Heat (wide bank, batched fit)': ('#eb6834', 'X'), 'Burgers': ('#1baf7a', '^')}
INK = '#2b2b2b'; MUTED = '#6b6b6b'
LABEL = {'Heat (wide bank)': 'Heat, wide bank', 'Heat (wide bank, batched fit)': 'Heat, wide bank, batched fit'}
NUDGE = {'Poisson': (6, 6), 'Burgers': (6, -8), 'Heat (wide bank, batched fit)': (6, -1), 'Heat (wide bank)': (6, -7)}   # end labels that would otherwise touch at 4096^2

ap = argparse.ArgumentParser()
ap.add_argument('--evidence', default=str(HERE.parent / 'tables/headline-provenance.json'))
ap.add_argument('--out', default=str(HERE / 'fig_speedup_resolution'))
a = ap.parse_args()
raw = Path(a.evidence).read_bytes(); prov = json.loads(raw)
for k, v in prov['sources'].items():
    assert v.get('sha256'), k  # refuse to draw from an unpinned source

series = {}
for r in prov['rows']:
    if r['problem'] not in STYLE: continue          # the earlier-model Burgers rows have two FOMs for one point; table only
    assert r['job_id']
    for s in ('fast', 'accurate'):
        if r[s]:
            series.setdefault((r['dim'], r['problem'], s), []).append(
                dict(intervals=r['intervals'], speedup=r[s]['speedup'], error_pct=r[s]['error_pct'], fom=r['fom']['name'], eq=r[s].get('eq'),
                     status=r['status'], job_id=r['job_id'], source=r['source']))

plt.rcParams.update({'font.size': 8, 'font.family': 'serif', 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.linewidth': 0.6})
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.1), sharey=True, gridspec_kw={'width_ratios': [1.55, 1]})
for ax, dim in zip(axes, (2, 3)):
    ax.axhline(1, color=MUTED, lw=0.8, ls=':'); ax.set_xscale('log', base=2); ax.set_yscale('log')
    for (d, p, s), pts in sorted(series.items()):
        if d != dim: continue
        pts.sort(key=lambda x: x['intervals']); c, mk = STYLE[p]
        prov_flag = any(x['status'].startswith('provisional') for x in pts)
        ax.plot([x['intervals'] for x in pts], [x['speedup'] for x in pts], color=c, lw=1.6 if s == 'fast' else 1.2,
                ls='-' if s == 'fast' else '--', marker=mk, ms=5.5, mfc=c if s == 'fast' else 'white', mec=c, mew=1.2, zorder=3)
        for x in pts:          # a dense-residual accurate point: marked, so it is not read as the method's quadrature path
            if x.get('eq') == 'dense':
                ax.plot([x['intervals']], [x['speedup']], marker=mk, ms=5.5, mfc='white', mec=MUTED, mew=1.0, ls='none', zorder=4)
                ax.annotate('dense residual', (x['intervals'], x['speedup']), xytext=(6, -3), textcoords='offset points', fontsize=6.5, color=MUTED)
        if s == 'fast':
            # labels whose end point is crowded are anchored at the first point instead: (offset, horizontal alignment)
            AT_FIRST = {'Poisson, L-shape': ((-6, 9), 'right'), 'Heat': ((4, -13), 'left')}
            AT_LAST = {'Poisson (dev. sources)': ((-8, 7), 'right')}
            first = p in AT_FIRST
            x = pts[0] if first else pts[-1]
            off, ha = AT_FIRST[p] if first else AT_LAST.get(p, (NUDGE.get(p, (6, 2)), 'left'))
            ax.annotate(LABEL.get(p, p) + (' (provisional)' if prov_flag else ''), (x['intervals'], x['speedup']), xytext=off,
                        ha=ha, textcoords='offset points', fontsize=7, color=INK)
    ticks = sorted({x['intervals'] for (d, _, _), pts in series.items() if d == dim for x in pts})
    ax.set_xticks(ticks); ax.set_xticklabels([f'${t}^{dim}$' for t in ticks]); ax.minorticks_off()
    ax.set_xlim(ticks[0] / 1.35, ticks[-1] * (2.6 if dim == 2 else 2.2))
    ax.set_xlabel(f'mesh ({dim}D)'); ax.grid(True, which='major', color='#e6e6e6', lw=0.5); ax.set_axisbelow(True)
    for sp_ in ('top', 'right'): ax.spines[sp_].set_visible(False)
axes[0].set_ylabel('speedup over the named FOM')
axes[0].text(0.01, 1.0, ' FOM parity', transform=axes[0].get_yaxis_transform(), fontsize=6.5, color=MUTED, va='bottom')
axes[1].legend(handles=[Line2D([], [], color=INK, lw=1.6, marker='o', ms=5, mfc=INK, label='fast setting'),
                        Line2D([], [], color=INK, lw=1.2, ls='--', marker='o', ms=5, mfc='white', label='accurate setting')],
               frameon=False, fontsize=7, loc='lower right')
fig.tight_layout(pad=0.4)
for ext in ('pdf', 'png'):
    fig.savefig(f'{a.out}.{ext}', dpi=300)
Path(a.out + '.json').write_text(json.dumps(dict(evidence=Path(a.evidence).name, evidence_sha256=hashlib.sha256(raw).hexdigest(),
    series={f'{d}D|{p}|{s}': pts for (d, p, s), pts in sorted(series.items())},
    note='Development, final and provisional cohorts are mixed across series; each point carries its status. Points within a series share one frozen model; FOM settings are those of the headline table.'), indent=2) + '\n')
print(f'fig_speedup_resolution: {sum(len(v) for v in series.values())} points in {len(series)} series')
