"""Figure 2 -- the family, from the three 2026-09-17 lanes.

Three panels, each read from one lane's machine-readable output, none from a
number typed here:

  A  Rank against error (b-qxm analysis.json): worst evolved error against the
     correction rank q for the fixed-M = 1088 ladder, the fixed-M = 256 ladder
     and the scheduled M = 4(K+q) ladder.  Errors are comparable across the
     lane's three jobs (shared cells agree to 1e-9); costs are not, so this
     panel has no cost axis and the within-job cost span of the fixed-M ladder
     is written as text from analysis.json.
  B  The 256^2 same-allocation panel (b-panel summary.json): median GPU ms
     against worst evolved error for every admissible subject, one job, one GPU.
     The non-dominated set over admissible subjects is drawn as a step line.
     Filled markers = converged (the pre-registered completion rule); hollow =
     not.
  C  The primary-rule EQ ladder against its dense twins in one allocation
     (b-eqtop summary.json, ladder rows).  PROVISIONAL: every certified flag is
     one draw; the replication job listed in summary.json['pending'] is pending.

Writes fig_tunability_family.{pdf,png,json}; the json holds every plotted point
with the file and SHA256 it came from.

Usage (CPU only):  /home/tahmid/Dev/.venv/bin/python paper/figures/gen_fig_tunability_family.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

DEFAULT_WORKTREES = Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees')
SOURCES = {
    'qxm': '2026-09-17-b-qxm/experiments/b-qxm/reports/analysis.json',
    'panel': '2026-09-17-b-panel/experiments/b-panel/reports/summary.json',
    'eqtop': '2026-09-17-b-eqtop/experiments/b-eqtop/reports/summary.json',
}
# Validated categorical palette (dataviz six-checks; see the 2026-09-16 version).
C_DENSE = '#3b6ea5'; C_EQ = '#4f8a3d'; C_FOM = '#d17f2a'; C_FNO = '#7a5ba6'
C_POD = '#9a9a9a'; C_FREE = '#2b2b2b'; C_SCHED = '#b5443c'; INK = '#2b2b2b'; MUTED = '#6b6b6b'


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as fh:
        for c in iter(lambda: fh.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def load(root, key):
    p = root / SOURCES[key]
    if not p.exists():
        raise SystemExit(f'missing source {p}')
    return json.loads(p.read_text()), {'path': str(p), 'sha256': sha256(p)}


def pareto(pts):
    out = []
    for p in pts:
        if not any(q is not p and q['cost_ms'] <= p['cost_ms'] and q['err'] <= p['err']
                   and (q['cost_ms'] < p['cost_ms'] or q['err'] < p['err']) for q in pts):
            out.append(p)
    return sorted(out, key=lambda p: p['cost_ms'])


def panel_a(ax, a):
    ev = a['spans']['worst_evolved_percent']
    series = []
    for M, col, lab in [('1088', C_DENSE, 'fixed $M{=}1088$'), ('256', '#7fa7d1', 'fixed $M{=}256$')]:
        d = ev['fixed_M'][M]
        series.append({'name': lab, 'q': d['q'], 'err': d['values'], 'converged': d['all_converged']})
        ax.plot(d['q'], d['values'], '-o', color=col, ms=4, lw=1.4, label=lab)
    s = ev['scheduled']['4x']
    qs = [c[0] for c in s['cells']]
    series.append({'name': 'scheduled M=4(K+q)', 'q': qs, 'err': s['values']})
    ax.plot(qs, s['values'], '--s', color=C_SCHED, ms=4, lw=1.2, label='scheduled $M{=}4(K{+}q)$')
    wj = ev['fixed_M']['1088']['within_job']
    ax.text(0.03, 0.05, f"fixed-$M$ ladder inside job {wj['job_id']}:\n"
            f"error span {wj['error_span']:.2f}$\\times$, cost span {wj['cost_span']:.2f}$\\times$,\n"
            f"{wj['non_dominated_points']} non-dominated, all converged",
            transform=ax.transAxes, fontsize=6.5, color=INK, va='bottom')
    ax.set_xscale('symlog', linthresh=16); ax.set_xticks([0, 16, 32, 64, 128, 256]); ax.set_xticklabels(['0', '16', '32', '64', '128', '256'])
    ax.set_xlabel('correction rank $q$'); ax.set_ylabel('worst evolved error (%)')
    ax.set_title('A  rank against error, dense (b-qxm)', fontsize=8, loc='left')
    ax.legend(fontsize=6.5, frameon=False, loc='upper right')
    return series


def panel_b(ax, s):
    rows = [r for r in s['rows'] if int(r['mesh']) == 256]   # the 256^2 job; the 1024^2 job is another GPU
    P = defaultdict(dict); fam = {}
    for r in rows:
        P[r['subject']][r['metric']] = r['value']; fam[r['subject']] = r['family']
    pts = []
    style = {'rom': (C_DENSE, 'o'), 'fast': (C_EQ, 'D'), 'pod': (C_POD, 's'), 'free': (C_FREE, '^'), 'fno': (C_FNO, 'P'), 'fom': (C_FOM, 'v')}
    for sub, d in P.items():
        if sub == '*' or not d.get('admissible'):
            continue
        f = fam[sub]; col, mk = style[f]
        if f == 'rom' and ('eqcert' in sub or 'eqtop' in sub):
            col = C_EQ
        # DESIGN §5 as written (fixed 1e-6 at every step), the lane's primary flag since its §A13;
        # the loose-tolerance arms are not admissible under it and are filtered out above
        conv = d.get('converged_design5'); conv = True if conv is None else bool(conv)
        p = {'subject': sub, 'family': f, 'cost_ms': d['median_gpu_ms'], 'err': d['worst_evolved_percent'],
             'err_all': d.get('worst_all_times_percent'), 'converged': conv}
        pts.append(p)
        ax.plot(p['cost_ms'], max(p['err'], 1e-3), mk, color=col, ms=4.5, mfc=col if conv else 'white', mew=1.0, alpha=0.95)
    front = pareto(pts)
    ax.step([p['cost_ms'] for p in front], [max(p['err'], 1e-3) for p in front], where='post', color=INK, lw=0.8, alpha=0.6)
    for p in front:
        if p['family'] == 'fom' and p['err'] == 0:
            ax.annotate('reference (0 by construction)', (p['cost_ms'], 1e-3), fontsize=5.5, color=MUTED, xytext=(3, 3), textcoords='offset points')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(8e-4, 80)
    ax.set_xlabel('median GPU ms, one allocation'); ax.set_ylabel('worst evolved error (%)')
    ax.set_title(f"B  the $256^2$ panel, job {rows[0]['job_id']} (b-panel)", fontsize=8, loc='left')
    handles = [Line2D([], [], marker='o', color=C_DENSE, ls='', label='rungs, dense'),
               Line2D([], [], marker='o', color=C_EQ, ls='', label='rungs, EQ (both rule sets)'),
               Line2D([], [], marker='s', color=C_POD, ls='', label='POD-LSPG'),
               Line2D([], [], marker='^', color=C_FREE, ls='', label='free bank'),
               Line2D([], [], marker='P', color=C_FNO, ls='', label='FNO'),
               Line2D([], [], marker='v', color=C_FOM, ls='', label='full-order Newton'),
               Line2D([], [], color=INK, lw=0.8, alpha=0.6, label='non-dominated set')]
    ax.legend(handles=handles, fontsize=6, frameon=False, loc='upper right')
    return pts, [p['subject'] for p in front]


def panel_c(ax, e):
    rows = e['rows']
    L = defaultdict(dict); meta = {}
    for r in rows:
        if r['table'] == 'ladder':
            L[(r['ladder'], r['arm'])][r['metric']] = r['value']; meta[(r['ladder'], r['arm'])] = r
    prim = sorted([k for k in L if k[0] == 'primary'], key=lambda k: meta[k]['q'])
    dense = sorted([k for k in L if k[0] == 'dense'], key=lambda k: meta[k]['q'])
    out = []
    for keys, col, lab, mk in [(prim, C_EQ, 'primary-certified EQ (one draw)', 'o'), (dense, C_DENSE, 'dense twins', 's')]:
        xs = [L[k]['median_gpu_ms'] for k in keys]; ys = [L[k]['worst_evolved_percent'] for k in keys]
        ax.plot(xs, ys, '-' + mk, color=col, ms=4.5, lw=1.2, label=lab)
        for k, x, y in zip(keys, xs, ys):
            ax.annotate(f"$q{{=}}{meta[k]['q']}$", (x, y), fontsize=5.5, color=MUTED, xytext=(3, -7 if col == C_EQ else 3), textcoords='offset points')
            out.append({'ladder': k[0], 'arm': k[1], 'q': meta[k]['q'], 'm': meta[k]['m'], 'rho_max': meta[k]['rho_max'],
                        'certified_primary': meta[k]['certified_primary'], 'cost_ms': x, 'err': y, 'job_id': meta[k]['job_id']})
    ax.set_xscale('log'); ax.set_xlabel('median GPU ms, one allocation'); ax.set_ylabel('worst evolved error (%)')
    pend = ', '.join(p['job_id'] for p in e.get('pending', []))
    ax.set_title(f"C  EQ ladder, job {meta[prim[0]]['job_id']} (b-eqtop)" + (f" -- PROVISIONAL, {pend} pending" if pend else ''), fontsize=8, loc='left')
    ax.legend(fontsize=6.5, frameon=False, loc='upper right')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--worktrees', default=str(DEFAULT_WORKTREES))
    ap.add_argument('--out', default=str(Path(__file__).with_name('fig_tunability_family')))
    args = ap.parse_args()
    root = Path(args.worktrees); out = Path(args.out)
    qxm, prov_q = load(root, 'qxm'); panel, prov_p = load(root, 'panel'); eqtop, prov_e = load(root, 'eqtop')
    plt.rcParams.update({'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
                         'axes.edgecolor': MUTED, 'axes.linewidth': 0.6, 'figure.facecolor': 'white'})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.2))
    for ax in axes:
        ax.grid(True, which='major', color='#e6e6e6', lw=0.5); ax.set_axisbelow(True)
    a = panel_a(axes[0], qxm); b, front = panel_b(axes[1], panel); c = panel_c(axes[2], eqtop)
    fig.tight_layout(w_pad=1.2)
    fig.savefig(out.with_suffix('.pdf')); fig.savefig(out.with_suffix('.png'), dpi=220)
    out.with_suffix('.json').write_text(json.dumps({
        'sources': {'qxm': prov_q, 'panel': prov_p, 'eqtop': prov_e},
        'panel_a': a, 'panel_b': {'points': b, 'non_dominated_admissible': front},
        'panel_c': c, 'status': {'panel_c': eqtop.get('status')},
        'labels': {'seed': 'single training seed', 'cohort': 'development cohort', 'cost': 'never compared across panels'},
    }, indent=1))
    print('wrote', out.with_suffix('.pdf'), out.with_suffix('.json'))


if __name__ == '__main__':
    main()
