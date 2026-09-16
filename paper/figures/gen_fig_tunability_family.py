"""Figure F6 — the inference-time operating-point family, error against cost.

Supports claims C0 (three-layer error decomposition) and C6 (a single trained
decoder exposes a monotone family of offline-prepared inference-time operating
points).  Three panels:

  A  Burgers 2D, 256 intervals: the correction-rank ladder against the POD rank
     ladder, with the same-job full-order controls as vertical references.
  B  Poisson 2D, 1024 intervals: the same, where the corrections are eliminated
     exactly and the nonlinear iteration stays K-dimensional.
  C  Solver-side tuning (stopping tolerance, iteration cap, quadrature rule) on
     the 32 held-out validation cases.  DIFFERENT METRIC AND DIFFERENT COHORT
     from A and B; it is drawn in its own panel for that reason and its points
     must never be compared with A's by eye.

PROVISIONAL.  Every number is read from an audited run JSON and nothing is typed
into this script, but the panels combine FOUR CLUSTER JOBS:

    3734098  cheap-corrections Burgers   (A: correction rungs, FOM verticals)
    3734084  cheap-corrections Poisson   (B: correction rungs, DST vertical)
    3711424  head-ablation Burgers       (A: POD rank ladder)
    3711736  head-ablation Poisson       (B: POD rank ladder)
    3712269  fixed-checkpoint tuning     (C)

Timings from different allocations may not be divided.  The error axis is
same-metric and same-cohort within each panel and is comparable; **the cost axis
across series within a panel is cross-job and therefore provisional.**  The
same-job envelope — correction rungs, the POD ladder and the full-order controls
interleaved in one job on one GPU — is a separate experiment and is not this
figure.

Usage (CPU only, no GPU, no cluster):

    /home/tahmid/Dev/.venv/bin/python paper/figures/gen_fig_tunability_family.py

Writes, beside itself:  fig_tunability_family.{png,pdf}  and
                        fig_tunability_family.json  (every plotted point, with
                        the file and SHA256 it came from).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Validated categorical palette (dataviz six-checks, light surface #fcfcfb:
# lightness band PASS, chroma floor PASS, CVD separation PASS worst dE 18.1
# deutan / 9.7 tritan, normal-vision floor PASS 24.4, contrast PASS).
C_DENSE = '#3b6ea5'   # correction rungs, exact dense quadrature
C_EQ    = '#4f8a3d'   # correction rungs, fitted empirical quadrature
C_FOM   = '#d17f2a'   # full-order controls
C_TUNE  = '#7a5ba6'   # solver-side tuning arms
C_POD   = '#9a9a9a'   # POD rank ladder: recessive baseline, not an identity hue
INK     = '#2b2b2b'
INK_MUTED = '#6b6b6b'

DEFAULT_WORKTREES = Path(
    '/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees')

SOURCES = {
    'burgers_rungs': ('2026-09-15-cheap-corrections/experiments/cheap-corrections/'
                      'checks/cclad01-audit.json'),
    'burgers_result': ('2026-09-15-cheap-corrections/experiments/cheap-corrections/'
                       'artifacts/cclad01/result.json'),
    'poisson_rungs': ('2026-09-15-cheap-corrections/experiments/cheap-corrections/'
                      'checks/ccpoi01-audit.json'),
    'poisson_result': ('2026-09-15-cheap-corrections/experiments/cheap-corrections/'
                       'artifacts/ccpoi01/result.json'),
    'burgers_pod': ('2026-09-14-head-ablation/experiments/head-ablation/'
                    'checks/abl01-audit.json'),
    'poisson_pod': ('2026-09-14-head-ablation/experiments/head-ablation/'
                    'checks/pabl01-audit.json'),
    'tuning': ('2026-09-14-no-burgers/experiments/neural-operator-burgers/'
               'artifacts/tuning02/output-index.json'),
}

POD_ORDER = ['e_pod8', 'e_pod16', 'e_pod32', 'e_pod64', 'e_pod128']
# The Burgers ablation names each POD rung by its quadrature; the dense rows are
# the ones run with exact advection, which is the arm that favours POD.
POD_ORDER_BURGERS = [a + '_dense' for a in POD_ORDER]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load(root: Path, key: str):
    path = root / SOURCES[key]
    if not path.exists():
        return None, {'key': key, 'path': str(path), 'reachable': False}
    return json.loads(path.read_text()), {
        'key': key, 'path': str(path), 'reachable': True, 'sha256': sha256(path)}


def pareto(points):
    """Non-dominated set on (cost, error), both minimised."""
    out = []
    for p in points:
        if not any(q is not p and q['cost_ms'] <= p['cost_ms']
                   and q['error_percent'] <= p['error_percent']
                   and (q['cost_ms'] < p['cost_ms'] or q['error_percent'] < p['error_percent'])
                   for q in points):
            out.append(p)
    return sorted(out, key=lambda p: p['cost_ms'])


# --------------------------------------------------------------------------- A
def burgers_series(audit, pod_audit):
    """Correction rungs (cheap-corrections) and the POD rank ladder (abl01)."""
    rungs, foms = [], []
    for row in audit['checks']['arm_table']:
        if row['kind'] == 'rom':
            rungs.append({
                'arm': row['arm'], 'q': row['q'], 'rule': row['rule'],
                'quadrature': row['quadrature'], 'variant': row['variant'],
                'error_percent': row['worst_same_grid_percent'],
                'cost_ms': row['median_gpu_ms'],
                'converged': bool(row['converged']),
                'budget_exits': row['total_budget_exits'],
                'bank_projection_percent': row['worst_bank_projection_percent'],
                'best_found_percent': row['worst_best_found_percent'],
            })
        else:
            foms.append({'arm': row['arm'],
                         'error_percent': row['worst_same_grid_percent'],
                         'cost_ms': row['median_gpu_ms']})
    pod = []
    if pod_audit is not None:
        by_arm = {r['arm']: r for r in pod_audit['checks']['arm_table']
                  if r.get('intervals') == 256}
        for arm in POD_ORDER_BURGERS:
            if arm in by_arm:
                r = by_arm[arm]
                pod.append({'arm': arm,
                            'rank': int(arm.split('pod')[1].split('_')[0]),
                            'error_percent': r['worst_same_grid_percent'],
                            'cost_ms': r['median_gpu_ms']})
    return rungs, foms, pod


# --------------------------------------------------------------------------- B
def poisson_series(audit, pod_audit):
    rungs, foms = [], []
    for row in audit['checks']['arm_table']:
        if row['kind'] == 'rom':
            rungs.append({
                'arm': row['arm'], 'q': row['q'], 'rule': row['rule'],
                'error_percent': row['worst_physical_percent'],
                'cost_ms': row['median_device_ms'],
                'solver_valid': bool(row['all_solver_valid']),
                'degenerate_full_bank': bool(row['degenerate_full_bank']),
                'bank_projection_percent': row['worst_bank_projection_percent'],
                'best_found_percent': row['worst_best_found_percent'],
            })
        else:
            foms.append({'arm': row['arm'],
                         'error_percent': row['worst_physical_percent'],
                         'cost_ms': row['median_device_ms']})
    pod = []
    if pod_audit is not None:
        by_arm = {r['arm']: r for r in pod_audit['checks']['arm_table']
                  if r.get('intervals') == 1024}
        for arm in POD_ORDER:
            if arm in by_arm:
                r = by_arm[arm]
                pod.append({'arm': arm, 'rank': int(arm.split('pod')[1]),
                            'error_percent': r['worst_error_percent'],
                            'cost_ms': r['median_query_ms']})
    return rungs, foms, pod


# --------------------------------------------------------------------------- C
def tuning_series(index):
    rom, fom = [], []
    for name, s in index['summaries'].items():
        point = {
            'arm': s['method'],
            'error_percent': 100.0 * s['worst_fixed_initial_error'],
            'cost_ms': 1000.0 * s['median_gpu_seconds'],
            'early_stopped': int(s['early_stopped_invocations']),
            'invocations': s['invocations'],
        }
        (fom if s['method'].startswith(('same_', 'coarse_')) else rom).append(point)
    return sorted(rom, key=lambda p: p['cost_ms']), sorted(fom, key=lambda p: p['cost_ms'])


# ------------------------------------------------------------------- plotting
def draw_pod(ax, pod, label):
    if not pod:
        return False
    ax.plot([p['cost_ms'] for p in pod], [p['error_percent'] for p in pod],
            '-o', color=C_POD, lw=1.4, ms=6, mfc=C_POD, alpha=.75, zorder=2)
    for i, p in enumerate(pod):
        ax.annotate(f"POD r={p['rank']}", (p['cost_ms'], p['error_percent']),
                    textcoords='offset points',
                    xytext=(9, 3) if i % 2 == 0 else (-9, -13),
                    ha='left' if i % 2 == 0 else 'right',
                    fontsize=6.5, color=INK_MUTED)
    return True


def draw_foms(ax, foms, floor=None):
    for f in foms:
        e = f['error_percent']
        if floor is not None and e is not None and 0 < e < floor:
            ax.axvline(f['cost_ms'], color=C_FOM, ls='-.', lw=1.4, alpha=.9, zorder=1)
            ax.annotate(f"FOM {f['arm']}\n{e:.2g}% error, off scale",
                        (f['cost_ms'], 0.0), xycoords=('data', 'axes fraction'),
                        textcoords='offset points', xytext=(4, 6),
                        fontsize=6.5, color=C_FOM, va='bottom')
            continue
        if e is not None and e > 0:
            ax.plot(f['cost_ms'], e, 'X', ms=11, color=C_FOM, mec='white', mew=.8,
                    zorder=6)
            ax.annotate(f"FOM {f['arm']}", (f['cost_ms'], e),
                        textcoords='offset points', xytext=(7, 4),
                        fontsize=6.5, color=C_FOM)
        else:
            ax.axvline(f['cost_ms'], color=C_FOM, ls='-.', lw=1.4, alpha=.9, zorder=1)
            ax.annotate(f"FOM {f['arm']}\n(exact, cost only)", (f['cost_ms'], 0.0),
                        xycoords=('data', 'axes fraction'),
                        textcoords='offset points', xytext=(4, 6),
                        fontsize=6.5, color=C_FOM, va='bottom')


def panel_a(ax, rungs, foms, pod, front):
    drew_pod = draw_pod(ax, pod, 'POD-LSPG rank ladder')
    for r in rungs:
        col = C_EQ if r['quadrature'] == 'eq' else C_DENSE
        ax.plot(r['cost_ms'], r['error_percent'], 'o', ms=6.5, color=col,
                mfc=col if r['converged'] else 'none', mew=1.3, alpha=.55, zorder=3)
    ax.plot([p['cost_ms'] for p in front], [p['error_percent'] for p in front],
            '-', color=C_EQ, lw=2.0, zorder=4)
    for i, p in enumerate(front):
        ax.plot(p['cost_ms'], p['error_percent'], 'o', ms=9, color=C_EQ,
                mfc=C_EQ if p['converged'] else 'none', mew=1.6, zorder=5)
        ax.annotate(f"q={p['q']}", (p['cost_ms'], p['error_percent']),
                    textcoords='offset points',
                    xytext=(-2, 13) if i % 2 == 0 else (2, -17), ha='center',
                    fontsize=7.5, color=C_EQ, weight='bold')
    draw_foms(ax, foms)
    ax.set_title('A · Burgers 2D, 256 intervals\ncorrection rank vs POD rank',
                 fontsize=9, color=INK)
    ax.set_ylabel('worst same-grid error vs converged FOM (%)', fontsize=8)
    return drew_pod


def panel_b(ax, rungs, foms, pod):
    draw_pod(ax, pod, 'POD ladder')
    shown = [r for r in rungs if r['rule'] == 'm256']
    shown.sort(key=lambda r: r['q'])
    ax.plot([r['cost_ms'] for r in shown], [r['error_percent'] for r in shown],
            '-', color=C_DENSE, lw=2.0, zorder=4)
    for r in rungs:
        main = r['rule'] == 'm256'
        ok = r['solver_valid'] and not r['degenerate_full_bank']
        ax.plot(r['cost_ms'], r['error_percent'], 'o',
                ms=9 if main else 6, color=C_DENSE,
                mfc=C_DENSE if ok else 'none', mew=1.5 if main else 1.1,
                alpha=1.0 if main else .45, zorder=5 if main else 3)
    for i, r in enumerate(shown):
        ax.annotate(f"q={r['q']}", (r['cost_ms'], r['error_percent']),
                    textcoords='offset points',
                    xytext=(-26, 2) if i % 2 == 0 else (8, -3),
                    fontsize=7.5, color=C_DENSE, weight='bold')
    floor = 0.05 * min(r['error_percent'] for r in rungs)
    draw_foms(ax, foms, floor=floor)
    ax.set_title('B · Poisson 2D, 1024 intervals\ncorrections eliminated exactly',
                 fontsize=9, color=INK)
    ax.set_ylabel('worst physical error (%)', fontsize=8)


def panel_c(ax, rom, fom):
    for p in rom:
        ax.plot(p['cost_ms'], p['error_percent'], 's', ms=8, color=C_TUNE,
                mfc=C_TUNE if p['early_stopped'] == 0 else 'none', mew=1.4, zorder=4)
        ax.annotate(p['arm'], (p['cost_ms'], p['error_percent']),
                    textcoords='offset points', xytext=(0, 13), ha='center',
                    fontsize=6.5, color=C_TUNE)
    for p in fom:
        ax.plot(p['cost_ms'], p['error_percent'], 'X', ms=10, color=C_FOM,
                mec='white', mew=.8, zorder=5)
        ax.annotate(p['arm'], (p['cost_ms'], p['error_percent']),
                    textcoords='offset points', xytext=(0, -17), ha='center',
                    fontsize=6.5, color=C_FOM)
    ax.set_title('C · solver-side tuning, 32 held-out cases\n'
                 'DIFFERENT metric and cohort from A — do not compare by eye',
                 fontsize=9, color=INK)
    ax.set_ylabel('worst fixed-initial error (%)', fontsize=8)


def build(root: Path, out_stem: Path):
    prov = []
    b_audit, p1 = load(root, 'burgers_rungs'); prov.append(p1)
    b_result, p2 = load(root, 'burgers_result'); prov.append(p2)
    p_audit, p3 = load(root, 'poisson_rungs'); prov.append(p3)
    p_result, p4 = load(root, 'poisson_result'); prov.append(p4)
    b_pod, p5 = load(root, 'burgers_pod'); prov.append(p5)
    p_pod, p6 = load(root, 'poisson_pod'); prov.append(p6)
    tune, p7 = load(root, 'tuning'); prov.append(p7)

    if b_audit is None or p_audit is None:
        raise SystemExit('the cheap-corrections audit JSONs are required and were '
                         'not found; nothing is drawn from a fallback')

    b_rungs, b_foms, b_podpts = burgers_series(b_audit, b_pod)
    p_rungs, p_foms, p_podpts = poisson_series(p_audit, p_pod)
    b_front = pareto([r for r in b_rungs if r['converged']])
    t_rom, t_fom = tuning_series(tune) if tune is not None else ([], [])

    ncol = 3 if tune is not None else 2
    fig, axes = plt.subplots(1, ncol, figsize=(4.9 * ncol, 5.4),
                             constrained_layout=True)
    axes = list(axes)
    drew_pod = panel_a(axes[0], b_rungs, b_foms, b_podpts, b_front)
    panel_b(axes[1], p_rungs, p_foms, p_podpts)
    if tune is not None:
        panel_c(axes[2], t_rom, t_fom)

    for ax in axes:
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('median complete-query GPU time (ms), log', fontsize=8)
        ax.grid(True, which='both', alpha=.18, lw=.6)
        ax.margins(x=.16, y=.20)
        ax.tick_params(labelsize=7.5, colors=INK_MUTED)
        for spine in ('top', 'right'):
            ax.spines[spine].set_visible(False)

    handles = [
        Line2D([], [], color=C_EQ, marker='o', lw=2.0, ms=8,
               label='correction rungs, fitted quadrature (Burgers)'),
        Line2D([], [], color=C_DENSE, marker='o', lw=2.0, ms=8,
               label='correction rungs, exact algebra (Poisson m256)'),
        Line2D([], [], color=C_POD, marker='o', lw=1.4, ms=6,
               label='POD-LSPG rank ladder (cross-job baseline)'),
        Line2D([], [], color=C_TUNE, marker='s', lw=0, ms=8,
               label='solver-side tuning arm (panel C)'),
        Line2D([], [], color=C_FOM, marker='X', lw=0, ms=10,
               label='same-job full-order control'),
        Line2D([], [], color=INK, marker='o', lw=0, ms=8, mfc=INK,
               label='filled = converged / solver-valid / not early-stopped'),
        Line2D([], [], color=INK, marker='o', lw=0, ms=8, mfc='none', mew=1.4,
               label='hollow = budget exit, degenerate, or early-stopped'),
    ]
    if not drew_pod:
        handles = [h for h in handles if 'POD' not in (h.get_label() or '')]
    fig.legend(handles=handles, loc='lower center', ncol=4, fontsize=7.2,
               frameon=False, bbox_to_anchor=(0.5, -0.075))
    fig.suptitle(
        'A single trained decoder exposes a monotone family of offline-prepared '
        'inference-time operating points\n'
        'PROVISIONAL: the cost axis combines separate cluster jobs and may not be '
        'divided; the same-job envelope is a separate experiment',
        fontsize=10, color=INK)

    fig.savefig(out_stem.with_suffix('.png'), dpi=220, bbox_inches='tight')
    fig.savefig(out_stem.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)

    payload = {
        'figure': out_stem.name,
        'generated_by': 'paper/figures/gen_fig_tunability_family.py',
        'supports': ['C0', 'C6'],
        'provisional': True,
        'provisional_reason': (
            'Panels combine measurements from four cluster jobs. Error is '
            'same-metric and same-cohort within each panel; the cost axis is '
            'cross-job and no ratio across series may be taken. Panel C uses a '
            'different error metric (worst fixed-initial) and a different cohort '
            '(32 held-out validation cases) from panels A and B (worst same-grid '
            'or worst physical, on the opened development cohorts).'),
        'jobs': {
            'burgers_correction_rungs': b_audit.get('job_id'),
            'poisson_correction_rungs': p_audit.get('job_id'),
            'burgers_pod_ladder': (b_pod or {}).get('job_id'),
            'poisson_pod_ladder': (p_pod or {}).get('job_id'),
            'solver_side_tuning': (tune or {}).get('provenance', {}).get('job_id'),
        },
        'gpus': {
            'burgers_correction_rungs': b_audit.get('gpu'),
            'poisson_correction_rungs': p_audit.get('gpu'),
            'burgers_pod_ladder': (b_pod or {}).get('gpu'),
            'poisson_pod_ladder': (p_pod or {}).get('gpu'),
            'solver_side_tuning': (tune or {}).get('provenance', {}).get('gpu'),
        },
        'checkpoint_sha256': {
            'burgers': (b_result or {}).get('checkpoint_sha256'),
        },
        'sources': prov,
        'panels': {
            'A_burgers': {
                'intervals': 256,
                'error_metric': 'worst same-grid error vs converged same-mesh FOM (%)',
                'cost_metric': 'median complete-query GPU ms',
                'correction_rungs': b_rungs,
                'non_dominated_converged': b_front,
                'pod_ladder': b_podpts,
                'full_order_controls': b_foms,
            },
            'B_poisson': {
                'intervals': 1024,
                'error_metric': 'worst physical error (%)',
                'cost_metric': 'median device ms',
                'correction_rungs': p_rungs,
                'pod_ladder': p_podpts,
                'full_order_controls': p_foms,
            },
            'C_solver_side_tuning': {
                'cohort': '32 held-out validation cases',
                'error_metric': 'worst fixed-initial error (%)',
                'cost_metric': 'median GPU ms',
                'tuned_arms': t_rom,
                'full_order_controls': t_fom,
            },
        },
    }
    if b_front:
        payload['panels']['A_burgers']['cost_span'] = (
            max(p['cost_ms'] for p in b_front) / min(p['cost_ms'] for p in b_front))
        payload['panels']['A_burgers']['error_span'] = (
            max(p['error_percent'] for p in b_front)
            / min(p['error_percent'] for p in b_front))
    out_stem.with_suffix('.json').write_text(json.dumps(payload, indent=1) + '\n')
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--worktrees', default=str(DEFAULT_WORKTREES),
                    help='directory holding the sibling experiment worktrees')
    ap.add_argument('--out', default=str(Path(__file__).with_name(
        'fig_tunability_family')), help='output stem (no extension)')
    a = ap.parse_args()
    payload = build(Path(a.worktrees), Path(a.out))
    A = payload['panels']['A_burgers']
    print(f"Burgers: {len(A['correction_rungs'])} correction arms, "
          f"{len(A['non_dominated_converged'])} non-dominated converged, "
          f"cost span {A.get('cost_span', float('nan')):.2f}x, "
          f"error span {A.get('error_span', float('nan')):.2f}x")
    print(f"Poisson: {len(payload['panels']['B_poisson']['correction_rungs'])} arms")
    print(f"Tuning : {len(payload['panels']['C_solver_side_tuning']['tuned_arms'])} "
          f"tuned arms, "
          f"{len(payload['panels']['C_solver_side_tuning']['full_order_controls'])} "
          f"full-order controls")
    print('wrote', a.out + '.png/.pdf/.json')


if __name__ == '__main__':
    main()
