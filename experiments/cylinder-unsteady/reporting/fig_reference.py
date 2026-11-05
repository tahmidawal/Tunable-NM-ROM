"""Reference-stage figure (G5 R1) generated only from saved arrays.

Panels: (a) Re 100 spin-up lift history; (b,c) Re 100 recorded-window lift and drag vs the
Featflow 2014 finest level (Featflow time-shifted so the first lift minima coincide: a
phase-ALIGNED visual comparison); (d) R1b time-refinement gaps normalized by their tolerance.
Usage: /home/tahmid/Dev/.venv/bin/python fig_reference.py <out.png>
"""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
LANE = HERE.parent
REF = LANE.parents[2] / '2026-11-03-cylinder-unsteady-ref/experiments/cylinder-unsteady-ref'
sys.path.insert(0, str(LANE / 'sources'))
import cycles as cyc  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
TOL = {'e_u_max': 2e-3, 'e_p_max': 4e-3, 'e_D': 3e-3, 'e_L': 2e-4}
LABEL = {'e_u_max': 'velocity $e_u$', 'e_p_max': 'pressure $e_p$', 'e_D': 'drag $e_D$', 'e_L': 'lift $e_L$'}


def style(ax):
    ax.set_facecolor(SURF)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def main(out):
    plt.rcParams.update({'font.size': 9, 'text.color': INK, 'axes.labelcolor': INK2})
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), facecolor=SURF)
    # (a) spin-up
    h = np.load(REF / 'artifacts/r1-spinup01/results/spinup_re100/history.npz')['rows']
    ok = np.isfinite(h[:, 2])
    ax = axs[0, 0]; style(ax)
    ax.plot(h[ok, 0], h[ok, 2], color=BLUE, lw=1.0)
    ax.axvspan(0, 1, color=GRID, alpha=0.6, lw=0); ax.axvspan(1, 1.5, color=YELLOW, alpha=0.18, lw=0)
    ax.text(0.5, 1.15, 'ramp', ha='center', color=INK2, fontsize=8); ax.text(1.25, 1.15, 'kick', ha='center', color=INK2, fontsize=8)
    ax.set_ylim(-1.3, 1.3); ax.set_xlabel('t [s]'); ax.set_ylabel('$C_L$')
    ax.set_title('(a) Re 100 spin-up from the Re 20 state (P3/P2, dt = 1/800)', loc='left', fontsize=9.5)
    # (b, c) recorded window vs Featflow finest, phase-aligned
    r = np.load(REF / 'artifacts/r1-record-re100/results/record_re100_p4/history.npz')['rows']
    r = r[(r[:, 8] == 1) & np.isfinite(r[:, 1])]
    ff = LANE / 'sources/featflow_2014_official/raw/logs/results_cc2d_bench2_referencenew/bdforces_q2_lv6_dt4'
    f = np.loadtxt(ff, comments='#')
    m_ours = cyc.lift_minima(r[:, 0], r[:, 2])[0][1]; m_ff = cyc.lift_minima(f[:, 1], f[:, 4])[0][1]
    tf = f[:, 1] - m_ff + m_ours
    for ax, col, fcol, name, lab in ((axs[0, 1], 2, 4, '$C_L$', '(b) Re 100 lift, recorded window'),
                                     (axs[1, 0], 1, 3, '$C_D$', '(c) Re 100 drag, recorded window')):
        style(ax)
        ax.plot(r[:, 0], r[:, col], color=BLUE, lw=2.0, label='this reference (P4/P3, dt = 1/3200)')
        ax.plot(tf, f[:, fcol], color=ORANGE, lw=1.0, ls=(0, (4, 2)), label='Featflow 2014 finest (Q2/P1disc, level 6), phase-aligned')
        ax.set_xlim(r[0, 0], r[0, 0] + 1.0); ax.set_xlabel('t [s]'); ax.set_ylabel(name)
        ax.set_title(lab, loc='left', fontsize=9.5)
    h1, l1 = axs[0, 1].get_legend_handles_labels()
    fig.legend(h1, l1, frameon=False, fontsize=8, loc='upper right', ncol=2, bbox_to_anchor=(0.995, 0.965))
    # (d) R1b convergence
    ax = axs[1, 1]; style(ax)
    cmp = REF / 'artifacts/r1b-compare01/results/compare'
    pairs = [(800, 1600), (1600, 3200), (3200, 6400)]
    dts = np.array([1 / c for c, _ in pairs])
    for k, colr in zip(TOL, (BLUE, ORANGE, AQUA, YELLOW)):
        v = np.array([json.loads((cmp / f'compare_{c}_{f2}.json').read_text())[k] for c, f2 in pairs]) / TOL[k]
        ax.loglog(dts, v, color=colr, lw=2.0, marker='o', ms=6, mec=SURF, mew=1.5, label=LABEL[k])
    ax.axhline(1.0, color=INK2, lw=1.0, ls='--'); ax.text(dts[0] * 0.97, 1.15, 'tolerance', ha='right', color=INK2, fontsize=8)
    guide = 2e-3 * (dts / dts[-1]) ** 2
    ax.loglog(dts, guide, color=INK2, lw=0.8, ls=':')
    ax.text(dts[1] * 1.05, guide[1] * 0.55, 'slope 2', color=INK2, fontsize=8)
    ax.set_xlabel('coarser dt of the pair  [s]'); ax.set_ylabel('gap / tolerance (max over W = 0.35 s)')
    ax.minorticks_off(); ax.set_xticks(dts); ax.set_xticklabels(['1/800', '1/1600', '1/3200'])
    ax.set_xlim(dts[-1] * 0.8, dts[0] * 1.25)
    ax.set_title('(d) R1b time refinement: dt$_0$ = 1/3200 is the first passing step', loc='left', fontsize=9.5)
    ax.legend(frameon=False, fontsize=7.5, loc='lower right')
    fig.suptitle('G5 reference (DFG 2D-2, Re 100): spin-up, benchmark agreement and time refinement', x=0.01, ha='left', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out, dpi=200, facecolor=SURF)


if __name__ == '__main__':
    main(sys.argv[1])
