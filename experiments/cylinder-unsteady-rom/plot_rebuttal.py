"""Rebuttal figures: reference / ROM / error fields at matched colour scales and lift/drag traces.

Usage: plot_rebuttal.py --frames frames.npz --series-dir DIR --case re105 --bank bank_r24 --pod pod_r24 --out DIR
Inputs are the small arrays written by rom_frames.py (one point per element) and rom_r4.py (series).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

INK, MUTED = '#1a1a1a', '#6b6b6b'
C_REF, C_BANK, C_POD = '#1a1a1a', '#1f6fb5', '#d1602a'


def main():
    ap = argparse.ArgumentParser()
    for k in ('--frames', '--series-dir', '--case', '--bank', '--pod', '--out'):
        ap.add_argument(k, required=True)
    ap.add_argument('--label', default='')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    z = np.load(a.frames)
    pts, ts, ref = z['points'], z['t'], z['reference']
    tri = mtri.Triangulation(pts[:, 0], pts[:, 1])
    T = tri.triangles
    cx, cy = pts[T, 0].mean(1), pts[T, 1].mean(1)
    edge = np.max([np.hypot(pts[T[:, i], 0] - pts[T[:, j], 0], pts[T[:, i], 1] - pts[T[:, j], 1]) for i, j in ((0, 1), (1, 2), (2, 0))], axis=0)
    tri.set_mask((np.hypot(cx - 0.2, cy - 0.2) < 0.05) | (edge > 0.06))
    names = {'bank': a.bank, 'pod': a.pod}
    rom = {k: z[f'rom_{v}'] for k, v in names.items()}
    fi = len(ts) - 1
    speed = lambda U: np.hypot(U[..., 0], U[..., 1])
    for comp, lab, f in (('speed', '|u|', speed), ('v', 'v (cross-stream)', lambda U: U[..., 1])):
        fields = [f(ref[fi]), f(rom['bank'][fi]), f(rom['pod'][fi])]
        errs = [speed(rom['bank'][fi] - ref[fi]), speed(rom['pod'][fi] - ref[fi])]
        vmin = min(x.min() for x in fields); vmax = max(x.max() for x in fields)
        emax = max(x.max() for x in errs)
        fig, axs = plt.subplots(5, 1, figsize=(9, 9.2), constrained_layout=True)
        titles = [f'Reference (FE P4/P3), {lab}', f'Neural-bank ROM ({a.bank}), {lab}', f'POD-Galerkin ROM ({a.pod}), {lab}',
                  f'|u_ROM - u_ref|, neural bank (max {errs[0].max():.2e})', f'|u_ROM - u_ref|, POD (max {errs[1].max():.2e})']
        cmap_f = 'cividis' if comp == 'speed' else 'RdBu_r'
        if comp == 'v':
            vmax = max(abs(vmin), abs(vmax)); vmin = -vmax
        for ax, val, ti in zip(axs[:3], fields, titles[:3]):
            m1 = ax.tripcolor(tri, val, shading='gouraud', cmap=cmap_f, vmin=vmin, vmax=vmax, rasterized=True)
            ax.set_title(ti, fontsize=9, color=INK, loc='left')
        for ax, val, ti in zip(axs[3:], errs, titles[3:]):
            m2 = ax.tripcolor(tri, val, shading='gouraud', cmap='Greys', vmin=0, vmax=emax, rasterized=True)
            ax.set_title(ti, fontsize=9, color=INK, loc='left')
        for ax in axs:
            ax.set_aspect('equal'); ax.set_xlim(0, 2.2); ax.set_ylim(0, 0.41)
            ax.add_patch(plt.Circle((0.2, 0.2), 0.05, color='white', ec=MUTED, lw=0.6))
            ax.tick_params(labelsize=7, colors=MUTED)
            for s in ax.spines.values():
                s.set_color(MUTED); s.set_linewidth(0.5)
        fig.colorbar(m1, ax=axs[:3], shrink=0.8, pad=0.01, label=f'{lab}, one scale for the three panels')
        fig.colorbar(m2, ax=axs[3:], shrink=0.8, pad=0.01, label='error magnitude, one scale for both panels')
        fig.suptitle(f'{a.label} t = {ts[fi]:.4f} s (end of the rollout window)', fontsize=10, color=INK)
        for ext in ('png', 'pdf'):
            fig.savefig(out / f'fields_{a.case}_{comp}.{ext}', dpi=200)
        plt.close(fig)
    # ---- lift and drag traces
    sb = np.load(Path(a.series_dir) / f'series_{a.case}_{a.bank}.npz'); sp = np.load(Path(a.series_dir) / f'series_{a.case}_{a.pod}.npz')
    fig, axs = plt.subplots(2, 2, figsize=(10, 5.2), constrained_layout=True, sharex=True)
    for col, (key, lab) in enumerate((('CL', 'lift coefficient $C_L$'), ('CD', 'drag coefficient $C_D$'))):
        t = sb['t']; ok = np.isfinite(sb[key]) & np.isfinite(sp[key])
        ax = axs[0, col]
        ax.plot(t[ok], sb[f'{key}_ref'][ok], color=C_REF, lw=2.4, alpha=0.35, label='reference FE')
        ax.plot(t[ok], sb[key][ok], color=C_BANK, lw=1.2, label=f'neural-bank ROM ({a.bank})')
        ax.plot(t[ok], sp[key][ok], color=C_POD, lw=1.2, ls=(0, (4, 2)), label=f'POD-Galerkin ROM ({a.pod})')
        ax.set_ylabel(lab, fontsize=9)
        ax = axs[1, col]
        ax.plot(t[ok], (sb[key] - sb[f'{key}_ref'])[ok], color=C_BANK, lw=1.2, label='neural bank - reference')
        ax.plot(t[ok], (sp[key] - sp[f'{key}_ref'])[ok], color=C_POD, lw=1.2, ls=(0, (4, 2)), label='POD - reference')
        ax.axhline(0, color=MUTED, lw=0.5)
        # the first rollout steps carry a start-up spike in the reaction force; keep the data, set the axis by the rest
        skip = 40
        for row, series in ((0, [sb[f'{key}_ref'], sb[key], sp[key]]), (1, [sb[key] - sb[f'{key}_ref'], sp[key] - sp[f'{key}_ref']])):
            body = np.concatenate([x[ok][skip:] for x in series]); lo, hi = body.min(), body.max(); pad = 0.08 * (hi - lo)
            axs[row, col].set_ylim(lo - pad, hi + pad)
        spike = max(np.abs((x[key] - x[f'{key}_ref'])[ok][:skip]).max() for x in (sb, sp))
        ax.text(0.99, 0.03, f'first {skip} steps (start-up) reach |error| {spike:.2g}, off scale', transform=ax.transAxes,
                ha='right', va='bottom', fontsize=7, color=MUTED)
        ax.set_ylabel(f'error in {key[0]}$_{key[1]}$', fontsize=9); ax.set_xlabel('time (s)', fontsize=9)
    for ax in axs.ravel():
        ax.grid(True, color='#e3e3e3', lw=0.5); ax.tick_params(labelsize=8, colors=MUTED)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axs[0, 0].legend(fontsize=8, frameon=False, loc='upper left', ncol=1)
    axs[1, 0].legend(fontsize=8, frameon=False, loc='upper left')
    fig.suptitle(f'{a.label} lift and drag over the whole rollout window', fontsize=10, color=INK)
    for ext in ('png', 'pdf'):
        fig.savefig(out / f'forces_{a.case}.{ext}', dpi=200)
    plt.close(fig)
    print('figures in', out)


if __name__ == '__main__':
    main()
