"""R4 animation (DESIGN-G5 5.4): reference, ROM and pointwise vector error, development Re 95.

Reads the frames saved by the ROM lane (one quadrature point per element, velocity vectors,
every 4th save over the last two shedding periods) and writes a GIF.
Usage: /home/tahmid/Dev/.venv/bin/python anim_rom_dev.py <out.gif>
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

HERE = Path(__file__).resolve().parent
FR = HERE.parents[3] / '2026-11-03-cylinder-unsteady-rom/experiments/cylinder-unsteady-rom/artifacts/r4-frames02/results/frames_re95.npz'
SURF, INK2 = '#fcfcfb', '#52514e'


def main(out):
    z = np.load(FR)
    p, t, ref = z['points'], z['t'], z['reference']
    roms = {'POD r=16': z['rom_pod_r16'], 'retrained bank r=16': z['rom_bank_r16']}
    tri = mtri.Triangulation(p[:, 0], p[:, 1])
    cx = p[tri.triangles].mean(axis=1)
    edge = np.max(np.linalg.norm(p[tri.triangles] - p[tri.triangles][:, [1, 2, 0]], axis=2), axis=1)
    tri.set_mask((np.hypot(cx[:, 0] - 0.2, cx[:, 1] - 0.2) < 0.05) | (edge > 0.03))
    speed = np.linalg.norm(ref, axis=2)
    vmax = float(np.percentile(speed, 99.5))
    errs = {k: np.linalg.norm(v - ref, axis=2) for k, v in roms.items()}
    emax = float(max(np.percentile(e, 99.5) for e in errs.values()))
    fig, axs = plt.subplots(3, 2, figsize=(12, 5.4), facecolor=SURF)
    arts = []
    def draw(i):
        for ax in axs.ravel():
            ax.clear(); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlim(0, 2.2); ax.set_ylim(0, 0.41)
        for j, (name, U) in enumerate(roms.items()):
            axs[0, j].tripcolor(tri, speed[i], shading='gouraud', cmap='viridis', vmin=0, vmax=vmax)
            axs[1, j].tripcolor(tri, np.linalg.norm(U[i], axis=1), shading='gouraud', cmap='viridis', vmin=0, vmax=vmax)
            axs[2, j].tripcolor(tri, errs[name][i], shading='gouraud', cmap='magma', vmin=0, vmax=emax)
            axs[0, j].set_title(f'reference |u|   (t = {t[i]:.3f} s)', fontsize=9, loc='left', color=INK2)
            axs[1, j].set_title(f'{name}: |u|', fontsize=9, loc='left', color=INK2)
            axs[2, j].set_title(f'{name}: pointwise error |u_ROM - u_ref|  (colour max {emax:.3f})', fontsize=9, loc='left', color=INK2)
        return []
    fig.suptitle('G5 development Re 95: last two shedding periods (speed colour max %.2f)' % vmax, x=0.01, ha='left', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    anim = FuncAnimation(fig, draw, frames=len(t), blit=False)
    anim.save(out, writer=PillowWriter(fps=8), dpi=90)
    draw(len(t) // 2); fig.savefig(str(out).replace('.gif', '_frame.png'), dpi=150, facecolor=SURF)


if __name__ == '__main__':
    main(sys.argv[1])
