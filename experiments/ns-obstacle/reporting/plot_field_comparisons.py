"""Plot existing development states on a shared deterministic display subset."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from matplotlib.patches import Circle
from matplotlib.ticker import ScalarFormatter, MaxNLocator


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--data', type=Path, required=True)
    a.add_argument('--metadata', type=Path, required=True)
    a.add_argument('--sha256', required=True)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--models', nargs='+', help='Explicit saved model keys to display')
    args = a.parse_args()
    if sha(args.data) != args.sha256:
        raise ValueError('Plot dataset identity differs')
    data = np.load(args.data, allow_pickle=False)
    meta = json.loads(args.metadata.read_text())
    xy = data['points_physical']; g = meta['geometry']
    center = np.array(g['center']); radius = g['radius']
    tri = Triangulation(xy[:, 0], xy[:, 1])
    vertices = xy[tri.triangles]
    mask = np.linalg.norm(vertices.mean(axis=1)-center, axis=1) < radius
    # Remove edges crossing the solid so display interpolation cannot span it.
    for j in range(3):
        p = vertices[:, j]; v = vertices[:, (j+1)%3]-p
        t = np.clip(np.sum((center-p)*v, axis=1)/np.sum(v*v, axis=1), 0, 1)
        mask |= np.linalg.norm(p+t[:, None]*v-center, axis=1) < radius
    tri.set_mask(mask)
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'pdf.fonttype': 42})
    models = [('bank_r8_projection', 'Neural bank: best-fit reconstruction',
               'Eight neural velocity modes plus shared pressure/stability spaces. Not an online flow solve.'),
              ('POD_r2_mixed_Galerkin', 'Standard POD–Galerkin: solved flow',
               'Two POD velocity modes plus pressure-stability enrichment. Actual converged flow solve.')]
    if 'bank_r8_solve_velocity' in data.files:
        models.insert(1, ('bank_r8_solve', 'Original neural-bank ROM: flow-solve output',
                         'User-authorized diagnostic with original solver; representation gate remains failed. See recorded convergence status.'))
    defaults = [m[0] for m in models]
    if 'pod_guided_neural_r4_solve_velocity' in data.files:
        models.append(('pod_guided_neural_r4_solve', 'First improved neural-bank ROM: flow-solve output',
                       'Four POD-guided neural velocity modes plus FE pressure/stability spaces. Lift, divergence and mass targets still fail.'))
        defaults.append('pod_guided_neural_r4_solve')
    if 'refined_neural_r4_solve_velocity' in data.files:
        models.append(('refined_neural_r4_solve', 'Refined neural-bank ROM: flow-solve output',
                       'Longer-trained neural velocity bank plus FE pressure/stability spaces. Divergence remains above target.'))
        defaults.append('refined_neural_r4_solve')
    for key, label in [('trace_neural_r4_solve', 'Trace-trained neural-bank ROM'),
                       ('energy_neural_r4_solve', 'Energy-weighted neural-bank ROM'),
                       ('traction_neural_r4_solve', 'Surface-traction-trained neural-bank ROM'),
                       ('residual_r4_solve', 'Neural-bank ROM with correction heads'),
                       ('coordinate_r4_solve', 'Neural-bank ROM with scaled-coordinate training'),
                       ('galerkin_r4_solve', 'Neural-bank ROM with mixed Galerkin solve')]:
        if key + '_velocity' in data.files:
            models.append((key, label + ': flow-solve output',
                           'Four neural velocity modes plus FE pressure/stability spaces. '
                           'Actual reduced solve; consult full-rule metrics for accuracy and conservation.'))
            defaults.append(key)
    for rank in (4, 8):
        for method, label in [('mixed_Galerkin', 'Galerkin solved flow'),
                              ('exact_tensor_weak_LS', 'weak least-squares solved flow'),
                              ('coefficient_PCHIP', 'coefficient interpolation')]:
            key = f'POD_r{rank}_{method}'
            if key+'_velocity' not in data.files:
                continue
            note = (f'{rank} POD velocity modes plus shared pressure/stability spaces. '
                    + ('Interpolation control; no PDE solve.' if method == 'coefficient_PCHIP'
                       else 'Actual reduced flow-solve output.'))
            models.append((key, f'Standard POD r{rank}: {label}', note))
            if method == 'mixed_Galerkin':
                defaults.append(key)
    registry = {m[0]: m for m in models}
    requested = args.models or defaults
    if len(set(requested)) != len(requested) or any(k not in registry for k in requested):
        raise ValueError('Requested model is duplicated or unavailable in this saved dataset')
    models = [registry[k] for k in requested]
    speed_max = max(float(np.linalg.norm(data[key+'_velocity'], axis=-1).max())
                    for key in ['reference']+[m[0] for m in models])
    pmin = min(float(data[key+'_pressure'].min()) for key in ['reference']+[m[0] for m in models])
    pmax = max(float(data[key+'_pressure'].max()) for key in ['reference']+[m[0] for m in models])
    outputs = []
    def decorate(ax, zoom=True):
        ax.add_patch(Circle(center, radius, facecolor='white', edgecolor='#222', linewidth=.7, zorder=5))
        ax.set(xlim=(g['xmin'], .8 if zoom else g['xmax']), ylim=(0, g['H']),
               xlabel='x [m]', ylabel='y [m]', aspect='equal')
    for i, reynolds in enumerate(data['Re']):
        tag = str(float(reynolds)).replace('.', 'p')
        for key, title, note in models:
            uref = data['reference_velocity'][i]; u = data[key+'_velocity'][i]
            pref = data['reference_pressure'][i]; p = data[key+'_pressure'][i]
            values = [[np.linalg.norm(uref, axis=-1), np.linalg.norm(u, axis=-1), np.linalg.norm(u-uref, axis=-1)],
                      [pref, p, np.abs(p-pref)]]
            fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), layout='constrained')
            for row in range(2):
                for col in range(3):
                    ax = axes[row, col]; z = values[row][col]
                    low, high = ((0, speed_max) if row == 0 else (pmin, pmax)) if col < 2 else (0, max(float(z.max()), 1e-16))
                    artist = ax.tripcolor(tri, z, shading='gouraud', cmap='viridis' if col < 2 else 'magma', vmin=low, vmax=high)
                    artist.set_rasterized(True)
                    decorate(ax)
                    field_label = ('Best-fit field' if key.endswith('_projection') else
                                   'Interpolated field' if key.endswith('_PCHIP') else 'Predicted flow')
                    ax.set_title(['Qualified fine reference', field_label, 'Pointwise error'][col])
                    label = (r'$|u|/U$' if row == 0 else r'$p/U^2$') if col < 2 else (r'$|u_{ROM}-u_{ref}|/U$' if row == 0 else r'$|p_{ROM}-p_{ref}|/U^2$')
                    bar = fig.colorbar(artist, ax=ax, orientation='horizontal', fraction=.08, pad=.07, label=label)
                    formatter = ScalarFormatter(useMathText=True)
                    formatter.set_powerlimits((-2, 2))
                    bar.formatter = formatter
                    bar.locator = MaxNLocator(nbins=4)
                    bar.update_ticks()
            fig.suptitle(f'{title} — Re = {float(reynolds):g}\nVelocity (top) and pressure (bottom); near-cylinder view', fontsize=14)
            fig.supxlabel(note+'\nDisplay interpolation; error scales differ. Errors against stored reference do not certify equal continuum accuracy. Development cases.', fontsize=9)
            stem = f'{key}_Re{tag}'
            fig.savefig(args.output/(stem+'.png'), dpi=155)
            fig.savefig(args.output/(stem+'.pdf'), dpi=155)
            plt.close(fig); outputs.append(stem+'.png')
        fig, axes = plt.subplots(len(models)+1, 1, figsize=(13, 2*(len(models)+1)+2), layout='constrained')
        for ax, key, title in zip(axes, ['reference']+[m[0] for m in models],
                                  ['Qualified fine reference']+[m[1] for m in models]):
            z = np.linalg.norm(data[key+'_velocity'][i], axis=-1)
            artist = ax.tripcolor(tri, z, shading='gouraud', cmap='viridis', vmin=0, vmax=speed_max)
            artist.set_rasterized(True)
            decorate(ax, False); ax.set_title(title, loc='left')
        fig.colorbar(artist, ax=list(axes), shrink=.65, label=r'Speed $|u|/U$')
        fig.suptitle(f'Full-channel velocity comparison — Re = {float(reynolds):g}', fontsize=15)
        fig.supxlabel('Same speed scale throughout. Best-fit projection and actual flow-solve outputs are labeled separately.', fontsize=10)
        stem = f'full_channel_Re{tag}'
        fig.savefig(args.output/(stem+'.png'), dpi=155)
        fig.savefig(args.output/(stem+'.pdf'), dpi=155); plt.close(fig); outputs.append(stem+'.png')
    manifest = {'data_sha256':sha(args.data), 'metadata_sha256':sha(args.metadata),
                'generator_sha256':sha(Path(__file__)), 'outputs':outputs,
                'selected_models':requested,
                'global_metrics_source':'Original full-rule records in input metadata; no global error inferred from plotting subset',
                'display':'Linear triangulation of deterministic saved quadrature samples, masked at obstacle; no PDE solve or fit'}
    (args.output/'PLOT-MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
