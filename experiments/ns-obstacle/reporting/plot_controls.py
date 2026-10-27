"""Plot saved 2D FE controls without interpolation or admission claims.

Use the prescribed local Python. No solver, training or synthetic field is run.
Inputs remain untouched. Copy this generator into reports/ when publishing.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def plot_case(case: Path, destination: Path) -> dict:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    result_path = case / 'result.json'
    rule_path = case / 'full_rule.npz'
    result = json.loads(result_path.read_text())
    cfg = result['configuration']
    with np.load(rule_path, allow_pickle=False) as archive:
        points = archive['points'].copy()
        velocity = archive['velocity'].copy()
        gradient = archive['velocity_gradient'].copy()
        pressure = archive['pressure'].copy()
    count = len(points)
    expected = {'points': (count, 2), 'velocity': (count, 2),
                'gradient': (count, 2, 2), 'pressure': (count,)}
    arrays = dict(points=points, velocity=velocity, gradient=gradient, pressure=pressure)
    for name, array in arrays.items():
        if array.shape != expected[name] or not np.isfinite(array).all():
            raise ValueError(f'{name}: cannot render malformed/nonfinite saved data')
    diameter, speed = cfg['D'], cfg['U']
    if diameter <= 0 or speed <= 0:
        raise ValueError('positive recorded physical scales required')
    xy = points / diameter
    fields = [np.linalg.norm(velocity, axis=1) / speed,
              pressure / speed**2,
              diameter * np.trace(gradient, axis1=1, axis2=2) / speed]
    labels = [r'Speed $|\mathbf{u}|/U$', r'Pressure $p/U^2$',
              r'Divergence $D\nabla\!\cdot\!\mathbf{u}/U$']
    cmaps = ['viridis', 'coolwarm', 'RdBu_r']
    destination.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 8.4), layout='constrained')
    ranges = []
    for axis, values, label, cmap in zip(axes, fields, labels, cmaps):
        options = {}
        if label == labels[-1]:
            bound = max(float(np.max(np.abs(values))), np.finfo(float).eps)
            options.update(vmin=-bound, vmax=bound)
        artist = axis.scatter(xy[:, 0], xy[:, 1], c=values, s=2,
                              marker='s', linewidths=0, cmap=cmap,
                              rasterized=True, **options)
        if cfg['kind'] in ('dfg', 'manufactured'):
            center = np.asarray(cfg['center']) / diameter
            axis.add_patch(Circle(center, cfg['radius'] / diameter,
                                  facecolor='#f4f4f4', edgecolor='#303030', linewidth=1))
        axis.set(xlim=(cfg['xmin'] / diameter, cfg['xmax'] / diameter),
                 ylim=(0, cfg['H'] / diameter), aspect='equal',
                 xlabel=r'$x/D$', ylabel=r'$y/D$', title=label)
        fig.colorbar(artist, ax=axis, fraction=.026, pad=.015)
        ranges.append({'quantity': label, 'minimum': float(values.min()),
                       'maximum': float(values.max())})
    kind = {'dfg': 'Cylinder benchmark control', 'channel': 'Empty-channel analytic control',
            'manufactured': 'Manufactured curved-domain control'}.get(cfg['kind'], cfg['kind'])
    fig.suptitle(f'{kind} — provisional, no reference admission\n'
                 f"P{cfg['velocity_order']}/P{cfg['pressure_order']}, "
                 f"geometry order {cfg['geometry_order']}; {count:,} saved quadrature points",
                 fontsize=13, fontweight='bold')
    outputs = []
    for extension in ('png', 'pdf'):
        target = destination / ('fields.' + extension)
        fig.savefig(target, dpi=180)
        outputs.append({'path': str(target), 'sha256': digest(target)})
    plt.close(fig)
    metadata = {'status': 'provisional_control_visualization_not_reference_admission',
                'source_case': str(case.resolve()),
                'inputs': [{'path': str(path.resolve()), 'sha256': digest(path)}
                           for path in (result_path, rule_path)],
                'generator_sha256': digest(Path(__file__)),
                'rendering': 'all saved fluid quadrature points; no interpolation or outlier removal',
                'point_count': count, 'ranges': ranges, 'outputs': outputs,
                'source_provenance': result.get('provenance')}
    (destination / 'PROVENANCE.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    print(json.dumps(plot_case(args.case, args.destination), indent=2))
