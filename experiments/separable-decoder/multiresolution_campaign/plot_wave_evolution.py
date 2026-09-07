"""Export still-image sequences from the fresh multiresolution wave pilot."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_directory', type=Path)
    parser.add_argument('--out-directory', type=Path, required=True)
    args = parser.parse_args()
    result_path = args.input_directory/'result.json'
    result = json.loads(result_path.read_text())
    cfg = result['config']
    n = max(cfg['meshes'])
    case = cfg['validation_indices'][-1]
    dt = cfg['rom_dts'][0]
    frames = np.rint(np.linspace(0, round(cfg['end_time']/cfg['observation_dt']), 5)).astype(int)
    provenance = {'result_sha256': hashlib.sha256(result_path.read_bytes()).hexdigest(),
                  'solver_intervals': n, 'display_intervals': cfg['saved_intervals'],
                  'case': case, 'rom_dt': dt, 'frames': frames.tolist(), 'sources': {}}
    for boundary in cfg['boundaries']:
        reference_path = args.input_directory/f'reference_{boundary}_{case}.npz'
        predicted_path = args.input_directory/f'{boundary}_{n}_{case}_0_rom_{dt}.npz'
        truth = np.load(reference_path)['u']
        prediction = np.load(predicted_path)['u']
        if boundary == 'dirichlet':
            truth = np.pad(truth, ((0, 0), (1, 1), (1, 1)))
            prediction = np.pad(prediction, ((0, 0), (1, 1), (1, 1)))
        assert truth.shape == prediction.shape
        difference = np.abs(prediction-truth)
        amplitude = max(float(np.max(np.abs(truth))), float(np.max(np.abs(prediction))))
        error_max = float(np.max(difference))
        fig, axes = plt.subplots(3, len(frames), figsize=(13, 7.2), sharex=True, sharey=True,
                                 layout='constrained')
        for col, frame in enumerate(frames):
            for row, fields in enumerate((truth, prediction, difference)):
                options = dict(cmap='RdBu_r', vmin=-amplitude, vmax=amplitude) if row < 2 else dict(cmap='magma', vmin=0, vmax=error_max)
                artist = axes[row, col].imshow(fields[frame].T, extent=(0, 1, 0, 1), origin='lower', **options)
                axes[row, col].set_xticks([0, .5, 1])
                axes[row, col].set_yticks([0, .5, 1])
                if row == 0:
                    axes[row, col].set_title(f't = {frame*cfg["observation_dt"]:g}')
                    field_artist = artist
                if row == 2:
                    axes[row, col].set_xlabel('x')
                    error_artist = artist
        for row, label in enumerate(('Reference u', 'NM-ROM u', '|NM-ROM − reference|')):
            axes[row, 0].set_ylabel(label+'\ny')
        fig.colorbar(field_artist, ax=axes[:2, :], shrink=.8, label='Displacement; fixed scale')
        fig.colorbar(error_artist, ax=axes[2, :], shrink=.8, label='Absolute error; fixed scale')
        label = 'Reflective wave' if boundary == 'dirichlet' else 'Absorbing wave'
        fig.suptitle(f'{label} evolution — development case {case}\n'
                     f'ROM solves at {n} intervals; display at {cfg["saved_intervals"]}; frozen weights', fontsize=12)
        stem = args.out_directory/f'{boundary}-wave-evolution-case{case}'
        for suffix in ('.png', '.pdf'):
            fig.savefig(stem.with_suffix(suffix), dpi=180)
        plt.close(fig)
        for path in (reference_path, predicted_path):
            provenance['sources'][str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.out_directory/'wave-evolution-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')


if __name__ == '__main__':
    main()
