"""Independent NumPy audit of persisted worst-time nested-grid field pairs.

No wave implementation is imported and no trajectory is simulated. Reconstruct
the declared edge energy directly from saved fields and compare raw JSON values.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def independent_energy(u, v, c, n, bc):
    if bc == 'dirichlet':
        potential = np.sum(np.diff(np.pad(u, 1), axis=0)**2)
        potential += np.sum(np.diff(np.pad(u, 1), axis=1)**2)
        kinetic = np.sum(v*v)/n**2
    elif bc == 'absorbing':
        weights = np.ones(n+1)
        weights[[0, -1]] = .5
        potential = np.sum(np.diff(u, axis=0)**2*weights[None, :])
        potential += np.sum(np.diff(u, axis=1)**2*weights[:, None])
        kinetic = np.sum(v*v*weights[:, None]*weights[None, :])/n**2
    else:
        raise ValueError(bc)
    return .5*(kinetic+c*c*potential)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('result', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    records = result['smooth_family']
    audits = []
    for row in records:
        if 'coarse_n' not in row:
            continue
        bc, pi, name = row['bc'], row['parameter_index'], row['family']
        coarse, fine = row['coarse_n'], row['fine_n']
        init = next(r for r in records if r.get('n') == coarse and r['bc'] == bc and r['parameter_index'] == pi)
        c = init['parameters'][5]
        with np.load(args.result.parent/f'{name}_{bc}_{pi}_arrays.npz') as saved:
            key = f'difference_{coarse}_{fine}'
            worst = int(saved[key+'_worst_index'])
            ratio = fine//coarse
            sl = slice(ratio-1, -1, ratio) if bc == 'dirichlet' else slice(None, None, ratio)
            du = saved[key+'_coarse_u']-saved[key+'_fine_u'][sl, sl]
            dv = saved[key+'_coarse_v']-saved[key+'_fine_v'][sl, sl]
            prefix = f'{name}_{bc}_{pi}_{coarse}'
            e0 = independent_energy(saved[prefix+'_u'][0], saved[prefix+'_v'][0], c, coarse, bc)
            recomputed = float(np.sqrt(independent_energy(du, dv, c, coarse, bc)/e0))
        original = row['max_energy_state_difference']
        assert worst == int(np.argmax(row['state_difference_array']))
        assert abs(e0-init['initial_energy']) <= 1e-12*max(1., e0)
        assert abs(recomputed-original) <= 1e-12*max(1., original)
        audits.append(dict(bc=bc, parameter_index=pi, coarse_n=coarse, fine_n=fine,
                           worst_observation=worst, reported=original,
                           independent_recomputed=recomputed,
                           absolute_discrepancy=abs(recomputed-original)))
    report = dict(source_result=str(args.result.resolve()), all_checked=True,
                  checked_field_pairs=len(audits), rows=audits)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'rows'}, indent=2))


if __name__ == '__main__':
    main()
