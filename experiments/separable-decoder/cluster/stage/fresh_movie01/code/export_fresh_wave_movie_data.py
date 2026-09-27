"""Regenerate only the illustrated validation references at every saved time.

Frozen fresh-wave FOM and seed generator are supplied beside this committed
coordinator script. No training, ROM solve, new cohort or changed FOM step.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from fresh_fom import Grid, localized_initial, integrate_balance, energy, provenance
from fresh_learning import parameter_rows


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--config', required=True, type=Path)
    p.add_argument('--inputs', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    meta = provenance()
    assert meta['jax_backend']=='gpu' and meta['x64'] and meta['matmul_precision']=='highest'
    args.out.mkdir(parents=True, exist_ok=False)
    rows = parameter_rows(config['validation_seed'], config['validation_count'])
    case = 0
    result = dict(provenance=meta, coordinator_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), config=config, case=case, validation_seed=config['validation_seed'], final_test_opened=False, training_performed=False, rom_recomputed=False, boundaries={})
    for boundary, bc in [('reflective', 'dirichlet'), ('absorbing', 'absorbing')]:
        source = args.inputs/(boundary+'_manifest.json')
        manifest = json.loads(source.read_text())
        expected = manifest['splits']['validation']
        assert manifest['bc']==bc and manifest['n']==config['n'] and not manifest['final_test_opened']
        np.testing.assert_array_equal(rows, np.asarray(expected['parameters']))
        grid = Grid(config['n'], bc, bc)
        stride = int(np.ceil(config['observation_dt']/(config['fom_cfl']*grid.h/1.15)))
        dt = config['observation_dt']/stride
        assert dt == expected['dt']
        observations = int(round(config['end_time']/config['observation_dt']))
        initial_u, initial_v = localized_initial(grid, rows[case])
        u, v, flux = integrate_balance(initial_u, initial_v, rows[case][5], dt, grid=grid, steps=observations*stride, stride=stride)
        e = np.asarray(energy(u, v, grid, rows[case][5]))
        u, v, flux = np.asarray(u), np.asarray(v), np.asarray(flux)
        assert u.dtype==np.float64 and v.dtype==np.float64
        assert all(np.all(np.isfinite(x)) for x in (u, v, e, flux))
        old_energy = np.asarray(expected['truth_audits'][case]['energy_trace'])
        energy_parity = float(np.max(abs(e-old_energy))/max(np.max(abs(old_energy)), 1e-30))
        balance = float(np.max(abs(e+flux-e[0]))/e[0])
        assert energy_parity<1e-10 and balance<1e-5
        times = np.arange(observations+1)*config['observation_dt']
        np.savez_compressed(args.out/(boundary+'.npz'), u=u, v=v, times=times, energy=e, outflux=flux, parameters=rows[case], mass=grid.mass())
        result['boundaries'][boundary] = dict(bc=bc, n=grid.n, shape=list(grid.shape), dt=dt, observation_dt=config['observation_dt'], frames=len(times), energy_trace_relative_parity=energy_parity, energy_balance_relative=balance, parent_manifest_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        print(boundary, 'frames', len(times), 'energy_parity', energy_parity, flush=True)
    (args.out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print('fresh_wave_movie_data_complete', flush=True)


if __name__=='__main__':
    main()
