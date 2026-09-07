"""Independent saved-array audit; no wave simulation or training imports."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from audit_model_arrays import head_value_jacobian, relative


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('job', type=Path)
    parser.add_argument('--parent', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    root = args.job/'cluster'
    result = json.loads((root/'out/refinement/result.json').read_text())
    assert result['old_fine_parity_passed']
    assert not any(result[k] for k in ('retraining_performed', 'initial_fitting_performed', 'final_test_opened'))
    origin = json.loads((root/'in/origin.json').read_text())
    for name, meta in origin['files'].items():
        assert sha(root/'in'/name) == meta['sha256']
        assert sha(args.parent/'cluster'/meta['remote_relative']) == meta['sha256']
        assert result['input_sha256'][name] == meta['sha256']
    directory = args.parent/'cluster/out/campaign/dirichlet'
    manifest = json.loads((directory/'data_manifest.json').read_text())
    validation = manifest['splits']['validation']
    scales, e0, parameters = map(np.asarray, (validation['scales'], validation['initial_energy'], validation['parameters']))
    with np.load(root/'in/bank_tables.npz') as bank:
        g, mass = bank['g'], bank['mass']
        width = manifest['n']-1
        fields = np.pad(g.reshape(width, width, -1), ((1, 1), (1, 1), (0, 0)))
        dx, dy = np.diff(fields, axis=0).reshape(-1, g.shape[1]), np.diff(fields, axis=1).reshape(-1, g.shape[1])
        stiffness = dx.T@dx+dy.T@dy
    with np.load(directory/'truth_spotchecks.npz') as truth:
        truth_u, truth_v, times = truth['u'], truth['v'], truth['times']
    checks = dict(parent_archive_inputs_reconciled=True, head=0., tangent=0., energy=0., displacement_metric=0., velocity_metric=0., energy_state_metric=0., refinement_summary=0.)
    fine_rows = []
    with np.load(root/'out/refinement/rollouts.npz') as rollouts, np.load(root/'in/rollouts.npz') as old, np.load(root/'in/head.npz') as head:
        def differences(first, second, case):
            da = first[0]-second[0]
            db = first[1]-second[1]
            return (np.linalg.norm(da, axis=-1)/scales[case, 0], np.linalg.norm(db, axis=-1)/scales[case, 1], np.sqrt(np.maximum(0., np.sum(db*db, axis=-1)+parameters[case, 5]**2*np.einsum('tr,rs,ts->t', da, stiffness, da))/(2*e0[case])))
        def state(store, dt, case):
            prefix = f'dt{dt}_case{case}'
            assert np.all(store[prefix+'_completed'])
            return store[prefix+'_coefficients'], store[prefix+'_physical_velocity_coefficients']
        for case in range(len(scales)):
            for dt in result['diagnostic_config']['rom_dts']:
                prefix = f'dt{dt}_case{case}'
                a, b = state(rollouts, dt, case)
                z, w = rollouts[prefix+'_z'], rollouts[prefix+'_w']
                np.testing.assert_array_equal(z[0], old[f'dt0.00125_case{case}_z'][0])
                np.testing.assert_array_equal(w[0], old[f'dt0.00125_case{case}_w'][0])
                aa, jac = head_value_jacobian(head, z, result['kind'])
                bb = np.einsum('trk,tk->tr', jac, w)
                checks['head'] = max(checks['head'], relative(aa, a))
                checks['tangent'] = max(checks['tangent'], relative(bb, b))
                energy = .5*(np.sum(b*b, axis=-1)+parameters[case, 5]**2*np.einsum('tr,rs,ts->t', a, stiffness, a))
                checks['energy'] = max(checks['energy'], float(np.max(abs(energy-rollouts[prefix+'_rom_energy']))/e0[case]))
                indices = np.rint(times/result['original_config']['observation_dt']).astype(int)
                du, dv = a[indices]@g.T-truth_u[case], b[indices]@g.T-truth_v[case]
                uerr = np.sqrt(np.sum(du*du*mass, axis=-1))/scales[case, 0]
                verr = np.sqrt(np.sum(dv*dv*mass, axis=-1))/scales[case, 1]
                field = np.pad(du.reshape(len(indices), width, width), ((0, 0), (1, 1), (1, 1)))
                potential = parameters[case, 5]**2*(np.sum(np.diff(field, axis=1)**2, axis=(1, 2))+np.sum(np.diff(field, axis=2)**2, axis=(1, 2)))
                eerr = np.sqrt((np.sum(dv*dv*mass, axis=-1)+potential)/(2*e0[case]))
                for metric, expected in [('displacement', uerr), ('velocity', verr), ('energy_state', eerr)]:
                    discrepancy = float(np.max(abs(expected-rollouts[prefix+'_'+metric+'_error'][indices])))
                    checks[metric+'_metric'] = max(checks[metric+'_metric'], discrepancy)
                for metric in ('displacement_error', 'velocity_error', 'energy_state_error', 'rom_energy'):
                    assert np.all(np.isfinite(rollouts[prefix+'_'+metric]))
                phase = rollouts[prefix+'_unwrapped_phase_error_valid_segments']
                mask = rollouts[prefix+'_modal_phase_defined']
                assert np.all(np.isfinite(phase[mask])) and np.all(np.isnan(phase[~mask]))
            for dt1, dt2 in ((.00125, .000625), (.000625, .0003125)):
                values = differences(state(rollouts, dt1, case), state(rollouts, dt2, case), case)
                saved = next(r for r in result['adjacent_refinement'] if r['case']==case and r['coarse_dt']==dt1)
                row = dict(case=case, coarse_dt=dt1, fine_dt=dt2)
                for metric, value in zip(('displacement', 'velocity', 'energy_state'), values):
                    key = 'max_'+metric+'_difference'
                    row[key] = float(value.max())
                    checks['refinement_summary'] = max(checks['refinement_summary'], abs(row[key]-saved[key]))
                row['passed'] = max(float(v.max()) for v in values)<=result['original_config']['rom_refinement_target']
                assert row['passed']==saved['passed']
                fine_rows.append(row)
    assert max(v for k, v in checks.items() if k!='parent_archive_inputs_reconciled') < 1e-8, checks
    record = dict(job=str(args.job.resolve()), result_sha256=sha(root/'out/refinement/result.json'), passed=True, checks=checks, adjacent_refinement=fine_rows, original_normalizations_used_in_independent_audit=True, regenerated_scale_relative_parity=result['truth_parity']['scales_relative_parity'])
    args.out.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(passed=True, checks=checks), indent=2))


if __name__=='__main__':
    main()
