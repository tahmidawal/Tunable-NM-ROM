"""Plot checked saved wave fields; no PDE solve, retraining or latent fitting.

Run with the absolute project Python, --runs <fresh_wave_campaign>, --out <dir>.
The first validation case and first optimizer repeat are chosen by index before
examining errors, identically in both boundary conditions. Full-grid truth is
available at four saved times; no interpolated time frames are manufactured.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_inputs(cluster, paths):
    manifest = {}
    for line in (cluster/'PULL.sha256').read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        manifest[name.removeprefix('*').removeprefix('./')] = digest
    hashes = {}
    for path in paths:
        name = path.relative_to(cluster).as_posix()
        digest = sha(path)
        assert digest == manifest[name], name
        hashes[name] = digest
    return hashes


def state_metrics(du, dv, mass, n, reflective, speed, scales, initial_energy):
    width = n-1 if reflective else n+1
    field = du.reshape(-1, width, width)
    if reflective:
        field = np.pad(field, ((0, 0), (1, 1), (1, 1)))
    weights = np.ones(n+1)
    weights[[0, -1]] = .5
    dx, dy = np.diff(field, axis=1), np.diff(field, axis=2)
    potential = speed**2*(np.sum(dx*dx*weights[None, None, :], axis=(1, 2))+np.sum(dy*dy*weights[None, :, None], axis=(1, 2)))
    kinetic = np.sum(dv*dv*mass, axis=-1)
    return dict(displacement=np.sqrt(np.sum(du*du*mass, axis=-1))/scales[0],
                velocity=np.sqrt(kinetic)/scales[1],
                energy_state=np.sqrt((kinetic+potential)/(2*initial_energy)))


def plot_fields(out, boundary, quantity, times, truth, linear, decoder, metrics, meta):
    n = meta['intervals']
    reflective = boundary == 'reflective'
    width = n-1 if reflective else n+1
    def field(value):
        value = value.reshape(width, width)
        return np.pad(value, ((1, 1), (1, 1))) if reflective else value
    symbol = 'u' if quantity == 'displacement' else 'v'
    title = 'Displacement' if symbol == 'u' else 'Velocity'
    figure = plt.figure(figsize=(12.8, 12), layout='constrained')
    grid = figure.add_gridspec(6, len(times), height_ratios=[1, 1, 1, .055, 1, .055])
    labels = [f'Reference {symbol}', f'Linear bank\n{meta["rank"]} coefficients', f'MLP decoder\n{meta["latent"]} coordinates', f'Absolute MLP error\n|{symbol} prediction − {symbol} reference|']
    axes_values = np.linspace(0, 1, n+1)
    limits, error_limits = [], []
    for col, t in enumerate(times):
        fields = [field(x[col]) for x in (truth, linear, decoder)]
        limit = max(float(np.max(abs(x))) for x in fields)
        # Initial velocity can vanish exactly. Give a zero field a meaningful
        # physical range, rather than an arbitrary microscopic color scale.
        if limit == 0:
            limit = max(float(np.max(abs(x))) for x in (truth, linear, decoder))
        limit = max(limit, np.finfo(float).tiny)
        limits.append(limit)
        for row, value in enumerate(fields):
            ax = figure.add_subplot(grid[row, col])
            artist = ax.pcolormesh(axes_values, axes_values, value.T, cmap='RdBu_r', norm=Normalize(-limit, limit), shading='nearest', rasterized=True)
            ax.set(xlim=(0, 1), ylim=(0, 1), aspect='equal', xticks=[0, .5, 1], yticks=[0, .5, 1])
            ax.tick_params(labelsize=8)
            if col == 0:
                ax.set_ylabel(labels[row]+'\ny', fontsize=10)
            if row == 0:
                ax.set_title(f't = {t:g}', fontsize=12)
            else:
                which = 'linear' if row==1 else 'mlp'
                current = metrics[which]['current_field_'+quantity][col]
                current_label = 'undefined' if current is None else f'{100*current:.2f}%'
                ax.set_title(f'Fixed-scale L2: {100*metrics[which][quantity][col]:.3f}%\nRelative to current field: {current_label}', fontsize=8.5)
            if row == 2:
                ax.set_xlabel('x')
        cb = figure.colorbar(artist, cax=figure.add_subplot(grid[3, col]), orientation='horizontal')
        cb.set_ticks(np.linspace(-limit, limit, 5), labels=[f'{x:.2g}' for x in np.linspace(-limit, limit, 5)])
        cb.set_label(f'{symbol}: shared field scale in this column', fontsize=8)
        cb.ax.tick_params(labelsize=7)
        ax = figure.add_subplot(grid[4, col])
        error = abs(fields[2]-fields[0])
        # Error range is at least the physical-field amplitude; increase it if
        # necessary to avoid clipping a large difference. Never magnify a tiny
        # error by stretching its own small maximum across the color scale.
        error_limit = max(limit, float(error.max()))
        error_limits.append(error_limit)
        error_artist = ax.pcolormesh(axes_values, axes_values, error.T, cmap='magma', norm=Normalize(0, error_limit), shading='nearest', rasterized=True)
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect='equal', xticks=[0, .5, 1], yticks=[0, .5, 1], xlabel='x')
        ax.tick_params(labelsize=8)
        if col==0:
            ax.set_ylabel(labels[3]+'\ny', fontsize=10)
        ax.set_title(f'Max absolute difference: {float(error.max()):.3g}', fontsize=9)
        cb = figure.colorbar(error_artist, cax=figure.add_subplot(grid[5, col]), orientation='horizontal')
        cb.set_ticks(np.linspace(0, error_limit, 3), labels=[f'{x:.2g}' for x in np.linspace(0, error_limit, 3)])
        cb.set_label('Absolute difference; no pointwise division', fontsize=8)
        cb.ax.tick_params(labelsize=7)
    description = 'Reflective fixed walls' if reflective else 'Absorbing boundary'
    normalization = 'Fixed scale: initial displacement L2 norm.' if symbol=='u' else 'Fixed scale: square root of twice the initial total state energy; current relative error is undefined at zero velocity.'
    figure.suptitle(f'{description} — {title} on the same unseen initial condition\nValidation case {meta["case"]}; optimizer seed {meta["optimizer_seed"]}; MLP time step {meta["primary_dt"]:g}\nField colors match within each time column; column scales may differ.\n{normalization}', fontsize=11)
    stem = f'2026-09-07-fresh-wave-{boundary}-{quantity}-fields'
    for suffix in ('png', 'pdf'):
        figure.savefig(out/(stem+'.'+suffix), dpi=170)
    plt.close(figure)
    return dict(png=stem+'.png', pdf=stem+'.pdf', symmetric_field_limits=limits, absolute_error_limits=error_limits)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--runs', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    evidence = dict(selection='First validation case and first optimizer repeat, fixed by index for both boundaries; illustrative case, not a cohort summary or a best/worst selection.', numerical_work='Saved-array matrix products, plotting and independent metric checks only. No PDE solve, training, fitting or new test data.', boundaries={})
    common_parameters = None
    for label, boundary, bc in [('reflective01', 'reflective', 'dirichlet'), ('absorbing02', 'absorbing', 'absorbing')]:
        cluster = args.runs/label/'cluster'
        campaign = cluster/'out/campaign'
        directory = campaign/bc
        run = json.loads((campaign/'result.json').read_text())
        config = run['config']
        assert run['completed'] and not run['final_test_opened']
        case, seed = 0, config['optimizer_seeds'][0]
        arm = next(a for a in run['boundary_results'][bc]['arms'] if a['name']=='mlp' and a['optimizer_seed']==seed)
        assert arm['rollout']['refinement_passed']
        dt = arm['rollout']['primary_dt']
        files = [campaign/'result.json', directory/'data_manifest.json', directory/'truth_spotchecks.npz', directory/'bank_tables.npz', directory/'learned_bank_linear_r.npz', directory/f'mlp_{seed}/rollouts.npz']
        hashes = checked_inputs(cluster, files)
        manifest = json.loads((directory/'data_manifest.json').read_text())
        validation = manifest['splits']['validation']
        parameters = np.asarray(validation['parameters'][case])
        if common_parameters is None:
            common_parameters = parameters
        else:
            np.testing.assert_array_equal(parameters, common_parameters)
        scales, e0 = np.asarray(validation['scales'][case]), validation['initial_energy'][case]
        with np.load(directory/'truth_spotchecks.npz') as saved:
            times, truth_u, truth_v = saved['times'], saved['u'][case], saved['v'][case]
            np.testing.assert_array_equal(parameters, saved['parameters'][case])
        indices = np.rint(times/config['observation_dt']).astype(int)
        np.testing.assert_allclose(indices*config['observation_dt'], times, rtol=0, atol=1e-14)
        with np.load(directory/'bank_tables.npz') as bank:
            g, mass = bank['g'], bank['mass']
        predictions, metric_audit = {}, {}
        for name, file in [('linear', directory/'learned_bank_linear_r.npz'), ('mlp', directory/f'mlp_{seed}/rollouts.npz')]:
            with np.load(file) as saved:
                if name=='linear':
                    prefix = f'case{case}_'
                    np.testing.assert_array_equal(saved['g'], g)
                    a, b = saved[prefix+'a'], saved[prefix+'b']
                else:
                    prefix = f'dt{dt}_case{case}_'
                    assert np.all(saved[prefix+'completed'])
                    a, b = saved[prefix+'coefficients'], saved[prefix+'physical_velocity_coefficients']
                u, v = a[indices]@g.T, b[indices]@g.T
                assert all(np.all(np.isfinite(x)) for x in (u, v, truth_u, truth_v))
                measured = state_metrics(u-truth_u, v-truth_v, mass, config['n'], bc=='dirichlet', parameters[5], scales, e0)
                discrepancy = {key:float(np.max(abs(value-saved[prefix+key+'_error'][indices]))) for key, value in measured.items()}
                assert max(discrepancy.values())<1e-10, discrepancy
                metric_audit[name] = dict(maximum_absolute_metric_discrepancy=discrepancy, **{key:value.tolist() for key,value in measured.items()})
                for quantity, difference, truth in [('displacement', u-truth_u, truth_u), ('velocity', v-truth_v, truth_v)]:
                    numerator = np.sqrt(np.sum(difference*difference*mass, axis=-1))
                    denominator = np.sqrt(np.sum(truth*truth*mass, axis=-1))
                    threshold = scales[0 if quantity=='displacement' else 1]*1e-12
                    metric_audit[name]['current_field_'+quantity] = [float(a/b) if b>threshold else None for a,b in zip(numerator, denominator)]
                predictions[name] = (u, v)
        meta = dict(case=case, optimizer_seed=seed, intervals=config['n'], rank=config['rank'], latent=config['latent'], primary_dt=dt, initial_displacement_norm=float(scales[0]), initial_state_norm=float(scales[1]), initial_energy=float(e0), parameters=parameters.tolist(), times=times.tolist(), source_job=run['provenance']['job_id'], source_commit=run['provenance']['source_commit'], input_sha256=hashes, metrics=metric_audit, mlp_primary_time_refinement_passed=True)
        meta['plots'] = {}
        for quantity, index, truth in [('displacement', 0, truth_u), ('velocity', 1, truth_v)]:
            meta['plots'][quantity] = plot_fields(args.out, boundary, quantity, times, truth, predictions['linear'][index], predictions['mlp'][index], metric_audit, meta)
        evidence['boundaries'][boundary] = meta
    (args.out/'2026-09-07-fresh-wave-fields.json').write_text(json.dumps(evidence, indent=2)+'\n')
    print(json.dumps({bc:{q:p['png'] for q,p in meta['plots'].items()} for bc,meta in evidence['boundaries'].items()}, indent=2))


if __name__=='__main__':
    main()
