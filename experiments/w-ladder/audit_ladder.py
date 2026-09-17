"""Independent NumPy/SciPy audit of one collected w-ladder attempt. No JAX, no driver import.

    python experiments/w-ladder/audit_ladder.py <attempt>

Recomputes every reported error from the saved coefficient trajectories (ROM arms, full
grid, fields = coefficients @ bank.T from mesh_<n>.npz) and saved fields (FOM arms: full grid
on the designated case, common grid otherwise) against a reference regenerated here with
SciPy's DST from the saved initial fields; recomputes reduced-energy drifts and the retained
value gates; recomputes per-arm timing medians. Writes runs/<attempt>/audit.json.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.fft import dstn

ROOT = Path(__file__).resolve().parents[2]
OBS = 49


def restrict(field, n, target):
    ratio = n // target
    sl = slice(ratio - 1, None, ratio)
    return field[..., sl, sl]


def reference(u0, v0, c, n, times):
    """Exact semidiscrete Dirichlet solution: same eigenvalues as pilot.spectral_propagate, SciPy transforms."""
    modes = np.arange(1, n, dtype=np.float64)
    eigen = 4 * n * n * np.sin(np.pi * modes / (2 * n)) ** 2
    omega = c * np.sqrt(eigen[:, None] + eigen[None, :])
    a, b = dstn(u0, type=1, norm='ortho'), dstn(v0, type=1, norm='ortho')
    us, vs = np.empty((len(times), n - 1, n - 1)), np.empty((len(times), n - 1, n - 1))
    for i, t in enumerate(times):
        co, si = np.cos(omega * t), np.sin(omega * t)
        us[i] = dstn(co * a + si / omega * b, type=1, norm='ortho')
        vs[i] = dstn(-omega * si * a + co * b, type=1, norm='ortho')
    return us, vs


def energy(u, v, h, c):
    uu = np.pad(u, [(0, 0)] * (u.ndim - 2) + [(1, 1), (1, 1)])
    potential = np.sum(np.diff(uu, axis=-2) ** 2, axis=(-2, -1)) + np.sum(np.diff(uu, axis=-1) ** 2, axis=(-2, -1))
    return .5 * (h * h * np.sum(v * v, axis=(-2, -1)) + c * c * potential)


def worst_errors(u, v, ut, vt, h, c):
    l2 = lambda x: np.sqrt(h * h * np.sum(x * x, axis=(-2, -1)))
    e0 = float(energy(ut[0], vt[0], h, c)); u0, v0 = float(l2(ut[0])), np.sqrt(2 * e0)
    du, dv = l2(u - ut) / u0, l2(v - vt) / v0
    de = np.sqrt(np.maximum(0., 2 * energy(u - ut, v - vt, h, c))) / v0
    return dict(displacement=float(np.max(du)), velocity=float(np.max(dv)), energy_state=float(np.max(de)),
                energy_fraction=(energy(u, v, h, c) / e0).tolist())


def decode(coefficients, g, shape, chunk=7):
    out = np.empty((len(coefficients), *shape))
    for i in range(0, len(coefficients), chunk):
        out[i:i + chunk] = (coefficients[i:i + chunk] @ g.T).reshape(-1, *shape)
    return out


def main():
    attempt = sys.argv[1]
    run = ROOT / 'experiments/w-ladder/runs' / attempt
    out = run / 'archive/output'
    result = json.loads((out / 'result.json').read_text())
    n = result['mesh']; h = 1. / n; cfg = result['config']
    times = np.arange(OBS) * cfg['observation_dt']
    with np.load(out / f'mesh_{n}.npz') as f:
        g, phi, kk, kk_pod = f['g'], f['pod_phi'], f['stiffness'], f['pod_stiffness']
    bank_of = lambda name: (phi[:, :int(name.removeprefix('pod_k'))], kk_pod[:int(name.removeprefix('pod_k')), :int(name.removeprefix('pod_k'))]) if name.startswith('pod_k') else (g, kk)
    checks = dict(attempt=attempt, mesh=n, reference=[], rows=[], energy=[], gates=[], timing={})
    worst = 0.
    for ref in result['references']:
        case = ref['case']
        with np.load(out / f'reference_{n}_{case}.npz') as f:
            u0, v0, par, su, sv = f['u0'], f['v0'], f['parameters'], f['u'], f['v']
        c = float(par[5]); full = ref['fields_saved'] == 'full'
        ru, rv = reference(u0, v0, c, n, times)
        saved = (ru, rv) if full else (restrict(ru, n, cfg['saved_intervals']), restrict(rv, n, cfg['saved_intervals']))
        dref = float(max(np.max(abs(saved[0] - su)) / np.max(abs(su)), np.max(abs(saved[1] - sv)) / np.max(abs(sv))))
        checks['reference'].append(dict(case=case, fields=ref['fields_saved'], regenerated_vs_saved_relative=dref,
                                        u0_sha256=hashlib.sha256(np.ascontiguousarray(u0).tobytes()).hexdigest()))
        common_n = cfg['saved_intervals']; hc = 1. / common_n
        cru, crv = restrict(ru, n, common_n), restrict(rv, n, common_n)
        for row in result['invocations']:
            if row['case'] != case or row['repetition'] != 0:
                continue
            name = row['method']; reported = row['same_grid_discrepancy']; reported_common = row['common_grid_discrepancy']
            rom = name.startswith(('head_', 'trained_', 'nested_', 'linear_', 'pod_'))
            if rom:
                with np.load(out / row['field_artifact']) as f:
                    a, b = f['coefficients'], f['velocity_coefficients']
                gg, stiff = bank_of(name)
                u, v = decode(a, gg, (n - 1, n - 1)), decode(b, gg, (n - 1, n - 1))
                got = worst_errors(u, v, ru, rv, h, c)
                scope = 'full'
                if 'reduced_energy' in row:
                    e = .5 * (np.sum(b * b, axis=1) + c * c * np.einsum('ti,ij,tj->t', a, stiff, a))
                    drift = float(np.max(abs(e / e[0] - 1.)))
                    checks['energy'].append(dict(case=case, method=name, recomputed_max_relative_drift=drift,
                                                 reported=row['reduced_energy']['max_relative_drift'], difference=abs(drift - row['reduced_energy']['max_relative_drift'])))
            elif name == 'dst':
                got = worst_errors(ru, rv, ru, rv, h, c); scope = 'full(identity)'
            else:
                with np.load(out / row['field_artifact']) as f:
                    u, v = f['u'], f['v']
                if full:
                    got = worst_errors(u, v, ru, rv, h, c); scope = 'full'
                else:
                    got = worst_errors(u, v, cru, crv, hc, c); scope = 'common'; reported = reported_common
            diffs = {k: abs(got[k] - reported[k]['max_initial_normalized']) for k in ('displacement', 'velocity', 'energy_state')}
            ef = float(np.max(abs(np.asarray(got['energy_fraction']) - np.asarray(reported['energy_fraction']))))
            worst = max(worst, max(diffs.values()), ef)
            checks['rows'].append(dict(case=case, method=name, scope=scope, recomputed={k: got[k] for k in diffs}, differences=diffs, energy_fraction_difference=ef))
        print('audited', case, flush=True)
    # timing medians (all repetitions) and retained gates
    for name in result['arms']:
        rows = [r for r in result['invocations'] if r['method'] == name]
        gpu = [r['seconds']['complete_device_query'] * 1e3 for r in rows]
        total = [r['device_plus_output_transfer_seconds'] * 1e3 for r in rows]
        errors = [r['same_grid_discrepancy']['energy_state']['max_initial_normalized'] for r in rows if r['repetition'] == 0]
        checks['timing'][name] = dict(count=len(rows), median_gpu_ms=float(np.median(gpu)), median_complete_ms=float(np.median(total)),
                                      worst_energy_state=float(np.max(errors)), median_energy_state=float(np.median(errors)))
    gates = json.loads((ROOT / 'experiments/w-ladder/retained-gates.json').read_text())['gates']
    for gate in gates:
        if gate['intervals'] != n:
            continue
        if gate['metric'] == 'selected_fit_objective':
            # Not independently recomputable here: it would require rerunning the 8-start LM fit,
            # i.e. the driver's own code. The in-job gate covers it; this audit covers the errors.
            checks['gates'].append(dict(**gate, recomputed=None, relative_difference=None, passed=None,
                                        note='skipped: fit objective is not recomputable without the solver'))
            continue
        rows = [r for r in checks['rows'] if r['case'] == gate['case'] and r['method'] == gate['method']]
        if rows:
            rel = abs(rows[0]['recomputed'][gate['metric']] - gate['value']) / abs(gate['value'])
            # DESIGN A4, applied independently of the driver: an arm carrying the 8-start cold fit
            # inherits a start-tie flip (the starts reach the same minimum; argmin breaks the tie by
            # index), which moves the trajectory at ~1e-8. Linear arms are deterministic.
            fit_arm = gate['method'].startswith(('head_', 'trained_', 'nested_'))
            tol = 1e-7 if fit_arm else 1e-9
            checks['gates'].append(dict(**gate, recomputed=rows[0]['recomputed'][gate['metric']],
                                        relative_difference=rel, tolerance=tol, passed=rel <= tol))
    checks['max_error_difference'] = worst
    checks['reference_max_relative_difference'] = max(r['regenerated_vs_saved_relative'] for r in checks['reference'])
    checks['energy_max_difference'] = max([e['difference'] for e in checks['energy']] or [0.])
    checks['gates_all_passed'] = all(x['passed'] for x in checks['gates'] if x['passed'] is not None)
    checks['passed'] = bool(worst <= 1e-10 and checks['reference_max_relative_difference'] <= 1e-10 and checks['energy_max_difference'] <= 1e-12 and checks['gates_all_passed'])
    checks['result_sha256'] = hashlib.sha256((out / 'result.json').read_bytes()).hexdigest()
    (run / 'audit.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps({k: checks[k] for k in ('max_error_difference', 'reference_max_relative_difference', 'energy_max_difference', 'gates_all_passed', 'passed')}))


if __name__ == '__main__':
    main()
