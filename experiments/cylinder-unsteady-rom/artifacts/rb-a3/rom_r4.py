"""G5 R4/R5 rollouts (DESIGN-G5 §5.3–5.5): every arm on one trajectory.

Usage: rom_r4.py --export <dir> --manifest <json> --case <name> --dt <dt0> --arms <json list>
       --out <dir> [--trained-bank <npz from bank_eval>]
Arm spec: {"name": "pod_r16", "kind": "pod", "r": 16} or {"name": "bank_r16", "kind": "bank",
"r": 16, "key": "trained"} or {"name": "frozen_bank_r4", "kind": "bank", "r": 4, "key": "frozen"}.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np

import cycles as cyc
import rom_core as rc

TARGETS = {'e_u_rms': 1e-2, 'e_u_max': 3e-2, 'e_p_rms': 2e-2, 'St_rel': 1e-2, 'CD_max_rel': 2e-2, 'CL_max_abs': 2e-2}


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def reference_history(run_dir, stage):
    st = json.loads((Path(run_dir) / 'stages.json').read_text())
    (s,) = [x for x in st['stages'] if x['name'] == stage]
    h = np.load(Path(run_dir) / 'history.npz')['rows']
    keep = (h[:, 0] >= s['t_start'] - 1e-9) & (h[:, 0] <= s['t_end'] + 1e-9)
    return h[keep]


def stats(t, cd, cl, dp):
    recs = cyc.cycles(t, cd, cl, dp)
    if not recs:
        return None
    keys = ('CD_max', 'CL_max', 'CL_min', 'St', 'dp_at_t0_plus_half_period', 'period')
    return {k: float(np.mean([r[k] for r in recs])) for k in keys} | {'cycles': len(recs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', required=True); ap.add_argument('--manifest', required=True)
    ap.add_argument('--case', required=True); ap.add_argument('--dt', type=float, required=True)
    ap.add_argument('--arms', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--trained-bank'); ap.add_argument('--trained-proj')
    ap.add_argument('--uncertainty', required=True, help='R1DE.json with passing R1b/R1c/R1d/R1e and bound inputs')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    man = json.loads(Path(a.manifest).read_text())
    (case,) = [c for c in man['cases'] if c['name'] == a.case]
    nu = 0.1 / case['Re']
    ex = rc.Export(a.export)
    unc = json.loads(Path(a.uncertainty).read_text())
    if not all(unc.get(k) == 'pass' for k in ('R1b', 'R1c', 'R1d_gate', 'R1e_gate')) or not unc.get('uncertainty_inputs_bound'):
        raise PermissionError('reference not admitted or uncertainty inputs not bound; R4 refuses to run')
    U = unc['uncertainty']
    if a.trained_bank:
        z = np.load(a.trained_bank)
        ex.add_bank('trained', z['V'], z['G'], z['outlet_V'],
                    {lab: (z[f'{lab}_V'], z[f'{lab}_G']) for lab in ('inlet', 'outlet')})
    proj = dict(np.load(Path(a.export) / f"proj_{a.case}.npz"))
    if a.trained_bank:
        import hashlib
        tp = dict(np.load(a.trained_proj))
        if str(tp['bank_eval_sha256']) != hashlib.sha256(Path(a.trained_bank).read_bytes()).hexdigest():
            raise ValueError('trained-bank projections are not bound to this bank evaluation')
        if not np.array_equal(tp['t'], proj['t']):
            raise ValueError('trained-bank projections are on different saved times')
        proj['uv_trained'] = tp['uv_trained']
    ts = proj['t']; t0 = ts[0]
    idx = np.rint((ts - t0) / a.dt).astype(int)
    if np.max(np.abs(t0 + idx * a.dt - ts)) > 1e-9:
        raise ValueError('save times are not on the rollout grid')
    n_steps = int(idx[-1])
    href = reference_history(case['run_dir'], case['stage'])
    tr = href[:, 0]
    ref_stats = stats(tr[1:], href[1:, 1], href[1:, 2], href[1:, 5] - href[1:, 6])
    period = ref_stats['period'] if ref_stats else 1 / 3.0
    import hashlib
    summary = {'export': str(a.export), 'export_manifest_sha256': hashlib.sha256((Path(a.export) / 'EXPORT.json').read_bytes()).hexdigest(),
               'trained_bank_sha256': hashlib.sha256(Path(a.trained_bank).read_bytes()).hexdigest() if a.trained_bank else None,
               'case': a.case, 'Re': case['Re'], 'nu': nu, 'dt': a.dt, 'steps': n_steps, 'reference_cycles': ref_stats,
               'targets': TARGETS, 'arms': {}}
    for spec in json.loads(a.arms):
        name = spec['name']; rec = {'spec': spec}
        try:
            key = spec.get('key')
            if spec['kind'] == 'bank' and key == 'trained' and spec['r'] < z['V'].shape[-1]:
                # PCA prefix of the trained bank (rebuttal push): first r functions, own identity-tagged key
                key = f"trained_p{spec['r']}"; n_ = spec['r']
                if key not in ex.bank:
                    ex.add_bank(key, z['V'][..., :n_], z['G'][..., :n_], z['outlet_V'][..., :n_],
                                {lab: (z[f'{lab}_V'][..., :n_], z[f'{lab}_G'][..., :n_]) for lab in ('inlet', 'outlet')})
                    proj[f'uv_{key}'] = proj['uv_trained'][:, :n_]
            arm = rc.Arm(ex, spec['kind'], spec['r'], key)
            rec['floor'] = {k_: {'median': float(np.median(v_)), 'max': float(np.max(v_))} for k_, v_ in arm.floors(proj).items()}
            rec.update(velocity_dimension=arm.d, pressure_dimension=arm.dp, total_unknowns=arm.d + arm.dp,
                       retained=arm.retained, raw=arm.raw_count, orthonormality_defect=arm.orth_defect)
            full, _ = arm.ref_coeffs(proj)
            a0 = full[0]
            w0 = time.perf_counter()
            A, C, F, probes, its = arm.rollout(nu, a.dt, a0, n_steps)
            rec['rollout_seconds'] = time.perf_counter() - w0
            ini = dict(arm.initial)
            ini['relative_continuity_defect'] = ini['continuity_defect'] / ini['continuity_scale']
            ini['flag'] = 'qualified' if ini['relative_continuity_defect'] <= 1e-3 else 'large_initial_defect_reported'
            rec['initial'] = ini
            rec['newton_iterations'] = {'median': float(np.median(its[2:])), 'max': int(its.max())}
            err = arm.errors(proj, A, C, idx)
            t_rom = t0 + np.arange(n_steps + 1) * a.dt
            # forces on common step times (reference rows are every reference step = dt)
            ref_on = np.interp(t_rom, tr, href[:, 1]), np.interp(t_rom, tr, href[:, 2])
            ok = np.isfinite(F[:, 0])
            rec['force'] = {'CD_maxabs_rel': float(np.max(np.abs(F[ok, 0] - ref_on[0][ok])) / np.max(np.abs(ref_on[0][ok]))),
                            'CL_maxabs': float(np.max(np.abs(F[ok, 1] - ref_on[1][ok])))}
            st = stats(t_rom[ok], F[ok, 0], F[ok, 1], (probes[:, 0] - probes[:, 1])[ok])
            rec['cycles'] = st
            if st and ref_stats:
                rec['cycle_errors'] = {'St_rel': abs(st['St'] - ref_stats['St']) / ref_stats['St'],
                                       'CD_max_rel': abs(st['CD_max'] - ref_stats['CD_max']) / ref_stats['CD_max'],
                                       'CL_max_abs': abs(st['CL_max'] - ref_stats['CL_max']),
                                       'dp_rel': abs(st['dp_at_t0_plus_half_period'] - ref_stats['dp_at_t0_plus_half_period'])
                                       / abs(ref_stats['dp_at_t0_plus_half_period'])}
            last = ts >= ts[-1] - period
            rec['e_u'] = {'rms': rms(err['e_u']), 'max': float(err['e_u'].max()), 'last_period_mean': float(err['e_u'][last].mean())}
            rec['e_p'] = {'rms': rms(err['e_p']), 'max': float(err['e_p'].max()), 'last_period_mean': float(err['e_p'][last].mean()),
                          'rms_excluding_t0_supplemental': rms(err['e_p'][1:])}
            # reduced energy balance every 4th save (G5-A5 rev.3)
            eb = []
            for i in range(0, len(idx), 4):
                n = int(idx[i])
                if n < 2:
                    continue
                adot = (3 * A[n] - 4 * A[n - 1] + A[n - 2]) / (2 * a.dt)
                eb.append(arm.energy_balance(nu, A[n], adot, C[n])['residual'])
            rec['energy_balance_residual'] = {'median': float(np.median(eb)), 'max': float(np.max(eb)), 'count': len(eb)}
            rec['divergence'] = {'median': float(np.median(err['divergence'])), 'max': float(err['divergence'].max())}
            rec['mass'] = {'max': float(err['mass'].max())}
            # Phase-aligned (supplemental, G5-A5): one fixed support for every lag, the saves with
            # t in [t_end - 2T + T/2, t_end - T/2]; lags |k| <= T/2 on the save grid.
            ds = ts[1] - ts[0]
            kmax = int(round(0.5 * period / ds))
            I = np.where((ts >= ts[-1] - 1.5 * period) & (ts <= ts[-1] - 0.5 * period))[0]
            if I.min() - kmax < 0 or I.max() + kmax >= len(ts):
                raise ValueError('horizon too short for the fixed phase-search support')
            cl_rom_s = np.interp(ts, t_rom[ok], F[ok, 1]); cl_ref_s = np.interp(ts, tr, href[:, 2])
            best = min(((k, rms(cl_rom_s[I] - cl_ref_s[I + k])) for k in range(-kmax, kmax + 1)), key=lambda z: z[1])
            k = best[0]
            shifted = {kk: vv[I + k] for kk, vv in proj.items() if kk.startswith('uv') or kk in ('uu', 'pp', 'pv')}
            err_al = arm.errors(shifted, A, C, idx[I])
            rec['phase_aligned'] = {'lag_saves': int(k), 'lag_seconds': float(k * (ts[1] - ts[0])), 'lift_rms_at_lag': best[1],
                                    'e_u_rms': rms(err_al['e_u']), 'e_p_rms': rms(err_al['e_p'])}
            ce = rec.get('cycle_errors', {})
            got = {'e_u_rms': rec['e_u']['rms'], 'e_u_max': rec['e_u']['max'], 'e_p_rms': rec['e_p']['rms'], **ce}
            fh, cs = U['field_and_force_over_horizon'], U['cycle_statistics']
            delta = {'e_u_rms': fh['e_u_max'], 'e_u_max': fh['e_u_max'], 'e_p_rms': fh['e_p_max'], 'St_rel': cs['St_rel'],
                     'CD_max_rel': cs['CD_max_abs'] / ref_stats['CD_max'], 'CL_max_abs': cs['CL_max_abs']}
            rec['targets'] = {}
            for k2, v in TARGETS.items():
                val = got.get(k2)
                rec['targets'][k2] = {'value': val, 'target': v, 'met': val is not None and val <= v,
                                      'reference_uncertainty': delta[k2], 'resolved': delta[k2] <= v / 5,
                                      'classification': 'resolved under the linear-propagation estimate' if delta[k2] <= v / 5
                                      else 'unresolved (reference uncertainty > target/5)'}
            rec['status'] = 'completed'
            np.savez(out / f'series_{a.case}_{name}.npz', t_save=ts, e_u=err['e_u'], e_p=err['e_p'], divergence=err['divergence'],
                     mass=err['mass'], t=t_rom, CD=F[:, 0], CL=F[:, 1], p_front=probes[:, 0], p_back=probes[:, 1],
                     CD_ref=ref_on[0], CL_ref=ref_on[1], A=A[idx], C=C[idx], newton=its)
        except Exception as exc:  # retained failure, never dropped
            rec['status'] = 'failed'; rec['error'] = repr(exc)
        summary['arms'][name] = rec
        print(name, rec['status'], rec.get('e_u'), rec.get('error', ''), flush=True)
    import resource
    summary['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    summary['uncertainty_source'] = {'file': a.uncertainty, 'kind': U['kind']}
    (out / f'R4-{a.case}.json').write_text(json.dumps(summary, indent=2, default=float) + '\n')


if __name__ == '__main__':
    main()
