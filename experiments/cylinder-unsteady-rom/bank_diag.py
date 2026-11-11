"""Rebuttal push diagnosis (PROTOCOL.md step 1): where a bank misses, against POD at the same rank.

Static part (no trajectory): per-mode regression errors of bank function i against POD mode i,
L2 span miss of every exported POD mode onto span(bank), divergence, all split by region.
Development part: projection floor and rollout error of bank and POD arms split by region at a few
saved times (reference evaluated on Q).  Development split only.
Regions: cylinder (within 0.05 of the surface), walls (within 0.03 of a channel wall),
wake (x > 0.25 otherwise), upstream (rest).
"""
import argparse
import json
from pathlib import Path

import numpy as np

import rom_core as rc
import rom_export as rx

REG = ('cylinder', 'walls', 'wake', 'upstream')


def regions(x):
    rad = np.linalg.norm(x - np.array([0.2, 0.2]), axis=1)
    cyl = rad - 0.05 < 0.05
    wall = ~cyl & ((x[:, 1] < 0.03) | (x[:, 1] > 0.41 - 0.03))
    wake = ~cyl & ~wall & (x[:, 0] > 0.25)
    return {'cylinder': cyl, 'walls': wall, 'wake': wake, 'upstream': ~cyl & ~wall & ~wake}


def main():
    ap = argparse.ArgumentParser()
    for k in ('--export', '--manifest', '--case', '--bank-eval', '--bank-proj', '--out'):
        ap.add_argument(k, required=True)
    ap.add_argument('--rank', type=int, required=True); ap.add_argument('--frames', type=int, default=12)
    ap.add_argument('--dt', type=float, default=0.0003125)
    a = ap.parse_args()
    man = json.loads(Path(a.manifest).read_text())
    (case,) = [c for c in man['cases'] if c['name'] == a.case]
    if case['split'] != 'dev':
        raise PermissionError('diagnosis uses the development trajectory only')
    ex = rc.Export(a.export); r = a.rank
    q = np.load(Path(a.export) / 'quadrature.npz'); x, w = q['points'], q['weights']
    reg = regions(x)
    z = np.load(a.bank_eval)
    V, G = z['V'][..., :r], z['G'][..., :r]
    sig = np.load(Path(a.export) / 'pod.npz')['singular_values']
    npod = sum(n.startswith('pod') for n in ex.names)
    pc = [ex.col(f'pod{i}') for i in range(npod)]
    Vt, Gt = ex.V[..., pc], ex.G[..., pc]
    out = {'rank': r, 'region_area_fraction': {k: float(w[m].sum() / w.sum()) for k, m in reg.items()}}
    # ---- static -------------------------------------------------------------
    dv2 = np.einsum('qcr->qr', (V - Vt[..., :r]) ** 2); dg2 = np.einsum('qcdr->qr', (G - Gt[..., :r]) ** 2)
    tv2 = np.einsum('qcr->qr', Vt[..., :r] ** 2); tg2 = np.einsum('qcdr->qr', Gt[..., :r] ** 2)
    div2 = (G[:, 0, 0] + G[:, 1, 1]) ** 2
    s2 = sig[:r] ** 2
    st = {'value_rel_per_mode': np.sqrt(w @ dv2 / (w @ tv2)).tolist(), 'gradient_rel_per_mode': np.sqrt(w @ dg2 / (w @ tg2)).tolist(),
          'divergence_over_gradient_per_mode': np.sqrt(w @ div2 / (w @ np.einsum('qcdr->qr', G ** 2))).tolist(), 'regions': {}}
    for k, m in reg.items():
        st['regions'][k] = {
            'share_of_sigma2_weighted_value_error': float((w[m] @ dv2[m]) @ s2 / ((w @ dv2) @ s2)),
            'share_of_sigma2_weighted_gradient_error': float((w[m] @ dg2[m]) @ s2 / ((w @ dg2) @ s2)),
            'share_of_sigma2_weighted_divergence': float((w[m] @ div2[m]) @ s2 / ((w @ div2) @ s2)),
            'value_rel_in_region_sigma2': float(np.sqrt((w[m] @ dv2[m]) @ s2 / ((w[m] @ tv2[m]) @ s2))),
            'gradient_rel_in_region_sigma2': float(np.sqrt((w[m] @ dg2[m]) @ s2 / ((w[m] @ tg2[m]) @ s2)))}
    Mb = rc.wdot(w, V, V); R = rc.wdot(w, V, Vt)                       # (r, npod)
    coef = np.linalg.solve(Mb, R); miss = 1 - np.einsum('ij,ij->j', R, coef) / np.einsum('q,qcj->j', w, Vt ** 2)
    st['l2_span_miss_per_pod_mode'] = miss.tolist()
    st['sigma2_weighted_span_miss_first_r'] = float(miss[:r] @ s2 / s2.sum())
    st['snapshot_energy_equivalent_error_first_r'] = float(np.sqrt(miss[:r] @ s2 / (sig ** 2).sum()))
    st['pod_truncation_tail_same_units'] = float(np.sqrt((sig[r:] ** 2).sum() / (sig ** 2).sum()))
    res = Vt - np.einsum('qci,ij->qcj', V, coef); res2 = np.einsum('qcj->qj', res ** 2)[:, :r]
    for k, m in reg.items():
        st['regions'][k]['share_of_sigma2_weighted_span_miss'] = float((w[m] @ res2[m]) @ s2 / ((w @ res2) @ s2))
    out['static'] = st
    del dv2, dg2, tv2, tg2, res, res2
    # ---- development trajectory ---------------------------------------------
    ex.add_bank('trained', V, G, z['outlet_V'][..., :r], {lab: (z[f'{lab}_V'][..., :r], z[f'{lab}_G'][..., :r]) for lab in ('inlet', 'outlet')})
    proj = dict(np.load(Path(a.export) / f'proj_{a.case}.npz'))
    proj['uv_trained'] = np.load(a.bank_proj)['uv_trained'][:, :r]
    ts = proj['t']; idx = np.rint((ts - ts[0]) / a.dt).astype(int)
    sel = np.linspace(0, len(ts) - 1, a.frames + 1).astype(int)[1:]
    arms = {'bank': rc.Arm(ex, 'bank', r, 'trained'), 'pod': rc.Arm(ex, 'pod', r)}
    dev = {}; roll = {}
    for name, arm in arms.items():
        fl = arm.floors(proj)
        full, _ = arm.ref_coeffs(proj)
        A, C, F, _, _ = arm.rollout(0.1 / case['Re'], a.dt, full[0], int(idx[-1]))
        err = arm.errors(proj, A, C, idx)
        dev[name] = {'floor_basis_only_median': float(np.median(fl['velocity_basis_only'])),
                     'floor_with_supremizers_median': float(np.median(fl['velocity_with_supremizers'])),
                     'floor_with_supremizers_rms': float(np.sqrt(np.mean(fl['velocity_with_supremizers'] ** 2))),
                     'rollout_e_u_rms': float(np.sqrt(np.mean(err['e_u'] ** 2))), 'rollout_e_u_max': float(err['e_u'].max()),
                     'rollout_divergence_median': float(np.median(err['divergence'])),
                     'e_u_at_quarters': [float(err['e_u'][i]) for i in np.linspace(0, len(ts) - 1, 5).astype(int)],
                     'regions': {k: {'projection_err2': 0.0, 'rollout_err2': 0.0, 'rollout_div2': 0.0} for k in REG}}
        roll[name] = (A[idx], full)
    import ngsolve as ng
    uu_tot = 0.0
    with ng.TaskManager():
        fe = rx.FE(ng, man['mesh_path'], man['mesh_sha256'], k=man['velocity_order'], qorder=man['quadrature_order'])
        fe.rule()
        want = {round(float(ts[i]), 9): i for i in sel}
        for t, xs in rx.recorded(case['run_dir'], case['stage']):
            i = want.get(round(t, 9))
            if i is None:
                continue
            u, _ = fe.eval_v(xs[:fe.nv]); uu_tot += float(w @ np.einsum('qc,qc->q', u, u))
            for name, arm in arms.items():
                Aidx, full = roll[name]
                cproj = np.linalg.solve(arm.M, full[i])
                e_p = np.einsum('qc,qc->q', *(2 * [u - arm.lV - arm.Vb @ cproj]))
                e_r = np.einsum('qc,qc->q', *(2 * [u - arm.lV - arm.Vb @ Aidx[i]]))
                g = arm.lG + arm.Gb @ Aidx[i]; d2 = (g[:, 0, 0] + g[:, 1, 1]) ** 2
                for k, m in reg.items():
                    rr = dev[name]['regions'][k]
                    rr['projection_err2'] += float(w[m] @ e_p[m]); rr['rollout_err2'] += float(w[m] @ e_r[m]); rr['rollout_div2'] += float(w[m] @ d2[m])
    for name in arms:
        tot = {f: sum(dev[name]['regions'][k][f] for k in REG) for f in ('projection_err2', 'rollout_err2', 'rollout_div2')}
        dev[name]['frames_relative_error'] = {'projection': float(np.sqrt(tot['projection_err2'] / uu_tot)), 'rollout': float(np.sqrt(tot['rollout_err2'] / uu_tot))}
        for k in REG:
            rr = dev[name]['regions'][k]
            dev[name]['regions'][k] = {'projection_share': rr['projection_err2'] / tot['projection_err2'],
                                       'rollout_share': rr['rollout_err2'] / tot['rollout_err2'],
                                       'divergence_share': rr['rollout_div2'] / tot['rollout_div2']}
    out['development'] = dev; out['frames'] = [float(ts[i]) for i in sel]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps({'static_span_miss': st['sigma2_weighted_span_miss_first_r'], 'dev': {n: {k: v for k, v in d.items() if k != 'regions'} for n, d in dev.items()}}, indent=1))


if __name__ == '__main__':
    main()
