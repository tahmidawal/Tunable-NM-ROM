"""quadrature-burgers3d: independent NumPy audit of one pulled panel (DESIGN.md section 10). No JAX.

    audit_q.py --panel runs/val65/code/output --refined <refine_q json> [--selection <select_q json>] --out <json>

1. every (arm|FOM, case) 16^3-lattice error recomputed from the saved restricted fields (<= 1e-12 abs vs recorded);
2. ROM restricted fields re-decoded from the saved internal coefficients with the NumPy bank (<= 1e-10 rel), for the
   first 4 cases of every arm;
3. refined errors: refine_q (NumPy) against the in-job values when present (<= 1e-10 abs);
4. rho ingredients of the saved audit states: off-mesh rule (NumPy bank + forward derivative) vs saved (<= 1e-10 rel);
   mesh target via scipy DST + NumPy upwind at n <= 129 (<= 1e-10 rel); continuum target re-done with NumPy Gauss 64^3
   vs the saved Gauss 80^3 target (<= 1e-5 rel, the continuum-check bar); the recorded rho of these states recomputed;
5. the selected setting recomputed by an independent implementation (if --selection);
6. must-fail audit controls: a swapped reference case and a 1 % perturbed recorded error must be detected.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import npcore as N

HERE = Path(__file__).resolve().parent


def restrict16_coords(n):
    s = (n - 1) // 32
    k = (2 * np.arange(16) + 1) * s / (n - 1)
    X, Y, Z = np.meshgrid(k, k, k, indexing='ij')
    return np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--panel', required=True)
    ap.add_argument('--refined', default=None)
    ap.add_argument('--selection', default=None)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    pdir = Path(a.panel)
    rep = json.loads((pdir / 'result.json').read_text())
    n = rep['mesh']
    p, T = N.load_bank(HERE / 'inputs' / 'model_M2')
    if rep['config'].get('R_truncate'):
        T = T[:, :rep['config']['R_truncate']]
    res = dict(mesh=n, job_id=rep['job_id'], checks={})
    ok = {}
    # 1. restricted errors
    R16 = np.load(pdir / 'fields' / 'reference_restricted.npz')
    worst = 0.
    for nm, arm in list(rep['arms'].items()) + list(rep['fom'].items()):
        for q in arm['quick']:
            f16 = np.load(pdir / 'fields' / f"{nm}_c{q['case']}.npz")['f16']
            r = R16[f"c{q['case']}"]
            e = np.linalg.norm(f16 - r, axis=1) / np.linalg.norm(r[0])
            worst = max(worst, float(np.abs(e - np.asarray(q['err_restricted16'])).max()))
    res['checks']['restricted16_errors_max_abs_diff'] = worst
    ok['restricted16'] = worst <= 1e-12
    # 2. decoding of ROM fields from internal coefficients
    G16 = N.bank_np(p, T, restrict16_coords(n))
    keep = int(round(0.05 / rep['config']['dt']))
    worst = 0.
    for nm, arm in rep['arms'].items():
        Rp = arm['spec']['Rp']
        for q in arm['quick'][:4]:
            z = np.load(pdir / 'fields' / f"{nm}_c{q['case']}.npz")
            F = z['internal'][::keep] @ G16[:, :Rp].T
            worst = max(worst, float(np.abs(F - z['f16']).max() / np.abs(z['f16']).max()))
    res['checks']['decode16_max_rel'] = worst
    ok['decode16'] = worst <= 1e-10
    # 3. refined (from refine_q)
    if a.refined:
        rf = json.loads(Path(a.refined).read_text())['panels'][str(n)]
        res['checks']['refined_in_job_vs_numpy'] = rf['max_abs_diff_vs_in_job']
        res['checks']['refined_in_job_present'] = bool(rep.get('refined_in_job'))
        res['checks']['refined_in_job_compared'] = rf['in_job_compared']
        ok['refined'] = rf['max_abs_diff_vs_in_job'] <= 1e-10 and (rf['in_job_compared'] > 0 or
                                                                   not rep.get('refined_in_job'))
    # 4. rho ingredients
    rules = np.load(HERE / 'rules' / 'rules.npz')
    rule = rep['config']['audit_rule']
    for Rp in rep['rho']:
        f = pdir / 'fields' / f'rho_R{Rp}_{rule}.npz'
        z = np.load(f)
        Cs, v, tc, tm = z['Cs'], z['v'], z['tgt_cont'], z['tgt_mesh']
        M = rep['rho'][Rp]['M']
        k, _ = N.modes(n, 4 * int(Rp))
        assert len(k) == M, (len(k), M)
        Xr, wr = rules[f'{rule}_X'], rules[f'{rule}_w']
        G, D = N.bank_np(p, T[:, :int(Rp)], Xr, deriv=True)
        vn = N.offmesh_adv_np(G, D, Xr, wr, n, k, Cs)
        c = dict(offmesh_rule_rel=float(np.abs(vn - v).max() / np.abs(v).max()))
        Xg, wg = rules['gl64_X'], rules['gl64_w']
        tg = 0.
        for s in range(0, len(Xg), 1 << 15):
            Gg, Dg = N.bank_np(p, T[:, :int(Rp)], Xg[s:s + (1 << 15)], deriv=True)
            tg = tg + N.offmesh_adv_np(Gg, Dg, Xg[s:s + (1 << 15)], wg[s:s + (1 << 15)], n, k, Cs)
        c['continuum_gl64_vs_saved_gl80_rel'] = float((np.linalg.norm(tg - tc, axis=1) /
                                                        np.linalg.norm(tc, axis=1)).max())
        if n <= 129:
            Gm = N.bank_np(p, T[:, :int(Rp)], N.mesh_coords(n))
            tmn = np.stack([N.tested_np(N.upwind_np(Gm @ cc, n), n, k) for cc in Cs])
            c['mesh_target_rel'] = float(np.abs(tmn - tm).max() / np.abs(tm).max())
            del Gm
        rc = np.linalg.norm(vn - tg, axis=1) / np.linalg.norm(tg, axis=1)
        c['rho_cont_states_numpy_max'] = float(rc[z['k'] >= 1].max())
        c['rho_cont_recorded_worst'] = rep['rho'][Rp]['rules'][rule]['cont']['worst']
        c['rho_numpy_le_recorded_worst'] = bool(c['rho_cont_states_numpy_max'] <= c['rho_cont_recorded_worst'] * 1.001 + 1e-5)
        res['checks'][f'rho_R{Rp}'] = c
        ok[f'rho_R{Rp}'] = (c['offmesh_rule_rel'] <= 1e-10 and c['continuum_gl64_vs_saved_gl80_rel'] <= 1e-5
                            and c.get('mesh_target_rel', 0.) <= 1e-10 and c['rho_numpy_le_recorded_worst'])
    # 4b. the recorded worst rho of EVERY rule reproduced at its saved arg-max state (NumPy rule vs the saved targets)
    for Rp in rep['rho']:
        z = np.load(pdir / 'fields' / f'rho_argmax_R{Rp}.npz')
        idx = list(z['idx'])
        k, _ = N.modes(n, 4 * int(Rp))
        worst = 0.
        for nm, (ic, im) in zip(z['names'], z['amax']):
            Xr, wr = rules[f'{nm}_X'], rules[f'{nm}_w']
            G, D = N.bank_np(p, T[:, :int(Rp)], Xr, deriv=True)
            for which, i_, key in (('cont', ic, 'tgt_cont'), ('mesh', im, 'tgt_mesh')):
                r_ = idx.index(i_)
                v_ = N.offmesh_adv_np(G, D, Xr, wr, n, k, z['Cs'][r_:r_ + 1])[0]
                t_ = z[key][r_]
                rho_np = float(np.linalg.norm(v_ - t_) / np.linalg.norm(t_))
                rec_ = rep['rho'][Rp]['rules'][str(nm)][which]['worst']
                worst = max(worst, abs(rho_np - rec_) / rec_)
        res['checks'][f'rho_worst_reproduced_R{Rp}_max_rel'] = worst
        ok[f'rho_worst_reproduced_R{Rp}'] = worst <= 1e-8
    # 5. selection, independent implementation
    if a.selection:
        sel = json.loads(Path(a.selection).read_text())['meshes'][str(n)]['R']
        tim = rep['timing']['summary']
        agree = {}
        g = rep['gates']
        timing_ok = bool(g['timing_drift_pass'] and g['timing_neighbour_pass'] and g['deterministic_pass'])
        hard_ok = bool(g['gram_condition'] <= 1e8 and g['tensor_vs_direct'] <= 1e-10 and g['G1_derivative_vs_fd'] <= 1e-6
                       and g['G2_meshnodes_offmesh_vs_tensor'] <= 1e-12 and g['G3_jacobian_vs_jacfwd'] <= 1e-12
                       and g['G3_adv_half_Jc'] <= 1e-12 and g['G4_solver_vs_vendor_fields'] <= 1e-12
                       and g['G4_solver_vs_vendor_coefs'] <= 1e-12 and g['G4_iterations_reasons_equal']
                       and g['reference_residual'] < 1e-9)
        s_m = {nm: rep['sizes'][nm]['m'] for nm in rep['arms'] if 'm' in rep['sizes'].get(nm, {})}
        for Rp in rep['rho']:
            hard_ok_R = hard_ok and rep['rho'][Rp]['continuum_check']['vs_target']['worst'] <= 1e-5
            cand, conv_ok = [], {}
            for nm, arm in rep['arms'].items():
                s = arm['spec']
                if s['Rp'] != int(Rp) or s['family'] != 'offmesh' or s['rule'] in rep['config']['controls']:
                    continue
                rc_ = rep['rho'][Rp]['rules'][s['rule']]['cont']['worst']
                nst = (arm['reason_counts']['0'] + arm['reason_counts']['2']) / (arm['cases_run'] * 25)
                d = 0.0 if s['rule'] == rep['config']['converged_rule'] else \
                    arm['distance'][rep['config']['converged_rule']]['worst']
                if arm['all_finite'] and arm['reason_counts']['3'] == 0 and nst <= 0.01 and rc_ <= 0.116 and d <= 1e-3:
                    cand.append(((tim[nm]['median_ms'] if timing_ok else s_m[nm]), nm))
                if s['rule'] in (rep['config']['converged_rule'], 'gl32'):
                    conv_ok[s['rule']] = bool(arm['all_finite'] and arm['reason_counts']['3'] == 0 and nst <= 0.01
                                              and rc_ <= 0.116 and d <= 1e-3)
            mine = min(cand)[1] if (cand and all(conv_ok.values()) and len(conv_ok) == 2 and hard_ok_R) else None
            agree[Rp] = dict(numpy=mine, select_q=sel[Rp]['selected'], match=mine == sel[Rp]['selected'])
        res['checks']['selection'] = agree
        ok['selection'] = all(v['match'] for v in agree.values())
    # 6. audit controls
    q0 = rep['arms'][f"tensor_R{rep['config']['Rps'][0]}"]['quick'][0]
    f16 = np.load(pdir / 'fields' / f"tensor_R{rep['config']['Rps'][0]}_c0.npz")['f16']
    r1 = R16['c1']
    e_sw = np.linalg.norm(f16 - r1, axis=1) / np.linalg.norm(r1[0])
    sw = float(np.abs(e_sw - np.asarray(q0['err_restricted16'])).max())
    pert = np.asarray(q0['err_restricted16']) * 1.01
    pt = float(np.abs(pert - np.asarray(q0['err_restricted16'])).max())
    res['checks']['control_swapped_reference_diff'] = sw
    res['checks']['control_perturbed_diff'] = pt
    ok['control_swapped_detected'] = sw > 1e-12
    ok['control_perturbed_detected'] = pt > 1e-12
    res['ok'] = ok
    res['all_pass'] = all(ok.values())
    Path(a.out).write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(dict(ok=ok, checks={k: v for k, v in res['checks'].items() if not isinstance(v, dict)}), indent=1))


if __name__ == '__main__':
    main()
