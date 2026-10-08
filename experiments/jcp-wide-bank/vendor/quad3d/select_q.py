"""quadrature-burgers3d: the pre-registered rules of DESIGN.md sections 6-7, applied mechanically.

    select_q.py --panels <result.json> ... --refined <refine_q json> --out <json> [--frozen selection.json]

Per mesh and R': eligibility of every off-mesh arm, the converged-rollout check, the selected off-mesh setting
(validation; with --frozen the held-out run reuses the frozen selection and only reports), the (i) lattice-vs-Gauss
verdict, the smallest lattice m reaching the tensor and reaching the converged rollout, the must-fail controls, every
gate, timings, both speedup rules. Uses only recorded numbers; no thresholds other than DESIGN's.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

BAR_RHO = 0.116
BAR_CONV = 1e-3
LADDER = [4096, 8192, 16384, 32768]
STEPS = 25                       # dt = 0.01 over t = 0.25


def gates_ok(g):
    return dict(table=bool(g['gram_condition'] <= 1e8 and g['tensor_vs_direct'] <= 1e-10),
                G1=bool(g['G1_derivative_vs_fd'] <= 1e-6), G2=bool(g['G2_meshnodes_offmesh_vs_tensor'] <= 1e-12),
                G3=bool(g['G3_jacobian_vs_jacfwd'] <= 1e-12 and g['G3_adv_half_Jc'] <= 1e-12),
                reference=bool(g['reference_residual'] < 1e-9),
                G4=bool(g['G4_solver_vs_vendor_fields'] <= 1e-12 and g['G4_solver_vs_vendor_coefs'] <= 1e-12
                        and g['G4_iterations_reasons_equal']),
                timing=bool(g['timing_drift_pass'] and g['timing_neighbour_pass'] and g['deterministic_pass']),
                timing_neighbour_untested=g.get('timing_neighbour_untested', []))


def fastest(foms, key, bar):
    ok = [(v['ms'], k) for k, v in foms.items() if v[key] is not None and v[key] <= bar]
    return min(ok)[1] if ok else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--panels', nargs='+', required=True)
    ap.add_argument('--refined', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--frozen', default=None)
    a = ap.parse_args()
    ref = json.loads(Path(a.refined).read_text())
    frozen = json.loads(Path(a.frozen).read_text()) if a.frozen else None
    out = dict(rule='DESIGN.md sections 6-7, R1, R2 (select_q.py)', refined=a.refined, frozen=frozen is None,
               cohort_seed=None,
               refined_sha256=hashlib.sha256(Path(a.refined).read_bytes()).hexdigest(), meshes={})
    for pth in a.panels:
        rep = json.loads(Path(pth).read_text())
        n = rep['mesh']
        out['cohort_seed'] = rep['cohort']['seed']
        rf = ref['panels'][str(n)]
        tim = rep['timing']['summary']
        g = gates_ok(rep['gates'])
        cont_chk = {Rp: rep['rho'][Rp]['continuum_check']['vs_target']['worst'] for Rp in rep['rho']}
        foms = {k: dict(ms=tim[k]['median_ms'], same=v['worst_same_grid'], refined=rf['fom'][k]['worst'])
                for k, v in rep['fom'].items()}
        mres = dict(job_id=rep['job_id'], gpu=rep['gpu'], commit=rep['commit'], gates=g, gate_values=rep['gates'],
                    continuum_check=cont_chk, continuum_check_pass={k: v <= 1e-5 for k, v in cont_chk.items()},
                    fom=foms, refined_same_grid_reference_worst=max(rf['same_grid_reference']), R={})
        for Rp in sorted({v['spec']['Rp'] for v in rep['arms'].values()}, reverse=True):
            rho = rep['rho'][str(Rp)]['rules']
            ctrls = rep['config']['controls']
            rows = {}
            for nm, arm in rep['arms'].items():
                s = arm['spec']
                if s['Rp'] != Rp:
                    continue
                rule = s['rule']
                steps = arm['cases_run'] * STEPS
                dist = arm.get('distance', {})
                rows[nm] = dict(
                    rule=rule, family=s['family'], control=rule in ctrls, m=rep['sizes'].get(nm, {}).get('m'),
                    finite=arm['all_finite'], reason3=arm['reason_counts']['3'],
                    nonstationary_frac=(arm['reason_counts']['0'] + arm['reason_counts']['2']) / steps,
                    lm_its_median=arm['lm_iterations_per_query_median'], lm_its_max=arm['lm_iterations_per_query_max'],
                    reasons=arm['reason_counts'], cases=arm['cases_run'],
                    rho_cont=rho.get(rule if s['family'] == 'offmesh' else ('tensor' if s['family'] == 'tensor'
                                                                            else 'dense_upwind'), {}).get('cont'),
                    rho_mesh=rho.get(rule if s['family'] == 'offmesh' else ('tensor' if s['family'] == 'tensor'
                                                                            else 'dense_upwind'), {}).get('mesh'),
                    same_worst=arm['worst_same_grid'], same_median=arm['median_same_grid'],
                    refined_worst=rf['arms'][nm]['worst'], refined_median=rf['arms'][nm]['median'],
                    dist_tensor=dist.get('tensor', {}).get('worst'), dist_dense=dist.get('dense', {}).get('worst'),
                    dist_conv=0.0 if rule == rep['config']['converged_rule'] else dist.get(
                        rep['config']['converged_rule'], {}).get('worst'),
                    ms=tim.get(nm, {}).get('median_ms'), jac_ms=rep['microbench'].get(nm, {}).get('jacobian_ms_median'),
                    bytes=rep['sizes'].get(nm, {}).get('bytes'), flops_jacobian=rep['sizes'].get(nm, {}).get('flops_jacobian'))
                r = rows[nm]
                r['eligible'] = bool(r['family'] == 'offmesh' and not r['control'] and r['finite'] and r['reason3'] == 0
                                     and r['nonstationary_frac'] <= 0.01 and r['rho_cont']['worst'] <= BAR_RHO)
                if r['ms'] is not None:
                    fs, fr = fastest(foms, 'same', r['same_worst']), fastest(foms, 'refined', r['refined_worst'])
                    r['fom_same_rule'], r['fom_refined_rule'] = fs, fr
                    r['speedup_same_rule'] = foms[fs]['ms'] / r['ms'] if fs else None
                    r['speedup_refined_rule'] = foms[fr]['ms'] / r['ms'] if fr else None
            conv = f"{rep['config']['converged_rule']}_R{Rp}"
            g32 = rows.get(f'gl32_R{Rp}', {})
            conv_valid = bool(rows[conv]['eligible'] and g32.get('eligible') and g32.get('dist_conv') is not None
                              and g32['dist_conv'] <= BAR_CONV)
            hard_gates = all(v for k, v in g.items() if k not in ('timing', 'timing_neighbour_untested')) and \
                mres['continuum_check_pass'][str(Rp)]
            by = 'median_ms' if g['timing'] else 'm (timing gate failed, R2-1)'
            if frozen:
                sel = frozen['meshes'][str(n)]['R'][str(Rp)]['selected']
            else:
                key = (lambda nm, r: (r['ms'], nm)) if g['timing'] else (lambda nm, r: (r['m'], nm))
                cands = [key(nm, r) for nm, r in rows.items() if r['eligible'] and r['dist_conv'] is not None
                         and r['dist_conv'] <= BAR_CONV]
                sel = min(cands)[1] if (cands and conv_valid and hard_gates) else None
            tens = rows[f'tensor_R{Rp}']
            lat = {mm: rows.get(f'lat{mm}_R{Rp}') for mm in LADDER}
            reach_t = [mm for mm in LADDER if lat[mm] and lat[mm]['refined_worst'] <= tens['refined_worst']]
            reach_c = [mm for mm in LADDER if conv_valid and lat[mm] and lat[mm]['dist_conv'] is not None
                       and lat[mm]['dist_conv'] <= BAR_CONV]
            rc = lambda k: rows[f'{k}_R{Rp}']['rho_cont']['worst']
            res_floor = 10 * cont_chk[str(Rp)]
            unresolved = {pair: bool(rc(a_) < res_floor and rc(b_) < res_floor)
                          for pair, (a_, b_) in dict(m4096=('lat4096', 'gl16'), m32768=('lat32768', 'gl32')).items()}
            controls = {c: dict(rho_cont=rows[f'{c}_R{Rp}']['rho_cont']['worst'],
                                dist_conv=rows[f'{c}_R{Rp}']['dist_conv'],
                                fired_rho=bool(rows[f'{c}_R{Rp}']['rho_cont']['worst'] > BAR_RHO),
                                fired_dist=(bool(rows[f'{c}_R{Rp}']['dist_conv'] > BAR_CONV) if conv_valid
                                            and rows[f'{c}_R{Rp}']['dist_conv'] is not None else 'withheld'),
                                fired=bool(rows[f'{c}_R{Rp}']['rho_cont']['worst'] > BAR_RHO and conv_valid and
                                           rows[f'{c}_R{Rp}']['dist_conv'] is not None and
                                           rows[f'{c}_R{Rp}']['dist_conv'] > BAR_CONV))
                        for c in ctrls}
            controls['tensor_mesh_rho_le_1e-2'] = dict(value=tens['rho_mesh']['worst'],
                                                       fired=bool(tens['rho_mesh']['worst'] <= 1e-2))
            dn = rep['rho'][str(Rp)]['rules']['dense_upwind']['cont']['worst']
            controls['dense_cont_rho_gt_bar'] = dict(value=dn, fired=bool(dn > BAR_RHO))
            mres['R'][str(Rp)] = dict(
                selected=sel, selected_row=rows[sel] if sel else None, tensor_row=tens, rho_resolution=res_floor,
                lattice_vs_gauss_unresolved=unresolved, converged=conv, converged_valid=conv_valid,
                hard_gates=hard_gates, selected_by=by,
                lattice_beats_gauss_4096=('unresolved' if unresolved['m4096'] else bool(rc('lat4096') < rc('gl16'))),
                lattice_beats_gauss_32768=('unresolved' if unresolved['m32768'] else bool(rc('lat32768') < rc('gl32'))),
                smallest_m_reaching_tensor=reach_t[0] if reach_t else None,
                smallest_m_reaching_converged=reach_c[0] if reach_c else None,
                controls=controls, rows=rows)
        out['meshes'][str(n)] = mres
        for Rp, v in mres['R'].items():
            sr = v['selected_row'] or dict(ms=float('nan'), refined_worst=float('nan'))
            print(f"n={n} R'={Rp}: selected {v['selected']} ({sr['ms']:.1f} ms, refined "
                  f"{sr['refined_worst']:.3%}) tensor ({v['tensor_row']['ms']:.1f} ms, refined "
                  f"{v['tensor_row']['refined_worst']:.3%}); lat>gauss {v['lattice_beats_gauss_4096']}/"
                  f"{v['lattice_beats_gauss_32768']}; m->tensor {v['smallest_m_reaching_tensor']} m->conv "
                  f"{v['smallest_m_reaching_converged']}; controls " +
                  ', '.join(f"{k}:{c['fired']}" for k, c in v['controls'].items()))
        print(f'n={n} gates {g}')
    out['lattice_beats_gauss'] = all(v['lattice_beats_gauss_4096'] is True and v['lattice_beats_gauss_32768'] is True
                                     for m in out['meshes'].values() for v in m['R'].values())
    Path(a.out).write_text(json.dumps(out, indent=1) + '\n')


if __name__ == '__main__':
    main()
