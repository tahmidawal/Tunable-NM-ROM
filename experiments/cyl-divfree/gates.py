"""Pre-registered Stage-2 gates (DESIGN.md section 5), computed only from saved JSON.

Usage: gates.py --evaluation runs/<eval>/EVALUATION.json --out runs/<eval>/GATES.json
"""
from pathlib import Path
import argparse, hashlib, json

DEV = ('16.875', '23.125')
TARGETS = ('velocity', 'pressure', 'drag', 'transverse', 'divergence', 'mass', 'boundary_trace')


def main():
    a = argparse.ArgumentParser(); a.add_argument('--evaluation', type=Path, required=True); a.add_argument('--out', type=Path, required=True)
    a = a.parse_args()
    base_path = Path(__file__).with_name('BASELINES.json'); base = json.loads(base_path.read_text())
    ev = json.loads(a.evaluation.read_text())
    res = {'evaluation_sha256': hashlib.sha256(a.evaluation.read_bytes()).hexdigest(),
           'baselines_sha256': hashlib.sha256(base_path.read_bytes()).hexdigest(), 'cases': {}}
    # validity precondition: harness reproduces recorded run 09
    rep = {str(c['Re']): c for c in ev['models']['run09_replay']['cases']}
    res['harness_equivalence'] = {re: {'recorded': base['run09'][re]['velocity_relative'], 'replayed': rep[re]['scores']['velocity_relative'],
                                       'relative_gap': abs(rep[re]['scores']['velocity_relative']/base['run09'][re]['velocity_relative']-1)} for re in DEV}
    res['harness_valid'] = all(v['relative_gap'] <= 1e-8 for v in res['harness_equivalence'].values())
    df = {str(c['Re']): c for c in ev['models']['divfree']['cases']}
    ok_all = True
    for re in DEV:
        c = df[re]; s = c.get('scores')
        if s is None:
            res['cases'][re] = {'passed': False, 'reason': 'no scores', 'row': {k: c.get(k) for k in ('failure', 'diagnostic_failure')}}; ok_all = False; continue
        div_ref = min(base['POD4']['exact_tensor_weak_LS'][re]['divergence'], base['POD4']['mixed_Galerkin'][re]['divergence'])
        g1 = bool(s['accuracy_targets_passed'] and all(s['target_checks'][t] for t in TARGETS)
                  and c['solver_status'] in ('stationary', 'root_converged') and ev['models']['divfree']['stability']['passed'])
        g2 = s['divergence'] <= 10*div_ref
        g3 = s['velocity_relative'] <= base['run09'][re]['velocity_relative']
        res['cases'][re] = {'G1_all_seven_targets': g1, 'G2_divergence': g2, 'G3_velocity_vs_run09': g3,
                            'divergence': s['divergence'], 'divergence_limit_10x_POD4': 10*div_ref,
                            'velocity_relative': s['velocity_relative'], 'velocity_limit_run09': base['run09'][re]['velocity_relative'],
                            'velocity_vs_POD4_weak_LS_ratio': s['velocity_relative']/base['POD4']['exact_tensor_weak_LS'][re]['velocity_relative'],
                            'velocity_vs_POD4_mixed_Galerkin_ratio': s['velocity_relative']/base['POD4']['mixed_Galerkin'][re]['velocity_relative'],
                            'target_checks': s['target_checks'], 'solver_status': c['solver_status'],
                            'passed': bool(g1 and g2 and g3)}
        ok_all &= res['cases'][re]['passed']
    res['success'] = bool(ok_all and res['harness_valid'])
    a.out.write_text(json.dumps(res, indent=1)+'\n'); print(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
