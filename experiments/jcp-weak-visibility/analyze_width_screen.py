"""Independent NumPy W0 retention/numerics and original-R128 reproduction audit."""
from pathlib import Path
import hashlib
import json
import argparse
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    out, dest = args.output, args.destination
    dest.mkdir(parents=True, exist_ok=True)
    lane = Path(__file__).resolve().parent
    s = json.loads((out/'summary.json').read_text())
    env = json.loads((out/'environment.json').read_text())
    assert set(s['cases']) == set(map(str, range(6)))
    G = np.concatenate([np.load(p)['bank'] for p in sorted(out.glob('bank-columns-*.npz'))], axis=1)
    n = env['config']['mesh_intervals']
    R = np.load(out/'qr-factor.npz')['Rmass']
    gram = G.T @ G/n**2
    factor_error = float(np.linalg.norm(R.T @ R-gram)/np.linalg.norm(gram))
    ranks = {}
    for width in [128, 256, 512]:
        singular = np.linalg.svd(R[:width, :width], compute_uv=False)
        ranks[str(width)] = {str(t):int(np.sum(singular**2 > t*singular[0]**2)) for t in (1e-10, 1e-12, 1e-14)}
    records, maximum_metric_error, maximum_baseline_difference = [], 0., 0.
    per_prefix_factor = {str(w):float(np.linalg.norm(R[:w, :w].T @ R[:w, :w]-gram[:w, :w])/np.linalg.norm(gram[:w, :w])) for w in (128,256,512)}
    per_prefix_control = {k:v <= 1e-10 for k,v in per_prefix_factor.items()}
    for case_id in range(6):
        folder = out/f'case-{case_id:02d}'
        ref = np.load(folder/'reference-1.npz')['fields']
        norm0 = np.linalg.norm(ref[0])
        prev = None
        for width in [128, 256, 512]:
            z = np.load(folder/f'projection-r{width}.npz')
            basis = G[:, :width]
            field = z['coeff'] @ basis.T
            error = np.linalg.norm(field-ref, axis=1)/norm0
            metric_delta = float(np.max(np.abs(error-z['error'])))
            ortho = float(np.linalg.norm((field-ref) @ basis/n**2)/(np.linalg.norm(ref)*np.linalg.norm(basis)/n**2))
            maximum_metric_error = max(maximum_metric_error, metric_delta)
            nested = prev is None or bool(np.max(error-prev) <= 1e-10)
            row = dict(case=case_id, width=width, initial_error=float(error[0]), maximum_all_times_error=float(error.max()),
                       errors=error.tolist(), metric_disagreement=metric_delta, orthogonality_defect=ortho,
                       nested_projection_check=nested)
            if width == 128:
                accepted = lane/'runs'/('p03' if case_id < 2 else 'p04b')/'output'/f'case-{case_id:02d}'/'projection.npz'
                old = np.load(accepted)['error']
                row['accepted_R128_maximum_difference'] = float(np.max(np.abs(error-old)))
                maximum_baseline_difference = max(maximum_baseline_difference, row['accepted_R128_maximum_difference'])
            records.append(row)
            per_prefix_control[str(width)] = bool(per_prefix_control[str(width)] and metric_delta <= 1e-10 and ortho <= 1e-9 and nested)
            prev = error
    baseline_reproduced = maximum_baseline_difference <= 1e-10
    collection = json.loads((out.parent/'collection.json').read_text())
    environment_ok = bool(collection['checksum_verified'] and collection['gpu_log_valid'] and collection['no_environmental_warnings'] and collection['scheduler_state'] == 'COMPLETED')
    complete = bool(per_prefix_control['128'] and baseline_reproduced and environment_ok and s['all_references_eligible'] and all(r == 128 for r in ranks['128'].values()))
    admitted = [width for width in (256, 512) if complete and per_prefix_control[str(width)] and s['provisional_admission'][str(width)] and all(r == width for r in ranks[str(width)].values())]
    contract_keys = ['mesh_intervals','dt','horizon','cohort','case_ids','seed_first_draw','seed_first_count',
        'seed_second_draw','seed_second_count','local_parameter_sha256','reference_arithmetic','reference_tolerances',
        'reference_max_disagreement','solver_gtols','solver_max_iterations','tangent_relative_eigenvalue_threshold']
    audit = dict(W0_raw_audit_passed=complete, R128_reproduction_passed=baseline_reproduced,
        admitted_widths=admitted, source_job_id=env['job_id'], source_output=str(out.resolve()),
        summary_sha256=hashlib.sha256((out/'summary.json').read_bytes()).hexdigest(),
        source_checkpoint_sha256=env['checkpoint_sha256'], source_rotation_sha256=env['rotation_sha256'],
        source_core_sha256=env['source_sha256']['core.py'], fixed_contract={k:env['config'][k] for k in contract_keys},
        allowed_test_spaces={'128':[2048], '256':[2048,4096], '512':[2048,8192]},
        required_observer_start=8192, required_observer_rows=1024, required_initialization='mass-qr',
        qr_gram_relative_error=factor_error, qr_prefix_ranks=ranks, maximum_saved_metric_difference=maximum_metric_error,
        per_prefix_factor_error=per_prefix_factor, per_prefix_audit_passed=per_prefix_control,
        maximum_R128_reproduction_difference=maximum_baseline_difference, records=records,
        scope='post-pilot frozen-prefix screen; same-grid same-step reference; all six exposed development cases')
    (dest/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    (dest/'admission.json').write_text(json.dumps(audit, indent=2)+'\n')
    (lane/'inputs/width-admission.json').write_text(json.dumps(audit, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 2, figsize=(11, 10.5))
    for case_id, ax in enumerate(axs.flat):
        for width in (128, 256, 512):
            row = next(v for v in records if v['case'] == case_id and v['width'] == width)
            ax.plot(np.arange(len(row['errors']))*env['config']['dt'], 100*np.asarray(row['errors']), label=f'R={width}')
        ax.set(title=f'Development case {case_id}', xlabel='Time', ylabel='Projection error / initial norm (%)')
        ax.legend()
    fig.suptitle('MEASURED frozen-prefix projection screen — same-grid/same-step discrete references')
    fig.tight_layout()
    fig.savefig(dest/'projection-screen.png', dpi=180)
    fig.savefig(dest/'projection-screen.svg')
    print(json.dumps({k:audit[k] for k in ['W0_raw_audit_passed','R128_reproduction_passed','admitted_widths','qr_gram_relative_error','maximum_R128_reproduction_difference']}))


if __name__ == '__main__':
    main()
