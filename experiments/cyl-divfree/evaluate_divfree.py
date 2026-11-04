"""Development-only actual PDE solves with the unchanged run-09 code path, for three models.

For each model (run-09 ResidualHeadBank replay, the new stream-function bank, POD-4) and each
development Re (16.875, 23.125) this calls the unchanged accuracy_v2.diagnostic.diagnose
(model.query weak-LS solve from the zero state, pilot2d.score_state full-rule metrics, Jacobian
check, direct/tensor gap), then pilot2d.projection_diagnostic (best-fit floor of the full
8-column velocity trial space, same weights as the scored metric). Adds, as diagnostics only:
the neural-only 4-column velocity floor, a cylinder-ring error breakdown, and solved fields on
the 18000-point plot subset used by generated/neural-guided-diagnostic09.
Validation cases are never opened.
"""
from pathlib import Path
import argparse, gc, hashlib, json, os, sys, time
here = Path(__file__).resolve().parent
sys.path[:0] = [str(here), str(here.parent/'ns-obstacle-rom')]
import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import pilot2d as pilot
from spaces import projection_floor
from accuracy_v2.diagnostic import diagnose
from accuracy_v2.residual_head_bank import ResidualHeadBank
from divfree_bank import DivFreeBank
from geometry import CHANNEL


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    a = argparse.ArgumentParser()
    for k in ('prepared', 'release', 'probes', 'development', 'run09_bank', 'divfree_bank', 'out'):
        a.add_argument('--'+k.replace('_', '-'), type=Path, required=True)
    a.add_argument('--skip', nargs='*', default=[])
    args = a.parse_args()
    out = args.out; out.mkdir(parents=True, exist_ok=False)
    runtime = {'jax': jax.__version__, 'backend': jax.default_backend(), 'devices': [d.device_kind for d in jax.devices()],
               'x64': bool(jax.config.x64_enabled), 'matmul_precision': str(jax.config.jax_default_matmul_precision),
               'job_id': os.environ.get('SLURM_JOB_ID'),
               'inputs_sha256': {k: sha(getattr(args, k)) for k in ('release', 'probes', 'development', 'run09_bank', 'divfree_bank')},
               'source_sha256': {f: sha(here/f) for f in ('divfree_bank.py', 'evaluate_divfree.py')},
               'pilot_source_digest': pilot.source_digest()}
    (out/'RUNTIME.json').write_text(json.dumps(runtime, indent=1)+'\n')
    arrays, metadata, _ = pilot.load_preparation(args.prepared, args.release)
    probes = pilot._read_arrays(args.probes, 2**30)
    rule = pilot.array_digest({k: arrays['volume'][k] for k in ('points_dimensionless', 'weights_dimensionless')})
    population = pilot.load_reference_population(args.development, sha(args.development), 'development', rule)
    if [r['Re'] for r in population] != [16.875, 23.125]: raise ValueError('development cases only')
    N = arrays['volume'].point_count
    idx = np.floor(np.arange(18000)*(N-1)/(18000-1)).astype(np.int64)
    pts = np.asarray(arrays['volume']['points_physical'])
    ring = (np.linalg.norm(pts-np.asarray(CHANNEL.center), axis=1)-CHANNEL.obstacle_radius) < 0.01
    result = {'schema': 'cyl-divfree-development-evaluation-v1', 'validation_seen': False,
              'plot_indices_sha256': hashlib.sha256(idx.tobytes()).hexdigest(), 'ring_definition': 'r - R < 0.01 m (0.1 D)',
              'ring_volume_fraction': None, 'models': {}}
    plot = {'indices': idx, 'points_physical': pts[idx], 'Re': np.array([16.875, 23.125])}
    for name, kind, loader, path in (('run09_replay', 'bank', ResidualHeadBank.load, args.run09_bank),
                                     ('divfree', 'bank', DivFreeBank.load, args.divfree_bank),
                                     ('POD_r4_weak_LS', 'POD', None, None)):
        if name in args.skip: continue
        tic = time.monotonic()
        bank = loader(path) if loader else None
        model = pilot.build_model(kind, 4, arrays, metadata, probes, bank)
        w = model.rule.weights
        if result['ring_volume_fraction'] is None:
            result['ring_volume_fraction'] = float(w[ring].sum()/w.sum())
        rec = {'build_seconds': time.monotonic()-tic, 'stability': model.stability, 'cases': []}
        sub = out/name; sub.mkdir()
        vel_plot, pre_plot = [], []
        for reference in population:
            row = diagnose(model, reference['Re'], reference, arrays['boundary'], sub, lambda: None)
            row['representation_diagnostic'] = pilot.projection_diagnostic(model, reference, arrays['boundary'], w)
            f = model.fields['volume']; uc = np.asarray(reference['velocity'])-f['lift']
            _, neural = projection_floor(f['velocity'][..., :4], uc[None], w)
            row['neural_only_velocity_floor'] = {'velocity_relative': float(np.sqrt(neural['squared_errors'][0])/row['scores']['velocity_denominator']), **neural}
            state = np.asarray(row_state(sub, reference['Re']))
            fields = model.reconstruct(state, reference['Re'], gradients=True)
            err = np.sum((fields['velocity']-reference['velocity'])**2, axis=1)
            row['ring_breakdown'] = {'ring_velocity_error_squared_fraction': float(w[ring]@err[ring]/(w@err)),
                                     'velocity_relative_outside_ring': float(np.sqrt(w[~ring]@err[~ring])/row['scores']['velocity_denominator']),
                                     'velocity_relative_inside_ring': float(np.sqrt(w[ring]@err[ring])/row['scores']['velocity_denominator'])}
            vel_plot.append(fields['velocity'][idx]); pre_plot.append(fields['pressure'][idx])
            rec['cases'].append(row)
            print(name, reference['Re'], row.get('solver_status'), row.get('scores', {}).get('velocity_relative'),
                  row.get('scores', {}).get('divergence'), flush=True)
        plot[name+'_velocity'] = np.stack(vel_plot); plot[name+'_pressure'] = np.stack(pre_plot)
        rec['seconds'] = time.monotonic()-tic
        result['models'][name] = rec
        (out/'EVALUATION.json').write_text(json.dumps(pilot_clean(result), indent=1)+'\n')
        del model, bank; gc.collect()
    ref_v = [np.asarray(r['velocity'])[idx] for r in population]; ref_p = [np.asarray(r['pressure'])[idx] for r in population]
    plot['reference_velocity'] = np.stack(ref_v); plot['reference_pressure'] = np.stack(ref_p)
    np.savez_compressed(out/'plot-fields.npz', **plot)
    result['plot_fields_sha256'] = sha(out/'plot-fields.npz')
    (out/'EVALUATION.json').write_text(json.dumps(pilot_clean(result), indent=1)+'\n')


def row_state(sub, re):
    rec = json.loads((sub/f'pod_guided_neural_r4_{re}_raw_invocation.json').read_text())
    key = rec['record']['evidence']['state']['array']
    with np.load(sub/rec['array_file'], allow_pickle=False) as a:
        return a[key].copy()


def pilot_clean(o):
    if isinstance(o, dict): return {str(k): pilot_clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [pilot_clean(v) for v in o]
    if isinstance(o, np.ndarray): return o.tolist() if o.size <= 64 else {'array_shape': list(o.shape)}
    if isinstance(o, np.generic): return o.item()
    return o


if __name__ == '__main__':
    main()
