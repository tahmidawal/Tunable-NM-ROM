"""Regenerate checkpoint-budget, precision and shared-family training audit."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np


def leaves(x):
    if isinstance(x, dict):
        for v in x.values():
            yield from leaves(v)
    elif isinstance(x, (tuple, list)):
        for v in x:
            yield from leaves(v)
    else:
        yield np.asarray(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=Path, default=Path(__file__).parent/'runs')
    args = ap.parse_args()
    rows, parameters = [], []
    for label in ('train_reflective01', 'train_absorbing01'):
        d = args.runs/label/'cluster/out/training'
        result = json.loads((d/'training_result.json').read_text())
        cfg = result['config']
        m = json.loads((d/'data/training_data.json').read_text())
        parameters.append(np.asarray(m['parameters']))
        nt = int(round(cfg['end_time']/cfg['observation_dt']))+1
        assert m['complete'] and m['shape'][0] == cfg['train_count']*nt and m['evaluation_generated'] is False
        for architecture in ('cp', 'modcp', 'film'):
            path = d/'checkpoints'/f'{architecture}.pkl'
            with path.open('rb') as f:
                q = pickle.load(f)
            assert q['extra']['complete'] and q['extra']['total_training_steps'] == cfg['pretrain_steps']+cfg['comparison_steps']
            assert q['Z'].shape == (cfg['train_count']*nt, cfg['latent']) and np.isfinite(q['Z']).all()
            arrays = list(leaves(q['params']))
            assert all(a.dtype == np.float64 and np.isfinite(a).all() for a in arrays)
            rows.append({'attempt': label, 'architecture': architecture,
                         'parameter_count': sum(a.size for a in arrays), 'codes_shape': list(q['Z'].shape),
                         'checkpoint_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'fixed_training_scales': q['extra']['fixed_component_scales']})
    assert np.array_equal(*parameters)
    result = {'passed': True, 'matching_physical_training_rows_across_boundaries': True,
              'evaluation_opened': False, 'models': rows}
    (args.runs/'training-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print('WAVE_TRAINING_AUDIT_PASSED', flush=True)


if __name__ == '__main__':
    main()
