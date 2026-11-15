"""Freeze all available validation-selected weights, including interrupted runs."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('training', type=Path)
    parser.add_argument('accounting', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    states = {}
    for line in args.accounting.read_text().splitlines():
        job, state, code, end = line.split('|')
        assert state == 'COMPLETED' or state.startswith('CANCELLED'), line
        states[job] = {'state': state, 'exit_code': code, 'end': end}

    entries = {}
    for directory in sorted(args.training.glob('*/runs/*')):
        if not directory.is_dir():
            continue
        meta = json.loads((directory / 'meta.json').read_text())
        history = json.loads((directory / 'history.json').read_text())
        assert not meta['test_cohort_seen'] and history
        job = str(meta['job_id'])
        result_path = directory / 'result.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        if not result:
            assert states[job]['state'].startswith('CANCELLED')
        train_commit = (directory.parents[1] / 'COMMIT.txt').read_text().strip()
        for weights in sorted(directory.glob('best*.pt')):
            snapshot = weights.name != 'best.pt'
            cut = int(weights.stem.split('_')[-1]) if snapshot else history[-1]['step']
            expected = min((h for h in history if h['step'] <= cut), key=lambda h: h['val_mean'])
            # Memory-map the original weights; the validation metadata are tiny.
            ck = torch.load(weights, map_location='cpu', weights_only=False, mmap=True)
            assert (ck['step'], ck['validation']['mean']) == (expected['step'], expected['val_mean'])
            for stat in ['worst', 'median']:
                assert ck['validation'][stat] == expected['val_' + stat]
            assert ck['config'] == meta['config'] and ck['normalization'] == meta['normalization']
            tag = f'{cut // 1000}k' if snapshot else 'final'
            key = directory.parents[1].name + '/' + directory.name
            if snapshot:
                key += '@' + tag
            entries[key] = dict(
                path=str(directory), checkpoint=weights.name, budget=tag,
                best_sha256=sha(weights), best_step=ck['step'], validation=ck['validation'],
                family='FNO' if ck['arm'].startswith('F') else 'U-Net',
                arm=ck['arm'], rung=ck['rung'], seed=ck['seed'], config=ck['config'],
                train_cases=meta['train_cases'], real_parameters=meta['real_parameters'],
                last_logged_step=history[-1]['step'], stop_reason=result.get('stop_reason', 'interrupted_by_cancellation'),
                converged=result.get('converged', False), training_job=job, accounting=states[job],
                training_commit=train_commit, validation_history_sha256=sha(directory/'history.json'),
                training_meta_sha256=sha(directory/'meta.json'))
            print(key, weights.name, flush=True)
        assert (directory/'best.pt').exists()
    selected = {}
    for family in ['FNO', 'U-Net']:
        candidates = {k: v['validation']['mean'] for k, v in entries.items()
                      if v['family'] == family and v['budget'] == 'final'}
        selected[family] = {'selected': min(candidates, key=candidates.get),
                            'rule': 'lowest validation mean across all available final best checkpoints',
                            'candidates': candidates}
    manifest = dict(test_cohort_seen=False, runs=entries, selection=selected)
    with args.destination.open('x') as stream:
        json.dump(manifest, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
