"""Write `reports/timing-handoff.json`: what `ops-timing-panel` needs to time this lane's arms.

No speed number is admissible from this lane (DESIGN section 7). The only admissible
construction is that lane's same-allocation panel, so everything it needs is handed over in the
format it already consumes (`experiments/ops-deeponet-b2d/reports/timing-handoff.json` is the
pattern), with every checkpoint hash **re-verified here against the hash `train.py` recorded**.

    python reports/timing_handoff.py <attempt> [<attempt> ...]
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
LANE = HERE.parent
WORKTREE = 'worktrees/2026-09-22-ops-tune-deeponet'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main(attempts):
    audits = {a: json.loads((LANE / 'runs' / a / 'audit.json').read_text()) for a in attempts
              if (LANE / 'runs' / a / 'audit.json').exists()}
    checkpoints, unverified = [], []
    for attempt, audit in audits.items():
        for name, arm in audit['arms'].items():
            if not arm.get('complete'):
                continue
            path = LANE / 'runs' / attempt / 'archive' / attempt / 'out' / name / 'best.pt'
            entry = dict(name=name, family='deeponet', sha256=arm['best_checkpoint_sha256'],
                         path=f'{WORKTREE}/{path.relative_to(LANE.parents[1])}',
                         recorded_in=f'{attempt}/OUTPUTS.sha256', source_job=audit['job_id'],
                         training_cases=arm['training_cases'],
                         validation_mean=arm['fixed_initial']['mean'],
                         validation_worst=arm['fixed_initial']['maximum'],
                         parameters=arm['real_parameter_count'])
            if path.exists():
                entry['reverified'] = sha(path) == arm['best_checkpoint_sha256']
                if not entry['reverified']:
                    unverified.append(name)
            else:
                entry['reverified'] = None
                entry['note'] = 'extracted tree absent; rebuild from archive-parts before timing'
            checkpoints.append(entry)
    checkpoints.sort(key=lambda c: c['validation_mean'])
    out = dict(
        purpose="Everything the ops-timing-panel harness needs to time this lane's DeepONet arms "
                "in ONE allocation beside the NM-ROM, POD and the full-order solver — the only "
                "construction from which a speed number for these checkpoints is admissible. "
                "This lane makes no speed claim and ran no timing block.",
        add_to='experiments/ops-timing-panel/operators.json -> checkpoints',
        requires=dict(
            families_py=dict(path=f'{WORKTREE}/experiments/ops-tune-deeponet/families.py',
                             sha256=sha(LANE / 'families.py'),
                             why='DeepONet2d here gains a `trunk_layers` argument (default 3, the '
                                 'inherited depth); tuned arms may set trunk_layers, pool_bins or '
                                 'trunk_frequencies, which the parent lane\'s copy cannot build'),
            model_py=dict(path=f'{WORKTREE}/experiments/ops-tune-deeponet/model.py',
                          sha256=sha(LANE / 'model.py'),
                          why='byte-identical to the parent lane\'s; listed so the panel can pin it'),
            output_scale=dict(why='an arm with output_scale_mode "per_time" carries a (1, 5, 1, 1) '
                                  'output scale in its checkpoint normalisation instead of a scalar. '
                                  '`model.predict` broadcasts either without change, but a harness '
                                  'that assumes a scalar must not squeeze it'),
            unpack='checkpoint paths live in a .gitignored extracted tree; if it is gone, rebuild it '
                   'by concatenating runs/<attempt>/archive-parts/part-* (manifest.json has each '
                   'part\'s SHA256) and untarring, then re-verify every best.pt against the sha256 here'),
        caveat='Every arm here was given MORE than the U-Net, Transolver and FNO arms it would be '
               'tabulated beside — more training data, a tuned schedule, and for the final arms a '
               'longer budget. A speed row for one of these must carry its training-set size and '
               'wall budget beside it, and the accuracy asymmetry must be stated.',
        checkpoints=checkpoints, hash_mismatches=unverified,
        generated_from={a: str((LANE / 'runs' / a / 'audit.json').relative_to(LANE.parents[1]))
                        for a in audits})
    (HERE / 'timing-handoff.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(dict(checkpoints=len(checkpoints), mismatches=unverified,
                          attempts=sorted(audits)), indent=2))


if __name__ == '__main__':
    main(sys.argv[1:] or ['lad01', 'tun01', 'fin01'])
