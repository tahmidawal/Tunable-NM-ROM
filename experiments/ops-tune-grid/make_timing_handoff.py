"""Write `reports/timing-handoff.json` — everything `ops-timing-panel` needs to time this
lane's arms, and nothing this lane is allowed to say about speed itself.

    python make_timing_handoff.py

**No speed number from this lane is admissible.** The only construction from which a speed
number for these checkpoints can be formed is a same-allocation panel in which the operator,
the NM-ROM and the full-order solver are timed in one job on one GPU. This file hands that
lane the checkpoint paths and SHA256s, the two modules its harness needs, and — the part that
must travel with a timing row — **the accuracy values each checkpoint was measured at**, so a
speed row can never be published without its error beside it.

Follows the format of `experiments/ops-deeponet-b2d/reports/timing-handoff.json`.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

LANE = Path(__file__).resolve().parent
REPO = LANE.parents[1]
RELATIVE = 'worktrees/2026-09-22-ops-tune-grid/experiments/ops-tune-grid'
OUT = LANE / 'reports/timing-handoff.json'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def module(name, why):
    path = LANE / name
    return dict(path=f'{RELATIVE}/{name}', sha256=sha256(path), why=why)


def collect(attempt):
    """Every complete arm of one attempt, with its checkpoint and its measured accuracy."""
    audit_path = LANE / 'runs' / attempt / 'audit.json'
    if not audit_path.exists():
        return [], None
    audit = json.loads(audit_path.read_text())
    root = LANE / 'runs' / attempt / 'archive' / attempt / 'out'
    rows = []
    for name, arm in sorted((audit.get('arms') or {}).items()):
        if not arm.get('complete'):
            continue
        checkpoint = root / name / 'best.pt'
        recorded = arm['best_checkpoint_sha256']
        cohort = ((audit.get('cohort') or {}).get('models') or {}).get(name, {}).get('fixed_initial')
        rows.append(dict(
            name=name, family=arm['family'], attempt=attempt, job_id=audit['job_id'],
            gpu=audit['gpu'], source_commit=audit['source_commit'],
            path=f'{RELATIVE}/runs/{attempt}/archive/{attempt}/out/{name}/best.pt',
            sha256=recorded,
            sha256_reverified=sha256(checkpoint) if checkpoint.exists() else None,
            sha256_matches=(sha256(checkpoint) == recorded) if checkpoint.exists() else None,
            config=arm['config'], real_parameter_count=arm['real_parameter_count'],
            training_cases=arm.get('training_cases'),
            optimisation_steps=arm.get('optimisation_steps'),
            epochs_completed=arm['epochs_completed'], stop_reason=arm['stop_reason'],
            # The accuracy that MUST travel with any timing row for this checkpoint.
            accuracy=dict(
                cohort='validation-32 and diagnosis-8, fixed-initial relative error, scored '
                       'against the pinned 4096/1.5625e-4 reference restricted to 256',
                validation_32=arm['fixed_initial'], diagnosis_8=cohort)))
    return rows, audit


def main():
    checkpoints = []
    for attempt in ('grid01', 'ladder01'):
        rows, _ = collect(attempt)
        checkpoints.extend(rows)
    handoff = dict(
        purpose='Everything the ops-timing-panel harness needs to time this lane\'s arms in ONE '
                'allocation beside the NM-ROM, POD and the full-order solver — the only '
                'construction from which a speed number for these checkpoints may be formed. '
                'This lane states no speed number and forms no cross-job ratio.',
        add_to='experiments/ops-timing-panel/operators.json -> checkpoints',
        source_commit=subprocess.check_output(
            ['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip(),
        requires=dict(
            families_py=module('families.py',
                               'unchanged from the no-second/ops-deeponet lineage; listed so the '
                               'harness can confirm it is the same file it already carries'),
            model_py=module('model.py',
                            'adds norm=config.get("norm") to the neuralop FNO constructor; an '
                            'arm whose config carries "norm" cannot be rebuilt without it'),
            train_py=module('train.py',
                            'adds the cosine schedule and the configurable plateau patience; '
                            'needed only to re-train, not to load a checkpoint'),
            unpack='the checkpoint paths below live in a .gitignored extracted tree; if it is '
                   'gone, rebuild it by concatenating runs/<attempt>/archive-parts/part-* '
                   '(manifest.json carries each part\'s SHA256) and untarring, then re-verify '
                   'every best.pt against the sha256 recorded here'),
        must_travel_with_any_timing_row=
            'the accuracy block of each checkpoint below. Two of this lane\'s arms differ only '
            'in training-set size, so a speed row without its error and its training_cases is '
            'not interpretable.',
        cohort_caveat='These accuracy numbers are on validation-32 and diagnosis-8 against the '
                      'pinned fine reference. The panel\'s own operator percentages are on six '
                      'development cases against a same-job converged 256-grid solve. The two '
                      'are NOT subtractable: the same unet-refine checkpoint scores 1.7110 % on '
                      'diagnosis-8 and 4.5529 % on the panel.',
        checkpoints=checkpoints)
    OUT.write_text(json.dumps(handoff, indent=2) + '\n')
    print(json.dumps(dict(out=str(OUT), checkpoints=len(checkpoints),
                          arms=[c['name'] for c in checkpoints]), indent=2))


if __name__ == '__main__':
    main()
