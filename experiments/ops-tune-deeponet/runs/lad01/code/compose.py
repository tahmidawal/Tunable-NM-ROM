"""The composition rule of DESIGN.md section 5.3, as code, so it cannot drift.

`tuned` takes every knob whose one-factor sweep arm improved the selection metric by at least
`threshold` relative to the sweep's own reference arm, each at the value that arm used; where
two arms vary the same knob, only the better one is taken. If no arm clears the threshold the
composed configuration is the reference configuration itself and T2 fails by construction.

The arm -> knob -> override mapping is declared in the job spec before the job runs; nothing
here chooses which knobs exist or what values they take.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def compose(sweep, scores, reference, threshold):
    """sweep: [{name, knob, override}]; scores: {arm: selection score}; reference: arm name."""
    if reference not in scores:
        raise RuntimeError(f'the sweep reference arm {reference} has no score')
    base_score = scores[reference]
    winners, considered = {}, []
    for arm in sweep:
        name = arm['name']
        if name == reference or name not in scores:
            considered.append(dict(arm=name, knob=arm.get('knob'), score=scores.get(name),
                                   status='missing' if name not in scores else 'reference'))
            continue
        relative = (base_score - scores[name]) / base_score
        accepted = relative >= threshold
        considered.append(dict(arm=name, knob=arm['knob'], score=scores[name],
                               relative_improvement=relative, accepted=accepted))
        if accepted:
            previous = winners.get(arm['knob'])
            if previous is None or scores[name] < scores[previous['name']]:
                winners[arm['knob']] = arm
    override = {}
    for knob in sorted(winners):
        override.update(winners[knob]['override'])
    return dict(override=override, winners={k: v['name'] for k, v in winners.items()},
                reference=reference, reference_score=base_score, threshold=threshold,
                arms=considered,
                rule='DESIGN section 5.3: every knob whose one-factor arm improved the selection '
                     'metric by at least the threshold, best arm per knob')


def scores_from_runs(runs, prefix):
    out = {}
    for result in sorted(Path(runs).glob(f'{prefix}-*/result.json')):
        name = result.parent.name[len(prefix) + 1:]
        out[name] = json.loads(result.read_text())['validation']['mean_case_max']
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--runs', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    sweep = [a for a in spec['arms'] if 'knob' in a]
    rule = spec['compose']
    decision = compose(sweep, scores_from_runs(args.runs, spec['prefix']),
                       rule['reference'], rule['threshold'])
    Path(args.out).write_text(json.dumps(decision, indent=2) + '\n')
    print(json.dumps({k: v for k, v in decision.items() if k != 'arms'}, indent=2), flush=True)
