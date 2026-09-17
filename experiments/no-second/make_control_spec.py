"""Write the job-3 control spec (DESIGN §3.4 / §A1) from the audited screen results.

    python make_control_spec.py unet01 [tsol01]

Reads each attempt's `audit.json` (never a raw result file), takes the validation-selected
capacity by the pre-registered rule (argmin of the batched validation mean case-max over
every complete arm, refine included; if that is `refine`, the capacity it refined) and
emits `specs/ctrl01.json` with: the selected U-Net capacity in float64 (precision control),
the same capacity at seed 20260915 in float32 (seed control), and, if a Transolver attempt
is given, its selected capacity in float64. Same 3000 s budget each, no refinement.
"""
import json
from pathlib import Path
import sys

LANE = Path(__file__).resolve().parent
PER_ARM = 3000.0


def selected(attempt):
    audit = json.loads((LANE / 'runs' / attempt / 'audit.json').read_text())
    assert audit['passed']
    prefix = audit['spec']['prefix']
    arms = {k: v for k, v in audit['arms'].items() if v.get('complete')}
    best = min(arms, key=lambda k: arms[k]['batched_selection_score'])
    name = best[len(prefix) + 1:]
    if name == 'refine':
        refined = json.loads((LANE / 'runs' / attempt / 'archive' / attempt / 'out/refine-config.json').read_text())
        name = refined['refines_capacity']
    config = next(a['config'] for a in audit['spec']['arms'] if a['name'] == name)
    return audit, prefix, name, config, best, arms[best]['batched_selection_score']


def main(attempts):
    arms, notes = [], {}
    audit, prefix, name, config, best, score = selected(attempts[0])
    assert audit['spec']['family'] == 'unet'
    notes['unet'] = dict(selected_arm=best, selected_capacity=name, validation_mean_case_max=score, source_job=audit['job_id'])
    arms.append(dict(name=f'{name}-f64', config=config, override=dict(dtype='float64'), seconds=PER_ARM,
                     role='precision control: selected U-Net capacity, network in float64, seed 20260914'))
    arms.append(dict(name=f'{name}-seed2', config=config, override=dict(seed=20260915), seconds=PER_ARM,
                     role='seed control: selected U-Net capacity, float32, seed 20260915'))
    if len(attempts) > 1:
        t_audit, t_prefix, t_name, t_config, t_best, t_score = selected(attempts[1])
        assert t_audit['spec']['family'] == 'transolver'
        notes['transolver'] = dict(selected_arm=t_best, selected_capacity=t_name, validation_mean_case_max=t_score, source_job=t_audit['job_id'])
        arms.append(dict(name=f'tsol-{t_name}-f64', config=t_config, override=dict(dtype='float64'), seconds=PER_ARM,
                         role='precision control: selected Transolver capacity, network in float64, seed 20260914'))
    n = len(arms)
    # `family` only selects the in-job training smoke; each arm's config carries its own family.
    spec = dict(family='unet', prefix='ctrl', pde='burgers', job_name='ctol_nos_ctrl01',
                time='03:20:00' if n == 2 else '04:20:00', arms=arms, refine_learning_rate=None,
                per_arm_seconds=PER_ARM, global_seconds=n * PER_ARM + 2400.0, reserve_seconds=900.0,
                selection=notes, design='pre-registered controls (DESIGN §3.4/§A1): same protocol and budget as the screen; '
                                        'no refinement; matched-cohort scoring and same-job timing as in the screen')
    (LANE / 'specs/ctrl01.json').write_text(json.dumps(spec, indent=2) + '\n')
    print(json.dumps(spec, indent=2))


if __name__ == '__main__':
    main(sys.argv[1:])
