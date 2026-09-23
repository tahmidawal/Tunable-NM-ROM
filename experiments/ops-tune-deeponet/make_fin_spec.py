"""Write `specs/fin01.json` from the audited sweep, so no override is ever hand-copied.

S4 of DESIGN section 5.1: the selected learned-trunk recipe and the selected POD recipe, each
retrained at the final budget at the top rung, plus the best-worst-case arm of the sweep if the
budget allows. Every arm's override is read out of `runs/tun01/audit.json`, never typed.

    python make_fin_spec.py            # after tun01 is collected and audited
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LANE = Path(__file__).resolve().parent
# The knobs a POD arm can inherit from the sweep. The trunk knobs cannot: a POD trunk has no
# coordinate MLP, and its rank is its number of modes, fixed at 64 by DESIGN A4.
POD_INHERITABLE = ('schedule', 'min_learning_rate', 'warmup_steps', 'learning_rate',
                   'pool_bins', 'batch_size', 'weight_decay', 'output_scale_mode')
POD = dict(trunk='pod', rank=64, mesh_intervals=256)


def main(args):
    audit = json.loads((LANE / 'runs/tun01/audit.json').read_text())
    spec = json.loads((LANE / 'specs/tun01.json').read_text())
    selection = audit['decisions']['selection']
    arms = {name.split('-', 1)[1]: value for name, value in audit['arms'].items() if value.get('complete')}
    decision = json.loads((LANE / 'runs/tun01/archive/tun01/out/schedule-decision.json').read_text())['applied'] \
        if (LANE / 'runs/tun01/archive/tun01/out/schedule-decision.json').exists() else {}

    def override_of(name):
        """Exactly the delta this arm trained with, relative to the base config."""
        base = json.loads((LANE / 'configs/deeponet/base.json').read_text())
        return {k: v for k, v in arms[name]['config'].items()
                if k not in ('arm', 'knob', 'derived_from', 'applied_schedule_decision')
                and base.get(k) != v}

    chosen = selection['selected']
    final = [dict(name='final-learned', mount='ext', seconds=args.seconds,
                  override=override_of(chosen), refines=chosen)]
    pod_arms = [n for n in arms if arms[n].get('trunk') == 'pod']
    pod_override = dict(POD)
    pod_override.update({k: v for k, v in override_of(chosen).items() if k in POD_INHERITABLE})
    final.append(dict(name='final-pod', mount='ext', seconds=args.seconds, override=pod_override,
                      refines='the tuned recipe restricted to the knobs a POD trunk can take'))
    if selection['best_worst_case_arm'] != chosen:
        final.append(dict(name='final-bestworst', mount='ext', seconds=args.seconds,
                          override=override_of(selection['best_worst_case_arm']),
                          refines=selection['best_worst_case_arm']))
    out = {k: v for k, v in spec.items()
           if k in ('family', 'prefix', 'pde', 'base_config', 'validation_index', 'reference_index',
                    'cohort_train_mount', 'mounts', 'worker', 'mem')}
    out.update(job_name='opstune_don_fin01', time=args.time, arms=final,
               global_seconds=args.seconds * len(final) + 3600, reserve_seconds=900,
               design=('DESIGN 5.1 S4: the validation-selected learned-trunk recipe and the POD '
                       'recipe, each retrained at the top rung for %d s -- three times what every '
                       'published operator arm was given, which the report states -- plus the '
                       'best-worst-case sweep arm when the mean-selected one is not it. Every '
                       'override is generated from runs/tun01/audit.json by make_fin_spec.py.'
                       % args.seconds),
               generated_from=dict(audit='runs/tun01/audit.json', selected=chosen,
                                   best_worst_case=selection['best_worst_case_arm'],
                                   schedule_decision=decision, pod_arms_in_sweep=pod_arms))
    (LANE / 'specs/fin01.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(dict(arms=[(a['name'], a['override']) for a in final],
                          global_seconds=out['global_seconds']), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=9000.)
    parser.add_argument('--time', default='09:00:00')
    main(parser.parse_args())
