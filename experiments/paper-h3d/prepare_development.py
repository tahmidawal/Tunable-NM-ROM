"""Write the declared two-seed frozen development comparison configurations."""
import copy, json
from pathlib import Path


def main():
    lane = Path(__file__).resolve().parent
    base = json.loads((lane/'coverage05.json').read_text())
    for setting in base['seeds'][:2]:
        label = setting['name']
        cfg = copy.deepcopy(base); cfg.pop('seeds'); cfg.update(setting['overrides'])
        cfg['seed_role'] = setting['role']
        cfg.update(driver='run.py', frozen_source_attempt='coverage05', frozen_source_seed=label,
            frozen_input_directory=f'inputs/coverage05_{label}',
            frozen_operators=[operator['name'] for operator in cfg['operators']], operators=[],
            representation_oracles=True, fit_quadrature=False, retain_solver_states=True,
            evaluation_cohort='development')
        if label == 'seedA':
            cfg['companion_config_file'] = 'code/developmentB.json'
        else:
            cfg.update(confirmation_role='independent_seed_accuracy', include_linear_controls=False,
                       include_half_step_control=False, evaluation_intervals=[32], repetitions=1)
        path = lane/('developmentA.json' if label == 'seedA' else 'developmentB.json')
        assert not path.exists(), path
        path.write_text(json.dumps(cfg, indent=2)+'\n')


if __name__ == '__main__':main()
