"""Bounded end-to-end development smoke; scientific evaluation seed stays closed."""
import argparse
import json
from pathlib import Path
import sys
import compare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', type=Path, required=True)
    ap.add_argument('--boundary', choices=('dirichlet', 'absorbing'), required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    cfg = json.loads((Path(__file__).parent/'config.json').read_text())
    cfg.update(name='development_only_wave_smoke', smoke_only=True, meshes=[16],
               validation_seed=910611, evaluation_seed=910612, validation_count=2, evaluation_count=2,
               end_time=.01, observation_dt=.005, time_steps=[.005], gn_caps=[2], gn_tolerances=[1e-3],
               eq_multipliers=[4], eq_fit_codes=4, eq_candidate_count=4096, initial_fit_cap=3,
               initial_fit_starts=2, cg_tolerances=[1e-6], rk4_cfls=[.3], warmups=1, repetitions=2)
    path = args.out/'config.json'
    path.write_text(json.dumps(cfg, indent=2)+'\n')
    sys.argv = ['compare.py', '--config', str(path), '--inputs', str(args.inputs),
                '--boundary', args.boundary, '--out', str(args.out/'comparison')]
    compare.main()
    result = json.loads((args.out/'comparison/handoff.json').read_text())
    if result['status'] != 'complete' or len(result['checkpoint_sha256']) != 3:
        raise RuntimeError('End-to-end smoke did not finish all three trained architecture arms')
    print('WAVE_END_TO_END_SMOKE_PASSED', args.boundary, flush=True)


if __name__ == '__main__':
    main()
