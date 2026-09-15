"""Bounded local check that the ladder's staged Burgers solver is the retained one.

The consolidated baseline keeps one opened development case at 64 intervals, with
its archived mesh operators and its saved output field. This runs that exact case
through both the retained selected solver and the ladder's staged reformulation,
and requires the ladder to reproduce the saved field within the same tolerance the
consolidated replay declares. No new mesh operators are fitted here, so the check
is fast and isolates the solver reformulation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'experiments/mr-burgers2d'))

import engines as e                      # noqa: E402
import accuracy_paths as ap              # noqa: E402
import ladder_common as lc               # noqa: E402
from ladder_burgers import make_rom_parts, reference_settings   # noqa: E402

TOLERANCE = 1e-8          # the consolidated replay's declared cross-device threshold
PARITY = 1e-12            # staged vs retained, same device, same operators


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    begin = time.perf_counter()
    environment = lc.preflight()

    manifest = json.loads((ROOT / 'consolidated/manifest.json').read_text())
    model = manifest['models']['burgers']
    raw = json.loads((ROOT / model['replay']['raw_result']).read_text())
    checkpoint = ROOT / model['artifacts'][0]
    params, _, _ = e.sc.load_pkl(checkpoint)
    archived = {k: jnp.asarray(v) for k, v in np.load(ROOT / model['replay']['operators']).items()}

    L, cfg = raw['config']['meshes'][0], raw['config']
    assert L == model['replay']['intervals'] == 64
    bank = e.sc.SeparableDecoder(params, raw['K'], raw['R']).feat_at(e.coords(L), chunk=8192)
    data = (bank, archived['A'], archived['lam'], archived['G5'], archived['Pq'],
            None, None, archived['candidate_Z'])
    rotated = e.sc.head(params, archived['candidate_Z']) @ archived['cold_R'].T
    cold = tuple(archived[k] for k in ('cold_xy', 'cold_w', 'cold_Q', 'cold_R')) + (
        rotated, jnp.sum(rotated * rotated, axis=1))
    setup = next(r for r in raw['mesh_setup'] if (r['intervals'], r['model']) == (L, 'frozen'))
    physical = raw['physical_cases'][0]
    supplied = jnp.asarray(e.initial(L, physical))
    nu = float(physical[4])

    retained = ap.make_rom(params, L, cfg['dt'], setup['trust_radius'], **cfg['strict'])
    staged, parts = make_rom_parts(params, L, cfg['dt'], setup['trust_radius'], **cfg['strict'])
    retained_value = lc.host(retained(supplied, nu, data, cold))
    staged_value = lc.host(staged(supplied, nu, data, cold))
    expected = np.load(ROOT / model['replay']['expected'])['fields']

    def relative(a, b):
        return float(np.linalg.norm(np.asarray(a) - b) / np.linalg.norm(b))

    # The three stages must compose into exactly the fused staged query.
    state = lc.host(parts['initialize'](supplied, data, cold))
    evolved = lc.host(parts['evolve'](jnp.asarray(state[0]), nu, float(state[3]), data))
    decoded = lc.host(parts['decode'](jnp.asarray(evolved[0]), data))

    checks = dict(
        scope=('one opened development case at 64 intervals with archived mesh operators; '
               'a solver-reformulation check, not a timing or accuracy measurement'),
        intervals=L, case=model['replay']['case'], dt=cfg['dt'], strict=cfg['strict'],
        trust_radius=setup['trust_radius'], checkpoint=str(model['artifacts'][0]),
        checkpoint_sha256=lc.sha_file(checkpoint), K=raw['K'], R=raw['R'],
        tolerance=TOLERANCE, parity_tolerance=PARITY,
        retained_vs_expected_relative=relative(retained_value[0], expected),
        staged_vs_expected_relative=relative(staged_value[0], expected),
        staged_vs_retained_relative=relative(staged_value[0], retained_value[0]),
        staged_vs_retained_bitwise_identical=bool(np.array_equal(staged_value[0], retained_value[0])),
        stages_compose_relative=relative(decoded[0], staged_value[0]),
        latent_relative=relative(evolved[0], staged_value[7]),
        max_step_stationarity=float(np.max(staged_value[8])),
        initial_fit_stationarity=float(staged_value[9]),
        stationary=bool(max(float(np.max(staged_value[8])), float(staged_value[9]))
                        <= cfg['strict']['gtol'] * (1 + 1e-7)),
        expected_sha256=lc.sha_array(expected), staged_sha256=lc.sha_array(staged_value[0]),
        reference_settings_for_4096=[list(x) for x in reference_settings(4096, 0.0003125)],
        restriction=lc.validate_restriction(expected, [64, 128, 256, 512, 1024]),
        seconds=time.perf_counter() - begin, **environment)

    # errors() must agree with a plain NumPy recomputation of the declared metric.
    reference = np.asarray(expected)
    metric = e.errors(staged_value[0], reference, L)
    difference = np.linalg.norm((np.asarray(staged_value[0]) - reference).reshape(len(reference), -1), axis=1)
    checks['metric_recomputation_difference'] = float(
        abs(metric['fixed_initial_max'] - np.max(difference / np.linalg.norm(reference[0]))))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps(checks), flush=True)
    assert checks['retained_vs_expected_relative'] <= TOLERANCE, checks['retained_vs_expected_relative']
    assert checks['staged_vs_expected_relative'] <= TOLERANCE, checks['staged_vs_expected_relative']
    assert checks['staged_vs_retained_relative'] <= PARITY, checks['staged_vs_retained_relative']
    assert checks['stages_compose_relative'] <= PARITY, checks['stages_compose_relative']
    assert checks['latent_relative'] <= PARITY, checks['latent_relative']
    assert checks['metric_recomputation_difference'] <= 1e-15
    print('SMOKE BURGERS OK', flush=True)


if __name__ == '__main__':
    main()
