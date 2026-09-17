"""Generate every configuration this lane runs, from the two audited parents.

  * the ladder configs are the exact qtd02 configuration (`comparators/qtd02-config-dense.json`,
    job 3757505) with only the fields listed in DESIGN.md section 4 changed;
  * the EQ-certification configs are q-ridge's `config-r3.json` (job 3768168, qrg304)
    restricted to q <= 64 and m = 1024 (DESIGN.md section 5);
  * the sealed cohort is drawn here, once, with NumPy only, and its disjointness from
    every draw any Burgers checkpoint was trained, selected or diagnosed on is asserted
    and written to `checks/sealed-cohort.json` BEFORE any job runs.

Run:  python experiments/b-seeds/make_configs.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SEEDS = [1, 2, 3]
SEALED_SEED = 17092026
SEALED_CASES = 6
INCUMBENT = 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
INCUMBENT_SHA = '18f0266ae6f0454200ec0b7bf94a18cde531feac9d3170d5099adc5d68d6b589'


def params_draw(seed, count):
    """`engines.params_draw`, replicated so this file imports no JAX."""
    r = np.random.default_rng(seed)
    return np.stack([r.uniform(.15, .85, count), r.uniform(.15, .85, count),
                     r.uniform(.05, .20, count), r.uniform(.5, 2., count),
                     np.exp(r.uniform(np.log(.01), np.log(.1), count))], axis=1)


def sha_array(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2) + '\n')


def sealed_cohort():
    sealed = params_draw(SEALED_SEED, SEALED_CASES)
    # every draw a Burgers checkpoint in this lineage was trained, selected or measured on
    others = {
        'bank+head training, canonical seed 0 (576)': params_draw(0, 576),
        'head training, appended seed 1000 (4032)': params_draw(1000, 4032),
        'r3 fresh test cohort, seed 1 (8)': params_draw(1, 8),
        'direction / EQ training family, seed 0 (128)': params_draw(0, 128),
        'opened development, seed 7090702 (4)': params_draw(7090702, 4),
        'fresh development, seed 911702 (2)': params_draw(911702, 2),
        'b-head-train holdout, seed 20260916 (64)': params_draw(20260916, 64),
    }
    dev = np.concatenate((others['opened development, seed 7090702 (4)'],
                          others['fresh development, seed 911702 (2)']))
    dist = {}
    for name, arr in others.items():
        d = np.linalg.norm(sealed[:, None, :] - arr[None, :, :], axis=2)
        dist[name] = float(d.min())
    lo = np.array([.15, .15, .05, .5, .01])
    hi = np.array([.85, .85, .20, 2., .1])
    inside = bool(np.all(sealed >= lo) and np.all(sealed <= hi))
    assert inside and min(dist.values()) > 1e-6, dist
    out = dict(seed=SEALED_SEED, cases=SEALED_CASES, physical_cases=sealed.tolist(),
               sha256_local=sha_array(sealed),
               sha256_note=('computed on the GB10; the viscosity column passes through np.exp, '
                            'which differs by one ulp between the GB10 and the cluster NumPy, so '
                            'the in-job gate is left unset and the audit compares VALUES to 1 ulp'),
               development_cases=dev.tolist(), development_sha256=sha_array(dev),
               min_distance_to=dist, inside_declared_ranges=inside,
               declared=('opened only in the last job of this lane (DESIGN.md section 3); '
                         'never used for any selection, diagnosis or fit'))
    return out


def ladder_config(base, attempt, label, ckpt_role, cohort, sealed):
    c = json.loads(json.dumps(base))
    c['attempt'] = attempt
    c['checkpoint_label'] = label
    c['lane'] = 'b-seeds'
    c['parent_configuration'] = 'q-trajdirs config-dense.json, job 3757505 (qtd02)'
    if ckpt_role == 'seed':
        c['expected_directions_sha256'] = None
        c['fidelity_expectations'] = {}
        c['btq201_expectations'] = {}
    else:
        # the incumbent: every dense_m4 / dense_fixedM arm must reproduce qtd02 to 1e-9
        c['qtd02_expectations'] = {a: dict(reproduces=a, tolerance=1e-9) for a in [
            'q0_M64_dense', 'q0_M256_dense', 'old_q16_M128_dense', 'old_q16_M256_dense',
            'old_q32_M192_dense', 'old_q32_M256_dense', 'old_q64_M256_dense',
            'old_q64_M320_dense', 'old_q128_M256_dense', 'old_q128_M576_dense',
            'old_q256_M1088_dense']}
    if cohort == 'sealed':
        c['eval_seed'] = sealed['seed']
        c['eval_cases'] = sealed['cases']
        c['eval_fresh_cases'] = 0
        c['cohort_roles'] = ['sealed final'] * sealed['cases']
        c['final_cohort_unopened'] = False
        c['expected_cohort_sha256'] = None
        c['expected_reference_sha256'] = {}
        c['fidelity_expectations'] = {}
        c['btq201_expectations'] = {}
        c.pop('qtd02_expectations', None)
        c['cohort_note'] = (f'THE SEALED FINAL COHORT: params_draw({sealed["seed"]}, '
                            f'{sealed["cases"]}), opened in this job for the first time '
                            '(DESIGN.md section 3); disjoint from every training, direction, '
                            'EQ and development draw by construction, checked in '
                            'checks/sealed-cohort.json')
        c['purpose'] = ('Sealed-cohort confirmation of the dense correction ladder for one '
                        f'checkpoint ({label}): the qtd02 configuration verbatim, on the six '
                        'sealed cases instead of the six development cases.')
    else:
        c['final_cohort_unopened'] = True
        c['purpose'] = (f'Development-cohort dense correction ladder for one checkpoint ({label}): '
                        'the qtd02 configuration verbatim (same six opened cases, same ladders, '
                        'same budget-600 block-damped variable-projection solver, same-job '
                        'fft_tight / fft_loose / nt1e-2_dt01 controls, three timed repetitions).')
    return c


def eqcert_config(base, attempt, label):
    c = json.loads(json.dumps(base))
    c['attempt'] = attempt
    c['checkpoint_label'] = label
    c['lane'] = 'b-seeds'
    c['parent_configuration'] = 'q-ridge config-r3.json, job 3768168 (qrg304)'
    c['q_ladder'] = [0, 16, 32, 64]
    c['m_grid'] = [1024]
    c['m_grid_requested'] = [1024]
    c['dense_twins'] = []
    c['reproduction_arms'] = []
    c['expectations'] = {}
    c['expected_directions_sha256'] = None
    c['question'] = 'EQCERT-SEED'
    c['purpose'] = ('EQ rule certification for a retrained seed checkpoint, q <= 64, m = 1024 '
                    'only (DESIGN.md section 5): the reachable-state rule refit and the '
                    'held-out rho certification of qrg304, restricted to what fits the budget. '
                    'The dense numbers come from the ladder job, not from here.')
    return c


def main():
    base = json.loads((HERE / 'comparators/qtd02-config-dense.json').read_text())
    r3 = json.loads((HERE.parent / 'q-ridge/config-r3.json').read_text())
    sealed = sealed_cohort()
    dump(HERE / 'checks/sealed-cohort.json', sealed)
    dump(HERE / 'config-dev-incumbent.json',
         ladder_config(base, 'dev-incumbent', 'incumbent', 'incumbent', 'dev', sealed))
    for s in SEEDS:
        dump(HERE / f'config-dev-seed{s}.json',
             ladder_config(base, f'dev-seed{s}', f'seed{s}', 'seed', 'dev', sealed))
        dump(HERE / f'config-eqcert-seed{s}.json', eqcert_config(r3, f'eqcert-seed{s}', f'seed{s}'))
        dump(HERE / f'config-sealed-seed{s}.json',
             ladder_config(base, f'sealed-seed{s}', f'seed{s}', 'seed', 'sealed', sealed))
    dump(HERE / 'config-sealed-incumbent.json',
         ladder_config(base, 'sealed-incumbent', 'incumbent', 'incumbent', 'sealed', sealed))
    print('sealed cohort', sealed['sha256_local'][:16], 'min distance',
          min(sealed['min_distance_to'].values()))
    print('wrote', sorted(p.name for p in HERE.glob('config-*.json')))


if __name__ == '__main__':
    main()
