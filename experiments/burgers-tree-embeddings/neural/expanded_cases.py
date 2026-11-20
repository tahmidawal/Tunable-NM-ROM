"""Independently seeded B8 training/development families; final cohort stays sealed."""
import numpy as np

SEEDS = dict(train=2026111981, dev=2026111982, final_test=2026111983)
COUNTS = dict(train=32, dev=8)
NU = .005


def definitions(cohort, start=0, count=None):
    """Even indices are ridges, odd indices two pulses; chunking changes no draw."""
    if cohort == 'final_test':
        raise PermissionError('Expanded final-test cohort remains sealed until method freeze.')
    if cohort not in COUNTS:
        raise ValueError(f'Unknown expanded cohort: {cohort}')
    if count is None:
        count = COUNTS[cohort] - start
    if not (isinstance(start, int) and isinstance(count, int)
            and 0 <= start < COUNTS[cohort] and 0 < count <= COUNTS[cohort] - start):
        raise ValueError('Chunk must lie inside the registered cohort.')
    rows = []
    for index in range(start, start + count):
        rng = np.random.default_rng(np.random.SeedSequence([SEEDS[cohort], index]))
        family = 'ridge' if index % 2 == 0 else 'two_pulse'
        if family == 'ridge':
            gaussian = dict(center=rng.uniform(.3, .6, 2).tolist(),
                            sigma_t=float(rng.uniform(.18, .28)),
                            sigma_n=float(rng.uniform(.025, .04)),
                            angle_degrees=float(rng.uniform(-60., -30.)),
                            amplitude=float(rng.uniform(.8, 1.4)))
            gaussians = [gaussian]
        else:
            gaussians = []
            for low, high, amin, amax in [(.25, .4, 1., 1.5), (.5, .65, .5, 1.)]:
                center = rng.uniform(low, high, 2).tolist()
                width = float(rng.uniform(.04, .075))
                gaussians.append(dict(center=center, sigma_t=width, sigma_n=width,
                                      angle_degrees=0., amplitude=float(rng.uniform(amin, amax))))
        rows.append(dict(case=f'b8_{cohort}{index:03d}', cohort=cohort, index=index,
                         family=family, seed=SEEDS[cohort], nu=NU,
                         definition=dict(name=f'B8 {cohort} {family} {index:03d}',
                                         background=.15, gaussians=gaussians)))
    return rows
