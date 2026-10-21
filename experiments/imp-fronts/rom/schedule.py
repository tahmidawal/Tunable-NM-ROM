"""imp-fronts Phase B schedule definition: ONE source for resources.py, mkconfigs_pol.py and fpolicy.py's production
validation (DESIGN Amendments 10-12; Amendment 12 governs). Pure Python.

Ceilings (GPU-h; wall = ceiling / GPUs): training 4.6 per arm (1 GPU, no self-timer: certified against 100 %),
references 3.25 (2 H200, 1.625 h wall = 01:37:30; Amendment 13), selection 0.4 per arm, policy 0.5 per bank; every self-timed item is certified
against 90 % of its ceiling (the work cutoff). Smokes: SM-a 0.2 (arm a), SM-b 0.4 (arm b, Amendment 14), SM-r 0.4 (ROM controls,
16-case K1 dry run, 2 dev references; H200).
Schedules: S1 both arms = 9.2 + 3.25 + 0.8 + 1.5 + 0.85 = 15.75 (> 15: not fundable); S3 arm (b) + K1 = 4.6 + 3.25 + 0.4 +
1.0 + 0.8 = 10.05 (Amendment 14); S4 no Phase B. S2 withdrawn (Amendment 10).
Policy population per bank: 40 cases = the 16 fresh dev/val cases (j = 0..3) + 24 test cases (FI 6, FN0s 8, FC1 8,
FC2 2; lowest test indices j = 4..), identical in S1 and S3.
"""
from __future__ import annotations

CAP = 15.
CUTOFF = .9
CEIL = {'train_a': 4.6, 'train_b': 4.6, 'references': 3.25, 'select_a': .4, 'select_b': .4,
        'policy_K1': .5, 'policy_a': .5, 'policy_b': .5}
CUTOFF_OF = {k: (1. if k.startswith('train_') else CUTOFF) for k in CEIL}
GPUS = {k: (2 if k == 'references' else 1) for k in CEIL}
SMOKE = {'SM-a': .2, 'SM-b': .4, 'SM-r': .4}
COHORTS = ('FI', 'FN0s', 'FC1', 'FC2')
TEST = dict(FI=6, FN0s=8, FC1=8, FC2=2)
DEVVAL_IDS = tuple(f'{c}|{j}' for c in COHORTS for j in range(4))
TEST_IDS = tuple(f'{c}|{4 + j}' for c in COHORTS for j in range(TEST[c]))
POLICY_IDS = DEVVAL_IDS + TEST_IDS
POLICY_CEILING_H = CEIL['policy_K1']
SCHEDULES = {
    'S1': dict(items=tuple(CEIL), smokes=('SM-a', 'SM-b', 'SM-r'), banks=('1', 'a', 'b')),
    'S3': dict(items=('train_b', 'references', 'select_b', 'policy_K1', 'policy_b'), smokes=('SM-b', 'SM-r'),
               banks=('1', 'b')),
    'S4': dict(items=(), smokes=(), banks=()),
}
ORDER = ('S1', 'S3', 'S4')
POLICY_ITEM = {'1': 'policy_K1', 'a': 'policy_a', 'b': 'policy_b'}


def total(name):
    s = SCHEDULES[name]
    return sum(CEIL[k] for k in s['items']) + sum(SMOKE[k] for k in s['smokes'])


def check_policy_config(cfg):
    """Production (test_once) policy configs must equal the registered schedule (Amendment 10/12)."""
    errs = []
    if tuple(cfg.get('cases', ())) != POLICY_IDS:
        errs.append(f'cases differ from the registered 40 (16 dev/val + 24 test): {len(cfg.get("cases", ()))}')
    if abs(float(cfg.get('ceiling_hours', -1)) - POLICY_CEILING_H) > 1e-12:
        errs.append(f'ceiling_hours {cfg.get("ceiling_hours")} != {POLICY_CEILING_H}')
    if float(cfg.get('stop_fraction', CUTOFF)) != CUTOFF:
        errs.append('stop_fraction != 0.9')
    if int(cfg.get('max_rounds', 10)) != 10 or cfg.get('anchors') is not None:
        errs.append('max_rounds must be 10 and anchors the frozen default')
    if cfg.get('schedule') not in ('S1', 'S3') or str(cfg.get('bank_label')) not in SCHEDULES[cfg.get('schedule', 'S4')]['banks']:
        errs.append(f'bank {cfg.get("bank_label")} not in schedule {cfg.get("schedule")}')
    if errs:
        raise ValueError('policy config violates the registered schedule: ' + '; '.join(errs))
    return True


assert abs(total('S1') - 15.75) < 1e-9 and abs(total('S3') - 10.05) < 1e-9
assert len(POLICY_IDS) == 40 and len(set(POLICY_IDS)) == 40
