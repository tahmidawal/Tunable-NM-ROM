"""imp-fronts Phase B schedule definition: ONE source for resources.py, mkconfigs_pol.py and fpolicy.py's production
validation (DESIGN Amendments 10-12; Amendment 12 governs). Pure Python.

Ceilings (GPU-h; wall = ceiling / GPUs): training 4.6 per arm (1 GPU; stopped 120 s before the wall, so certified
against 4.6 h - 120 s = 4.567 h, Amendment 15),
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
        'policy_K1': .5, 'policy_a': .5, 'policy_b': .5,
        'train_c': 4.6, 'select_c': .4, 'policy_c': .5}           # arm (c), Amendment 19 (E8 cap)
TRAIN_STOP_S = 120.               # Amendment 15: run_seed.sh is stopped 120 s before the hard wall (EXHAUSTED.json)
CUTOFF_OF = {k: ((CEIL[k] * 3600. - TRAIN_STOP_S) / (CEIL[k] * 3600.) if k.startswith('train_') else CUTOFF) for k in CEIL}
GPUS = {k: (2 if k == 'references' else 1) for k in CEIL}
SMOKE = {'SM-a': .2, 'SM-b': .4, 'SM-r': .4, 'SM-c': .45}
COHORTS = ('FI', 'FN0s', 'FC1', 'FC2')
TEST = dict(FI=6, FN0s=8, FC1=8, FC2=2)
DEVVAL_IDS = tuple(f'{c}|{j}' for c in COHORTS for j in range(4))
TEST_IDS = tuple(f'{c}|{4 + j}' for c in COHORTS for j in range(TEST[c]))
POLICY_IDS = DEVVAL_IDS + TEST_IDS
POLICY_CEILING_H = CEIL['policy_K1']
SCHEDULES = {
    'S1': dict(items=('train_a', 'train_b', 'references', 'select_a', 'select_b', 'policy_K1', 'policy_a', 'policy_b'),
               smokes=('SM-a', 'SM-b', 'SM-r'), banks=('1', 'a', 'b')),
    'S3': dict(items=('train_b', 'references', 'select_b', 'policy_K1', 'policy_b'), smokes=('SM-b', 'SM-r'),
               banks=('1', 'b')),
    # S3C (Amendment 19, coordinator release 2026-10-22): S3 + arm (c); fundable against the E8 cap only with the ledger
    # (S3's measured spend, not its ceilings): resources.certify / admit evaluate spent + live + unstarted ceilings.
    'S3C': dict(items=('train_b', 'references', 'select_b', 'policy_K1', 'policy_b', 'train_c', 'select_c', 'policy_c'),
                smokes=('SM-b', 'SM-r', 'SM-c'), banks=('1', 'b', 'c')),
    'S4': dict(items=(), smokes=(), banks=()),
}
ORDER = ('S1', 'S3C', 'S3', 'S4')
DEFAULT_PLAN = 'S3C'              # smoke admission plan before a certificate selects one (arm (c) released)
POLICY_ITEM = {'1': 'policy_K1', 'a': 'policy_a', 'b': 'policy_b', 'c': 'policy_c'}

# E10 (E10-DESIGN.md sections 8-9): a separate bucket of 10 GPU-h; E10-q resolved without a job, only E10-s runs
E10_CAP = 10.
E10_CEIL = {'policy_e10s': .8}
E10_SMOKE = {'SM-e': .3}
E10_FACTORS = (1., .5, .25, .125)
E10_ROUNDS = 11


def check_e10s_config(cfg):
    """POL-e10s: K1 (gauss160), the 40 Phase B cases RE-USED (marked), factor set {1, .5, .25, .125}, round cap 11,
    ceiling 0.8 h, schedule 'E10'."""
    errs = []
    if tuple(cfg.get('cases', ())) != POLICY_IDS:
        errs.append('cases differ from the 40 Phase B cases')
    if tuple(float(x) for x in cfg.get('factors', ())) != E10_FACTORS or int(cfg.get('max_rounds', 0)) != E10_ROUNDS:
        errs.append('factor set / round cap differ from E10-s')
    if abs(float(cfg.get('ceiling_hours', -1)) - E10_CEIL['policy_e10s']) > 1e-12 or float(cfg.get('stop_fraction', 0)) != CUTOFF:
        errs.append('ceiling / stop fraction differ from E10-s')
    if cfg.get('schedule') != 'E10' or str(cfg.get('bank_label')) != '1' or cfg.get('selected') != 'gauss160':
        errs.append('E10-s runs on K1 from gauss160 under schedule E10')
    if cfg.get('reused_test_cases') is not True or cfg.get('anchors') is not None:
        errs.append('E10-s must mark its test cases as re-used and keep the frozen anchors')
    if errs:
        raise ValueError('E10-s config violates the registered design: ' + '; '.join(errs))
    return True


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
    if cfg.get('factors') not in (None, [1, .5, .25], [1., .5, .25]):
        errs.append('Phase B policy uses the factor set {1, .5, .25}')
    if cfg.get('schedule') not in ('S1', 'S3', 'S3C') or str(cfg.get('bank_label')) not in SCHEDULES[cfg.get('schedule', 'S4')]['banks']:
        errs.append(f'bank {cfg.get("bank_label")} not in schedule {cfg.get("schedule")}')
    if errs:
        raise ValueError('policy config violates the registered schedule: ' + '; '.join(errs))
    return True


assert abs(total('S1') - 15.75) < 1e-9 and abs(total('S3') - 10.05) < 1e-9 and abs(total('S3C') - 16.) < 1e-9
assert len(POLICY_IDS) == 40 and len(set(POLICY_IDS)) == 40
