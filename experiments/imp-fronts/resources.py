"""imp-fronts Phase B cost certificate and admission rule (DESIGN B8 / B9 as replaced by Amendments 3-5 and 9-12;
Amendment 12 governs; pattern of jcp-speed3d resources.py). CPU only, no JAX. Ceilings and schedules come from
rom/schedule.py (shared with mkconfigs_pol.py and fpolicy.py). Projections are EXPECTED costs (Amendment 12): the hard
ceilings and the admission rule enforce spending; nothing guarantees completion.

Inputs
  runs/SM-a/timing.json, runs/SM-b/timing.json   the training smokes' timing records (cluster/train/phase_times.py
      smoke output; --train-timing SM-b=<path> overrides), schema imp-fronts-train-timing/3 documented in train_item().
      Anything missing, failed or empty -> the arm's training item is UNCERTIFIED (never zero).
  runs/SM-r/SMOKE-EVIDENCE.json (rom/evidence.py build): hash-bound C-B1 / C-B6 / dry-run records; the dry-run
      result.json itself is passed with --dry (the evidence check is re-run on it).
  ROM timing rows: the SM-r fpolicy result.json files (--rom, --dry) and jcp-robust e2s1 history (--history).
  jcp-regimes reference records (--regimes): runs/ref*/output/refjob_*/2d/*/ST/c*.json (seconds, job.gpu).
  runs/GPU-LEDGER.json: {"jobs": [{"name", "item", "ceiling_gpu_h", "state": pending|running|done|failed,
      "charged_gpu_h"}]}; item = a schedule item (train_b, references, ...) or a smoke name (SM-a, SM-b, SM-r).

Projection rules
  * training (Amendment 15): K1's slowest production retrain 15018 s + the measured same-population generation increment
    (arm - incumbent, scaled to the production trajectory counts, floored at 0); certified against 4.6 h - 120 s.
  * references: 24 test cases x (MEAN per-case seconds of the jcp-regimes reference records on H200, never below the
    smoke dev-reference mean) x 1.5; ceiling 3.25 (2 H200, 1.625 h wall; Amendment 13).
  * selection (R384 group): max(solve inventory at the pooled e2s1 + smoke rates, e2s1 measured R384 group + setup) x 1.5;
    ceiling 0.4.
  * policy per bank (Amendment 12): (mean per-case warm solve cost of the SM-r K1 dry run on the 16 fresh dev/val cases)
    x 40 x 1.5 + measured compilation per shape x shapes the bank's chain can reach + 4 tighter solves at the largest
    reachable rule (gauss256, f = 0.25); ceiling 0.5. The dry run must pass rom/evidence.check_dry (exact 16 ids, full
    policy, complete, no open case, every case charged). The K1 path distribution is ASSUMED to transfer to the arm
    banks (not verifiable in advance; arm jobs keep BUDGET-CAP). The worst-path cost projection is reported beside it.
  * every self-timed item passes only if projected <= 90 % of its ceiling (training: 4.6 h - 120 s).
Schedules (rom/schedule.py): S1 15.75 (over the cap -> never fundable), S3 10.05, S4 none; the first fundable schedule
whose items all pass is selected.
Admission (Amendments 5, 12): see admit().

    python resources.py [--rom r.json ...] [--dry dry.json] [--regimes dir] [--ref-gpu H200] [--out runs/COST-CERT.json]
    python resources.py --admit <item> <ceiling GPU-h>         (exit 0 admit, 3 refuse)
    python resources.py --selftest
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import tempfile

import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'rom'))
import policy_sm as SM  # noqa: E402
import schedule as SCH  # noqa: E402
import evidence as EV  # noqa: E402

CAP, CEIL = SCH.CAP, SCH.CEIL
REGIMES = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/runs')
E2S1 = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/runs/e2s1/archive/output/r2acc_e2s1/result.json')
EVIDENCE = HERE / 'runs' / 'SM-r' / 'SMOKE-EVIDENCE.json'
POLK1_RESULT = HERE / 'runs' / 'rom' / 'POL-K1-out' / 'output' / 'fpolicy_pol1' / 'result.json'
E10_EVIDENCE = HERE / 'runs' / 'SM-e' / 'E10-EVIDENCE.json'
C_REF, C_ROM = 1.5, 1.5
GAUSS = lambda p: p * p
M = {'fib4181': 4181, 'fib6765': 6765, 'gauss96': GAUSS(96), 'fib10946': 10946, 'fib17711': 17711, 'gauss128': GAUSS(128),
     'gauss160': GAUSS(160), 'gauss192': GAUSS(192), 'gauss256': GAUSS(256), 'gauss320': GAUSS(320), 'gauss384': GAUSS(384)}
ANCHORS = 4
K1_START = 'gauss160'
CANDIDATES = [('fib4181', 'fib6765'), ('fib6765', 'fib10946'), ('gauss96', 'gauss128'), ('fib10946', 'fib17711'),
              ('gauss128', 'gauss160'), ('gauss160', 'gauss192'), ('gauss192', 'gauss256')]


def steps(f):
    return int(round(.25 / (.005 * f)))


# --------------------------------------------------------------------------------------------- training
TRAIN_SCHEMA = 'imp-fronts-train-timing/3'
K1_TOTAL_S = 15018.               # jcp-robust e2t2d seed 2 (A100 80GB PCIe): stage A 11099 + B 3901 + rotation 18 s
ARM_OF_ITEM = {'train_a': ('a', 'fid'), 'train_b': ('b', 'cov')}
REGISTERED_POPS = {'seed0': (576, 1152), 'seed1000': (576, 4032)}   # measured / production trajectories (Amendments 15-16)


def train_item(timing, item='train_b'):
    """Amendment 15 training certificate from the smoke's timing record (cluster/train/phase_times.py smoke output), schema
      {"schema": "imp-fronts-train-timing/3", "job": "SM-b", "arm": "cov" | "fid" (or "b" | "a"),
       "backend": "gpu", "device": "<GPU name>",
       "all_done": bool, "gates_passed": bool, "chaining_completed": bool, "burn_in": bool,
       "timing": {"populations": {"seed0": {"trajectories": 576, "production_trajectories": 1152,
                                             "incumbent_s": float > 0, "arm_s": float > 0},
                                   "seed1000": {"trajectories": 576, "production_trajectories": 4032, ...}}}}
    (exactly these two populations and counts: seed 0 is generated by Stage A and Stage B, seed 1000 timed on one chunk)
    The incumbent and the arm generator are timed on the SAME populations in the same job after an untimed burn-in;
    increment = max(0, sum over populations of (arm_s - incumbent_s) * production_trajectories / trajectories);
    projection = K1's slowest production retrain 15018 s + increment, certified against 4.6 h - 120 s (schedule.py).
    Requirements (else UNCERTIFIED, never zero): schema, arm, GPU backend on K1's GPU type (A100), all_done, gates,
    chaining, burn-in, every population number finite and positive, sum of production_trajectories >= 1152 + 4032 (Amendment 16)."""
    if not timing:
        return dict(projected=None, certified=False, basis='no smoke timing record')
    why = []
    lab = ARM_OF_ITEM[item]
    if timing.get('schema') != TRAIN_SCHEMA:
        why.append(f"schema {timing.get('schema')} != {TRAIN_SCHEMA}")
    if timing.get('arm') not in lab:
        why.append(f"arm {timing.get('arm')} is not {lab}")
    if timing.get('backend') != 'gpu' or 'A100' not in str(timing.get('device', '')):
        why.append('not timed on a GPU of K1\'s type (A100)')
    for k in ('all_done', 'gates_passed', 'chaining_completed', 'burn_in'):
        if timing.get(k) is not True:
            why.append(f'{k} not true')
    pops = ((timing.get('timing') or {}).get('populations')) or {}
    inc, prod_n = 0., 0
    if set(pops) != set(REGISTERED_POPS):
        why.append(f'timed populations {sorted(pops)} != registered {sorted(REGISTERED_POPS)}')
    for name, (n_, N_) in REGISTERED_POPS.items():
        v = pops.get(name) or {}
        if (v.get('trajectories'), v.get('production_trajectories')) != (n_, N_):
            why.append(f'population {name}: counts {v.get("trajectories")}/{v.get("production_trajectories")} != {n_}/{N_}')
    for name, v in pops.items():
        try:
            n, N, i_, a_ = int(v['trajectories']), int(v['production_trajectories']), float(v['incumbent_s']), float(v['arm_s'])
        except (KeyError, TypeError, ValueError):
            why.append(f'population {name}: missing fields')
            continue
        if not (n > 0 and N > 0 and i_ > 0 and a_ > 0 and all(map(lambda x: x == x and x < float('inf'), (i_, a_)))):  # NaN-safe
            why.append(f'population {name}: non-positive or non-finite timing')
            continue
        inc += (a_ - i_) * N / n
        prod_n += N
    if pops and prod_n < 1152 + 4032:
        why.append(f'production trajectories {prod_n} < 5184')
    if why:
        return dict(projected=None, certified=False, basis='; '.join(why))
    return dict(projected=(K1_TOTAL_S + max(0., inc)) / 3600., certified=True, increment_s=inc,
                increment_credited_s=max(0., inc), k1_total_s=K1_TOTAL_S, gpu=timing.get('device'), job=timing.get('job'),
                basis='Amendment 15: K1 slowest retrain + measured same-population generation increment')


ARMC_SCHEMA = 'imp-fronts-train-timing-c/1'
ARMC_LIMIT_S = 16440.                                              # 4.6 h - 120 s (Amendments 15, 19-22)
ARMC_PARTS = ('sub', 'full', 'data_A', 'remA', 'B_gen_head', 'B_extract', 'rotation', 'compile')


def train_c_item(timing):
    """Arm (c) training certificate from SM-c's timing_c.json, written by cluster/train/phase_times.py smokec (schema
    'imp-fronts-train-timing-c/1'; the Amendment 21-22 projection with its two conservative choices is computed THERE:
    T = n_sub t_sub rho_sub + n_full t_full rho_full + T_B,gen+head + T_B,extract max(1, rho_ext) + T_rot + data_A
    + remA max(1, rho_full) + T_compile). Certified iff the record is that schema for arm c on K1's GPU type (A100), its
    gates passed (errors == [], all_done, certified), the eight parts are finite and non-negative and sum to projection_s,
    and projection_conservative_s (= projection_s + B_extract, audit codex-job-smC-2) <= 16 440 s. A missing file or a
    timing_c_INVALID.json -> UNCERTIFIED (no training job)."""
    if not timing:
        return dict(projected=None, certified=False, basis='no SM-c timing_c.json')
    why = []
    if timing.get('schema') != ARMC_SCHEMA or timing.get('arm') != 'c' or timing.get('job') != 'SM-c':
        why.append('schema / arm / job')
    if 'A100' not in str(timing.get('device', '')):
        why.append("not timed on a GPU of K1's type (A100)")
    if timing.get('errors') != [] or timing.get('all_done') is not True or timing.get('certified') is not True:
        why.append(f"SM-c gates not passed ({timing.get('errors')})")
    parts = timing.get('parts_s') or {}
    T = timing.get('projection_s')
    finite = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and x == x and abs(x) < float('inf')
    if set(parts) != set(ARMC_PARTS) or not all(finite(v) and v >= 0 for v in parts.values()):
        why.append('projection parts missing or invalid')
    elif not (finite(T) and T > 0 and abs(sum(parts.values()) - T) <= 1e-6 * T):
        why.append('projection_s does not equal the sum of its parts')
    Tc = timing.get('projection_conservative_s')        # audit codex-job-smC-2: extraction counted twice; certify on this
    if not why and not (finite(Tc) and abs(Tc - (T + parts['B_extract'])) <= 1e-6 * Tc):
        why.append('projection_conservative_s missing or != projection_s + B_extract')
    if why:
        return dict(projected=None, certified=False, basis='; '.join(why))
    return dict(projected=Tc / 3600., certified=bool(Tc <= ARMC_LIMIT_S), projection_s=T, projection_conservative_s=Tc,
                limit_s=ARMC_LIMIT_S,
                rho=timing.get('rho'), parts_s=parts, job_id=timing.get('job_id'),
                basis='SM-c timing_c.json (phase_times.py smokec, Amendments 21-22; certified on the conservative bound, '
                      'extraction x2)' + ('' if Tc <= ARMC_LIMIT_S else ': conservative projection over 16 440 s'))


# --------------------------------------------------------------------------------------------- references
def regimes_mean(regimes_dir, gpu):
    """Amendment 9: measured MEAN per-case seconds over every jcp-regimes reference case record on this GPU type."""
    xs = []
    for f in glob.glob(str(Path(regimes_dir) / 'ref*' / 'output' / 'refjob_*' / '2d' / '*' / 'ST' / 'c*.json')):
        d = json.loads(Path(f).read_text())
        if gpu.lower() in str(d.get('job', {}).get('gpu', '')).lower():
            xs.append((float(d['seconds']), bool(d.get('refined'))))
    if not xs:
        return None, dict(n_regimes_cases=0)
    return sum(x for x, _ in xs) / len(xs), dict(n_regimes_cases=len(xs), n_refined=sum(r for _, r in xs),
                                                  max_seconds=max(x for x, _ in xs))


def ref_item(regimes_dir, gpu, smr_timing):
    """24 test cases x max(historical H200 mean, smoke dev-reference mean) x 1.5. The smoke references are a cross-check:
    if SM-r's references are missing or finished INCOMPLETE (references_status != 'complete'), the historical bound is
    used and the item is FLAGGED (smoke_check), never silently passed."""
    n = len(SCH.TEST_IDS)
    m, info = regimes_mean(regimes_dir, gpu)
    t = smr_timing or {}
    refs = [r for r in t.get('references') or [] if isinstance(r.get('seconds'), (int, float)) and r['seconds'] > 0]
    status = t.get('references_status')
    if not t:
        check = 'MISSING: no SM-r reference timing; historical bound only'
    elif status != 'complete' or len(refs) < 2:
        check = f'INCOMPLETE: SM-r references {status!r}, {len(refs)} timed; historical bound only'
        refs = []
    else:
        check = 'ok'
    smm = sum(r['seconds'] for r in refs) / len(refs) if refs else None
    if m is None:
        return dict(projected=None, certified=False, basis=f'no jcp-regimes reference record on {gpu}', smoke_check=check, **info)
    per = max(m, smm or 0.) * C_REF
    return dict(projected=n * per / 3600., certified=True, per_case_seconds=per, regimes_mean_seconds=m,
                smoke_dev_mean_seconds=smm, smoke_check=check, n_cases=n, contingency=C_REF, gpu=gpu, gpus=2, **info)


# --------------------------------------------------------------------------------------------- ROM rates
def e2s1_rows(path=E2S1):
    """jcp-robust e2s1 (1024^2, A100-80G) timing rows in the smoke-row schema: prod rows -> ('prod', recorded), their
    seconds_tight -> ('tight', unrecorded), target rows -> ('prod', unrecorded) at gauss384. The first case of each
    (group, arm, rule) is marked first_of_shape (it carries the compilation)."""
    r = json.loads(Path(path).read_text())
    out, seen = [], set()
    for x in r['rows']:
        k = (x['group'], x['arm'], x['rule'], x['kind'])
        first = k not in seen
        seen.add(k)
        if x['kind'] == 'prod':
            out.append(dict(level='prod', recorded=True, rule=x['rule'], m=M[x['rule']], steps=x['steps'],
                            seconds=x['seconds'], first_of_shape=first, src='e2s1'))
            if 'seconds_tight' in x:
                out.append(dict(level='tight', recorded=False, rule=x['rule'], m=M[x['rule']], steps=x['steps'],
                                seconds=x['seconds_tight'], first_of_shape=first, src='e2s1'))
        elif x['kind'] == 'target':
            out.append(dict(level='prod', recorded=False, rule=x['rule'], m=M[x['rule']], steps=50, seconds=x['seconds'],
                            first_of_shape=first, src='e2s1'))
    return out, dict(gpu=r['gpu'], elapsed_R384=r['elapsed']['R384'],
                     setup_seconds=r['setup']['bank']['seconds'] + r['setup']['ops_frozen']['seconds'])


class Rates:
    """Per (level, recorded, rule): max over warm solves of seconds / (m * steps); a rule never measured at that level uses
    the rate of the largest measured rule (node-count and step-count scaling, Amendment 9). 'tighter' uses 'tight'.
    Compile per shape = max over (first-of-shape - median warm) of the same key. Rows from several sources (e2s1 history,
    smoke) are pooled, so every key takes the LARGER measured rate (the cross-check of Amendment 9)."""

    def __init__(self, rows):
        by = {}
        for x in rows:
            lv = 'tight' if x['level'] == 'tighter' else x['level']
            by.setdefault((lv, bool(x['recorded']), x.get('rule') or f"m{x['m']}"), []).append(x)
        self.r, self.top, self.compile, self.basis = {}, {}, 0., {}
        for k, xs in by.items():
            warm = [x for x in xs if not x['first_of_shape']] or xs
            self.r[k] = max(x['seconds'] / (x['m'] * x['steps']) for x in warm)
            med = sorted(x['seconds'] for x in warm)[len(warm) // 2]
            for x in xs:
                if x['first_of_shape']:
                    self.compile = max(self.compile, x['seconds'] - med)
            lvk = k[:2]
            m = warm[0]['m']
            if lvk not in self.top or m > self.top[lvk][0]:
                self.top[lvk] = (m, self.r[k])
            self.basis['/'.join(map(str, k))] = dict(rate=self.r[k], n=len(warm), sources=sorted({x.get('src', 'smoke') for x in xs}))

    def ok(self, lv, rec):
        lv = 'tight' if lv == 'tighter' else lv
        return (lv, rec) in self.top or (lv, not rec) in self.top

    def sec(self, lv, rec, rule, f):
        lv = 'tight' if lv == 'tighter' else lv
        for key in ((lv, rec, rule), (lv, not rec, rule)):
            if key in self.r:
                return self.r[key] * M[rule] * steps(f)
        t = self.top.get((lv, rec)) or self.top[(lv, not rec)]
        return t[1] * M[rule] * steps(f)


KIND = {'prod': ('prod', True), 'half': ('prod', True), 'succ': ('prod', True), 'tight': ('tight', False),
        'tighter': ('tighter', False)}


def solves_cost(rates, solves):
    if not all(rates.ok(*KIND[k]) for k, _, _ in solves):
        return None, 0
    s = sum(rates.sec(*KIND[k], r, f) for k, r, f in solves)
    return s, len({(r, f, KIND[k][1]) for k, r, f in solves})


def worst_path(rates, start):
    best = (-1., None, 0)
    for p in SM.paths(start):
        c, n = solves_cost(rates, SM.path_solves(p))
        if c is None:
            return None, p, n
        if c > best[0]:
            best = (c, p, n)
    return best


def selection_item(rates, hist):
    """Amendment 9: R384 group only (2 arms x [38 target gauss384 + 6 check gauss320 + 7 rules x (38 prod + 38 tight +
    6 tighter + 38 succ)] at factor 1). Projection = max(solve inventory at the pooled rates, e2s1's measured R384 group
    time + setup) x 1.5."""
    if rates is None:
        return dict(projected=None, certified=False, basis='no ROM timing')
    sv = []
    for _ in range(2):
        sv += [('succ', 'gauss384', 1.)] * 38 + [('succ', 'gauss320', 1.)] * 6
        for c, s in CANDIDATES:
            sv += [('prod', c, 1.)] * 38 + [('tight', c, 1.)] * 38 + [('tighter', c, 1.)] * 6 + [('succ', s, 1.)] * 38
    c, n = solves_cost(rates, sv)
    if c is None:
        return dict(projected=None, certified=False, basis='a solve level was never measured')
    inv = c + n * rates.compile
    h = (hist['elapsed_R384'] + hist['setup_seconds']) if hist else 0.
    sec = max(inv, h) * C_ROM
    return dict(projected=sec / 3600., certified=True, solves=len(sv), shapes=n, inventory_seconds=inv,
                e2s1_R384_seconds=h or None, contingency=C_ROM, groups=['R384'])


def reachable_shapes(starts):
    sh = set()
    for s in starts:
        for q in SM.paths(s):
            sh |= {(r, f, KIND[k][1]) for k, r, f in SM.path_solves(q)}
    return sh


def dry_run(dry):
    """Validated dry run -> (per-case seconds = the dry-run STEP's whole wall clock / 16, info) or (None, info). The step
    wall clock (fpolicy driver_wall_seconds) contains setup, the burn-in, compilation, solves, fresh-Jacobian
    certificates and control work (code audit SM-r-2), so nothing measured is left out of the projection."""
    if not dry:
        return None, dict(basis='no smoke policy dry run')
    chk = EV.check_dry(dry)
    if not all(chk.values()):
        return None, dict(basis='dry run fails the registered contract', failed=[k for k, v in chk.items() if not v])
    w = float(dry['driver_wall_seconds'])
    solve = sum(x['seconds'] for x in dry['rows'])
    return w / len(SCH.DEVVAL_IDS), dict(counts=dry.get('counts'), wall_seconds=w, solve_seconds=solve,
                                         non_solve_fraction=(w - solve) / w if w > 0 else None)


def worst_projection(rates, n_cases, starts):
    if rates is None:
        return dict(projected=None, basis='no ROM timing')
    worst = None
    for s in starts:
        c, p, _ = worst_path(rates, s)
        if c is None:
            return dict(projected=None, basis='a solve level was never measured')
        if worst is None or c > worst[0]:
            worst = (c, p, s)
    tighter = ANCHORS * rates.sec('tighter', False, 'gauss256', .25)
    comp = len(reachable_shapes(starts) | {('gauss256', .25, False)}) * rates.compile
    return dict(projected=((n_cases * worst[0] + tighter) * C_ROM + comp) / 3600., worst_start=worst[2],
                worst_path=worst[1], per_case_seconds=worst[0], basis='worst-path cost projection (every case worst path)')


def policy_item(rates, starts, dry):
    """Policy projection (Amendment 12 as restated after code audit SM-r-2): (dry-run step wall clock / 16 cases) x 40 x
    1.5 + 4 tighter solves at gauss256 f = 0.25 (the dry run has no anchors). Compilation, setup and certificates are
    inside the wall clock. The worst-path cost projection is reported beside it."""
    n = len(SCH.POLICY_IDS)
    mean, dinfo = dry_run(dry)
    wp = worst_projection(rates, n, starts)
    if mean is None or rates is None:
        return dict(projected=None, certified=False, basis=dinfo.get('basis', 'no ROM timing'), dry_run=dinfo,
                    worst_path_projection=wp)
    tighter = ANCHORS * rates.sec('tighter', False, 'gauss256', .25) * C_ROM
    sec = mean * n * C_ROM + tighter
    return dict(projected=sec / 3600., certified=True, per_case_seconds=mean, n_cases=n, contingency=C_ROM,
                tighter_seconds=tighter, dry_run=dinfo, worst_path_projection=wp,
                transfer_assumption=('K1 dry-run path distribution assumed to transfer' if starts != [K1_START] else None),
                basis='dry-run step wall clock / 16 x 40 x 1.5 + 4 tighter at gauss256 f=.25')


def dry_run_expected(rates, n=16, start=K1_START):
    """Expected cost of the SM-r dry run at the pooled rates: every case converging in round 1 (low) to every case on
    the worst path (high), plus compilation of the reachable shapes."""
    if rates is None:
        return None
    lo, _ = solves_cost(rates, SM.path_solves([(start, 1.)]))
    hi, p, _ = worst_path(rates, start)
    sh = len(reachable_shapes([start]))
    return dict(cases=n, start=start, low_gpu_h=(n * lo + 4 * rates.compile) / 3600., high_gpu_h=(n * hi + sh * rates.compile) / 3600.,
                per_case_low_s=lo, per_case_high_s=hi, worst_path=p)


# --------------------------------------------------------------------------------------------- certificate
def certify(timings, rom_rows, regimes_dir=REGIMES, ref_gpu='H200', ledger=None, hist_path=E2S1, dry=None, sme=None,
            k1pol=None):
    hrows, hist = e2s1_rows(hist_path) if hist_path and Path(hist_path).exists() else ([], None)
    rows = list(hrows) + list(rom_rows or [])
    rates = Rates(rows) if rows else None
    smr_timing = timings.get('SM-r')
    base = dict(train_a=train_item(timings.get('SM-a'), 'train_a'), train_b=train_item(timings.get('SM-b'), 'train_b'),
                train_c=train_c_item(timings.get('SM-c')),
                references=ref_item(regimes_dir, ref_gpu, smr_timing), select=selection_item(rates, hist),
                policy_K1=policy_item(rates, [K1_START], dry), policy_arm=policy_item(rates, [None] + list(SM.CHAIN), dry))
    pick = dict(train_a='train_a', train_b='train_b', references='references', select_a='select', select_b='select',
                policy_K1='policy_K1', policy_a='policy_arm', policy_b='policy_arm', train_c='train_c', select_c='select',
                policy_c='policy_arm')
    led = ledger_state(ledger)
    out = dict(schema='imp-fronts-cost-cert/3', amendment=12, cap=CAP, ceilings=CEIL, cutoff=SCH.CUTOFF_OF,
               smoke_ceilings=SCH.SMOKE, ledger=led, ref_gpu=ref_gpu, history=dict(e2s1=str(hist_path) if hist else None, **(hist or {})),
               smoke_rom_rows=len(rom_rows or []), rates=(rates.basis if rates else None),
               compile_seconds=(rates.compile if rates else None), dry_run_expected=dry_run_expected(rates),
               schedules={}, selected=None)
    for s in SCH.ORDER:
        items = {k: dict(base[pick[k]]) for k in SCH.SCHEDULES[s]['items']}
        for k, v in items.items():
            v.update(ceiling=CEIL[k], cutoff=SCH.CUTOFF_OF[k], limit=CEIL[k] * SCH.CUTOFF_OF[k])
            v['fits'] = bool(v['certified'] and v['projected'] is not None and v['projected'] <= v['limit'] + 1e-12)
        ok = all(v['fits'] for v in items.values())
        tot = SCH.total(s)
        S_ = SCH.SCHEDULES[s]        # fundability with the ledger: spent + live + ceilings of what is not yet started
        committed = led['spent'] + led['live'] + sum(CEIL[k] for k in S_['items'] if k not in led['items']) \
            + sum(SCH.SMOKE[k] for k in S_['smokes'] if k not in led['items'])
        if s == 'S1':                # S1 (both arms) was judged on its ceilings and is not resurrected by S3's savings
            committed = max(committed, tot)
        out['schedules'][s] = dict(items=items, all_fit=ok, ceiling_total=tot, committed=committed,
                                   fundable=bool(committed <= CAP + 1e-9),
                                   projected_sum=sum(v['projected'] or 0. for v in items.values()),
                                   uncertified=[k for k, v in items.items() if not v['certified']],
                                   over_cutoff=[k for k, v in items.items() if v['certified'] and not v['fits']])
        if out['selected'] is None and ok and committed <= CAP + 1e-9:
            out['selected'] = s
    out['e10'] = e10_item(sme, k1pol)
    return out


# --------------------------------------------------------------------------------------------- ledger / admission
class LedgerError(ValueError):
    pass


def ledger_state(ledger, bucket='E8'):
    """Fail closed: every done / failed job must carry a finite non-negative charged_gpu_h; every pending / running job a
    finite positive ceiling_gpu_h; states must be known. Jobs are counted in their bucket ('E8' default, 'E10')."""
    jobs = [j for j in (ledger or {}).get('jobs', []) if j.get('bucket', 'E8') == bucket]
    if any(j.get('bucket', 'E8') not in ('E8', 'E10') for j in (ledger or {}).get('jobs', [])):
        raise LedgerError('unknown ledger bucket')
    spent = live = 0.
    for j in jobs:
        st = j.get('state')
        if st in ('done', 'failed'):
            c = j.get('charged_gpu_h')
            if not (isinstance(c, (int, float)) and not isinstance(c, bool) and c == c and 0 <= c < float('inf')):
                raise LedgerError(f"job {j.get('name')}: {st} without a valid charged_gpu_h")
            spent += float(c)
        elif st in ('pending', 'running'):
            c = j.get('ceiling_gpu_h')
            if not (isinstance(c, (int, float)) and not isinstance(c, bool) and 0 < c < float('inf')):
                raise LedgerError(f"job {j.get('name')}: {st} without a valid ceiling_gpu_h")
            live += float(c)
        else:
            raise LedgerError(f"job {j.get('name')}: unknown state {st}")
    return dict(spent=spent, live=live, n=len(jobs),
                items={j.get('item') for j in jobs if j.get('state') in ('pending', 'running', 'done')})


def cert_ok(cert, item):
    """The certificate must select a fundable schedule whose items all fit, with this item certified and fitting."""
    sel = (cert or {}).get('selected')
    if sel not in ('S1', 'S3', 'S3C'):
        return False, f'certificate selects {sel}: no Phase B item is fundable'
    sc = (cert.get('schedules') or {}).get(sel) or {}
    if not (sc.get('fundable') is True and sc.get('all_fit') is True and abs(float(sc.get('ceiling_total', 99)) - SCH.total(sel)) < 1e-9):
        return False, f'schedule {sel} not fundable / not all fitting / ceilings differ from schedule.py'
    its = sc.get('items') or {}
    if set(its) != set(SCH.SCHEDULES[sel]['items']):
        return False, f'schedule {sel} items differ from schedule.py'
    v = its.get(item)
    if v is None:
        return False, f'{item} is not in schedule {sel}'
    if not (v.get('certified') is True and v.get('fits') is True and isinstance(v.get('projected'), (int, float))
            and v['projected'] <= CEIL[item] * SCH.CUTOFF_OF[item] + 1e-12 and abs(v.get('ceiling', -1) - CEIL[item]) < 1e-12):
        return False, f'{item} not certified within its cutoff in the certificate'
    return True, 'ok'


def admit(ledger, cert, item, ceiling, evidence=EVIDENCE, e10_evidence=E10_EVIDENCE):
    """Amendments 5 / 12 / 15. Smoke items (SM-a / SM-b / SM-r) are admitted against the plan (the certificate's selected
    schedule, S3 before any certificate exists) and need no evidence. Production items need a certificate that selects a
    fundable schedule with the item certified within its cutoff (cert_ok), the registered ceiling, the item not yet
    started, and passing hash-bound smoke evidence (C-B1, C-B6, dry run). A malformed ledger refuses everything. The sum
    spent (ledger charges) + ceilings of running / pending jobs + outstanding smoke reservations of the plan + the
    proposed ceiling + ceilings of the plan's remaining unstarted items must be <= 15."""
    why = []
    try:
        ceiling = float(ceiling)
    except (TypeError, ValueError):
        return dict(admit=False, reasons=[f'ceiling {ceiling!r} is not a number'])
    if not (ceiling == ceiling and 0 < ceiling < float('inf')):
        return dict(admit=False, reasons=[f'ceiling {ceiling} is not finite and positive'])
    try:
        led = ledger_state(ledger)
    except LedgerError as e:
        return dict(admit=False, reasons=[f'ledger: {e}'])
    if item in SCH.E10_CEIL or item in SCH.E10_SMOKE:
        return admit_e10(ledger, cert, item, ceiling, e10_evidence)
    sel = (cert or {}).get('selected')

    def tally(plan):
        S = SCH.SCHEDULES[plan]
        so = sum(SCH.SMOKE[k] for k in S['smokes'] if k not in led['items'] and k != item)
        rem = sum(CEIL[k] for k in S['items'] if k not in led['items'] and k != item)
        return so, rem, led['spent'] + led['live'] + so + float(ceiling) + rem

    if item in SCH.SMOKE:
        if abs(float(ceiling) - SCH.SMOKE[item]) > 1e-9:
            why.append(f'ceiling {ceiling} != registered {SCH.SMOKE[item]}')
        # the certificate's plan, else the pre-registered plans in order (S3C, then S3) that contain the smoke and fit
        cands = [sel] if sel in ('S1', 'S3', 'S3C') else [SCH.DEFAULT_PLAN, 'S3']
        cands = [c for c in cands if item in SCH.SCHEDULES[c]['smokes']]
        if not cands:
            why.append(f'{item} is not a smoke of the plan')
            plan = sel if sel in SCH.SCHEDULES else SCH.DEFAULT_PLAN
        else:
            plan = next((c for c in cands if tally(c)[2] <= CAP + 1e-9), cands[0])
    else:
        plan = sel if sel in ('S1', 'S3', 'S3C') else 'S4'
        if item not in CEIL:
            why.append(f'unknown item {item}')
        else:
            if abs(float(ceiling) - CEIL[item]) > 1e-9:
                why.append(f'ceiling {ceiling} != registered {CEIL[item]}')
            ok, msg = cert_ok(cert, item) if cert else (False, 'no cost certificate')
            if not ok:
                why.append(msg)
        ok, msg = EV.verify(evidence) if Path(evidence).exists() else (False, 'no smoke evidence file')
        if not ok:
            why.append(f'smoke evidence: {msg}')
    if item in led['items']:
        why.append(f'{item} already started (a retry needs its failed record and a new admission)')
    smokes_out, remaining, total = tally(plan)
    if total > CAP + 1e-9:
        why.append(f'total {total:.3f} > {CAP}')
    return dict(admit=not why, reasons=why, plan=plan, spent=led['spent'], live=led['live'], smoke_reserved=smokes_out,
                proposed=float(ceiling), remaining=remaining, total=total)


# --------------------------------------------------------------------------------------------- E10 bucket


def k1_paths_extended(k1pol):
    """K1's Phase B per-case state paths (POL-K1 policy histories) extended by the 0.125 level (E10-DESIGN.md section 9):
    every visited 0.25 state gains its half solve at 0.125 (path_solves with fmin = 0.125), and a case that ended
    STEP-UNCONVERGED at (rule, 0.25) continues to (rule, 0.125) (assumed to fail the 0.25 step check: conservative)."""
    out = {}
    for p in k1pol['policy']:
        path = [(h['rule'], float(h['f'])) for h in p.get('history') or [] if 'rule' in h and h.get('value') != 'BUDGET-CAP']
        if p.get('terminal') == 'STEP-UNCONVERGED' and path and path[-1][1] == .25:
            path.append((path[-1][0], .125))
        out[p['case']] = path
    return out


def e10_item(sme, k1pol=None):
    """POL-e10s projection (E10-DESIGN.md section 9): per-solve rates measured in SM-e (seconds / (m x steps) per level,
    node / step scaled to unmeasured rules), applied to K1's 40 Phase B paths extended by the 0.125 level (solve inventory
    of policy_sm.path_solves with fmin = 0.125) plus one tighter solve per anchor case (FI|4, FI|5, FC1|4, FC1|5) at its
    extended terminal state; times SM-e's whole-step overhead (step wall clock / charged solve seconds: setup, burn-in,
    compilation, certificates, controls) and 1.5; certified against 90 % of 0.8 GPU-h. The SM-e contract (control
    fired, all four factors, consistency, replay) must pass."""
    if not sme:
        return dict(projected=None, certified=False, basis='no SM-e result')
    chk = EV.check_e10_smoke(sme)
    if not all(chk.values()):
        return dict(projected=None, certified=False, basis='SM-e fails its contract', checks=chk,
                    failed=[k for k, v in chk.items() if not v])
    if not k1pol:
        return dict(projected=None, certified=False, basis="K1's Phase B policy result (POL-K1) unavailable")
    if tuple(p['case'] for p in k1pol['policy']) != SCH.POLICY_IDS:
        return dict(projected=None, certified=False, basis='POL-K1 cases differ from the 40 registered')
    R = Rates([dict(x, src='SM-e') for x in sme['rows']])
    solve_s = sum(x['seconds'] for x in sme['rows'])
    over = max(1., float(sme['driver_wall_seconds']) / solve_s) if solve_s > 0 else None
    if over is None or not (R.ok('prod', True) and R.ok('tight', False)):
        return dict(projected=None, certified=False, basis='SM-e did not measure prod / tight solves')
    paths = k1_paths_extended(k1pol)
    per = {c: sum(R.sec(*KIND[k], r, f) for k, r, f in SM.path_solves(pth, .125)) for c, pth in paths.items()}
    anchors = [c for c in ('FI|4', 'FI|5', 'FC1|4', 'FC1|5') if paths.get(c)]
    tighter = sum(R.sec('tighter', False, *paths[c][-1]) for c in anchors)
    sec = (sum(per.values()) + tighter) * over * C_ROM
    proj = sec / 3600.
    lim = SCH.E10_CEIL['policy_e10s'] * SCH.CUTOFF
    return dict(projected=proj, certified=True, fits=bool(proj <= lim + 1e-12), limit=lim, ceiling=SCH.E10_CEIL['policy_e10s'],
                overhead=over, solve_seconds=sum(per.values()), tighter_seconds=tighter, n_cases=len(per), anchors=anchors,
                extended_cases=sum(1 for c, pth in paths.items() if pth and pth[-1][1] == .125), control_fired=True,
                checks=chk, basis="SM-e rates on K1's Phase B paths extended by 0.125 + anchors, x SM-e overhead x 1.5")


def admit_e10(ledger, cert, item, ceiling, evidence=E10_EVIDENCE):
    """E10 bucket (cap 10 GPU-h; ledger jobs with bucket 'E10'). SM-e needs only its registered ceiling. POL-e10s needs
    the certificate's e10 projection certified and <= 90 % of 0.8 GPU-h (recomputed here, not trusted as a flag) and
    passing hash-bound SM-e evidence (evidence.verify_e10: control fired, all four factors, consistency, replay).
    spent + live + outstanding SM-e + proposed + remaining E10 items <= 10."""
    why = []
    try:
        led = ledger_state(ledger, 'E10')
    except LedgerError as e:
        return dict(admit=False, reasons=[f'ledger: {e}'])
    reg = dict(SCH.E10_CEIL, **SCH.E10_SMOKE)
    if abs(float(ceiling) - reg[item]) > 1e-9:
        why.append(f'ceiling {ceiling} != registered {reg[item]}')
    if item in SCH.E10_CEIL:
        e = (cert or {}).get('e10') or {}
        pj, lim = e.get('projected'), SCH.E10_CEIL[item] * SCH.CUTOFF
        if not (e.get('certified') is True and isinstance(pj, (int, float)) and not isinstance(pj, bool) and pj == pj
                and 0 < pj <= lim + 1e-12):
            why.append(f'E10-s projection {pj!r} not certified within 90 % of {SCH.E10_CEIL[item]} GPU-h')
        ok, msg = EV.verify_e10(evidence) if Path(evidence).exists() else (False, 'no SM-e evidence file')
        if not ok:
            why.append(f'SM-e evidence: {msg}')
    if item in led['items']:
        why.append(f'{item} already started')
    so = sum(v for k, v in SCH.E10_SMOKE.items() if k not in led['items'] and k != item)
    rem = sum(v for k, v in SCH.E10_CEIL.items() if k not in led['items'] and k != item)
    total = led['spent'] + led['live'] + so + float(ceiling) + rem
    if total > SCH.E10_CAP + 1e-9:
        why.append(f'E10 total {total:.3f} > {SCH.E10_CAP}')
    return dict(admit=not why, reasons=why, bucket='E10', spent=led['spent'], live=led['live'], total=total)


# --------------------------------------------------------------------------------------------- self-test
def selftest():
    def tim(job, inc_s=600., empty=False, **over):
        # two timed 576-trajectory populations; production counts 1152 (seed 0 in Stage A + B) and 4032 (seed 1000)
        pops = {} if empty else {
            'seed0': dict(trajectories=576, production_trajectories=1152, incumbent_s=100., arm_s=100. + inc_s / 2 * 576 / 1152),
            'seed1000': dict(trajectories=576, production_trajectories=4032, incumbent_s=100., arm_s=100. + inc_s / 2 * 576 / 4032)}
        r = dict(schema='imp-fronts-train-timing/3', job=job, arm='fid' if job == 'SM-a' else 'cov', backend='gpu',
                 device='NVIDIA A100 80GB PCIe', all_done=True, gates_passed=True, chaining_completed=True, burn_in=True,
                 timing=dict(populations=pops))
        r.update(over)
        return r

    def rows(rate):
        out = []
        for lv, rec, rule, f in (('prod', True, 'fib4181', 1.), ('prod', True, 'fib6765', .5), ('tight', False, 'fib4181', 1.),
                                 ('tight', False, 'fib6765', .5), ('prod', True, 'fib6765', .5)):
            out.append(dict(level=lv, recorded=rec, rule=rule, m=M[rule], steps=steps(f), seconds=rate * M[rule] * steps(f),
                            first_of_shape=False))
        out.append(dict(level='prod', recorded=True, rule='fib4181', m=4181, steps=50, seconds=rate * 4181 * 50 + 1.,
                        first_of_shape=True))
        return out

    k1 = json.loads(EV.K1_BANK.read_text())

    def dryres(per_case, ids=SCH.DEVVAL_IDS, rounds=10, term=None, complete=True, start='gauss160', fmin=.25, outcome=None,
               attempt='smkdryK1', factors=None):
        """A synthetic dry-run result produced by the real state machine: case 0 escalates once, case 1 halves once,
        the rest converge in round 1; every solve gets a row; per-case cost scaled to per_case seconds."""
        def outcome0(cid, rule, f):
            i = list(ids).index(cid)
            if i == 0 and rule == start:
                return 'q2bad'
            if i == 1 and f == 1.:
                return 'stepbad'
            return 'ok'
        outcome = outcome or outcome0
        rows, cases = [], [SM.Case(c, start, fmin) for c in ids]

        def ex(reqs):
            out = {}
            for q in reqs:
                o = outcome(q['cid'], q['state_rule'], q['state_f'])
                rec = dict(valid=True)
                if q['kind'] == 'succ':
                    rec.update(dist_succ=1e-3 if o == 'q2bad' else 1e-6, rho_succ_max=.01)
                if q['kind'] == 'half':
                    rec.update(dist_half=1e-3 if o == 'stepbad' else 1e-6)
                if q['kind'] == 'tight':
                    rec.update(dist_tight=1e-6)
                out[q['key']] = rec
                rows.append(dict(case=q['cid'], kind=q['kind'], rule=q['rule'], factor=q['f'],
                                 level=('tight' if q['kind'] == 'tight' else 'prod'), recorded=q['kind'] != 'tight',
                                 m=M[q['rule']], steps=steps(q['f']), valid=True, first_of_shape=False, seconds=1.))
            return out
        store = SM.drive(cases, ex, max_rounds=rounds)
        n_per = {c: sum(1 for x in rows if x['case'] == c) for c in ids}
        for x in rows:
            x['seconds'] = per_case / n_per[x['case']]
        pol = [c.as_dict() for c in cases]
        if term:
            for p_ in pol:
                p_['terminal'] = term
        return dict(complete=complete, backend='gpu', precision='highest',
                    bank=dict(ckpt_sha256=k1['ckpt_sha256'], rotation_sha256=k1['rotation_sha256']),
                    config=dict(attempt=attempt, cases=list(ids), selected=start, max_rounds=rounds,
                                **({'factors': factors} if factors else {})),
                    policy=pol, counts={}, rows=rows, records=store, driver_wall_seconds=per_case * len(ids) * 1.25)

    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / 'refA' / 'output' / 'refjob_x' / '2d' / 'C1' / 'ST'
        d.mkdir(parents=True)
        for i, (sx, r) in enumerate(((100., False), (200., True))):
            (d / f'c{i:03d}.json').write_text(json.dumps(dict(seconds=sx, refined=r, job=dict(job_id='1', gpu='NVIDIA H200'))))
        T = lambda **k: {'SM-a': k.get('a', tim('SM-a')), 'SM-b': k.get('b', tim('SM-b'))}
        D = dryres(10.)
        # (1) everything cheap -> S1 passes its items but is NOT fundable (15.75) -> S3 selected
        c = certify(T(), rows(1e-9), td, hist_path=None, dry=D)
        assert c['schedules']['S1']['all_fit'] and not c['schedules']['S1']['fundable'] and c['selected'] == 'S3', c['selected']
        it = c['schedules']['S3']['items']
        assert abs(it['train_b']['projected'] - (15018 + 600) / 3600) < 1e-12 and abs(it['train_b']['limit'] - (4.6 - 120 / 3600)) < 1e-12
        assert abs(it['references']['per_case_seconds'] - 225.) < 1e-9 and it['references']['n_cases'] == 24
        pk = it['policy_K1']
        assert pk['n_cases'] == 40 and abs(pk['per_case_seconds'] - 12.5) < 1e-12                # wall clock, not solves
        assert abs(pk['projected'] - (12.5 * 40 * 1.5 + pk['tighter_seconds']) / 3600) < 1e-12, (pk['projected'], pk['tighter_seconds'])
        assert abs(c['schedules']['S3']['ceiling_total'] - 10.05) < 1e-9 and abs(c['schedules']['S1']['ceiling_total'] - 15.75) < 1e-9
        # (2) training between the 4.567 h cutoff and the 4.6 h ceiling (inc 1500 s -> 4.588 h) fails -> S4
        c = certify(T(b=tim('SM-b', inc_s=1500.)), rows(1e-9), td, hist_path=None, dry=D)
        assert c['schedules']['S3']['over_cutoff'] == ['train_b'] and c['selected'] == 'S4'
        assert certify(T(b=tim('SM-b', inc_s=-100.)), rows(1e-9), td, hist_path=None, dry=D)['schedules']['S3']['items']['train_b']['projected'] == 15018 / 3600
        # (3) empty / missing / failed / wrong-GPU / wrong-arm / no burn-in timing records -> uncertified, never zero
        for tt in (T(b=tim('SM-b', empty=True)), {'SM-a': tim('SM-a')}, T(b=tim('SM-b', all_done=False)),
                   T(b=tim('SM-b', gates_passed=False)), T(b=tim('SM-b', device='NVIDIA H200')), T(b=tim('SM-b', arm='fid')),
                   T(b=tim('SM-b', burn_in=False)), T(b=tim('SM-b', schema='imp-fronts-timing/2')),
                   T(b=tim('SM-b', timing=dict(populations=dict(x=dict(trajectories=1, production_trajectories=5184,
                                                                     incumbent_s=1., arm_s=1.))))),
                   T(b=tim('SM-b', timing=dict(populations=dict(seed0=dict(trajectories=576, production_trajectories=576,
                                                                         incumbent_s=1., arm_s=1.),
                                                                seed1000=dict(trajectories=576, production_trajectories=4032,
                                                                              incumbent_s=1., arm_s=float('nan'))))))):
            c = certify(tt, rows(1e-9), td, hist_path=None, dry=D)
            assert 'train_b' in c['schedules']['S3']['uncertified'] and c['selected'] == 'S4'
        # (4) 90 % cutoff: references above 90 % of 3.25 fail (24 x p x 1.5 = 3.06 > 2.925 -> p = 306 s from the smoke dev mean)
        tb = tim('SM-b')
        c = certify({'SM-a': tim('SM-a'), 'SM-b': tb, 'SM-r': dict(references_status='complete',
                                                                   references=[dict(case='FC1|0', seconds=306.), dict(case='FC2|0', seconds=306.)])},
                    rows(1e-9), td, hist_path=None, dry=D)
        assert c['schedules']['S3']['over_cutoff'] == ['references'] and c['selected'] == 'S4'
        # (4b) incomplete / missing smoke references -> historical bound, flagged (not silent)
        for smr_t, flag in ((dict(references_status='INCOMPLETE', references=[dict(case='FC1|0', seconds=900.)]), 'INCOMPLETE'),
                            (None, 'MISSING')):
            tt = dict(T(), **({'SM-r': smr_t} if smr_t else {}))
            rf = certify(tt, rows(1e-9), td, hist_path=None, dry=D)['schedules']['S3']['items']['references']
            assert rf['smoke_check'].startswith(flag) and abs(rf['per_case_seconds'] - 225.) < 1e-9
        # (5) the dry-run contract: wrong ids / smoke round cap / open case / incomplete -> policy uncertified
        for bad in (dryres(10., ids=SCH.DEVVAL_IDS[:1]), dryres(10., rounds=1), dryres(10., term='ROUND-CAP'),
                    dryres(10., term='BUDGET-CAP'), dryres(10., complete=False), None):
            c = certify(T(), rows(1e-9), td, hist_path=None, dry=bad)
            assert 'policy_K1' in c['schedules']['S3']['uncertified'] and c['selected'] == 'S4'
        # (5b) audit B-3: skipped tight solves without rows, invented terminal labels, a decision the state machine would
        # not take -> the dry run fails the evidence contract
        assert all(EV.check_dry(D).values()), EV.check_dry(D)
        t1 = json.loads(json.dumps(D))
        for k in t1['records']:
            if '|tight|' in k:
                t1['records'][k] = dict(skipped='x', valid=None)
        t1['rows'] = [x for x in t1['rows'] if x['kind'] != 'tight']
        t2 = json.loads(json.dumps(D))
        t2['policy'][3]['terminal'] = 'SUPER-CONVERGED'
        t3 = json.loads(json.dumps(D))
        t3['policy'][2]['terminal'] = 'STEP-UNCONVERGED'
        t3['policy'][2]['history'][-1]['value'] = 'STEP-UNCONVERGED'
        t4 = json.loads(json.dumps(D))
        t4['records'][t4['policy'][0]['history'][0]['succ']]['dist_succ'] = 1e-9    # evidence says Q2 passed: no escalation
        for t_ in (t1, t2, t3, t4):
            assert not all(EV.check_dry(t_).values())
        # (6) slow dry run (28 s solves / case -> 35 s wall / case: 0.58 GPU-h > 0.45 = 90 % of 0.5) -> over the cutoff
        c = certify(T(), rows(1e-9), td, hist_path=None, dry=dryres(28.))
        assert 'policy_K1' in c['schedules']['S3']['over_cutoff'] and c['selected'] == 'S4'
        # (7) pooled rates take the larger source; scaling; tighter = tight
        slow = [dict(x, seconds=10 * x['seconds'], src='hist') for x in rows(1e-9)]
        assert abs(Rates(rows(1e-9) + slow).sec('prod', True, 'fib4181', 1.) - 10 * Rates(rows(1e-9)).sec('prod', True, 'fib4181', 1.)) < 1e-12
        R = Rates(rows(1e-9))
        assert abs(R.sec('prod', True, 'gauss256', .25) - 1e-9 * M['gauss256'] * 200) < 1e-12
        assert R.sec('tighter', False, 'fib4181', 1.) == R.sec('tight', False, 'fib4181', 1.)
        if E2S1.exists():
            hr, h = e2s1_rows()
            assert h['elapsed_R384'] > 0 and any(x['level'] == 'tight' for x in hr)
        # (8) admission
        c = certify(T(), rows(1e-9), td, hist_path=None, dry=D)
        assert c['selected'] == 'S3'
        nofile = Path(td) / 'none.json'
        assert not admit(None, None, 'train_b', 4.6, nofile)['admit']                     # no certificate
        assert not admit(None, dict(selected='S4'), 'train_b', 4.6, nofile)['admit']      # S4
        assert admit(None, None, 'SM-b', .4)['admit'] and admit(None, None, 'SM-r', .4)['admit']
        assert not admit(None, None, 'SM-a', .2)['admit']                                  # SM-a only under S1
        r = admit(None, c, 'train_b', 4.6, nofile)
        assert not r['admit'] and any('evidence' in x for x in r['reasons'])               # evidence missing
        # evidence: build a passing SM-r output and bind it
        o = Path(td) / 'smr'
        (o / 'fpolicy_smkdryK1').mkdir(parents=True)
        (o / 'fpolicy_smkpolK1').mkdir()
        (o / 'fpolicy_smkdryK1' / 'result.json').write_text(json.dumps(D))
        ctl = dict(D, config=dict(attempt='smkpolK1', cases=['FC1|0', 'FC1|1']),
                   controls=dict(cb1=dict(swapped_bank_json=dict(fired=True), config_hash_mismatch=dict(fired=True),
                                          matched_positive=dict(passed=True)),
                                 CB6=dict(run='FC1|0|cert|fib4181|1', step=10, fixture=dict(agree=True), fired=True, blocked=False)))
        (o / 'fpolicy_smkpolK1' / 'result.json').write_text(json.dumps(ctl))
        ev = EV.build(o)
        assert ev['passed'], ev['checks']
        evp = Path(td) / 'ev.json'
        evp.write_text(json.dumps(ev))
        r = admit(None, c, 'train_b', 4.6, evp)
        assert r['admit'] and abs(r['total'] - 10.05) < 1e-9, r                              # nothing spent: whole plan
        assert not admit(None, dict(selected='S3'), 'train_b', 4.6, evp)['admit']          # bare certificate refused
        bad = json.loads(json.dumps(c, default=sorted))
        bad['schedules']['S3']['items']['train_b']['certified'] = False
        assert not admit(None, bad, 'train_b', 4.6, evp)['admit']                           # item not certified
        bad = json.loads(json.dumps(c, default=sorted))
        bad['schedules']['S3']['fundable'] = False
        assert not admit(None, bad, 'train_b', 4.6, evp)['admit']
        assert not admit(dict(jobs=[dict(name='SM-b', item='SM-b', state='done')]), c, 'train_b', 4.6, evp)['admit']   # no charge
        assert not admit(dict(jobs=[dict(name='SM-b', item='SM-b', state='done', charged_gpu_h=float('nan'))]), c, 'train_b', 4.6, evp)['admit']
        assert not admit(dict(jobs=[dict(name='X', item='SM-r', state='weird')]), None, 'SM-b', .4)['admit']
        big = dict(jobs=[dict(name='SM-r', item='SM-r', state='done', charged_gpu_h=20.)])
        for x in (float('nan'), float('inf'), -1., 0., 'x', None):
            assert not admit(big, None, 'SM-b', x)['admit'] and not admit(None, None, 'SM-b', x)['admit']
        assert not admit(big, None, 'SM-b', .4)['admit']                                    # 20 h spent > cap
        for x in (float('inf'), -1.):
            assert not admit(dict(jobs=[dict(name='J', item='SM-r', state='done', charged_gpu_h=x)]), None, 'SM-b', .4)['admit']
            assert not admit(dict(jobs=[dict(name='J', item='SM-r', state='running', ceiling_gpu_h=x)]), None, 'SM-b', .4)['admit']
        assert not admit(None, c, 'train_a', 4.6, evp)['admit']                             # not in S3
        assert not admit(None, c, 'train_b', 4.4, evp)['admit']                             # wrong ceiling
        # tampering with a bound file breaks the evidence
        ctl['controls']['CB6']['fired'] = False
        (o / 'fpolicy_smkpolK1' / 'result.json').write_text(json.dumps(ctl))
        assert not admit(None, c, 'train_b', 4.6, evp)['admit']
        (o / 'fpolicy_smkpolK1' / 'result.json').write_text(json.dumps(dict(ctl, controls=dict(ctl['controls'], CB6=dict(ctl['controls']['CB6'], fired=True)))))
        # ledger: charges, live ceilings and outstanding smoke reservations count
        led = dict(jobs=[dict(name='SM-b', item='SM-b', ceiling_gpu_h=.4, state='done', charged_gpu_h=.4)])
        r = admit(led, c, 'train_b', 4.6, evp)
        assert r['admit'] and abs(r['smoke_reserved'] - .4) < 1e-12 and abs(r['total'] - 10.05) < 1e-9
        led['jobs'].append(dict(name='T', item='train_b', ceiling_gpu_h=4.6, state='running'))
        assert not admit(led, c, 'train_b', 4.6, evp)['admit']                              # already started
        led['jobs'][0]['charged_gpu_h'] = 6.                                                # failed retries charged
        r = admit(led, c, 'references', 3.25, evp)
        assert not r['admit'] and r['total'] > 15
        # (9) arm (c): training projection from SM-c paired ratios; S3C needs the ledger (S3's measured spend) to fund
        def armc(T_s=15000., **over):
            parts = dict(sub=8000., full=1000., data_A=125., remA=200., B_gen_head=2625., B_extract=1000., rotation=19.,
                         compile=31.)
            k = T_s / sum(parts.values())
            parts = {a_: v * k for a_, v in parts.items()}
            t = dict(schema='imp-fronts-train-timing-c/1', job='SM-c', job_id='1', arm='c', device='NVIDIA A100 80GB PCIe',
                     errors=[], parts_s=parts, projection_s=sum(parts.values()),
                     projection_conservative_s=sum(parts.values()) + parts['B_extract'],
                     certified=sum(parts.values()) + parts['B_extract'] <= 16440., all_done=True,
                     rho=dict(sub=1.1, full=1.2, extract=1.3))
            t.update(over)
            return t
        it = train_c_item(armc())
        assert it['certified'] and abs(it['projection_s'] - 15000.) < 1e-6
        assert not train_c_item(armc(T_s=16500.))['certified']                               # over 4.567 h
        assert not train_c_item(armc(T_s=16000.))['certified']               # T <= 16 440 but extraction x2 is over
        assert not train_c_item(armc(projection_conservative_s=15000.))['certified']        # inconsistent bound
        for bad in (None, armc(errors=['identity tests not passed']), armc(device='NVIDIA H200'), armc(certified=False),
                    armc(schema='imp-fronts-armc-timing/1'), armc(projection_s=1.), armc(parts_s=dict(sub=1.)),
                    armc(all_done=None)):
            assert not train_c_item(bad)['certified']
        TT = dict(T(), **{'SM-c': armc()})
        c = certify(TT, rows(1e-9), td, hist_path=None, dry=D)
        assert not c['schedules']['S3C']['fundable'] and c['selected'] == 'S3'            # ceilings 16 > 15 without a ledger
        led = dict(jobs=[dict(name=n, item=n, state='done', charged_gpu_h=h) for n, h in
                         (('SM-b', .3), ('SM-r', .3), ('train_b', 3.9), ('references', 1.), ('select_b', .3),
                          ('policy_K1', .1), ('policy_b', .1))])
        c = certify(TT, rows(1e-9), td, hist_path=None, dry=D, ledger=led)
        assert c['schedules']['S3C']['fundable'] and c['selected'] == 'S3C', c['schedules']['S3C']['committed']
        r = admit(led, c, 'train_c', 4.6, evp)
        assert r['admit'] and r['plan'] == 'S3C' and abs(r['total'] - (6. + 4.6 + .4 + .5 + .45)) < 1e-9, r
        assert admit(led, None, 'SM-c', .45)['admit']                                        # smoke before a certificate
        assert not admit(None, None, 'SM-c', .45)['admit']                                   # S3C unfundable on ceilings
        # (10) E10 bucket: SM-e result with the control firing at 0.5 and all four factors; separate cap
        def e10(fire=True, rule_ctl='gauss256'):
            # FC1|0 escalates gauss160 -> 192 -> 256 at f = 1 (as SM-r recorded), then fails the step checks from rule_ctl on
            def out(cid, rule, f):
                if cid == 'FC1|0' and rule != rule_ctl and f == 1.:
                    return 'q2bad'
                return 'stepbad' if (fire or f != .5) else 'ok'
            return dryres(10., ids=('FC1|0', 'FC1|1'), rounds=11, fmin=.125, outcome=out, attempt='smke10s',
                          factors=[1., .5, .25, .125])
        sm = e10()
        assert all(EV.check_e10_smoke(sm).values()), EV.check_e10_smoke(sm)
        assert not EV.check_e10_smoke(e10(fire=False))['control_fired']
        assert not EV.check_e10_smoke(e10(rule_ctl='gauss192'))['control_fired']             # fired, but not at the pinned state
        assert not EV.check_e10_smoke(dict(sm, precision='default'))['precision_highest']
        assert not EV.check_e10_smoke(dict(sm, bank=dict(sm['bank'], rotation_sha256='0' * 64)))['bank_is_K1']
        # K1's Phase B policy result: FC1 cases STEP-UNCONVERGED at 0.25 (extended to 0.125), the rest converge
        k1pol = dryres(5., ids=SCH.POLICY_IDS, rounds=10, outcome=lambda cid, r, f: 'stepbad' if cid.startswith('FC1') else 'ok',
                       attempt='pol1')
        assert sum(p_['terminal'] == 'STEP-UNCONVERGED' for p_ in k1pol['policy']) == 12
        paths = k1_paths_extended(k1pol)
        assert paths['FC1|4'][-2:] == [('gauss160', .25), ('gauss160', .125)] and paths['FI|4'] == [('gauss160', 1.)]
        c = certify(TT, rows(1e-9), td, hist_path=None, dry=D, ledger=led, sme=sm, k1pol=k1pol)
        e = c['e10']
        assert e['certified'] and e['fits'] and e['extended_cases'] == 12 and e['anchors'] == ['FI|4', 'FI|5', 'FC1|4', 'FC1|5'], e
        assert e['tighter_seconds'] > 0 and e['overhead'] >= 1.
        assert not certify(TT, rows(1e-9), td, hist_path=None, dry=D, sme=sm)['e10']['certified']   # no K1 paths
        # hash-bound SM-e evidence
        so = Path(td) / 'sme' / 'fpolicy_smke10s'
        so.mkdir(parents=True)
        (so / 'result.json').write_text(json.dumps(sm))
        eve = Path(td) / 'e10ev.json'
        eve.write_text(json.dumps(EV.build_e10(so.parent)))
        assert EV.verify_e10(eve)[0]
        assert admit(None, None, 'SM-e', .3)['admit'] and admit(None, c, 'policy_e10s', .8, e10_evidence=eve)['admit']
        assert not admit(None, c, 'policy_e10s', .8, e10_evidence=Path(td) / 'missing.json')['admit']        # no evidence
        assert not admit(None, dict(c, e10=dict(e, projected=99.)), 'policy_e10s', .8, e10_evidence=eve)['admit']  # audit probe
        assert not admit(None, dict(c, e10=dict(e, projected=float('nan'))), 'policy_e10s', .8, e10_evidence=eve)['admit']
        assert not admit(None, dict(c, e10=e10_item(e10(fire=False), k1pol)), 'policy_e10s', .8, e10_evidence=eve)['admit']
        (so / 'result.json').write_text(json.dumps(e10(fire=False)))                         # tampered bound file
        assert not EV.verify_e10(eve)[0] and not admit(None, c, 'policy_e10s', .8, e10_evidence=eve)['admit']
        (so / 'result.json').write_text(json.dumps(sm))
        e10led = dict(jobs=[dict(name='x', item='SM-e', state='done', charged_gpu_h=9.5, bucket='E10')])
        assert not admit(e10led, c, 'policy_e10s', .8, e10_evidence=eve)['admit']           # 9.5 + 0.8 > E10 cap 10
        assert admit(dict(jobs=led['jobs'] + e10led['jobs']), c, 'train_c', 4.6, evp)['admit']   # E10 charges not in E8
        assert not admit(dict(jobs=[dict(name='y', item='SM-e', state='done', charged_gpu_h=1., bucket='E9')]), None, 'SM-e', .3)['admit']
    print('resources selftest OK')


SMR_OUT = (HERE / 'runs' / 'rom' / 'SM-r' / 'output', HERE / 'runs' / 'SM-r' / 'output')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smr', help='pulled SM-r output dir (default: runs/rom/SM-r/output, else runs/SM-r/output); gives '
                                  'timing.json (references), fpolicy_smkpolK1/result.json (--rom) and '
                                  'fpolicy_smkdryK1/result.json (--dry)')
    ap.add_argument('--rom', action='append', default=[], help='extra smoke fpolicy result.json')
    ap.add_argument('--dry', help='override the dry-run result.json')
    ap.add_argument('--regimes', default=str(REGIMES))
    ap.add_argument('--ref-gpu', default='H200')
    ap.add_argument('--rom-gpu', default='H200', help='GPU type of the SM-r ROM runs (= the policy jobs\' GPU type)')
    ap.add_argument('--history', default=str(E2S1))
    ap.add_argument('--out', default=str(HERE / 'runs' / 'COST-CERT.json'))
    ap.add_argument('--admit', nargs=2)
    ap.add_argument('--evidence', default=str(EVIDENCE), help='built locally: rom/evidence.py build <SM-r output> --out this')
    ap.add_argument('--train-timing', action='append', default=[], help='SM-a=<path> / SM-b=<path> / SM-c=<timing_c.json>')
    ap.add_argument('--sme', help='SM-e fpolicy result.json (E10-s projection; default runs/rom/SM-e/output/fpolicy_smke10s/result.json)')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    lp = HERE / 'runs' / 'GPU-LEDGER.json'
    ledger = json.loads(lp.read_text()) if lp.exists() else None
    if a.admit:
        cp = Path(a.out)
        cert = json.loads(cp.read_text()) if cp.exists() else None
        r = admit(ledger, cert, a.admit[0], float(a.admit[1]), a.evidence)
        print(json.dumps(r))
        sys.exit(0 if r['admit'] else 3)
    timings = {}
    for j, fn in (('SM-a', 'timing.json'), ('SM-b', 'timing.json'), ('SM-c', 'timing_c.json')):
        p = HERE / 'runs' / j / fn
        if p.exists():
            timings[j] = json.loads(p.read_text())
    for kv in a.train_timing:
        j, p = kv.split('=', 1)
        timings[j] = json.loads(Path(p).read_text())
    smr = Path(a.smr) if a.smr else next((d for d in SMR_OUT if d.exists()), None)
    rom, dryp = list(a.rom), a.dry
    if smr:
        if (smr / 'timing.json').exists():
            timings['SM-r'] = json.loads((smr / 'timing.json').read_text())
        if (smr / 'fpolicy_smkpolK1' / 'result.json').exists():
            rom.append(str(smr / 'fpolicy_smkpolK1' / 'result.json'))
        if not dryp and (smr / 'fpolicy_smkdryK1' / 'result.json').exists():
            dryp = str(smr / 'fpolicy_smkdryK1' / 'result.json')
    rows = []
    for p in rom + ([dryp] if dryp else []):
        r = json.loads(Path(p).read_text())
        assert r.get('backend') == 'gpu' and a.rom_gpu.lower() in str(r.get('gpu', '')).lower(), f'{p}: not a {a.rom_gpu} GPU run'
        rows += [dict(x, src='smoke') for x in r['rows']]
    dry = json.loads(Path(dryp).read_text()) if dryp else None
    smep = Path(a.sme) if a.sme else HERE / 'runs' / 'rom' / 'SM-e' / 'output' / 'fpolicy_smke10s' / 'result.json'
    sme = json.loads(smep.read_text()) if smep.exists() else None
    if sme:
        assert sme.get('backend') == 'gpu' and a.rom_gpu.lower() in str(sme.get('gpu', '')).lower(), 'SM-e not on the ROM GPU'
    k1pol = json.loads(POLK1_RESULT.read_text()) if POLK1_RESULT.exists() else None
    cert = certify(timings, rows, a.regimes, a.ref_gpu, ledger, a.history, dry, sme, k1pol)
    cert['inputs'] = dict(timings=sorted(timings), smr=str(smr) if smr else None, rom=rom, dry=dryp, regimes=a.regimes,
                          history=a.history)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(cert, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else float(o)) + '\n')
    print('selected schedule', cert['selected'])
    for s_, v in cert['schedules'].items():
        print(s_, 'fundable' if v['fundable'] else 'NOT fundable', 'fit' if v['all_fit'] else 'NO', 'over', v['over_cutoff'],
              'uncertified', v['uncertified'],
              {k: (round(x['projected'], 3) if x['projected'] is not None else None) for k, x in v['items'].items()})


if __name__ == '__main__':
    main()
