"""imp-fronts Phase B cost certificate and admission rule (DESIGN B8 / B9 as replaced by Amendments 3, 4, 5; pattern of
jcp-speed3d resources.py). CPU only, no JAX.

Inputs
  runs/SM-a/timing.json, runs/SM-b/timing.json   per-phase training timings (written by the training driver), schema
      {"schema": "imp-fronts-timing/1", "job": "SM-a", "job_id": str, "gpu": "NVIDIA A100 80GB PCIe", "arm": "a"|"b",
       "phases": {"datagen" | "stageA" | "stageB" | "rotation":
                    {"fixed_seconds": float,                         (setup / compile, charged once)
                     "segments": [{"name": str, "seconds": float,    (measured wall seconds of the segment)
                                   "units": float,                    (work measured: trajectories / steps / ...)
                                   "production_units": float}]}},     (the same work at production size)
       "references": [{"case": "FC1|0", "seconds": float, "refined": bool}]   (SM-b only: the dev references)}
      A phase projects to fixed_seconds + sum(seconds * production_units / units). A missing phase -> the arm's training
      item is UNCERTIFIED.
  ROM timings: the fpolicy result.json files of the SM-b policy runs (--rom, repeatable): rows with level ('prod',
      'tight', 'tighter'), recorded, m (nodes), steps, seconds, first_of_shape, it_mean.
  jcp-regimes reference records (--regimes): per-case JSONs runs/ref*/output/refjob_*/2d/*/ST/c*.json with
      seconds, refined, job.gpu.
  runs/GPU-LEDGER.json: {"jobs": [{"name", "item", "ceiling_gpu_h", "state": pending|running|done|failed,
      "charged_gpu_h"}], ...}.

Projection rules (Amendments 3-5 as replaced by Amendments 9 and 10)
  * training (a), (b): sum of the four phases x 1.25 (contingency); ceiling 4.4 each.
  * references: 24 test cases (FI 6, FN0s 8, FC1 8, FC2 2) x (MEAN measured per-case seconds of the jcp-regimes
    reference records on the reference GPU type, H200; never below the smoke dev-reference mean) x 1.5; ceiling 2.9
    GPU-h on 2 GPUs (1.45 h wall).
  * ROM solve rates (selection, worst-path bound, dry-run expectation): seconds / (m * steps) per (level, recorded, rule)
    from the jcp-robust e2s1 selection job pooled with the smoke's solves (larger rate per key), node / step scaling for
    unmeasured rules, tighter = tight, + compile per shape; x 1.5.
  * selection (a), (b): R384 group only; max(solve inventory, e2s1 measured R384 group + setup) x 1.5; ceiling 0.35 each.
  * policy per bank (K1, (a), (b)), 40 cases (24 test + 16 dev/val): the smoke's reference-free dry run on the 16 fresh
    dev/val cases (--dry, K1) gives the measured mean per-case path cost (all charged solve seconds of a case, compile
    included); projection = mean x 40 x 1.5; ceiling 0.5 each. The worst-path bound (every case on the most expensive
    state path) is reported beside it. No dry run, or a dry run with BUDGET-CAP cases -> UNCERTIFIED.
  * an item with no measured or recorded bound is UNCERTIFIED. The smoke ceiling (0.9) is reserved until the ledger
    records the smoke jobs (item 'smoke').
Schedules (Amendment 10): S1 both arms (ceilings 13.9 + smoke 0.9 = 14.8), S3 arm (b) + K1 (8.65 + 0.9 = 9.55), S4 no
Phase B; S2 withdrawn. The first schedule whose every item is certified and projected <= its ceiling, and whose
ceilings + spent + smoke reservation <= 15, is selected. COST-CERT.json records every item of every schedule and the
expected cost of the SM-b dry run at the pooled rates (dry_run_expected).
Admission (Amendment 5): submit a job only if spent + ceilings of running / pending jobs + its ceiling + ceilings of the
remaining scheduled items <= 15.

    python resources.py [--rom <result.json> ...] [--regimes <dir>] [--ref-gpu H200] [--out runs/COST-CERT.json]
    python resources.py --admit <item> <ceiling GPU-h>         (exit 0 admit, 3 refuse)
    python resources.py --selftest
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'rom'))
import policy_sm as SM  # noqa: E402

CAP = 15.
REGIMES = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-regimes/experiments/jcp-regimes/runs')
CEIL = {'train_a': 4.4, 'train_b': 4.4, 'references': 2.9, 'select_a': .35, 'select_b': .35,     # Amendment 10
        'policy_K1': .5, 'policy_a': .5, 'policy_b': .5}
SMOKE_CEIL = .9
C_TRAIN, C_REF, C_ROM = 1.25, 1.5, 1.5
E2S1 = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-19-jcp-robust/experiments/jcp-robust/runs/e2s1/archive/output/r2acc_e2s1/result.json')
GAUSS = lambda p: p * p
M = {'fib4181': 4181, 'fib6765': 6765, 'gauss96': GAUSS(96), 'fib10946': 10946, 'fib17711': 17711, 'gauss128': GAUSS(128),
     'gauss160': GAUSS(160), 'gauss192': GAUSS(192), 'gauss256': GAUSS(256), 'gauss320': GAUSS(320), 'gauss384': GAUSS(384)}
TEST = dict(FI=6, FN0s=8, FC1=8, FC2=2)       # Amendment 10: 24 test references (lowest test indices), S1 and S3
SCHEDULES = ('S1', 'S3', 'S4')                 # Amendment 10: S2 withdrawn
DEVVAL = 16                         # 4 dev/val cases per fresh cohort
ANCHORS = 4
K1_START = 'gauss160'
CANDIDATES = [('fib4181', 'fib6765'), ('fib6765', 'fib10946'), ('gauss96', 'gauss128'), ('fib10946', 'fib17711'),
              ('gauss128', 'gauss160'), ('gauss160', 'gauss192'), ('gauss192', 'gauss256')]


def steps(f):
    return int(round(.25 / (.005 * f)))


# --------------------------------------------------------------------------------------------- training
def phase_seconds(ph):
    s = float(ph.get('fixed_seconds', 0.))
    for g in ph.get('segments', []):
        if not (g['units'] > 0 and g['seconds'] >= 0 and g['production_units'] >= 0):
            raise ValueError(f'bad segment {g}')
        s += g['seconds'] * g['production_units'] / g['units']
    return s


def train_item(timing):
    if timing is None:
        return dict(projected=None, certified=False, basis='no smoke timing')
    miss = [p for p in ('datagen', 'stageA', 'stageB', 'rotation') if p not in timing.get('phases', {})]
    if miss:
        return dict(projected=None, certified=False, basis=f'missing phases {miss}')
    per = {p: phase_seconds(timing['phases'][p]) / 3600. for p in ('datagen', 'stageA', 'stageB', 'rotation')}
    return dict(projected=sum(per.values()) * C_TRAIN, certified=True, phases_gpu_h=per, contingency=C_TRAIN,
                gpu=timing.get('gpu'), basis=f"{timing.get('job')} measured per-phase rates")


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


def ref_item(n, regimes_dir, gpu, smoke_refs):
    m, info = regimes_mean(regimes_dir, gpu)
    sm = [r['seconds'] for r in smoke_refs or []]
    smm = sum(sm) / len(sm) if sm else None
    if m is None:
        return dict(projected=None, certified=False, basis=f'no jcp-regimes reference record on {gpu}', **info)
    per = max(m, smm or 0.) * C_REF
    return dict(projected=n * per / 3600., certified=True, per_case_seconds=per, regimes_mean_seconds=m,
                smoke_dev_mean_seconds=smm, n_cases=n, contingency=C_REF, gpu=gpu, gpus=2, wall_hours=CEIL['references'] / 2,
                **info)


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


def dry_run_mean(dry):
    """Amendment 10: mean per-case path cost (all charged solve seconds of the case, compilation included) of the
    smoke's reference-free policy dry run on the 16 fresh dev/val cases. dry = list of fpolicy result.json dicts."""
    if not dry:
        return None, dict(basis='no smoke policy dry run')
    per, info = {}, dict(runs=[], complete=True)
    for r in dry:
        for x in r['rows']:
            per[(r['config']['attempt'], x['case'])] = per.get((r['config']['attempt'], x['case']), 0.) + x['seconds']
        bc = r.get('counts', {}).get('BUDGET-CAP', 0)
        info['runs'].append(dict(attempt=r['config']['attempt'], cases=len(r['config']['cases']), budget_cap=bc,
                                 complete=bool(r.get('complete')), counts=r.get('counts')))
        info['complete'] = info['complete'] and bool(r.get('complete')) and bc == 0
    return sum(per.values()) / len(per), dict(info, n_case_runs=len(per))


def policy_item(rates, n_cases, starts, dry=None):
    """Amendment 10: projection = smoke dry-run mean per-case path cost x n_cases x 1.5; the worst-path bound (every case
    on the most expensive state path at the pooled rates) is reported beside it. No (complete) dry run -> uncertified."""
    mean, dinfo = dry_run_mean(dry)
    wb = worst_bound(rates, n_cases, starts)
    if mean is None or not dinfo.get('complete'):
        return dict(projected=None, certified=False, basis=dinfo.get('basis', 'dry run incomplete or budget-capped'),
                    dry_run=dinfo, worst_path_bound=wb)
    return dict(projected=mean * n_cases * C_ROM / 3600., certified=True, per_case_seconds=mean, n_cases=n_cases,
                contingency=C_ROM, dry_run=dinfo, worst_path_bound=wb,
                basis='smoke dry-run mean per-case path cost (16 fresh dev/val cases)')


def worst_bound(rates, n_cases, starts):
    if rates is None:
        return dict(projected=None, certified=False, basis='no ROM timing (neither e2s1 history nor smoke)')
    worst, shapes_all = None, set()
    for s in starts:
        c, p, _ = worst_path(rates, s)
        if c is None:
            return dict(projected=None, certified=False, basis='a solve level was never measured')
        if worst is None or c > worst[0]:
            worst = (c, p, s)
        for q in SM.paths(s):
            shapes_all |= {(r, f, KIND[k][1]) for k, r, f in SM.path_solves(q)}
    tighter = ANCHORS * rates.sec('tighter', False, 'gauss256', .25)
    comp = len(shapes_all | {('gauss256', .25, False)}) * rates.compile
    sec = (n_cases * worst[0] + tighter + comp) * C_ROM
    return dict(projected=sec / 3600., certified=True, worst_start=worst[2], worst_path=worst[1],
                per_case_seconds=worst[0], tighter_seconds=tighter, compile_seconds=comp, n_cases=n_cases,
                contingency=C_ROM, basis='worst state path for every case (upper bound)')


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


def dry_run_expected(rates, n=16, start=K1_START):
    """Expected GPU cost of the SM-b dry run (16 dev/val cases, K1 from gauss160) at the pooled rates: every case
    converging in round 1 (lower) to every case on the worst path (upper), plus compilation of the shapes visited."""
    if rates is None:
        return None
    lo, _ = solves_cost(rates, SM.path_solves([(start, 1.)]))
    hi, p, _ = worst_path(rates, start)
    sh_lo = len({(r, f, KIND[k][1]) for k, r, f in SM.path_solves([(start, 1.)])})
    sh_hi = len({(r, f, KIND[k][1]) for q in SM.paths(start) for k, r, f in SM.path_solves(q)})
    return dict(cases=n, start=start, lower_gpu_h=(n * lo + sh_lo * rates.compile) / 3600.,
                upper_gpu_h=(n * hi + sh_hi * rates.compile) / 3600., per_case_lower_s=lo, per_case_upper_s=hi, worst_path=p)


# --------------------------------------------------------------------------------------------- schedules
def schedule_items(name, base):
    if name == 'S4':
        return {}
    n_test = sum(TEST.values())
    items = dict(train_a=dict(base['train_a']), train_b=dict(base['train_b']), references=base['ref'](n_test),
                 select_a=dict(base['select']), select_b=dict(base['select']), policy_K1=base['policy_K1'](n_test + DEVVAL),
                 policy_a=base['policy_arm'](n_test + DEVVAL), policy_b=base['policy_arm'](n_test + DEVVAL))
    if name == 'S3':
        for k in ('train_a', 'select_a', 'policy_a'):
            items.pop(k)
    return items


def certify(timings, rom_rows, regimes_dir=REGIMES, ref_gpu='H200', ledger=None, hist_path=E2S1, dry=None):
    hrows, hist = e2s1_rows(hist_path) if hist_path and Path(hist_path).exists() else ([], None)
    rows = list(hrows) + list(rom_rows or [])
    rates = Rates(rows) if rows else None
    smoke_only = Rates(rom_rows) if rom_rows else None
    smoke_refs = (timings.get('SM-b') or {}).get('references')
    base = dict(train_a=train_item(timings.get('SM-a')), train_b=train_item(timings.get('SM-b')),
                ref=lambda n: ref_item(n, regimes_dir, ref_gpu, smoke_refs), select=selection_item(rates, hist),
                policy_K1=lambda n: policy_item(rates, n, [K1_START], dry),
                policy_arm=lambda n: policy_item(rates, n, [None] + list(SM.CHAIN), dry))
    spent = spent_gpu_h(ledger)
    smoke_reserved = 0. if any(j.get('item') == 'smoke' for j in (ledger or {}).get('jobs', [])) else SMOKE_CEIL
    out = dict(schema='imp-fronts-cost-cert/2', amendment=9, cap=CAP, ceilings=CEIL, smoke_ceiling=SMOKE_CEIL,
               spent_gpu_h=spent, smoke_reserved_gpu_h=smoke_reserved, ref_gpu=ref_gpu,
               history=dict(e2s1=str(hist_path) if hist else None, **(hist or {})), smoke_rom_rows=len(rom_rows or []),
               rates=(rates.basis if rates else None), compile_seconds=(rates.compile if rates else None),
               smoke_rates=(smoke_only.basis if smoke_only else None), schedules={}, selected=None)
    out['dry_run_expected'] = dry_run_expected(rates)
    for s in SCHEDULES:
        items = schedule_items(s, base)
        for k, v in items.items():
            v['ceiling'] = CEIL[k]
            v['fits'] = bool(v['certified'] and v['projected'] is not None and v['projected'] <= CEIL[k])
        ok = all(v['fits'] for v in items.values())
        ceil_sum = sum(CEIL[k] for k in items)
        tot = spent + smoke_reserved + ceil_sum
        out['schedules'][s] = dict(items=items, all_fit=ok, ceiling_sum=ceil_sum, total_with_smoke=tot,
                                   within_cap=bool(tot <= CAP + 1e-9),
                                   projected_sum=sum(v['projected'] or 0. for v in items.values()),
                                   uncertified=[k for k, v in items.items() if not v['certified']],
                                   over_ceiling=[k for k, v in items.items() if v['certified'] and not v['fits']])
        if out['selected'] is None and ok and tot <= CAP + 1e-9:
            out['selected'] = s
    return out


# --------------------------------------------------------------------------------------------- ledger / admission
def spent_gpu_h(ledger):
    if not ledger:
        return 0.
    return float(sum(j.get('charged_gpu_h', 0.) for j in ledger.get('jobs', []) if j.get('state') in ('done', 'failed')))


def admit(ledger, cert, item, ceiling):
    """Amendment 5 B8: spent + ceilings of running / pending jobs + this job's ceiling + ceilings of the remaining
    scheduled items <= 15."""
    jobs = (ledger or {}).get('jobs', [])
    spent = spent_gpu_h(ledger)
    live = sum(float(j['ceiling_gpu_h']) for j in jobs if j.get('state') in ('pending', 'running'))
    started = {j.get('item') for j in jobs if j.get('state') in ('pending', 'running', 'done')}
    sel = (cert or {}).get('selected')
    items = (cert or {}).get('schedules', {}).get(sel, {}).get('items', {}) if sel else {}
    remaining = sum(CEIL[k] for k in items if k not in started and k != item)
    total = spent + live + float(ceiling) + remaining
    return dict(admit=bool(total <= CAP + 1e-9), spent=spent, live=live, proposed=float(ceiling), remaining=remaining,
                total=total, schedule=sel)


# --------------------------------------------------------------------------------------------- self-test
def selftest():
    def tim(job, train_h, ref_s=100.):
        sec = train_h * 3600. / 4
        ph = {p: dict(fixed_seconds=0., segments=[dict(name=p, seconds=sec / 100., units=1., production_units=100.)])
              for p in ('datagen', 'stageA', 'stageB', 'rotation')}
        return dict(schema='imp-fronts-timing/1', job=job, gpu='NVIDIA A100 80GB PCIe', phases=ph,
                    references=[dict(case='FC1|0', seconds=ref_s, refined=False)])

    def rows(rate):
        out = []
        for lv, rec, rule, f in (('prod', True, 'fib4181', 1.), ('prod', True, 'fib6765', .5), ('tight', False, 'fib4181', 1.),
                                 ('tight', False, 'fib6765', .5), ('prod', True, 'fib6765', .5)):
            out.append(dict(level=lv, recorded=rec, rule=rule, m=M[rule], steps=steps(f), seconds=rate * M[rule] * steps(f),
                            first_of_shape=False))
        out.append(dict(level='prod', recorded=True, rule='fib4181', m=4181, steps=50, seconds=rate * 4181 * 50 + 1.,
                        first_of_shape=True))
        return out

    def dryrun(per_case, n=16, capped=0):
        ids = [f'{c}|{j}' for c in ('FI', 'FN0s', 'FC1', 'FC2') for j in range(4)][:n]
        return [dict(config=dict(attempt='smkdryK1', cases=ids), complete=True, counts={'BUDGET-CAP': capped},
                     rows=[dict(case=i, seconds=per_case / 2, level='prod', recorded=True, rule='gauss160', m=M['gauss160'],
                                steps=50, first_of_shape=False) for i in ids for _ in range(2)])]

    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / 'refA' / 'output' / 'refjob_x' / '2d' / 'C1' / 'ST'
        d.mkdir(parents=True)
        for i, (sx, r) in enumerate(((100., False), (200., True))):
            (d / f'c{i:03d}.json').write_text(json.dumps(dict(seconds=sx, refined=r, job=dict(job_id='1', gpu='NVIDIA H200'))))
        A = dict(SMa=tim('SM-a', 3.), SMb=tim('SM-b', 3.))
        T = lambda **k: {'SM-a': k.get('a', A['SMa']), 'SM-b': k.get('b', A['SMb'])}
        D = dryrun(10.)
        # (1) cheap everything, no history -> S1; reference mean (150) x 1.5; policy = 10 s x 40 x 1.5
        c = certify(T(), rows(1e-9), td, hist_path=None, dry=D)
        assert c['selected'] == 'S1', c['schedules']['S1']['over_ceiling']
        it = c['schedules']['S1']['items']
        assert abs(it['train_a']['projected'] - 3.75) < 1e-9 and abs(it['references']['per_case_seconds'] - 225.) < 1e-9
        assert it['references']['n_cases'] == 24 and it['policy_K1']['n_cases'] == 40
        assert abs(it['policy_K1']['projected'] - 10 * 40 * 1.5 / 3600) < 1e-12 and it['policy_K1']['worst_path_bound']['certified']
        assert abs(c['compile_seconds'] - 1.) < 1e-6 and set(c['schedules']) == {'S1', 'S3', 'S4'}
        assert abs(c['schedules']['S1']['total_with_smoke'] - 14.8) < 1e-9          # Amendment 10 S1 total
        assert abs(c['schedules']['S3']['total_with_smoke'] - 9.55) < 1e-9          # Amendment 10 S3 total
        assert c['dry_run_expected']['lower_gpu_h'] <= c['dry_run_expected']['upper_gpu_h']
        # (2) arm (a) training over its ceiling -> S3
        c = certify(T(a=tim('SM-a', 4.)), rows(1e-9), td, hist_path=None, dry=D)
        assert c['schedules']['S1']['over_ceiling'] == ['train_a'] and c['selected'] == 'S3'
        # (3) missing SM-b timing -> S4
        c = certify({'SM-a': A['SMa']}, rows(1e-9), td, hist_path=None, dry=D)
        assert 'train_b' in c['schedules']['S3']['uncertified'] and c['selected'] == 'S4'
        # (4) references over 2.9 h (smoke dev mean 300 s dominates: 24 x 450 s = 3.0 h) -> S4 (S3 has the same refs)
        c = certify(T(b=tim('SM-b', 3., ref_s=300.)), rows(1e-9), td, hist_path=None, dry=D)
        assert c['schedules']['S1']['over_ceiling'] == ['references'] and c['selected'] == 'S4'
        # (5) the larger of two sources is used (pooled rates)
        slow = [dict(x, seconds=10 * x['seconds'], src='hist') for x in rows(1e-9)]
        assert abs(Rates(rows(1e-9) + slow).sec('prod', True, 'fib4181', 1.) - 10 * Rates(rows(1e-9)).sec('prod', True, 'fib4181', 1.)) < 1e-12
        # (6) unmeasured rules scale from the largest measured rule; tighter = tight
        R = Rates(rows(1e-9))
        assert abs(R.sec('prod', True, 'gauss256', .25) - 1e-9 * M['gauss256'] * 200) < 1e-12
        assert R.sec('tighter', False, 'fib4181', 1.) == R.sec('tight', False, 'fib4181', 1.)
        # (7) slow dry run (40 s / case -> 0.67 GPU-h) -> policy over 0.5 -> S4
        c = certify(T(), rows(1e-9), td, hist_path=None, dry=dryrun(40.))
        assert 'policy_K1' in c['schedules']['S1']['over_ceiling'] and c['selected'] == 'S4'
        # (8) no dry run -> policy uncertified (worst-path bound still reported); (9) budget-capped dry run -> uncertified
        for dd in (None, dryrun(10., capped=1)):
            c = certify(T(), rows(1e-9), td, hist_path=None, dry=dd)
            pk = c['schedules']['S1']['items']['policy_K1']
            assert 'policy_K1' in c['schedules']['S1']['uncertified'] and c['selected'] == 'S4' and pk['worst_path_bound']['certified']
        # (10) the real e2s1 history parses (if present)
        if E2S1.exists():
            hr, h = e2s1_rows()
            assert h['elapsed_R384'] > 0 and any(x['level'] == 'tight' for x in hr)
        # (11) admission rule (smoke charged -> no smoke reservation): S1 ceilings 13.9
        led = dict(jobs=[dict(name='SM-a', item='smoke', ceiling_gpu_h=.45, state='done', charged_gpu_h=.4),
                         dict(name='SM-b', item='smoke', ceiling_gpu_h=.45, state='done', charged_gpu_h=.45)])
        c = certify(T(), rows(1e-9), td, hist_path=None, ledger=led, dry=D)
        a = admit(led, c, 'train_a', 4.4)
        assert a['admit'] and abs(a['total'] - (.85 + 13.9)) < 1e-9
        led['jobs'].append(dict(name='X', item='train_b', ceiling_gpu_h=4.4, state='running'))
        led['jobs'][1]['charged_gpu_h'] = 2.2
        assert not admit(led, c, 'train_a', 4.4)['admit']            # 2.6 + 4.4 + 4.4 + 5.1 > 15
    print('resources selftest OK')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rom', action='append', default=[])
    ap.add_argument('--dry', action='append', default=[], help='fpolicy result.json of the SM-b policy dry run(s)')
    ap.add_argument('--regimes', default=str(REGIMES))
    ap.add_argument('--ref-gpu', default='H200')
    ap.add_argument('--rom-gpu', default='A100')
    ap.add_argument('--history', default=str(E2S1))
    ap.add_argument('--out', default=str(HERE / 'runs' / 'COST-CERT.json'))
    ap.add_argument('--admit', nargs=2)
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    lp = HERE / 'runs' / 'GPU-LEDGER.json'
    ledger = json.loads(lp.read_text()) if lp.exists() else None
    if a.admit:
        cert = json.loads(Path(a.out).read_text())
        r = admit(ledger, cert, a.admit[0], float(a.admit[1]))
        print(json.dumps(r))
        sys.exit(0 if r['admit'] else 3)
    timings = {}
    for j in ('SM-a', 'SM-b'):
        p = HERE / 'runs' / j / 'timing.json'
        if p.exists():
            timings[j] = json.loads(p.read_text())
    rows = []
    for p in a.rom:
        r = json.loads(Path(p).read_text())
        assert r.get('backend') == 'gpu' and a.rom_gpu.lower() in str(r.get('gpu', '')).lower(), f'{p}: not a {a.rom_gpu} GPU run'
        rows += [dict(x, src='smoke') for x in r['rows']]
    dry = [json.loads(Path(p).read_text()) for p in a.dry]
    for r in dry:
        assert r.get('backend') == 'gpu' and a.rom_gpu.lower() in str(r.get('gpu', '')).lower(), 'dry run not on the ROM GPU'
        rows += [dict(x, src='smoke-dry') for x in r['rows']]
    cert = certify(timings, rows, a.regimes, a.ref_gpu, ledger, a.history, dry)
    cert['inputs'] = dict(timings=sorted(timings), rom=a.rom, regimes=a.regimes, history=a.history)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(cert, indent=1, default=float) + '\n')
    print('selected schedule', cert['selected'])
    for s_, v in cert['schedules'].items():
        print(s_, 'fit' if v['all_fit'] else 'NO', 'over', v['over_ceiling'], 'uncertified', v['uncertified'],
              {k: (round(x['projected'], 3) if x['projected'] is not None else None) for k, x in v['items'].items()})


if __name__ == '__main__':
    main()
