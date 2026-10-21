"""imp-fronts Phase B cost certificate and admission rule (DESIGN B8 / B9 as replaced by Amendments 3-5 and 9-12;
Amendment 12 governs; pattern of jcp-speed3d resources.py). CPU only, no JAX. Ceilings and schedules come from
rom/schedule.py (shared with mkconfigs_pol.py and fpolicy.py). Projections are EXPECTED costs (Amendment 12): the hard
ceilings and the admission rule enforce spending; nothing guarantees completion.

Inputs
  runs/SM-a/timing.json, runs/SM-b/timing.json   training smoke timings (written by the training driver), schema
      {"schema": "imp-fronts-timing/2", "job": "SM-b", "job_id": str, "gpu": "NVIDIA A100 80GB PCIe", "arm": "a"|"b",
       "phases": {"datagen" | "stageA" | "stageB" | "rotation":
                    {"fixed_seconds": float, "segments": [{"name", "seconds" > 0, "units" > 0, "production_units" > 0}]}},
       "increment": {"arm_datagen_seconds": float,          (the arm's full 576 + 4 032 generation, measured)
                     "incumbent_datagen_seconds": float}}   (the incumbent upwind generation, same GPU type)
      Every phase must be present with at least one segment of positive seconds; anything empty or missing -> the arm's
      training item is UNCERTIFIED (never zero).
  runs/SM-r/SMOKE-EVIDENCE.json (rom/evidence.py build): hash-bound C-B1 / C-B6 / dry-run records; the dry-run
      result.json itself is passed with --dry (the evidence check is re-run on it).
  ROM timing rows: the SM-r fpolicy result.json files (--rom, --dry) and jcp-robust e2s1 history (--history).
  jcp-regimes reference records (--regimes): runs/ref*/output/refjob_*/2d/*/ST/c*.json (seconds, job.gpu).
  runs/GPU-LEDGER.json: {"jobs": [{"name", "item", "ceiling_gpu_h", "state": pending|running|done|failed,
      "charged_gpu_h"}]}; item = a schedule item (train_b, references, ...) or a smoke name (SM-a, SM-b, SM-r).

Projection rules
  * training: K1's measured maximum retrain 4.17 GPU-h + max(0, arm - incumbent data-generation seconds) (Amendment 11);
    certified against 100 % of the 4.6 ceiling (no self-timer). The per-phase extrapolation is reported beside it.
  * references: 24 test cases x (MEAN per-case seconds of the jcp-regimes reference records on H200, never below the
    smoke dev-reference mean) x 1.5; ceiling 3.2 (2 GPUs, 1.6 h wall).
  * selection (R384 group): max(solve inventory at the pooled e2s1 + smoke rates, e2s1 measured R384 group + setup) x 1.5;
    ceiling 0.4.
  * policy per bank (Amendment 12): (mean per-case warm solve cost of the SM-r K1 dry run on the 16 fresh dev/val cases)
    x 40 x 1.5 + measured compilation per shape x shapes the bank's chain can reach + 4 tighter solves at the largest
    reachable rule (gauss256, f = 0.25); ceiling 0.5. The dry run must pass rom/evidence.check_dry (exact 16 ids, full
    policy, complete, no open case, every case charged). The K1 path distribution is ASSUMED to transfer to the arm
    banks (not verifiable in advance; arm jobs keep BUDGET-CAP). The worst-path cost projection is reported beside it.
  * every self-timed item passes only if projected <= 90 % of its ceiling (training: 100 %).
Schedules (rom/schedule.py): S1 15.55 (over the cap -> never fundable), S3 9.85, S4 none; the first fundable schedule
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
K1_MAX_RETRAIN_H = 4.17           # jcp-robust measured K1-recipe retrains 3.69 / 4.17 / 3.76 GPU-h (Amendment 11)
C_TRAIN_DIAG, C_REF, C_ROM = 1.25, 1.5, 1.5
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
def phase_seconds(ph):
    segs = ph.get('segments') or []
    if not segs:
        raise ValueError('phase without measured segments')
    s = float(ph.get('fixed_seconds', 0.))
    for g in segs:
        if not (g['units'] > 0 and g['seconds'] > 0 and g['production_units'] > 0):
            raise ValueError(f'bad segment {g}')
        s += g['seconds'] * g['production_units'] / g['units']
    return s


def train_item(timing):
    if not timing:
        return dict(projected=None, certified=False, basis='no smoke timing')
    phases = timing.get('phases') or {}
    try:
        per = {p: phase_seconds(phases[p]) / 3600. for p in ('datagen', 'stageA', 'stageB', 'rotation')}
    except (KeyError, ValueError, TypeError) as e:
        return dict(projected=None, certified=False, basis=f'training phases missing or empty: {e}')
    inc = timing.get('increment') or {}
    a_, i_ = inc.get('arm_datagen_seconds'), inc.get('incumbent_datagen_seconds')
    if not (isinstance(a_, (int, float)) and isinstance(i_, (int, float)) and a_ > 0 and i_ > 0):
        return dict(projected=None, certified=False, basis='data-generation increment not measured', phases_gpu_h=per)
    return dict(projected=K1_MAX_RETRAIN_H + max(0., a_ - i_) / 3600., certified=True,
                basis=f"K1 max retrain {K1_MAX_RETRAIN_H} GPU-h + measured data-generation increment ({timing.get('job')})",
                increment_gpu_h=max(0., a_ - i_) / 3600., phases_gpu_h=per,
                per_phase_extrapolation_gpu_h=sum(per.values()) * C_TRAIN_DIAG, gpu=timing.get('gpu'))


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


def ref_item(regimes_dir, gpu, smoke_refs):
    n = len(SCH.TEST_IDS)
    m, info = regimes_mean(regimes_dir, gpu)
    sm = [r['seconds'] for r in smoke_refs or [] if r.get('seconds', 0) > 0]
    smm = sum(sm) / len(sm) if sm else None
    if m is None:
        return dict(projected=None, certified=False, basis=f'no jcp-regimes reference record on {gpu}', **info)
    per = max(m, smm or 0.) * C_REF
    return dict(projected=n * per / 3600., certified=True, per_case_seconds=per, regimes_mean_seconds=m,
                smoke_dev_mean_seconds=smm, n_cases=n, contingency=C_REF, gpu=gpu, gpus=2, **info)


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


def reachable_shapes(starts):
    sh = set()
    for s in starts:
        for q in SM.paths(s):
            sh |= {(r, f, KIND[k][1]) for k, r, f in SM.path_solves(q)}
    return sh


def dry_run(dry):
    """Validated dry run -> (mean warm per-case seconds, info) or (None, info)."""
    if not dry:
        return None, dict(basis='no smoke policy dry run')
    chk = EV.check_dry(dry)
    if not all(chk.values()):
        return None, dict(basis='dry run fails the registered contract', failed=[k for k, v in chk.items() if not v])
    per = {}
    for x in dry['rows']:
        if not x.get('first_of_shape'):
            per[x['case']] = per.get(x['case'], 0.) + x['seconds']
    for c in SCH.DEVVAL_IDS:
        per.setdefault(c, 0.)
    return sum(per.values()) / len(per), dict(counts=dry.get('counts'), n_cases=len(per),
                                              per_case_seconds=dict(sorted(per.items())))


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
    n = len(SCH.POLICY_IDS)
    mean, dinfo = dry_run(dry)
    wp = worst_projection(rates, n, starts)
    if mean is None or rates is None:
        return dict(projected=None, certified=False, basis=dinfo.get('basis', 'no ROM timing'), dry_run=dinfo,
                    worst_path_projection=wp)
    shapes = reachable_shapes(starts) | {('gauss256', .25, False)}
    comp = len(shapes) * rates.compile
    tighter = ANCHORS * rates.sec('tighter', False, 'gauss256', .25)
    sec = mean * n * C_ROM + comp + tighter
    return dict(projected=sec / 3600., certified=True, per_case_seconds=mean, n_cases=n, contingency=C_ROM,
                compile_seconds=comp, shapes=len(shapes), tighter_seconds=tighter, dry_run=dinfo, worst_path_projection=wp,
                transfer_assumption=('K1 dry-run path distribution assumed to transfer' if starts != [K1_START] else None),
                basis='Amendment 12: dry-run mean x 40 x 1.5 + compile x reachable shapes + 4 tighter at gauss256 f=.25')


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
def certify(timings, rom_rows, regimes_dir=REGIMES, ref_gpu='H200', ledger=None, hist_path=E2S1, dry=None):
    hrows, hist = e2s1_rows(hist_path) if hist_path and Path(hist_path).exists() else ([], None)
    rows = list(hrows) + list(rom_rows or [])
    rates = Rates(rows) if rows else None
    smoke_refs = (timings.get('SM-r') or timings.get('SM-b') or {}).get('references')
    base = dict(train_a=train_item(timings.get('SM-a')), train_b=train_item(timings.get('SM-b')),
                references=ref_item(regimes_dir, ref_gpu, smoke_refs), select=selection_item(rates, hist),
                policy_K1=policy_item(rates, [K1_START], dry), policy_arm=policy_item(rates, [None] + list(SM.CHAIN), dry))
    pick = dict(train_a='train_a', train_b='train_b', references='references', select_a='select', select_b='select',
                policy_K1='policy_K1', policy_a='policy_arm', policy_b='policy_arm')
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
        out['schedules'][s] = dict(items=items, all_fit=ok, ceiling_total=tot, fundable=bool(tot <= CAP + 1e-9),
                                   projected_sum=sum(v['projected'] or 0. for v in items.values()),
                                   uncertified=[k for k, v in items.items() if not v['certified']],
                                   over_cutoff=[k for k, v in items.items() if v['certified'] and not v['fits']])
        if out['selected'] is None and ok and tot <= CAP + 1e-9:
            out['selected'] = s
    return out


# --------------------------------------------------------------------------------------------- ledger / admission
def ledger_state(ledger):
    jobs = (ledger or {}).get('jobs', [])
    return dict(spent=float(sum(j.get('charged_gpu_h', 0.) for j in jobs if j.get('state') in ('done', 'failed'))),
                live=float(sum(float(j['ceiling_gpu_h']) for j in jobs if j.get('state') in ('pending', 'running'))),
                items={j.get('item') for j in jobs if j.get('state') in ('pending', 'running', 'done')}, n=len(jobs))


def admit(ledger, cert, item, ceiling, evidence=EVIDENCE):
    """Amendments 5 / 12. Smoke items (SM-a / SM-b / SM-r) are admitted against the pre-registered fundable plan (the
    certificate's selected schedule, S3 before any certificate exists) and need no evidence. Production items need: a
    certificate whose selected schedule is S1 or S3, the item in that schedule and not yet started, the registered
    ceiling, passing hash-bound smoke evidence (C-B1, C-B6, dry run). The sum spent (ledger charges) + ceilings of
    running / pending jobs + outstanding smoke reservations of the plan + the proposed ceiling + ceilings of the plan's
    remaining unstarted items must be <= 15."""
    led = ledger_state(ledger)
    sel = (cert or {}).get('selected')
    why = []
    if item in SCH.SMOKE:
        plan = sel if sel in ('S1', 'S3') else 'S3'
        if item not in SCH.SCHEDULES[plan]['smokes']:
            why.append(f'{item} is not a smoke of plan {plan}')
        if abs(float(ceiling) - SCH.SMOKE[item]) > 1e-9:
            why.append(f'ceiling {ceiling} != registered {SCH.SMOKE[item]}')
    else:
        plan = sel
        if not cert:
            why.append('no cost certificate')
        elif sel not in ('S1', 'S3'):
            why.append(f'certificate selects {sel}: no Phase B item is fundable')
        elif item not in SCH.SCHEDULES[sel]['items']:
            why.append(f'{item} is not in schedule {sel}')
        if item in CEIL and abs(float(ceiling) - CEIL[item]) > 1e-9:
            why.append(f'ceiling {ceiling} != registered {CEIL[item]}')
        if item not in CEIL:
            why.append(f'unknown item {item}')
        ok, msg = EV.verify(evidence) if Path(evidence).exists() else (False, 'no smoke evidence file')
        if not ok:
            why.append(f'smoke evidence: {msg}')
    if item in led['items']:
        why.append(f'{item} already started (a retry needs its failed record and a new admission)')
    S = SCH.SCHEDULES.get(plan or 'S4', SCH.SCHEDULES['S4'])
    smokes_out = sum(SCH.SMOKE[k] for k in S['smokes'] if k not in led['items'] and k != item)
    remaining = sum(CEIL[k] for k in S['items'] if k not in led['items'] and k != item)
    total = led['spent'] + led['live'] + smokes_out + float(ceiling) + remaining
    if total > CAP + 1e-9:
        why.append(f'total {total:.3f} > {CAP}')
    return dict(admit=not why, reasons=why, plan=plan, spent=led['spent'], live=led['live'], smoke_reserved=smokes_out,
                proposed=float(ceiling), remaining=remaining, total=total)


# --------------------------------------------------------------------------------------------- self-test
def selftest():
    def tim(job, inc_s=600., empty=False):
        ph = {p: dict(fixed_seconds=0., segments=[] if empty else [dict(name=p, seconds=10., units=1., production_units=100.)])
              for p in ('datagen', 'stageA', 'stageB', 'rotation')}
        return dict(schema='imp-fronts-timing/2', job=job, gpu='NVIDIA A100 80GB PCIe', phases=ph,
                    increment=dict(arm_datagen_seconds=1000. + inc_s, incumbent_datagen_seconds=1000.))

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

    def dryres(per_case, ids=SCH.DEVVAL_IDS, rounds=10, term='CONVERGED', complete=True):
        return dict(complete=complete, backend='gpu', precision='highest',
                    bank=dict(ckpt_sha256=k1['ckpt_sha256'], rotation_sha256=k1['rotation_sha256']),
                    config=dict(attempt='smkdryK1', cases=list(ids), selected='gauss160', max_rounds=rounds),
                    policy=[dict(case=c, terminal=term) for c in ids], counts={term: len(ids)},
                    rows=[dict(case=c, seconds=per_case / 2, valid=True, first_of_shape=False) for c in ids for _ in range(2)])

    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / 'refA' / 'output' / 'refjob_x' / '2d' / 'C1' / 'ST'
        d.mkdir(parents=True)
        for i, (sx, r) in enumerate(((100., False), (200., True))):
            (d / f'c{i:03d}.json').write_text(json.dumps(dict(seconds=sx, refined=r, job=dict(job_id='1', gpu='NVIDIA H200'))))
        T = lambda **k: {'SM-a': k.get('a', tim('SM-a')), 'SM-b': k.get('b', tim('SM-b'))}
        D = dryres(10.)
        # (1) everything cheap -> S1 passes its items but is NOT fundable (15.55) -> S3 selected
        c = certify(T(), rows(1e-9), td, hist_path=None, dry=D)
        assert c['schedules']['S1']['all_fit'] and not c['schedules']['S1']['fundable'] and c['selected'] == 'S3', c['selected']
        it = c['schedules']['S3']['items']
        assert abs(it['train_b']['projected'] - (4.17 + 600 / 3600)) < 1e-12 and it['train_b']['cutoff'] == 1.
        assert abs(it['references']['per_case_seconds'] - 225.) < 1e-9 and it['references']['n_cases'] == 24
        pk = it['policy_K1']
        assert pk['n_cases'] == 40 and abs(pk['projected'] - (10 * 40 * 1.5 + pk['compile_seconds'] + pk['tighter_seconds']) / 3600) < 1e-12
        assert abs(c['schedules']['S3']['ceiling_total'] - 9.85) < 1e-9 and abs(c['schedules']['S1']['ceiling_total'] - 15.55) < 1e-9
        # (2) training increment pushes the arm over 4.6 (100 % cutoff) -> S4
        c = certify(T(b=tim('SM-b', inc_s=1600.)), rows(1e-9), td, hist_path=None, dry=D)
        assert c['schedules']['S3']['over_cutoff'] == ['train_b'] and c['selected'] == 'S4'
        # (3) empty phases or a missing timing -> uncertified, never zero
        for tt in (T(b=tim('SM-b', empty=True)), {'SM-a': tim('SM-a')}):
            c = certify(tt, rows(1e-9), td, hist_path=None, dry=D)
            assert 'train_b' in c['schedules']['S3']['uncertified'] and c['selected'] == 'S4'
        # (4) 90 % cutoff: references at 95 % of 3.2 fail (24 x p x 1.5 = 3.04 -> p = 304 s from the smoke dev mean)
        tb = tim('SM-b')
        c = certify({'SM-a': tim('SM-a'), 'SM-b': tb, 'SM-r': dict(references=[dict(case='FC1|0', seconds=304.)])},
                    rows(1e-9), td, hist_path=None, dry=D)
        assert c['schedules']['S3']['over_cutoff'] == ['references'] and c['selected'] == 'S4'
        # (5) the dry-run contract: wrong ids / smoke round cap / open case / incomplete -> policy uncertified
        for bad in (dryres(10., ids=SCH.DEVVAL_IDS[:1]), dryres(10., rounds=1), dryres(10., term='ROUND-CAP'),
                    dryres(10., term='BUDGET-CAP'), dryres(10., complete=False), None):
            c = certify(T(), rows(1e-9), td, hist_path=None, dry=bad)
            assert 'policy_K1' in c['schedules']['S3']['uncertified'] and c['selected'] == 'S4'
        # (6) slow dry run (28 s / case: 0.467 GPU-h > 0.45 = 90 % of 0.5) -> over the cutoff
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
        assert admit(None, None, 'SM-b', .25)['admit'] and admit(None, None, 'SM-r', .4)['admit']
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
        assert r['admit'] and abs(r['total'] - 9.85) < 1e-9, r                              # nothing spent: whole plan
        assert not admit(None, c, 'train_a', 4.6, evp)['admit']                             # not in S3
        assert not admit(None, c, 'train_b', 4.4, evp)['admit']                             # wrong ceiling
        # tampering with a bound file breaks the evidence
        ctl['controls']['CB6']['fired'] = False
        (o / 'fpolicy_smkpolK1' / 'result.json').write_text(json.dumps(ctl))
        assert not admit(None, c, 'train_b', 4.6, evp)['admit']
        (o / 'fpolicy_smkpolK1' / 'result.json').write_text(json.dumps(dict(ctl, controls=dict(ctl['controls'], CB6=dict(ctl['controls']['CB6'], fired=True)))))
        # ledger: charges, live ceilings and outstanding smoke reservations count
        led = dict(jobs=[dict(name='SM-b', item='SM-b', ceiling_gpu_h=.25, state='done', charged_gpu_h=.25)])
        r = admit(led, c, 'train_b', 4.6, evp)
        assert r['admit'] and abs(r['smoke_reserved'] - .4) < 1e-12 and abs(r['total'] - 9.85) < 1e-9
        led['jobs'].append(dict(name='T', item='train_b', ceiling_gpu_h=4.6, state='running'))
        assert not admit(led, c, 'train_b', 4.6, evp)['admit']                              # already started
        led['jobs'][0]['charged_gpu_h'] = 6.                                                # failed retries charged
        r = admit(led, c, 'references', 3.2, evp)
        assert not r['admit'] and r['total'] > 15
    print('resources selftest OK')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rom', action='append', default=[], help='SM-r fpolicy result.json (smkpolK1)')
    ap.add_argument('--dry', help='SM-r dry-run fpolicy result.json (smkdryK1)')
    ap.add_argument('--regimes', default=str(REGIMES))
    ap.add_argument('--ref-gpu', default='H200')
    ap.add_argument('--rom-gpu', default='H200', help='GPU type of the SM-r ROM runs (= the policy jobs\' GPU type)')
    ap.add_argument('--history', default=str(E2S1))
    ap.add_argument('--out', default=str(HERE / 'runs' / 'COST-CERT.json'))
    ap.add_argument('--admit', nargs=2)
    ap.add_argument('--evidence', default=str(EVIDENCE))
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
    for j in ('SM-a', 'SM-b', 'SM-r'):          # SM-r/timing.json: {"references": [{"case", "seconds", "refined"}]}
        p = HERE / 'runs' / j / 'timing.json'
        if p.exists():
            timings[j] = json.loads(p.read_text())
    rows = []
    for p in a.rom + ([a.dry] if a.dry else []):
        r = json.loads(Path(p).read_text())
        assert r.get('backend') == 'gpu' and a.rom_gpu.lower() in str(r.get('gpu', '')).lower(), f'{p}: not a {a.rom_gpu} GPU run'
        rows += [dict(x, src='smoke') for x in r['rows']]
    dry = json.loads(Path(a.dry).read_text()) if a.dry else None
    cert = certify(timings, rows, a.regimes, a.ref_gpu, ledger, a.history, dry)
    cert['inputs'] = dict(timings=sorted(timings), rom=a.rom, dry=a.dry, regimes=a.regimes, history=a.history)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(cert, indent=1, default=lambda o: sorted(o) if isinstance(o, set) else float(o)) + '\n')
    print('selected schedule', cert['selected'])
    for s_, v in cert['schedules'].items():
        print(s_, 'fundable' if v['fundable'] else 'NOT fundable', 'fit' if v['all_fit'] else 'NO', 'over', v['over_cutoff'],
              'uncertified', v['uncertified'],
              {k: (round(x['projected'], 3) if x['projected'] is not None else None) for k, x in v['items'].items()})


if __name__ == '__main__':
    main()
