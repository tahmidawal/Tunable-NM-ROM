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

Projection rules (Amendments 3-5)
  * training (a), (b): sum of the four phases x 1.25 (contingency); ceiling 4.4 each.
  * references: n_test x per-case bound; bound = max per-case seconds of the jcp-regimes reference JOBS that exercised
    refinement, same GPU type, x 1.5 (and never below the smoke's own dev references x 1.5); ceiling 2.8 (2 GPUs).
  * ROM solves: per (level, recorded) the smoke's rate at its largest measured rule and finest factor,
    seconds / (m * steps), scaled to each solve's (m, steps); tight / tighter additionally x (iteration cap / measured
    mean iterations of that level) (Amendment 4 B9); + compile per distinct (rule, factor, recorded) shape at the
    largest measured compile overhead. Policy and selection x 1.5.
  * selection (a), (b): the registered e2s inventory (3 ranks x 2 arms x [38 target gauss384 + 6 check gauss320 + 7 rules
    x (38 prod + 38 tight + 6 tighter + 38 succ)] at factor 1), rates of r = 384 (an upper proxy for r = 128 / 256);
    ceiling 0.45 each.
  * policy per bank (K1, (a), (b)): every case x the most expensive state path of policy_sm.paths from the bank's start
    (K1: gauss160; arms: the worst over every possible start incl. the uncertified fib4181), Amendment 5 inventory
    (prod + tight + succ + half per visited state, promoted halves not re-solved) + 1 tighter per anchor at gauss256 f=.25;
    ceiling 0.4 each.
  * an item with no measured or recorded bound is UNCERTIFIED.
Schedules S1 -> S2 -> S3 -> S4 (Amendment 2 B8): the first whose every item is certified and projected <= its ceiling is
selected; S4 (no Phase B) always qualifies. COST-CERT.json records every item of every schedule.
Admission (Amendment 5): submit a job only if spent + ceilings of running / pending jobs + its ceiling + ceilings of the
remaining scheduled items <= 15.

    python resources.py [--rom <result.json> ...] [--regimes <dir>] [--gpu A100] [--out runs/COST-CERT.json]
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
CEIL = {'train_a': 4.4, 'train_b': 4.4, 'references': 2.8, 'select_a': .45, 'select_b': .45,
        'policy_K1': .4, 'policy_a': .4, 'policy_b': .4}
SMOKE_CEIL = .9
C_TRAIN, C_REF, C_ROM = 1.25, 1.5, 1.5
ITER_CAP = 600                      # LM iteration budget per step of r2acc's queries (QQ.get budget=600)
GAUSS = lambda p: p * p
M = {'fib4181': 4181, 'fib6765': 6765, 'gauss96': GAUSS(96), 'fib10946': 10946, 'fib17711': 17711, 'gauss128': GAUSS(128),
     'gauss160': GAUSS(160), 'gauss192': GAUSS(192), 'gauss256': GAUSS(256), 'gauss320': GAUSS(320), 'gauss384': GAUSS(384)}
TEST = {'S1': dict(FI=8, FN0s=8, FC1=8, FC2=4), 'S2': dict(FI=6, FN0s=8, FC1=8, FC2=2)}
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
def regimes_bound(regimes_dir, gpu):
    """Max per-case seconds in the jcp-regimes reference jobs that exercised refinement, same GPU type."""
    recs = []
    for f in glob.glob(str(Path(regimes_dir) / 'ref*' / 'output' / 'refjob_*' / '2d' / '*' / 'ST' / 'c*.json')):
        d = json.loads(Path(f).read_text())
        recs.append((str(d.get('job', {}).get('job_id')), str(d.get('job', {}).get('gpu')), bool(d.get('refined')), float(d['seconds'])))
    jobs = {j for j, g, r, _ in recs if r and gpu.lower() in g.lower()}
    xs = [s for j, g, r, s in recs if j in jobs and gpu.lower() in g.lower()]
    allg = [s for j, g, r, s in recs if gpu.lower() in g.lower()]
    return (max(xs) if xs else None), dict(jobs_with_refinement=sorted(jobs), n_regimes_cases=len(xs),
                                            mean_all_cases_same_gpu=(sum(allg) / len(allg) if allg else None))


def ref_item(n, regimes_dir, gpu, smoke_refs):
    b, info = regimes_bound(regimes_dir, gpu)
    sm = max((r['seconds'] for r in smoke_refs or []), default=None)
    if b is None:
        return dict(projected=None, certified=False, basis=f'no jcp-regimes refinement record on {gpu}', **info)
    per = max(b, sm or 0.) * C_REF
    return dict(projected=n * per / 3600., certified=True, per_case_seconds=per, regimes_max_seconds=b,
                smoke_dev_max_seconds=sm, n_cases=n, contingency=C_REF, gpu=gpu, **info)


# --------------------------------------------------------------------------------------------- ROM rates
class Rates:
    """Per (level, recorded): seconds / (m * steps) at the largest measured rule and finest factor; tight / tighter x
    (ITER_CAP / measured mean iterations); compile overhead per shape."""

    def __init__(self, rows):
        self.r, self.mult, self.compile, self.basis = {}, {}, 0., {}
        for lv, rec in {(x['level'], bool(x['recorded'])) for x in rows}:
            sel = [x for x in rows if x['level'] == lv and bool(x['recorded']) == rec]
            warm = [x for x in sel if not x['first_of_shape']] or sel
            mx = max(x['m'] for x in warm)
            top = [x for x in warm if x['m'] == mx]
            fin = max(x['steps'] for x in top)
            top = [x for x in top if x['steps'] == fin]
            self.r[(lv, rec)] = max(x['seconds'] / (x['m'] * x['steps']) for x in top)
            self.basis[f'{lv}/{"rec" if rec else "norec"}'] = dict(m=mx, steps=fin, n=len(top), rate=self.r[(lv, rec)])
            if lv in ('tight', 'tighter'):
                im = sum(x['it_mean'] for x in sel) / len(sel)
                self.mult[lv] = ITER_CAP / max(im, 1e-12)
        for x in rows:
            if x['first_of_shape'] and (x['level'], bool(x['recorded'])) in self.r:
                self.compile = max(self.compile, x['seconds'] - self.r[(x['level'], bool(x['recorded']))] * x['m'] * x['steps'])
        if 'tighter' not in self.mult and 'tight' in self.mult:   # tighter never measured: the tight rate and multiplier
            self.r[('tighter', False)], self.mult['tighter'] = self.r.get(('tight', False)), self.mult['tight']

    def ok(self, lv, rec):
        return self.r.get((lv, rec)) is not None

    def sec(self, lv, rec, rule, f):
        return self.r[(lv, rec)] * M[rule] * steps(f) * self.mult.get(lv, 1.)


KIND = {'prod': ('prod', True), 'half': ('prod', True), 'succ': ('prod', True), 'tight': ('tight', False),
        'tighter': ('tighter', False)}


def solves_cost(rates, solves):
    need = {KIND[k] for k, _, _ in solves}
    if not all(rates.ok(*n) for n in need):
        return None, len({(r, f, KIND[k][1]) for k, r, f in solves})
    s = sum(rates.sec(*KIND[k], r, f) for k, r, f in solves)
    shapes = {(r, f, KIND[k][1]) for k, r, f in solves}
    return s, len(shapes)


def worst_path(rates, start):
    best = (-1., None, 0)
    for p in SM.paths(start):
        c, n = solves_cost(rates, SM.path_solves(p))
        if c is None:
            return None, p, n
        if c > best[0]:
            best = (c, p, n)
    return best


def policy_item(rates, n_cases, starts):
    if rates is None:
        return dict(projected=None, certified=False, basis='no smoke ROM timing')
    worst, shapes_all = None, set()
    for s in starts:
        c, p, _ = worst_path(rates, s)
        if c is None:
            return dict(projected=None, certified=False, basis='a solve level was never measured in the smoke')
        if worst is None or c > worst[0]:
            worst = (c, p, s)
        for q in SM.paths(s):
            shapes_all |= {(r, f, KIND[k][1]) for k, r, f in SM.path_solves(q)}
    tighter = ANCHORS * rates.sec('tighter', False, 'gauss256', .25)
    comp = len(shapes_all | {('gauss256', .25, False)}) * rates.compile
    sec = (n_cases * worst[0] + tighter + comp) * C_ROM
    return dict(projected=sec / 3600., certified=True, worst_start=worst[2], worst_path=worst[1],
                per_case_seconds=worst[0], tighter_seconds=tighter, compile_seconds=comp, n_cases=n_cases,
                contingency=C_ROM, multipliers=rates.mult)


def selection_item(rates):
    if rates is None:
        return dict(projected=None, certified=False, basis='no smoke ROM timing')
    sv = []
    for _ in range(3 * 2):                   # ranks x arms (BE, CNR), all at factor 1
        sv += [('succ', 'gauss384', 1.)] * 38 + [('succ', 'gauss320', 1.)] * 6      # unrecorded prod-level solves (proxy: recorded rate)
        for c, s in CANDIDATES:
            sv += [('prod', c, 1.)] * 38 + [('tight', c, 1.)] * 38 + [('tighter', c, 1.)] * 6 + [('succ', s, 1.)] * 38
    c, n = solves_cost(rates, sv)
    if c is None:
        return dict(projected=None, certified=False, basis='a solve level was never measured in the smoke')
    sec = (c + n * rates.compile) * C_ROM
    return dict(projected=sec / 3600., certified=True, solves=len(sv), shapes=n, contingency=C_ROM, multipliers=rates.mult)


# --------------------------------------------------------------------------------------------- schedules
def schedule_items(name, base):
    if name == 'S4':
        return {}
    t = TEST['S1' if name == 'S1' else 'S2']
    n_test = sum(t.values())
    items = dict(train_a=dict(base['train_a']), train_b=dict(base['train_b']), references=base['ref'](n_test),
                 select_a=dict(base['select']), select_b=dict(base['select']), policy_K1=base['policy_K1'](n_test + DEVVAL),
                 policy_a=base['policy_arm'](n_test + DEVVAL), policy_b=base['policy_arm'](n_test + DEVVAL))
    if name == 'S3':
        for k in ('train_a', 'select_a', 'policy_a'):
            items.pop(k)
    return items


def certify(timings, rom_rows, regimes_dir=REGIMES, gpu='A100', ledger=None):
    rates = Rates(rom_rows) if rom_rows else None
    smoke_refs = (timings.get('SM-b') or {}).get('references')
    base = dict(train_a=train_item(timings.get('SM-a')), train_b=train_item(timings.get('SM-b')),
                ref=lambda n: ref_item(n, regimes_dir, gpu, smoke_refs), select=selection_item(rates),
                policy_K1=lambda n: policy_item(rates, n, [K1_START]),
                policy_arm=lambda n: policy_item(rates, n, [None] + list(SM.CHAIN)))
    spent = spent_gpu_h(ledger)
    out = dict(schema='imp-fronts-cost-cert/1', cap=CAP, ceilings=CEIL, smoke_ceiling=SMOKE_CEIL, spent_gpu_h=spent,
               gpu=gpu, rates=(rates.basis if rates else None), compile_seconds=(rates.compile if rates else None),
               iteration_multipliers=(rates.mult if rates else None), schedules={}, selected=None)
    for s in ('S1', 'S2', 'S3', 'S4'):
        items = schedule_items(s, base)
        for k, v in items.items():
            v['ceiling'] = CEIL[k]
            v['fits'] = bool(v['certified'] and v['projected'] is not None and v['projected'] <= CEIL[k])
        ok = all(v['fits'] for v in items.values())
        ceil_sum = sum(CEIL[k] for k in items)
        out['schedules'][s] = dict(items=items, all_fit=ok, ceiling_sum=ceil_sum,
                                   within_cap=bool(spent + ceil_sum <= CAP + 1e-9),
                                   uncertified=[k for k, v in items.items() if not v['certified']],
                                   over_ceiling=[k for k, v in items.items() if v['certified'] and not v['fits']])
        if out['selected'] is None and ok and spent + ceil_sum <= CAP + 1e-9:
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
    def tim(job, train_h):
        sec = train_h * 3600. / 4
        ph = {p: dict(fixed_seconds=0., segments=[dict(name=p, seconds=sec / 100., units=1., production_units=100.)])
              for p in ('datagen', 'stageA', 'stageB', 'rotation')}
        return dict(schema='imp-fronts-timing/1', job=job, gpu='NVIDIA A100 80GB PCIe', phases=ph,
                    references=[dict(case='FC1|0', seconds=100., refined=False)])

    def rows(rate, it_tight=10.):
        out = []
        for lv, rec, m, f in (('prod', True, 4181, 1.), ('prod', True, 6765, .5), ('tight', False, 4181, 1.),
                              ('tight', False, 6765, .5), ('prod', True, 6765, .5)):
            S = steps(f)
            out.append(dict(level=lv, recorded=rec, m=m, steps=S, seconds=rate * m * S, first_of_shape=False,
                            it_mean=it_tight if lv == 'tight' else 3.))
        out.append(dict(level='prod', recorded=True, m=4181, steps=50, seconds=rate * 4181 * 50 + 1., first_of_shape=True,
                        it_mean=3.))
        return out

    with tempfile.TemporaryDirectory() as td:
        d = Path(td) / 'refA' / 'output' / 'refjob_x' / '2d' / 'C1' / 'ST'
        d.mkdir(parents=True)
        for i, (s, r) in enumerate(((100., False), (150., True))):
            (d / f'c{i:03d}.json').write_text(json.dumps(dict(seconds=s, refined=r, job=dict(job_id='1', gpu='NVIDIA A100 80GB PCIe'))))
        # (1) cheap everything -> S1
        c = certify({'SM-a': tim('SM-a', 3.), 'SM-b': tim('SM-b', 3.)}, rows(1e-10, it_tight=600.), td)
        assert c['selected'] == 'S1', c['schedules']['S1']
        tr = c['schedules']['S1']['items']['train_a']
        assert abs(tr['projected'] - 3. * 1.25) < 1e-9
        rf = c['schedules']['S1']['items']['references']
        assert abs(rf['per_case_seconds'] - 150. * 1.5) < 1e-9 and rf['n_cases'] == 28
        assert abs(c['iteration_multipliers']['tight'] - 1.) < 1e-12 and abs(c['compile_seconds'] - 1.) < 1e-6
        # (2) arm (a) training over its ceiling -> S3 (S2 does not drop training)
        c = certify({'SM-a': tim('SM-a', 4.), 'SM-b': tim('SM-b', 3.)}, rows(1e-10, it_tight=600.), td)
        assert c['schedules']['S1']['over_ceiling'] == ['train_a'] and c['selected'] == 'S3'
        # (3) missing SM-b timing -> train_b uncertified everywhere -> S4
        c = certify({'SM-a': tim('SM-a', 3.)}, rows(1e-10, it_tight=600.), td)
        assert 'train_b' in c['schedules']['S3']['uncertified'] and c['selected'] == 'S4'
        # (4) references too slow for 28 but fine for 24 -> S2
        for i, s in enumerate((260.,)):
            (d / 'c001.json').write_text(json.dumps(dict(seconds=s, refined=True, job=dict(job_id='1', gpu='NVIDIA A100 80GB PCIe'))))
        c = certify({'SM-a': tim('SM-a', 3.), 'SM-b': tim('SM-b', 3.)}, rows(1e-10, it_tight=600.), td)
        n28, n24 = 28 * 260 * 1.5 / 3600, 24 * 260 * 1.5 / 3600
        assert n28 > 2.8 >= n24 and c['selected'] == 'S2', (n28, n24, c['selected'])
        # (5) tight iteration multiplier (cap / mean) is applied: mean 1 -> x600 makes the policy uncertifiable
        c = certify({'SM-a': tim('SM-a', 3.), 'SM-b': tim('SM-b', 3.)}, rows(1e-8, it_tight=1.), td)
        assert c['iteration_multipliers']['tight'] == 600. and 'policy_K1' in c['schedules']['S1']['over_ceiling']
        assert c['selected'] == 'S4'
        # (6) no ROM timing -> policy / selection uncertified
        c = certify({'SM-a': tim('SM-a', 3.), 'SM-b': tim('SM-b', 3.)}, [], td)
        assert 'policy_K1' in c['schedules']['S1']['uncertified'] and c['selected'] == 'S4'
        # (7) admission rule
        c = certify({'SM-a': tim('SM-a', 3.), 'SM-b': tim('SM-b', 3.)}, rows(1e-10, it_tight=600.), td)
        led = dict(jobs=[dict(name='SM-a', item='smoke', ceiling_gpu_h=.45, state='done', charged_gpu_h=.4),
                         dict(name='SM-b', item='smoke', ceiling_gpu_h=.45, state='done', charged_gpu_h=.45)])
        a = admit(led, c, 'train_a', 4.4)
        assert a['admit'] and abs(a['total'] - (.85 + 4.4 + sum(CEIL.values()) - 4.4)) < 1e-9
        led['jobs'].append(dict(name='X', item='train_b', ceiling_gpu_h=4.4, state='running'))
        led['jobs'][0]['charged_gpu_h'] = 1.
        a = admit(led, c, 'train_a', 4.4)
        assert abs(a['total'] - (1.45 + 14.6 - .9)) < 1e-9
        led['jobs'][1]['charged_gpu_h'] = 1.5
        assert not admit(led, c, 'train_a', 4.4)['admit']       # 2.5 + 4.4 (live) + 4.4 + 5.8 > 15
    print('resources selftest OK')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rom', action='append', default=[])
    ap.add_argument('--regimes', default=str(REGIMES))
    ap.add_argument('--gpu', default='A100')
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
        assert r.get('backend') == 'gpu' and a.gpu.lower() in str(r.get('gpu', '')).lower(), f'{p}: not a {a.gpu} GPU run'
        rows += r['rows']
    cert = certify(timings, rows, a.regimes, a.gpu, ledger)
    cert['inputs'] = dict(timings=sorted(timings), rom=a.rom, regimes=a.regimes)
    Path(a.out).write_text(json.dumps(cert, indent=1, default=float) + '\n')
    print('selected schedule', cert['selected'],
          {s: (v['all_fit'], v['over_ceiling'], v['uncertified']) for s, v in cert['schedules'].items()})


if __name__ == '__main__':
    main()
