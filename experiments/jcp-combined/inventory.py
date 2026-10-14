"""Resource gate (DESIGN A2.12, A4.6, A5.9; code audit 2 items 10 and 21). Fail-closed.

Projects a PRODUCTION attempt's wall time and peak device memory from pulled, complete cluster smokes run on the SAME GPU
type, and writes checks/inventory-<attempt>.json, bound to the attempt's config sha256, the sha256 of every staged
source file, the smoke result.json sha256s, job ids and GPU type. stage.py calls verify_certificate() and refuses on any
mismatch or on fits = false.

Model (per run signature):
  2D (c2grid): steady seconds per step measured per (setting, form, scheme, level) on the smoke's non-first case; a
     production run = rate x steps; compile = (first - steady) per production signature; FOM: evaluation = per scheme
     the measured per-invocation seconds scaled ~ 1/dt (max over measured dt, conservative) x cases; calibration = the
     recorded smoke calibration seconds x (production / smoke configurations) x (smallest-step ratio); order as measured.
  3D (c3run): steady seconds per step per (family, scheme, level, K) normalised by the rule cost proxy
     c = m R' (M + 2 R'); a production run = max smoke (rate / c) x c_prod x steps (conservative); compile per
     production signature = max smoke compile of its family.
  timing panels: invocations incl. warm-ups priced at the MOST expensive subject (3D family panels: their baseline A
     is priced at the largest escalation rule), including the common panels.
  memory: 2D = smoke whole-job peak + resident production initial fields; 3D = max smoke whole-job peak + tables of any
     bank the smokes never held + the chunked operator-check intermediates + rule-data bytes (8 (2 R' + M) m) of every
     production grid setting whose deployed rule the smokes did not hold.
  staging (verify_certificate) recomputes the inequalities, re-hashes every staged file and every smoke result, and
     re-checks the smoke machinery gates and GPU type; capacity comes from the allocation type (A7.4).
Anything missing (unmeasured family, incomplete smoke, GPU mismatch, non-positive peak) -> fits = false.
Projected total x 1.5 must fit the requested hours; projected peak must be <= 0.9 x the GPU memory.

    python inventory.py --attempt c3desc --smokes runs/smk3esc/archive/output runs/smk3grid/archive/output \
        --gpu h200 --gpu-gb 141 --hours 24 --mem 200G
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GPU_NAMES = {'h200': 'NVIDIA H200', 'a100-80G': 'NVIDIA A100 80GB PCIe', 'h100': 'NVIDIA H100', 'a100': 'NVIDIA A100'}
CAPACITY_GB = {'h200': 141., 'a100-80G': 80., 'h100': 80., 'a100': 40.}          # derived from the allocation type


def staged_sources(driver):
    """Every file stage.py ships for this driver (lane code, vendored dependencies, inputs) plus inventory.py itself."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('stage_', HERE / 'cluster' / 'stage.py')
    st = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st)
    files = st.FILES3 if driver == 'c3run' else st.FILES2
    return sorted(set(files + ['experiments/jcp-combined/inventory.py']))


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def rule_points(name):
    for pre, f in (('gl', lambda x: int(x) ** 3), ('lat', int), ('gauss', lambda x: int(x) ** 2), ('fib', int)):
        if name.startswith(pre):
            return f(name[len(pre):])
    raise KeyError(name)


def steps_of(spec):
    p = spec.split('|')
    if p[0] == 'vendor':
        return 25
    if p[0] == 'fs':
        return int(round(.25 / float(p[2])))
    if p[0] == 'ad':
        return int(round(.25 / float(p[3])))
    return int(round(.25 / float(p[2])))                     # 2D key form|scheme|dt|level


def fam3(spec):
    p = spec.split('|')
    if p[0] == 'fs':
        return ('fs', p[1], p[3])
    if p[0] == 'vendor':
        return ('vendor',)
    return ('ad', p[1], p[2], p[4])


def cost3(m, Rp, M):
    return float(m) * Rp * (M + 2 * Rp)


def fail(rep, why):
    rep.setdefault('failures', []).append(why)
    rep['fits'] = False
    return rep


def project_3d(cfg, smokes, inv, rep):
    rates, comps = {}, {}
    for sm in smokes:
        S = sm['settings']
        by = {}
        for r in sm['rows']:
            by.setdefault((r['setting'], r['rule'], r['spec']), []).append(r)
        for (s, rule, spec), rs in by.items():
            rs = sorted(rs, key=lambda x: x['case'])
            if len(rs) < 2:
                continue
            c = cost3(rule_points(rule), S[s]['Rp'], S[s]['M'])
            steady = float(np.median([x['seconds_first'] for x in rs[1:]]))
            f = fam3(spec)
            rates[f] = max(rates.get(f, 0.), steady / steps_of(spec) / c)
            comps[f] = max(comps.get(f, 0.), max(0., rs[0]['seconds_first'] - steady))
    ncases = inv['cases']
    M_of = lambda Rp: 4 * Rp + 64                  # complete_M(4R') exceeds 4R' by less than one shell (conservative)
    total, detail, allcalls = 0., {}, []
    for s in cfg['settings']:
        Rp = int(s['Rp'])
        rlist = inv['settings'][s['name']]['runs']
        tot, comp, calls, seen = 0., 0., [], set()
        for item in rlist:
            rule, spec = (item if isinstance(item, (list, tuple)) else (s.get('rule'), item))
            rule = rule.replace(' (conditional)', '')
            f = fam3(spec)
            if f not in rates:
                return fail(rep, f'no smoke measurement for family {f}')
            t = rates[f] * cost3(rule_points(rule), Rp, M_of(Rp)) * steps_of(spec)
            tot += t * ncases
            calls.append(t)
            if (rule, spec) not in seen:
                comp += comps[f]
                seen.add((rule, spec))
        nt = inv['settings'][s['name']]['timing_invocations']
        if cfg['mode'] == 'esc':           # family panels: A = the converged rule's fixed BE dt0 (priced at the largest rule)
            big = max((rule_points(x[0].replace(' (conditional)', '')) for x in rlist if isinstance(x, (list, tuple))
                       and x[0].startswith('gl')), default=0)
            tA = rates[('fs', 'BE', 'K1')] * cost3(big, Rp, M_of(Rp)) * 25
            tt = nt * (2 / 3 * tA + 1 / 3 * float(np.max(calls)) + cfg['timing']['burn']) * (1 + 1 / 3)
        else:
            tt = nt * (float(np.max(calls)) + cfg['timing']['burn']) * (1 + 1 / 3)
        detail[s['name']] = dict(rollouts_s=tot, compile_s=comp, timing_s=tt)
        total += tot + comp + tt
        allcalls += calls
    if cfg['mode'] == 'grid':
        tc = cfg['timing']
        nsub = sum(1 for s in cfg['settings'] for x in inv['settings'][s['name']]['runs'] if x.startswith('fs|'))
        total += (3 * nsub * tc['common_cases'] * tc['common_reps'] + nsub * tc['common_cases']) * \
            (float(np.mean(allcalls)) + tc['burn'])
    peaks = [sm.get('peak_bytes_job', -1) for sm in smokes]
    if min(peaks) <= 0:
        return fail(rep, 'a smoke lacks a positive peak')
    peak = max(peaks)
    extra = 0
    smoke_banks = {s_['bank'] for sm in smokes for s_ in sm['config']['settings']}
    for s in cfg['settings']:                         # tables of a bank the smokes never held (GTb + A + Gram), all R
        if s['bank'] not in smoke_banks:
            R = 1024 if s['bank'] == 'W1024' else 512
            extra += 8 * (250047 * R + M_of(R) * R + R * R)
            smoke_banks.add(s['bank'])
    if cfg['mode'] == 'esc':                          # chunked operator check: 2 x (16 states x m) intermediates at 80^3
        extra += 2 * 8 * 16 * 80 ** 3 * 2
    held = {(r['setting'], r['rule']) for sm in smokes for r in sm['rows']}
    if cfg['mode'] == 'grid':
        for s in cfg['settings']:
            if (s['name'], s['rule']) not in held:
                extra += 8 * (2 * int(s['Rp']) + M_of(int(s['Rp']))) * rule_points(s['rule'])
    rep.update(detail=detail, projected_seconds=total, projected_peak_gb=(peak + extra) / 1e9)
    return rep


def project_2d(cfg, smokes, inv, rep):
    sm = smokes[0]
    by = {}
    for r in sm['rows']:
        by.setdefault((r['setting'], r['run']), []).append(r)
    rates, comps = {}, {}
    for (s, k), rs in by.items():
        rs = sorted(rs, key=lambda x: (x['cohort'], x['case']))
        if len(rs) < 2:
            continue
        form, sc, dt, lev = k.split('|')
        steady = float(np.median([x['seconds_first'] for x in rs[1:]]))
        f = (s, form, sc, lev)
        rates[f] = max(rates.get(f, 0.), steady / steps_of(k))
        comps[f] = max(comps.get(f, 0.), max(0., rs[0]['seconds_first'] - steady))
    ncases = sum(len(cfg.get('case_subset', {}).get(c) or range(6 if c == 'dev6' else 32)) for c in cfg['cohorts'])
    total, detail, calls = 0., {}, {}
    for s in cfg['settings']:
        tot, comp = 0., 0.
        for k in inv['runs']:
            form, sc, dt, lev = k.split('|')
            f = (s['name'], form, sc, lev)
            if f not in rates:
                return fail(rep, f'no smoke measurement for {f}')
            t = rates[f] * steps_of(k)
            tot += t * ncases
            comp += comps[f]
            calls.setdefault(s['name'], []).append((k, t))
        tot += ncases * rates.get((s['name'], 'LSPG', 'BE', 'prod'), 0.) * 50          # the vendor G1a query per case
        detail[s['name']] = dict(rollouts_s=tot, compile_s=comp)
        total += tot + comp
    fe = {}
    for r in sm['fom'].get('rows', []):
        fe.setdefault(r['scheme'], []).append((r['dt'] / .005, r['seconds']))
    fc = cfg['fom']
    fom_eval = {}
    for scn in fc['schemes']:
        if scn not in fe:
            return fail(rep, f'no FOM smoke measurement for {scn}')
        for f in fc['dt_factors']:
            per = max(float(sec) * f_ref / f for f_ref, sec in fe[scn])                     # time ~ 1/dt, max (conservative)
            fom_eval[(scn, f)] = per
            total += per * ncases
    cal = sm['fom'].get('calibration', {})
    cal_s = sum(c_.get('seconds_tight', 0.) + sum(c_.get('seconds', {}).values()) for e_ in cal.values() for c_ in e_['cases'])
    if not cal or not any('seconds_tight' in c_ for e_ in cal.values() for c_ in e_['cases']):
        return fail(rep, 'the smoke recorded no calibration timings')
    sm_f = sm['config']['fom']['dt_factors']
    # production calibration = recorded smoke calibration x (production / smoke configurations) x (smallest-step ratio)
    total += cal_s * (len(fc['schemes']) * len(fc['dt_factors'])) / len(cal) * max(1., min(sm_f) / min(fc['dt_factors']))
    total += sum(sum(x['seconds'].values()) for x in sm['fom'].get('order', [])) * \
        len(fc['order_cases']) * len(fc['schemes']) / max(1, len(sm['fom'].get('order', [])))
    tc = cfg['timing']
    timed = [k for k in inv['runs'] if k.endswith('prod') and float(k.split('|')[2]) >= .0025 - 1e-12]
    mean_call = lambda s: float(np.mean([t for k, t in calls[s] if k in timed]))
    for s in cfg['settings']:
        n = len(timed) + 1                                                                  # + vendor subject
        total += (3 * n * tc['cases'] * tc['reps'] + (n + 1) * tc['cases']) * (mean_call(s['name']) + tc['burn'])
    nrom = len(timed) * len(cfg['settings'])
    nfom = len(fom_eval)
    mc = float(np.mean([mean_call(s['name']) for s in cfg['settings']]))
    mf = float(np.mean(list(fom_eval.values())))
    total += (3 * tc['cases'] * tc['reps'] + tc['cases']) * (nrom * (mc + tc['burn']) + nfom * (mf + tc['burn'])) + \
        2 * tc['cases'] * tc['reps'] * (nrom + nfom) * (mc + tc['burn'])                  # the two flanking baselines
    peak = sm.get('peak_bytes_job', -1)
    if peak <= 0:
        return fail(rep, 'no positive smoke peak')
    extra = 8 * ncases * (cfg['mesh'] + 1) ** 2                                         # resident initial fields
    rep.update(detail=detail, projected_seconds=total, projected_peak_gb=(peak + extra) / 1e9)
    return rep


def project_bud2(cfg, smokes, inv, rep):
    """2D budget (c2bud) projection over the COMPLETE call inventory of `c2bud.py --dry-run` (codex-results2d-1 item 1):
    every run on every case; on the check set the check-rule twin of every r_inf-dependent run, two B1 baseline calls
    and one control-rule call; and the conditional successor (conv runs at the check rule, twins at the successor rule,
    priced at (768/640)^2 x the check-rule rate) plus restoration (conv runs at the converged rule, twins at the check
    rule) phases, always included (worst case). Rate per signature (role, form, scheme, level) = steady seconds per step
    on the smoke (median over cases after the first; one case -> its time), max over the smoke's settings; a signature
    the smoke did not run is priced at the most expensive measured signature of the same role. A setting the smoke did
    not run is priced from a smoke setting with the SAME R' (hence the same M and the same converged/check/control
    rules), its 'main'-role rates scaled up by the main-rule point ratio when that ratio exceeds 1; with no same-R'
    smoke setting, from the most expensive smoke setting whose R' is >= its own (no downward R' scaling).
    Compile = max (first - steady) per distinct signature. Memory = max smoke per-setting peak x (768/640)^2."""
    rates, comps, by_setting = {}, {}, {}
    for sm in smokes:
        by = {}
        for r in sm['rows']:
            by.setdefault((r['setting'], r['run']), []).append(r)
        for (st, run), rs in by.items():
            role, form, sc, dt, lev = run.split('|')
            steps = int(round(.25 / float(dt)))
            steady = float(np.median([x['seconds_first'] for x in rs[1:]])) if len(rs) > 1 else float(rs[0]['seconds_first'])
            f = (role, form, sc, lev)
            rates[f] = max(rates.get(f, 0.), steady / steps)
            comps[f] = max(comps.get(f, 0.), max(0., rs[0]['seconds_first'] - steady))
            by_setting.setdefault(st, {}).setdefault(f, 0.)
            by_setting[st][f] = max(by_setting[st][f], steady / steps)
    fallbacks = set()

    def t(run, rate_tab, factor=1.):
        role, form, sc, dt, lev = run.split('|')
        f = (role, form, sc, lev)
        if f in rate_tab:
            rt = rate_tab[f]
        else:
            same = [v for k, v in rate_tab.items() if k[0] == role] or [v for k, v in rates.items() if k[0] == role]
            if not same:
                raise KeyError(f'no smoke measurement for role {role}')
            rt = max(same)
            fallbacks.add(run)
        return rt * int(round(.25 / float(dt))) * factor
    smk_cfg = {s_['name']: s_ for sm in smokes for s_ in sm['config']['settings']}

    def table_for(st):
        if st in by_setting:
            return by_setting[st], st
        me = next(s_ for s_ in cfg['settings'] if s_['name'] == st)
        same = [n for n, s_ in smk_cfg.items() if int(s_['Rp']) == int(me['Rp']) and n in by_setting]
        if same:
            n = same[0]
            fac = max(1., rule_points(me['rule']) / rule_points(smk_cfg[n]['rule']))
            return {k: (v * fac if k[0] == 'main' else v) for k, v in by_setting[n].items()}, f'{n} (main x {fac:.3g})'
        up = [n for n, s_ in smk_cfg.items() if int(s_['Rp']) >= int(me['Rp']) and n in by_setting]
        if not up:
            raise KeyError(f'no smoke setting with R\' >= {me["Rp"]} for {st}')
        n = max(up, key=lambda n_: sum(by_setting[n_].values()))
        return by_setting[n], n
    total, detail, seen = 0., {}, set()
    succ = (768 / 640) ** 2
    try:
        for st, c in inv['calls'].items():
            tab, priced_as = table_for(st)
            a_ = sum(t(r, tab) for r in c['all_cases']['runs']) * c['all_cases']['cases']
            b_ = sum(t(r, tab) for r in c['check_set']['runs']) * c['check_set']['cases']
            cond = c['conditional']
            conv_runs = cond['successor']['runs']
            twin = lambda r: 'check|' + r.split('|', 1)[1]
            d_ = sum(t(twin(r), tab) * (1 + succ) + t(r, tab) + t(twin(r), tab) for r in conv_runs) * cond['cases']
            comp = 0.
            for r in c['all_cases']['runs'] + c['check_set']['runs']:
                f = tuple(r.split('|')[i] for i in (0, 1, 2, 4))
                if f not in seen:
                    comp += comps.get(f, max(comps.values()))
                    seen.add(f)
            detail[st] = dict(all_cases_s=a_, check_set_s=b_, conditional_s=d_, compile_s=comp,
                              priced_as=priced_as)
            total += a_ + b_ + d_ + comp
    except KeyError as e:
        return fail(rep, str(e))
    peaks = [v.get('peak_bytes', -1) for sm in smokes for v in sm['setup'].values()]
    if not peaks or min(peaks) <= 0:
        return fail(rep, 'a budget smoke lacks a positive per-setting peak')
    rep.update(detail=detail, priced_by_role_fallback=sorted(fallbacks), projected_seconds=total,
               projected_peak_gb=max(peaks) * succ / 1e9)
    return rep


def certificate_binding(attempt, driver):
    commit = subprocess.check_output(['git', '-C', str(HERE), 'rev-parse', 'HEAD'], text=True).strip()
    root = HERE.parents[1]
    return dict(commit_at_certification=commit, driver=driver,
                config_sha256=sha_bytes((HERE / 'configs' / f'{attempt}.json').read_bytes()),
                sources_sha256={f: sha_bytes((root / f).read_bytes()) for f in staged_sources(driver)})


def offline_budget_gates(sm, path):
    """The smoke's committed offline budget analysis (checks/budget{2d,3d}-<attempt>-<ref>.json) must exist, be bound to
    THIS smoke's result.json bytes (sha256 of <path>/result.json) and to the current analyzer source (sha256), and pass
    its machinery gates with strict booleans and complete counts (codex-results2d-2/3): B0 (True), B0 analytic fixtures
    (True), the relative real-data mismatch control (True), B1 (True), job-row reproduction (True, with the expected number
    of comparisons, every comparison successful, and every G-score negative control present and breaking), inventory
    (True)."""
    two = sm['config']['driver'] == 'c2bud'
    f = HERE / 'checks' / f"budget{'2d' if two else '3d'}-{sm['config']['attempt']}-{'2d' if two else '3d'}_first_order.json"
    if path is None or not f.exists():
        return False
    o = json.loads(f.read_text())
    an = HERE / ('budget2d.py' if two else 'budget3d.py')
    if o.get('analyzer_sha256') != sha_bytes(an.read_bytes()):
        return False
    if o.get('result_sha256') != sha_bytes((Path(path) / 'result.json').read_bytes()):
        return False
    g = o.get('gates', {})
    t = lambda d, k='passed': isinstance(d, dict) and d.get(k) is True
    ok = t(g.get('B0')) and g.get('B0_fixtures') is True and t(g.get('B0_real_mismatch')) and g.get('B1') is True \
        and t(g.get('inventory'))
    rp = g.get('reproduction', {})
    nrows = len(sm['rows'])
    per = 18 if two else 6
    ok = ok and t(rp) and rp.get('applicable') is True and rp.get('compared') == per * nrows and nrows > 0 \
        and rp.get('failures') == 0
    if two:
        ctl = rp.get('controls', {})
        ok = ok and set(ctl) == {s_['name'] for s_ in sm['config']['settings']} and all(
            v.get('all_break') is True and all(k_ in v for k_ in ('case_swap', 'time_shift', 'ST_S_swap')) for v in ctl.values())
    else:
        ok = ok and all(o['settings'].get(s_['name'], {}).get('gscore_controls', {}).get('all_break') is True
                        for s_ in sm['config']['settings'])
    return bool(ok)


def budget_smoke_gates(sm, path=None):
    """Fail closed (codex-results2d-1/2/3): complete per-setting evidence and exact coverage DERIVED FROM THE CONFIG."""
    ok = bool(sm.get('complete')) and bool(sm['rows'])
    if sm['config']['driver'] == 'c2bud':
        import c2bud as CB
        cases = [(coh, c) for coh in sm['config']['cohorts']
                 for c in ((sm['config'].get('case_subset') or {}).get(coh) or range(6 if coh == 'dev6' else 32))]
        chk = [x for x in cases if x[0] == 'dev6'] + [x for x in cases if x[0] == 'val32'][:sm['config']['check_val32']]
        ok = ok and set(sm['setup']) == {s_['name'] for s_ in sm['config']['settings']}
        for s_ in sm['config']['settings']:
            v = sm['setup'].get(s_['name'], {})
            b1, b2 = v.get('B1_sham_bitwise', {}), v.get('B2_control_rule', {})
            ok = ok and v.get('conv_check', {}).get('passed') is True and b1.get('all_equal') is True and \
                b1.get('cases') == len(chk) and b2.get('complete') is True and len(b2.get('per_case', [])) == len(chk)
            rows = [r for r in sm['rows'] if r['setting'] == s_['name']]
            got = [(r['run'], r['cohort'], r['case']) for r in rows]
            runs = CB.budget_runs(s_['arms'])
            want = {(CB.key(r), coh, c) for r in runs for coh, c in cases}
            want |= {(CB.key(('check',) + r[1:]), coh, c) for r in runs if r[0] == 'conv' for coh, c in chk}
            want |= {(CB.key(('ctrl', 'LSPG', 'BE', CB.DT0, 'prod')), coh, c) for coh, c in chk}
            ok = ok and len(got) == len(set(got)) and set(got) == want
            ok = ok and all(r['verified'] is True for r in rows if not r['run'].startswith('ctrl|'))
    else:
        import c3run as CR
        cases = list(sm['config'].get('case_subset') or range(sm['config']['cohort_count']))
        ctrl = sm['config']['control_rule']
        nb = int(sm['config']['budget_cases'])
        ok = ok and 1 <= nb <= len(cases) and set(sm['settings']) == {s_['name'] for s_ in sm['config']['settings']}
        for s_ in sm['config']['settings']:
            v = sm['settings'].get(s_['name'], {})
            b1, b2 = v.get('B1_sham_bitwise', {}), v.get('B2_control_rule', {})
            ok = ok and v.get('budget_cases') == nb and b1.get('all_equal') is True and b1.get('cases') == nb and \
                b2.get('complete') is True and len(b2.get('per_case', [])) == nb
            want = CR.bud_case_plan(s_, cases, nb, ctrl)              # derived from the config, not from the result
            by = {}
            for r in sm['rows']:
                if r['setting'] == s_['name']:
                    by.setdefault((r['rule'], r['spec']), []).append(int(r['case']))
            ok = ok and set(by) == set(want) and all(by[k_] == w_ for k_, w_ in want.items() if k_ in by)
            cs = {tuple(k_.split('|', 1)): how for k_, how in v.get('case_sets', {}).items()}
            ok = ok and cs == {k_: ('all' if w_ == cases else f'first {nb}') for k_, w_ in want.items() if k_[0] != ctrl}
            ok = ok and all(r['valid'] is True for r in sm['rows'] if r['setting'] == s_['name'] and r['rule'] != ctrl)
    return bool(ok and offline_budget_gates(sm, path))


def smoke_gates(sm, path=None):
    """Machinery gates a smoke must have passed to be used as resource evidence."""
    if sm['config']['driver'] == 'c2bud' or sm['config'].get('mode') == 'bud':
        return budget_smoke_gates(sm, path)
    ext = all(s_.get('ext_rules') for s_ in sm['config']['settings'])   # A11 extension smoke: no timing panels by design
    ok = bool(sm.get('complete')) and all(v.get('valid') for v in sm.get('timing', {}).values()) and (ext or bool(sm.get('timing')))
    if ext:      # complete inventory: every planned rule (the last one omitted only after a passing A10 pair), every
                 # (rule, arm, case) row present once and valid, every consecutive pair recorded with all arms finite
        arms = ['fs|BE|0.01|K1', 'fs|CN|0.025|K2', 'ad|LSPG|BE|0.01|tight', 'ad|LSPG|CN|0.025|tight']
        cases = list(sm['config'].get('case_subset') or range(sm['config']['cohort_count']))
        for s_ in sm['config']['settings']:
            x = sm['settings'][s_['name']].get('extension', {})
            rr, plan = x.get('rules', []), s_['ext_rules']
            ok = ok and len(rr) >= 2 and rr == plan[:len(rr)]
            if ok and len(rr) < len(plan):
                ok = len(plan) >= 3 and len(rr) == len(plan) - 1 and \
                    x['pairs'].get(f'{rr[-2]}|{rr[-1]}', {}).get('posthoc_status') == 'pass'
            want = {f'{a_}|{b_}' for a_, b_ in zip(rr, rr[1:])}
            ok = ok and set(x.get('pairs', {})) == want
            for p in x.get('pairs', {}).values():
                for ax, ks in (('selection', arms[:2]), ('diagnostic', arms[2:]), ('posthoc', [arms[0], arms[3]])):
                    ok = ok and set(p.get(ax, {})) == set(ks) and all(np.isfinite(p[ax][k]['worst']) for k in ks)
            got = sorted((r['rule'], r['spec'], r['case']) for r in sm['rows'] if r['setting'] == s_['name'])
            ok = ok and got == sorted((r_, a_, c_) for r_ in rr for a_ in arms for c_ in cases)
        ok = ok and bool(sm['rows']) and all(r['valid'] is True for r in sm['rows'])
    if sm['config']['driver'] == 'c2grid':
        ok = ok and bool(sm['g1a']) and all(g.get('passed') for g in sm['g1a']) and all(r['verified'] for r in sm['rows'])
    else:
        ok = ok and all(v.get('G_FS1', {}).get('outcome', 'exact') != 'FAIL' for v in sm['settings'].values())
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--attempt', required=True)
    ap.add_argument('--smokes', nargs='+', required=True)
    ap.add_argument('--gpu', required=True, choices=sorted(GPU_NAMES))
    ap.add_argument('--gpu-gb', type=float, required=True)
    ap.add_argument('--hours', type=float, required=True)
    ap.add_argument('--mem', required=True)
    a = ap.parse_args()
    cfg = json.loads((HERE / 'configs' / f'{a.attempt}.json').read_text())
    assert not cfg.get('smoke'), 'a smoke attempt needs no certificate'
    assert a.gpu_gb == CAPACITY_GB[a.gpu], 'GPU capacity must be the allocation type capacity'
    rep = dict(attempt=a.attempt, gpu=a.gpu, gpu_gb=a.gpu_gb, requested_hours=a.hours, mem=a.mem, fits=True,
               **certificate_binding(a.attempt, cfg['driver']))
    smokes = []
    for p in map(Path, a.smokes):
        sm = json.loads((p / 'result.json').read_text())
        rep.setdefault('smokes', []).append(dict(path=str(p), job_id=sm.get('job_id'), gpu=sm.get('gpu'),
                                                  result_sha256=sha_bytes((p / 'result.json').read_bytes()),
                                                  complete=sm.get('complete'), smoke=sm['config'].get('smoke'),
                                                  gates_passed=smoke_gates(sm, p)))
        if not smoke_gates(sm, p):
            fail(rep, f'smoke {p} did not pass its machinery gates')
        if not (sm.get('complete') and sm['config'].get('smoke') and (p / 'COMPLETE').exists()):
            fail(rep, f'smoke {p} incomplete or not a smoke')
        if not str(sm.get('gpu', '')).startswith(GPU_NAMES[a.gpu].split(' 80GB')[0]):
            fail(rep, f'smoke GPU {sm.get("gpu")} differs from {a.gpu}')
        if sm['config']['driver'] != cfg['driver']:
            fail(rep, f'smoke driver {sm["config"]["driver"]} differs from {cfg["driver"]}')
        smokes.append(sm)
    inv = json.loads(subprocess.check_output(['/home/tahmid/Dev/.venv/bin/python', str(HERE / f"{cfg['driver']}.py"),
                                              '--config', str(HERE / 'configs' / f'{a.attempt}.json'), '--out', '/dev/null',
                                              '--dry-run'], text=True, cwd=HERE))
    if rep['fits']:
        (project_3d if cfg['driver'] == 'c3run' else project_bud2 if cfg['driver'] == 'c2bud' else project_2d)(cfg, smokes, inv, rep)
    if rep.get('projected_seconds') is not None:
        rep['projected_hours_with_margin'] = 1.5 * rep['projected_seconds'] / 3600
        rep['fits_time'] = bool(rep['projected_hours_with_margin'] <= a.hours)
        rep['fits_memory'] = bool(rep['projected_peak_gb'] <= .9 * a.gpu_gb)
        rep['fits'] = bool(rep['fits'] and rep['fits_time'] and rep['fits_memory'])
    dest = HERE / 'checks' / f'inventory-{a.attempt}.json'
    dest.write_text(json.dumps(rep, indent=1, default=float) + '\n')
    print(json.dumps({k: v for k, v in rep.items() if k not in ('detail', 'sources_sha256')}, indent=1, default=float))


def verify_certificate(inv, commit, cfg_bytes, gpu, hours, mem, git_show=None):
    """Called by stage.py: recompute the inequalities and check every binding (A7.4). Fail closed."""
    assert git_show is not None, 'source binding requires git_show'
    assert inv['config_sha256'] == sha_bytes(cfg_bytes), 'certificate is for a different config'
    assert inv['gpu'] == gpu and float(inv['gpu_gb']) == CAPACITY_GB[gpu], f"certificate is for GPU {inv['gpu']}"
    assert float(inv['requested_hours']) == float(hours) and inv['mem'] == mem, 'allocation differs from the certificate'
    srcs = staged_sources(inv['driver'])
    assert sorted(inv['sources_sha256']) == srcs, 'certificate source manifest incomplete'
    for f, h in inv['sources_sha256'].items():
        assert sha_bytes(git_show(f)) == h, f'certified source changed: {f}'
    for sm in inv['smokes']:
        p = Path(sm['path'])
        p = p if p.is_absolute() else HERE / p
        assert sha_bytes((p / 'result.json').read_bytes()) == sm['result_sha256'], f'smoke {p} changed'
        res = json.loads((p / 'result.json').read_text())
        assert res['config'].get('smoke') and (p / 'COMPLETE').exists() and smoke_gates(res, p), f'smoke {p} not usable'
        assert str(res.get('gpu', '')).startswith(GPU_NAMES[gpu].split(' 80GB')[0]), 'smoke GPU type differs'
    proj_h = 1.5 * float(inv['projected_seconds']) / 3600
    peak = float(inv['projected_peak_gb'])
    assert np.isfinite(proj_h) and np.isfinite(peak) and proj_h > 0 and peak > 0, 'non-finite projection'
    assert proj_h <= float(hours), f'projected {proj_h:.1f} h (x1.5) exceeds {hours} h'
    assert peak <= .9 * CAPACITY_GB[gpu], f'projected peak {peak:.1f} GB exceeds 90 % of {CAPACITY_GB[gpu]} GB'
    assert not inv.get('failures'), inv.get('failures')
    return True


if __name__ == '__main__':
    main()
