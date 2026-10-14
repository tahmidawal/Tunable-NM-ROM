"""jcp-combined 3D driver (DESIGN 3.2, 3.3, 8; A1.0, A1.2, A1.6, A1.7, A2.1, A2.4, A2.5, A2.11, A3.4, A4). Modes:

  esc   (`c3desc`) J4 escalation and rule-selection data. Per setting: escalation Gauss rules (W768/W1024: 48^3, 56^3,
        64^3; others 56^3, 64^3) under the two fixed-sweep SELECTION arms (BE dt0 K=1; LSPG-CN 2.5 dt0 K=2, A6.3) and the two
        adaptive DIAGNOSTIC arms (LSPG-BE dt0 tight; LSPG-CN 2.5 dt0 tight), all cases; the CN selection arm uses K = 2 (A6.3). Per rule pair (A2.5):
        selection_pair_status / adaptive_diagnostic_status in {pass, fail, unresolved}: pass iff every arm of the axis
        has eligible runs on both rules (A6.2: fixed-sweep arms valid and within 2.5e-4 of the adaptive tight solve of the
        same scheme and step; adaptive arms every step verified) and worst distance <= conv_bar; unresolved iff some arm
        is ineligible or a distance is non-finite; fail otherwise.
        Converged rule (A2.4) = 56^3 if (56^3, 64^3) passes the selection axis, else 64^3 if (64^3, 80^3) passes (80^3 is
        run only then), else none. Fixed-vs-adaptive sensitivities per rule (same scheme and step). Operator check on
        the reached states of the 64^3 adaptive LSPG-BE run (A1.7). Then the ladder under the two selection arms AND their
        adaptive counterparts (A6.2 eligibility per candidate rule, A7.1) and an A-B-A family panel per ladder rule
        (A = converged rule's fixed BE dt0).
  grid  (`c3grid`) per setting with its frozen deployed rule: G-FS1 first (vendor vs fs BE dt0 K=1, per case; A2.1:
        exact or approved-equivalent, otherwise the job stops) with the CN negative control on the first case; then
        fixed-sweep BE and CN / CN-R / BDF2 (K = 1, 2) on the step ladder, adaptive LSPG probes (tight, tighter) of every
        scheme and step, adaptive GAL-CN (tight), anchors GAL-BDF2 dt0/8 and dt0/16 (tight, tighter); per-setting A-B-A
        panel (A = fs BE dt0; B = every fixed-sweep candidate, the vendor, and the adaptive tight arms at dt0 and 2.5 dt0,
        A4.3); one common panel (baseline W512 fs BE dt0, fallback M2_512; every fs candidate with dt >= dt0).
Every timing panel checks that each timed output's W sha256 equals the accuracy-phase W (A and B, every invocation) and
applies the 10 % drift gate per subject (timing.panel_validity, A5.3); a failing panel is marked timing-invalid.
Residency: escalation rules are dropped after use (64^3 kept for the operator check); ladder rules one at a time; grid
mode keeps each setting's deployed rule for the common panel (deployed rules are <= the ladder sizes).
Saved: W (6, R') of every run per case (W_<mode>.npz, keys setting|rule|spec), w0 per setting, full per-step evidence,
refined errors per time on the 63^3 lattice (in-job mesh-row decode; score.py recomputes independently), peak device
memory per setting, the run inventory, result.json.

    python c3run.py --config configs/<attempt>.json --out <dir> [--dry-run]
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import c3common as C3
import timing as TM
from c3common import C, TB, OM, DT0, jax, jnp, sha

SEL = [('fs', 'BE', DT0, 1), ('fs', 'CN', 2.5 * DT0, 2)]          # A6.3: CN selection arm K = 2
DIAG = [('ad', 'LSPG', 'BE', DT0, 'tight'), ('ad', 'LSPG', 'CN', 2.5 * DT0, 'tight')]


def grid_specs(g):
    sp = []
    for f in g['dt_factors']:
        dt = DT0 * f
        sp.append(('fs', 'BE', dt, 1))
        for sc in g['fs_schemes']:
            for K in g['Ks']:
                sp.append(('fs', sc, dt, K))
        for sc in ['BE'] + g['fs_schemes']:
            for lev in ('tight', 'tighter'):
                sp.append(('ad', 'LSPG', sc, dt, lev))
    for f in g['gal_factors']:
        sp.append(('ad', 'GAL', 'CN', DT0 * f, 'tight'))
    for f in g['anchor_factors']:
        for lev in ('tight', 'tighter'):
            sp.append(('ad', 'GAL', 'BDF2', DT0 * f, lev))
    return sp


H8, H16 = DT0 / 8, DT0 / 16


def bud_specs(arms):
    """3D budget runs (A1.4, A1.5) as (role, spec): role 'r' = deployed rule, 'conv' = r_inf, 'finer' = next finer."""
    out = []
    add = lambda x: out.append(x) if x not in out else None
    for x in arms:
        sp = tuple(x['spec'])
        if sp[0] == 'vendor':
            sc, dt, K = 'BE', DT0, 1
        else:
            _, sc, dt, K = sp
        add(('r', sp))
        add(('conv', sp))
        add(('r', ('ad', 'LSPG', sc, dt, 'tight')))
        add(('r', ('ad', 'GAL', sc, dt, 'tight')))
        add(('r', ('fs', 'BDF2', H8, K)))
        add(('r', ('ad', 'LSPG', sc, dt / 2, 'tight')))
        add(('conv', ('fs', 'BDF2', H8, K)))
        add(('r', ('ad', 'LSPG', 'BDF2', H8, 'tight')))
        if x.get('selected'):
            add(('finer', sp))
    for r in [('conv', ('ad', 'GAL', 'BDF2', H8, 'tight')), ('conv', ('ad', 'GAL', 'BDF2', H16, 'tight')),
              ('conv', ('ad', 'GAL', 'BDF2', H8, 'tighter')), ('conv', ('ad', 'GAL', 'BDF2', H16, 'tighter')),
              ('r', ('ad', 'GAL', 'BDF2', H8, 'tight')), ('conv', ('ad', 'LSPG', 'BDF2', H8, 'tight'))]:
        add(r)
    return out


def bud_case_plan(s, cases, nb, control_rule):
    """Pure restatement of the bud-mode loop (used by the resource/smoke gates, codex-results2d-3): {(rule, spec_key):
    case list} of every rollout the 3D budget performs on one setting, incl. the B2 control-rule baseline runs."""
    rmap = dict(r=s['rule'], conv=s['conv'], finer=s['finer'])
    plan = bud_specs(s['arms'])
    sel_specs = [tuple(x['spec']) for x in s['arms'] if x.get('selected')]
    out = {}
    for role in ('r', 'conv', 'finer'):
        rule = rmap[role]
        specs_all = [sp for rl, sp in plan if rl == role and sp in sel_specs]
        specs_sub = [sp for rl, sp in plan if rl == role and sp not in specs_all]
        if role == 'finer':
            specs_sub += [sp for rl, sp in plan if rl == 'conv' and sp not in specs_all and sp not in specs_sub]
        for sp in specs_all:
            out[(rule, C3.spec_key(sp))] = list(cases)
        for sp in specs_sub:
            out[(rule, C3.spec_key(sp))] = list(cases[:nb])
    out[(control_rule, C3.spec_key(('fs', 'BE', DT0, 1)))] = list(cases[:nb])
    return out


def timed_adaptive(x):
    return x[0] == 'ad' and x[4] == 'tight' and (abs(x[3] - DT0) < 1e-12 or abs(x[3] - 2.5 * DT0) < 1e-12)


def nsteps(spec):
    return 25 if spec[0] == 'vendor' else int(round(.25 / (spec[2] if spec[0] == 'fs' else spec[3])))


def inventory(cfg):
    """A2.12 run inventory per case (incl. conditional successors) and timing invocations."""
    inv = dict(mode=cfg['mode'], cases=len(cfg.get('case_subset') or range(cfg['cohort_count'])), settings={})
    tc = cfg['timing']
    for s in cfg['settings']:
        if cfg['mode'] == 'grid':
            sp = [('vendor',)] + grid_specs(cfg['grid'])
            nt = len([x for x in sp if (x[0] in ('fs', 'vendor') and x != ('fs', 'BE', DT0, 1)) or timed_adaptive(x)])
            inv['settings'][s['name']] = dict(runs=[C3.spec_key(x) for x in sp], steps_per_case=sum(nsteps(x) for x in sp),
                                              timing_invocations=3 * nt * tc['cases'] * tc['reps'])
        elif cfg['mode'] == 'bud':
            runs = [(s[role] if role != 'r' else s['rule'], C3.spec_key(sp)) for role, sp in bud_specs(s['arms'])]
            runs += [(s['finer'], C3.spec_key(sp)) for role, sp in bud_specs(s['arms']) if role == 'conv']
            inv['settings'][s['name']] = dict(runs=runs, timing_invocations=0)
        elif s.get('ext_rules'):                      # A11 extension: last rule conditional, no panels
            runs = [(r, C3.spec_key(x)) for r in s['ext_rules'][:-1] for x in SEL + DIAG]
            runs += [(s['ext_rules'][-1] + (' (conditional)' if len(s['ext_rules']) >= 3 else ''), C3.spec_key(x))
                     for x in SEL + DIAG]
            inv['settings'][s['name']] = dict(runs=runs, timing_invocations=0)
        else:
            runs = [(r, C3.spec_key(x)) for r in s['esc_rules'] for x in SEL + DIAG]
            runs += [('gl80 (conditional)', C3.spec_key(x)) for x in SEL + DIAG]
            runs += [(r, C3.spec_key(x)) for r in s['ladder'] for x in SEL + DIAG]
            inv['settings'][s['name']] = dict(runs=runs, timing_invocations=3 * (2 * len(s['ladder']) + 1) * tc['cases'] * tc['reps'])
    return inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    mode = cfg['mode']
    if a.dry_run:
        print(json.dumps(inventory(cfg), indent=1))
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    C3.check_env(cfg)
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)
    n = int(cfg['mesh'])
    assert n == 65, 'the in-job lattice decode uses the 65-node mesh rows (interior = the 63^3 lattice)'
    rep = dict(config=cfg, mode=mode, commit=C3.os.environ.get('SOURCE_COMMIT'), job_id=C3.os.environ.get('SLURM_JOB_ID'),
               backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, nvidia_smi=C3.gpu_info(),
               jax_version=jax.__version__, mesh=n, offmesh_sha256=C3.sha_file(OM.__file__),
               fs3_sha256=C3.sha_file(C3.HERE / 'fs3.py'), inventory=inventory(cfg), settings={}, rows=[], timing={},
               complete=False)
    save = lambda: C.dump(out / 'result.json', C.clean(rep))
    tab, cases, REF = C3.load_cohort(cfg, rep)
    rules = C3.load_rules()
    U0 = {j: jax.block_until_ready(jnp.asarray(C.initial_interior(n, tab, j))) for j in cases}
    NU = {j: float(tab['nu'][j]) for j in cases}
    N0 = np.array([float(jnp.linalg.norm(U0[j])) for j in cases])
    N0R = {j: float(np.linalg.norm(REF[j][0])) for j in cases}
    i65 = np.asarray(C3.lattice65_index(n))
    for j in cases:
        assert float(np.abs(np.asarray(U0[j])[i65] - REF[j][0]).max()) <= 1e-12 * max(1., np.abs(REF[j][0]).max())
    tables, banks = {}, {}
    for bname in sorted({s['bank'] for s in cfg['settings']}):
        banks[bname] = C3.load_bank(bname)
        tables[bname] = C3.Tables(n, banks[bname], sorted({int(s['Rp']) for s in cfg['settings'] if s['bank'] == bname}),
                                  log=lambda s_: print(s_, flush=True))
    print('SETUP', el(), flush=True)
    Wstore, w0store, acc_sha, live, Cstore = {}, {}, {}, {}, {}
    tc = cfg['timing']

    def dump():
        np.savez_compressed(out / f'W_{mode}.npz', **Wstore, **{f'w0|{k}': v for k, v in w0store.items()},
                            **{f'cases|{k}': v for k, v in Cstore.items()})

    def rollouts(sname, sol, rule, prep, specs, lean):
        """Run specs on every case. Returns {speckey: dict(W (cases, 6, R'), recs, elig)}."""
        res = {}
        for spec in specs:
            sk = C3.spec_key(spec)
            arr = np.full((len(cases), 6, sol.Rp), np.nan)
            recs = []
            t_s = time.perf_counter()
            for ci, j in enumerate(cases):
                t1 = time.perf_counter()
                o, W, w0 = sol.run(spec, U0[j], NU[j], prep)
                jax.block_until_ready(o)
                secs = time.perf_counter() - t1
                W, w0 = np.asarray(W), np.asarray(w0)
                arr[ci] = W
                if sname not in w0store:
                    w0store[sname] = np.full((len(cases), sol.Rp), np.nan)
                if np.isnan(w0store[sname][ci]).all():
                    w0store[sname][ci] = w0
                assert np.array_equal(w0, w0store[sname][ci]), 'initial coefficients differ across runs (A1.4)'
                v = C3.run_valid(spec, o)
                ns_, st_ = C3.nonstationary(spec, o)
                recs.append((v, ns_, st_))
                F = np.asarray(TB.decode(jnp.asarray(W), lean))       # mesh rows; at n = 65 the interior is the lattice
                e = np.linalg.norm(F - REF[j], axis=1) / N0R[j]
                row = dict(setting=sname, rule=rule, spec=sk, case=j, valid=v, nonstationary=ns_, steps_n=st_,
                           seconds_first=secs, e_ref_per_time=e.tolist(), e_ref=float(e[1:].max()),
                           steps=C3.step_record(spec, o))
                if spec[0] == 'fs':
                    row['a_eff'] = sol.fs(spec[1], spec[2], spec[3]).a_eff
                rep['rows'].append(row)
                if ci < tc['cases']:
                    acc_sha[(sname, rule, sk, ci)] = sha(W)
                del o
            el_ = C3.pooled_eligible(recs)
            res[sk] = dict(W=arr, recs=recs, elig=el_)
            Wstore[f'{sname}|{rule}|{sk}'] = arr
            Cstore[f'{sname}|{rule}|{sk}'] = np.array(list(cases))           # explicit case ids (codex-code-4 item 3)
            print('RUN', sname, rule, sk, f'{time.perf_counter() - t_s:.1f}s', 'eligible', el_['eligible'],
                  el_['nonstationary'], '/', el_['steps'], el(), flush=True)
        return res

    def dist(A_, B_, Lt):
        """per case: max over evolved times of ||G^T (a - b)|| / ||u0||, Gram G G^T = L L^T, rows (a - b) @ L = @ Lt.T."""
        d = np.linalg.norm(np.einsum('cki,ji->ckj', A_ - B_, Lt), axis=2) / N0[:, None]
        per = d[:, 1:].max(1)
        return (float(per.max()) if np.isfinite(per).all() else float('nan')), per.tolist()

    def time_panel(label, A, Bs, get, seed, ncases, reps):
        """A-B-A panel; A, Bs = (setting, rule, spec); get(setting, rule) -> (sol, prep)."""
        name = lambda x: f'{x[0]}|{x[1]}|{C3.spec_key(x[2])}'

        def one(x, ci):
            sol, prep = get(x[0], x[1])
            j = cases[ci]
            C3.burn(tc['burn'])
            t1 = time.perf_counter()
            o, W, _ = sol.run(x[2], U0[j], NU[j], prep)
            jax.block_until_ready(o)
            secs = time.perf_counter() - t1
            return secs, sha(np.asarray(W)), acc_sha.get((x[0], x[1], C3.spec_key(x[2]), ci))
        for x in [A] + Bs:
            for ci in range(ncases):
                one(x, ci)
        rng = np.random.default_rng(seed)
        inv = []
        for rr in range(reps):
            order_ = [(b, ci) for b in range(len(Bs)) for ci in range(ncases)]
            rng.shuffle(order_)
            for b, ci in order_:
                ta1, h1, e1 = one(A, ci)
                tb, hb, eb = one(Bs[b], ci)
                ta2, h2, _ = one(A, ci)
                inv.append(dict(B=name(Bs[b]), case=cases[ci], rep=rr, tA1=ta1, tB=tb, tA2=ta2, ratio=tb / (.5 * (ta1 + ta2)),
                                drift=ta2 / ta1, A_sha=[h1, h2], A_expected=e1, B_sha=hb, B_expected=eb,
                                hashes_ok=bool(e1 is not None and eb is not None and h1 == e1 and h2 == e1 and hb == eb)))
        rep['timing'][label] = dict(A=name(A), subjects=[name(x) for x in Bs], invocations=inv, cases=cases[:ncases],
                                    reps=reps, burn=tc['burn'], seed=seed, gpu=C3.gpu_info(), **TM.panel_validity(inv))
        save()
        print('TIMING', label, len(inv), 'valid', rep['timing'][label]['valid'], el(), flush=True)

    for k_set, s in enumerate(cfg['settings']):
        sname, bname, Rp, a_ = s['name'], s['bank'], int(s['Rp']), int(s['a'])
        bk, tb = banks[bname], tables[bname]
        trust = float(cfg['trust_fraction'] * bk['spread'][str(Rp)])
        sol = C3.Solvers(n, Rp, trust, cfg['gtol'], a_)
        lean = tb.lean(Rp)
        Lt = np.asarray(lean['L']).T
        S_rep = rep['settings'][sname] = dict(bank=bname, Rp=Rp, M=tb.M_of[Rp], trust=trust, a=a_)
        preps = {}

        def prep_of(rule, _preps=preps, _tb=tb, _Rp=Rp):
            if rule not in _preps:
                X, w = rules[rule]
                _preps[rule] = C3.prepare(_tb.rule_data(_Rp, X, w))
            return _preps[rule]

        def drop(rule, _preps=preps):
            _preps.pop(rule, None)
            gc.collect()

        if mode == 'esc' and s.get('ext_rules'):
            # ---- A11 (POST HOC) escalation extension: rules run in order, each dropped after its rollouts; every
            # consecutive pair gets the selection axis (fixed BE, fixed CN K=2), the adaptive axis (both adaptive arms) and
            # the A10 post-hoc axis (fixed BE + adaptive CN); the last rule runs only if the previous pair fails the A10 axis.
            # No ladder, no operator check, no timing panel.
            ext, pairs = {}, {}

            def elig_x(rule, spec):
                e = dict(ext[rule][C3.spec_key(spec)]['elig'])
                if spec[0] == 'fs':
                    adx = DIAG[SEL.index(spec)]
                    dw, _ = dist(ext[rule][C3.spec_key(spec)]['W'], ext[rule][C3.spec_key(adx)]['W'], Lt)
                    adv = ext[rule][C3.spec_key(adx)]['elig']['all_valid']
                    e.update(solver_distance=dw, adaptive_valid=adv,
                             eligible=bool(e['all_valid'] and adv and np.isfinite(dw) and dw <= 2.5e-4))
                else:
                    e['eligible'] = bool(e['all_valid'])
                return e

            def axis_x(r1, r2, specs):
                det, sts = {}, []
                for spec in specs:
                    sk = C3.spec_key(spec)
                    worst, per = dist(ext[r1][sk]['W'], ext[r2][sk]['W'], Lt)
                    el1, el2 = elig_x(r1, spec), elig_x(r2, spec)
                    stt = ('unresolved' if not (el1['eligible'] and el2['eligible']) or not np.isfinite(worst)
                           else ('pass' if worst <= cfg['conv_bar'] else 'fail'))
                    det[sk] = dict(worst=worst, per_case=per, status=stt, eligibility=[el1, el2])
                    sts.append(stt)
                return ('unresolved' if 'unresolved' in sts else ('pass' if all(x == 'pass' for x in sts) else 'fail')), det
            rr = list(s['ext_rules'])
            for i, rule in enumerate(rr):
                if i == len(rr) - 1 and i >= 2 and pairs[f'{rr[i - 2]}|{rr[i - 1]}']['posthoc_status'] == 'pass':
                    break                                       # conditional last rule not needed
                res_ = rollouts(sname, sol, rule, prep_of(rule), SEL + DIAG, lean)
                ext[rule] = {k: dict(W=v['W'], elig=v['elig']) for k, v in res_.items()}
                drop(rule)
                if i:
                    r1 = rr[i - 1]
                    sp_, sd = axis_x(r1, rule, SEL)
                    ad_, ad = axis_x(r1, rule, DIAG)
                    ph_, phd = axis_x(r1, rule, [SEL[0], DIAG[1]])
                    pairs[f'{r1}|{rule}'] = dict(selection_pair_status=sp_, adaptive_diagnostic_status=ad_,
                                                 posthoc_status=ph_, selection=sd, diagnostic=ad, posthoc=phd)
                    print('PAIR', sname, r1, rule, sp_, ad_, ph_, el(), flush=True)
                save()
            S_rep['extension'] = dict(rules=list(ext), pairs=pairs, bar=cfg['conv_bar'],
                                      rule_eligibility={r_: {C3.spec_key(x): elig_x(r_, x) for x in SEL + DIAG} for r_ in ext})
            S_rep['peak_bytes'] = C3.peak_bytes()
            for rule in list(preps):
                drop(rule)
        elif mode == 'esc':
            esc = {}
            for rule in s['esc_rules']:
                esc[rule] = rollouts(sname, sol, rule, prep_of(rule), SEL + DIAG, lean)
                if rule != 'gl64':
                    drop(rule)

            def elig_of(rule, spec):
                """A6.2: fixed-sweep arms: all runs valid and worst distance to the adaptive tight solve of the same scheme
                and step <= 2.5e-4; adaptive arms: every step verified (pooled_eligible with zero allowance via nfail)."""
                e = dict(esc[rule][C3.spec_key(spec)]['elig'])
                if spec[0] == 'fs':
                    adx = DIAG[SEL.index(spec)]
                    dw, _ = dist(esc[rule][C3.spec_key(spec)]['W'], esc[rule][C3.spec_key(adx)]['W'], Lt)
                    adv = esc[rule][C3.spec_key(adx)]['elig']['all_valid']
                    e.update(solver_distance=dw, adaptive_valid=adv,
                             eligible=bool(e['all_valid'] and adv and np.isfinite(dw) and dw <= 2.5e-4))
                else:
                    e['eligible'] = bool(e['all_valid'])
                return e

            def axis_status(r1, r2, specs):
                det, sts = {}, []
                for spec in specs:
                    sk = C3.spec_key(spec)
                    worst, per = dist(esc[r1][sk]['W'], esc[r2][sk]['W'], Lt)
                    el1, el2 = elig_of(r1, spec), elig_of(r2, spec)
                    if not (el1['eligible'] and el2['eligible']) or not np.isfinite(worst):
                        stt = 'unresolved'
                    else:
                        stt = 'pass' if worst <= cfg['conv_bar'] else 'fail'
                    det[sk] = dict(worst=worst, per_case=per, status=stt, eligibility=[el1, el2])
                    sts.append(stt)
                total = 'unresolved' if 'unresolved' in sts else ('pass' if all(x == 'pass' for x in sts) else 'fail')
                return total, det

            pairs = {}

            def do_pair(r1, r2):
                sp_, sd = axis_status(r1, r2, SEL)
                ad_, ad = axis_status(r1, r2, DIAG)
                pairs[f'{r1}|{r2}'] = dict(selection_pair_status=sp_, adaptive_diagnostic_status=ad_, selection=sd,
                                           diagnostic=ad)
                return sp_
            rr = s['esc_rules']
            for i in range(len(rr) - 1):
                do_pair(rr[i], rr[i + 1])
            conv = None
            if pairs['gl56|gl64']['selection_pair_status'] == 'pass':
                conv = 'gl56'
            else:
                esc['gl80'] = rollouts(sname, sol, 'gl80', prep_of('gl80'), SEL + DIAG, lean)
                drop('gl80')
                if do_pair('gl64', 'gl80') == 'pass':
                    conv = 'gl64'
            sens = {}
            for rule in esc:
                for fsx, adx in zip(SEL, DIAG):
                    worst, per = dist(esc[rule][C3.spec_key(fsx)]['W'], esc[rule][C3.spec_key(adx)]['W'], Lt)
                    sens[f'{rule}|{C3.spec_key(fsx)}|vs|{C3.spec_key(adx)}'] = dict(worst=worst, per_case=per)
            # operator-level check on common states: the 64^3 adaptive LSPG-BE reached states (A1.7, A2.5)
            Cs = esc['gl64'][C3.spec_key(DIAG[0])]['W'][:, 1:].reshape(-1, Rp)
            keep = np.isfinite(Cs).all(1)
            Csj = jnp.asarray(Cs[keep])
            CH = 16                                          # chunked over states (A7.3: bounded intermediate memory)
            adv = lambda d_, X_: jnp.concatenate([OM.adv_offmesh_batch(d_, X_[i_:i_ + CH]) for i_ in range(0, X_.shape[0], CH)])
            Nf = adv(prep_of('gl64')['base'], Csj)
            den = np.asarray(jnp.linalg.norm(Nf, axis=1))
            opc = {}
            for rule in esc:
                if rule == 'gl64':
                    continue
                Nr = adv(prep_of(rule)['base'], Csj)
                num = np.asarray(jnp.linalg.norm(Nr - Nf, axis=1))
                ok = den > 0
                rel = num[ok] / den[ok]
                opc[f'{rule}|gl64'] = dict(worst=float(rel.max()) if ok.any() else None,
                                           median=float(np.median(rel)) if ok.any() else None, states=int(len(den)),
                                           undefined=int((~ok).sum()), nonfinite_states_dropped=int((~keep).sum()))
                del Nr
                if rule != conv:
                    drop(rule)
            del Nf
            if conv != 'gl64':
                drop('gl64')
            S_rep['escalation'] = dict(pairs=pairs, converged=conv, bar=cfg['conv_bar'], fixed_vs_adaptive=sens,
                                       operator_check=opc, status='pass' if conv else 'unresolved')
            save()
            # no converged rule: the ladder and the family panels still run against a DIAGNOSTIC baseline (the finest rule
            # run), with no selection claim (codex-job-smk item 1); S_rep['ladder_baseline'] records which
            base_rule = conv if conv is not None else ('gl80' if 'gl80' in esc else 'gl64')
            S_rep['ladder_baseline'] = dict(rule=base_rule, diagnostic_only=conv is None)
            if True:
                conv_ = conv
                conv = base_rule
                # bounded residency (code audit 2 item 11): only the converged rule and ONE ladder rule are resident; the
                # family panel runs per ladder rule (A = converged rule's fixed BE dt0, B = that rule's two selection arms)
                A = (sname, conv, SEL[0])
                time_panel(f'family|{sname}|{conv}', A, [(sname, conv, SEL[1])], lambda sn, rule: (sol, prep_of(rule)),
                           int(tc['seed']) + 100 * k_set, tc['cases'], tc['reps'])
                lad = {}
                for i_r, rule in enumerate(s['ladder']):
                    esc[rule] = rollouts(sname, sol, rule, prep_of(rule), SEL + DIAG, lean)     # A7.1: adaptive counterparts
                    lad[rule] = {C3.spec_key(x): elig_of(rule, x) for x in SEL}
                    time_panel(f'family|{sname}|{rule}', A, [(sname, rule, x) for x in SEL],
                               lambda sn, rl: (sol, prep_of(rl)), int(tc['seed']) + 100 * k_set + 1 + i_r, tc['cases'], tc['reps'])
                    drop(rule)
                conv = conv_
                S_rep['ladder_eligibility'] = lad
                S_rep['escalation_rule_eligibility'] = {r_: {C3.spec_key(x): elig_of(r_, x) for x in SEL}
                                                        for r_ in s['esc_rules'] + (['gl80'] if 'gl80' in esc else [])}
            S_rep['peak_bytes'] = C3.peak_bytes()
            for rule in list(preps):
                drop(rule)
        elif mode == 'bud':
            # ---- 3D error budget (A1.4, A1.5, A2.4, A3.6): all runs on the budget cases; selected arms' r_inf and finer
            # runs on every case; every r_inf-dependent run also with the finer rule on the budget cases (conv check)
            nb = int(cfg['budget_cases'])
            allc = list(cases)
            sub = cases[:nb]
            rmap = dict(r=s['rule'], conv=s['conv'], finer=s['finer'])
            plan = bud_specs(s['arms'])
            sel_specs = [tuple(x['spec']) for x in s['arms'] if x.get('selected')]
            for role in ('r', 'conv', 'finer'):
                rule = rmap[role]
                specs_all = [sp for rl, sp in plan if rl == role and (sp in sel_specs and role in ('r', 'conv', 'finer'))]
                specs_sub = [sp for rl, sp in plan if rl == role and sp not in specs_all]
                if role == 'finer':          # conv check: every r_inf-dependent run type on the budget cases
                    specs_sub += [sp for rl, sp in plan if rl == 'conv' and sp not in specs_all and sp not in specs_sub]
                prep = prep_of(rule)
                if specs_all:
                    rollouts(sname, sol, rule, prep, specs_all, lean)
                S_rep.setdefault('case_sets', {}).update({f'{rule}|{C3.spec_key(x)}': 'all' for x in specs_all})
                S_rep['case_sets'].update({f'{rule}|{C3.spec_key(x)}': f'first {nb}' for x in specs_sub})
                if specs_sub:
                    cases[:] = sub
                    try:
                        rollouts(sname, sol, rule, prep, specs_sub, lean)
                    finally:
                        cases[:] = allc
                prep = None              # release the local reference BEFORE the next rule is prepared (c3budW1024 5231969
                drop(rule)               # ran out of memory preparing Gauss 96^3 while Gauss 80^3 was still referenced here)
            # B1 sham (bitwise re-run) and B2 known effect (control rule lat256) of the BASELINE fs BE dt0 K=1 (the same
            # identity as analyze3d's comparator; G-FS1 makes it equal to the vendor) on the budget cases
            base_sp = ('fs', 'BE', DT0, 1)
            prep = prep_of(s['rule'])
            b1 = []
            for j in sub:
                o1 = sol.run(base_sp, U0[j], NU[j], prep)[1]
                o2 = sol.run(base_sp, U0[j], NU[j], prep)[1]
                b1.append(bool(np.array_equal(np.asarray(o1), np.asarray(o2))))
            if not all(b1):
                print('B1 FAILED: stopping', flush=True)
                dump()
                sys.exit(3)
            cases[:] = sub
            try:
                ctl = rollouts(sname, sol, cfg['control_rule'], prep_of(cfg['control_rule']), [base_sp], lean)
                ref_ = rollouts(sname, sol, s['rule'], prep, [base_sp], lean) if f"{sname}|{s['rule']}|fs|BE|0.01|K1" not in Wstore \
                    else None
            finally:
                cases[:] = allc
            drop(cfg['control_rule'])
            vk = f"{sname}|{s['rule']}|fs|BE|0.01|K1"
            base_W = Wstore[vk]
            ids = list(Cstore[vk])
            sel_rows = [ids.index(j) for j in sub]
            dctl = (np.linalg.norm(np.einsum('cki,ji->ckj', ctl['fs|BE|0.01|K1']['W'] - base_W[sel_rows], Lt), axis=2)
                    / N0[:nb, None])[:, 1:].max(1)
            S_rep.update(rules=rmap, budget_cases=nb, arms=s['arms'], peak_bytes=C3.peak_bytes(),
                         B1_sham_bitwise=dict(all_equal=all(b1), cases=len(b1)),
                         B2_control_rule=dict(rule=cfg['control_rule'], per_case=dctl.tolist(),
                                              complete=bool(ctl['fs|BE|0.01|K1']['elig']['all_valid'] and np.isfinite(dctl).all()),
                                              discriminating=bool(ctl['fs|BE|0.01|K1']['elig']['all_valid'] and np.isfinite(dctl).all()
                                                                  and np.mean(dctl > 2.5e-4) > .5)))
        else:
            rule = s['rule']
            prep = prep_of(rule)
            # ---- G-FS1 first (A2.1): vendor vs fs BE dt0 K=1 on every case; CN negative control on the first case
            g = rollouts(sname, sol, rule, prep, [('vendor',), ('fs', 'BE', DT0, 1)], lean)
            Wv, Wf = g['vendor|BE|0.01|K1']['W'], g['fs|BE|0.01|K1']['W']
            per = [float(np.max(np.linalg.norm(Wf[ci] - Wv[ci], axis=1)) / np.linalg.norm(Wv[ci][0])) for ci in range(len(cases))]
            rv = [r for r in rep['rows'] if r['setting'] == sname and r['spec'] == 'vendor|BE|0.01|K1']
            rf = [r for r in rep['rows'] if r['setting'] == sname and r['spec'] == 'fs|BE|0.01|K1']
            steps_eq = all(x['steps']['reason'] == y['steps']['reason'] and x['steps']['it'] == y['steps']['it']
                           for x, y in zip(rv, rf))
            finite_all = bool(np.isfinite(Wv).all() and np.isfinite(Wf).all() and np.isfinite(per).all())
            exact = bool(finite_all and all(x <= 1e-12 for x in per) and steps_eq)
            Fv = np.stack([np.asarray(TB.decode(jnp.asarray(Wv[ci]), lean)) for ci in range(len(cases))])
            Ff = np.stack([np.asarray(TB.decode(jnp.asarray(Wf[ci]), lean)) for ci in range(len(cases))])
            fper = (np.linalg.norm(Ff - Fv, axis=2) / N0[:, None]).max(1)
            exits = lambda r: np.bincount(np.asarray(r['steps']['reason']), minlength=5).tolist()
            approved = bool(finite_all and np.isfinite(fper).all() and all(x <= 1e-8 for x in per) and
                            all(x <= 1e-8 for x in fper) and
                            [x['valid'] for x in rv] == [x['valid'] for x in rf] and
                            [exits(x) for x in rv] == [exits(x) for x in rf] and
                            C3.pooled_eligible([(x['valid'], x['nonstationary'], x['steps_n']) for x in rv])['eligible'] ==
                            C3.pooled_eligible([(x['valid'], x['nonstationary'], x['steps_n']) for x in rf])['eligible'])
            cn = rollouts(sname, sol, rule, prep, [('fs', 'CN', DT0, 1)], lean)['fs|CN|0.01|K1']['W']
            c0 = cases.index(0)                    # named control case 923801/0 (A1.2)
            ctrl = float(np.max(np.linalg.norm(cn[c0] - Wv[c0], axis=1)) / np.linalg.norm(Wv[c0][0]))
            S_rep['G_FS1'] = dict(per_case_coef_rel=per, per_case_field=fper.tolist(), steps_equal=steps_eq,
                                  outcome='exact' if exact else ('approved-equivalent' if approved else 'FAIL'),
                                  cn_control=ctrl, cn_control_discriminating=bool(np.isfinite(ctrl) and ctrl > 1e-6),
                                  finite_all=finite_all)
            save()
            if not (exact or approved):
                dump()
                print('G-FS1 FAILED: stopping before any verdict', S_rep['G_FS1']['outcome'], flush=True)
                sys.exit(3)
            specs = [x for x in grid_specs(cfg['grid']) if C3.spec_key(x) not in ('fs|BE|0.01|K1', 'fs|CN|0.01|K1')]
            rollouts(sname, sol, rule, prep, specs, lean)
            S_rep['peak_bytes'] = C3.peak_bytes()
            live[sname] = (sol, rule, prep)
            allsp = [('vendor',)] + grid_specs(cfg['grid'])
            Bs = [(sname, rule, x) for x in allsp if (x[0] in ('fs', 'vendor') and x != ('fs', 'BE', DT0, 1)) or timed_adaptive(x)]
            time_panel(f'setting|{sname}', (sname, rule, ('fs', 'BE', DT0, 1)), Bs,
                       lambda sn, rl: (live[sn][0], live[sn][2]), int(tc['seed']) + 1 + k_set, tc['cases'], tc['reps'])
        dump()
        save()

    if mode == 'grid':
        base = cfg['common_baseline'] if cfg['common_baseline'] in live else cfg['common_baseline_fallback']
        A = (base, live[base][1], ('fs', 'BE', DT0, 1))
        Bs = [(sname, rule, x) for sname, (sol, rule, prep) in live.items() for x in grid_specs(cfg['grid'])
              if x[0] == 'fs' and x[2] >= DT0 - 1e-12 and (sname, rule, x) != A]
        time_panel('common', A, Bs, lambda sn, rl: (live[sn][0], live[sn][2]), int(tc['common_seed']),
                   tc['common_cases'], tc['common_reps'])
    rep['peak_bytes_job'] = C3.peak_bytes()
    rep['elapsed_seconds'] = el()
    rep['complete'] = True
    save()
    dump()
    (out / 'COMPLETE').write_text('complete\n')
    print('C3RUN COMPLETE', flush=True)


if __name__ == '__main__':
    main()
