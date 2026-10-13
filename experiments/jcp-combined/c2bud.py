"""jcp-combined 2D error-budget job `c2bud` (DESIGN 5 as amended by A1.4, A1.5, A2.3, A2.4, A3.6; runs after the frozen
selection). One GPU, all dev6 + val32 cases. Every run is an end-to-end query of t2core (C2) on one setting; only the
ingredient named changes; the initial coefficients are identical across every run of a setting and case (asserted).

Per setting (name, R', main rule r) and budget arm a = (form f, scheme s, step factor h, production tolerance) from
checks/FROZEN-SELECTION.json (selected arms, flagged) plus the pre-specified diagnostic arms (A1.5):
  a                               the arm
  a[r -> r_inf]                   quadrature ablation (r_inf = converged rule, Gauss 320^2)
  a[sigma -> tight]               solver ablation
  a[sigma -> tight, f -> GAL]     form sensitivity at tight tolerance (LSPG arms only)
  a[(s, h) -> (BDF2, h0/8)]       temporal-discretisation package (same form, rule, tolerance)
  a[sigma -> tight, h -> h/2]     within-scheme refinement (with a[tight])
  mixed-difference corners        (r_inf, f, BDF2, h0/8, prod) and (r, f, BDF2, h0/8, tight)
Per setting: the ideal star = (r_inf, GAL, BDF2, h0/8, tight) with its qualification (h0/16 tight; tighter replays of
both), the ideal-end changes star[r_inf -> r] (= the deployed-rule anchor) and star[GAL -> LSPG].
Converged-rule check (A2.4): on dev6 + the first 10 val32 cases, every run using r_inf is repeated with the check rule
(Gauss 640^2); bar 2.5e-5 (worst field distance on the 257^2 nodes / ||u0||). If it fails, the successor (r_inf = 640^2,
check 768^2) is run instead, decided in-job before the remaining cases; failing again -> 'converged rule unverified'.
Finalists (A3.6): every SELECTED arm also runs with the check rule on every case.
Outputs: result.json (rows: errors vs ST / S / TX per time, verification, statistics), WB_<setting>.npz (coefficients of
every run), COMPLETE. All distances and the budget are computed offline (budget2d.py) from the saved coefficients.

    python c2bud.py --config configs/c2bud.json --out <dir> [--dry-run]
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import c2common as CC
from c2common import Q, QS, T2, DT0, sha
import jax
import jax.numpy as jnp

H8 = DT0 / 8
H16 = DT0 / 16


def budget_runs(arms, conv='conv'):
    """Deduplicated run list (role, form, scheme, dt, level) for one setting."""
    R = []
    add = lambda r: R.append(r) if r not in R else None
    for x in arms:
        f, s, h = x['form'], x['scheme'], DT0 * x['f']
        add(('main', f, s, h, 'prod'))
        add((conv, f, s, h, 'prod'))
        add(('main', f, s, h, 'tight'))
        if f == 'LSPG':
            add(('main', 'GAL', s, h, 'tight'))
        add(('main', f, 'BDF2', H8, 'prod'))
        add(('main', f, s, h / 2, 'tight'))
        add((conv, f, 'BDF2', H8, 'prod'))
        add(('main', f, 'BDF2', H8, 'tight'))
        if x.get('selected'):
            add(('check', f, s, h, 'prod'))
    for r in [(conv, 'GAL', 'BDF2', H8, 'tight'), (conv, 'GAL', 'BDF2', H16, 'tight'), (conv, 'GAL', 'BDF2', H8, 'tighter'),
              (conv, 'GAL', 'BDF2', H16, 'tighter'), ('main', 'GAL', 'BDF2', H8, 'tight'), (conv, 'LSPG', 'BDF2', H8, 'tight')]:
        add(r)
    return R


def key(r):
    role, form, sc, dt, lev = r
    return f'{role}|{form}|{sc}|{dt:.8g}|{lev}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    if a.dry_run:
        inv = {s['name']: [key(r) for r in budget_runs(s['arms'])] for s in cfg['settings']}
        print(json.dumps(dict(runs=inv, steps={k: sum(int(round(.25 / float(x.split('|')[3]))) for x in v)
                                                for k, v in inv.items()}), indent=1))
        return
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    CC.check_env(cfg)
    begin = time.perf_counter()
    el = lambda: round(time.perf_counter() - begin, 1)
    L = int(cfg['mesh'])
    s256 = max(1, L // 256)
    rep = dict(config=cfg, commit=CC.os.environ.get('SOURCE_COMMIT'), job_id=CC.os.environ.get('SLURM_JOB_ID'),
               backend=jax.default_backend(), gpu=jax.devices()[0].device_kind, nvidia_smi=CC.gpu_info(),
               jax_version=jax.__version__, mesh=L, checkpoint_sha256=QS.sha_file(T2.CKPT), cohorts={}, references={},
               setup={}, rows=[], conv_check={}, complete=False)
    save = lambda: json.dump(QS.clean(rep), open(out / 'result.json', 'w'), indent=1)
    QS.burn(.05)
    cases = CC.load_cases(cfg, rep)
    refs = CC.load_refs(cfg, cases, rep)
    bank = CC.Bank(cfg, L)
    U0 = {}
    for coh, c, ph in cases:
        u0n = Q.e.initial(L, ph)
        U0[(coh, c)] = (jnp.asarray(u0n), float(ph[4]), float(np.linalg.norm(u0n[::s256, ::s256])))
    check_set = [i for i, (coh, c, _) in enumerate(cases) if coh == 'dev6'] + \
                [i for i, (coh, c, _) in enumerate(cases) if coh == 'val32'][:cfg['check_val32']]
    order = check_set + [i for i in range(len(cases)) if i not in check_set]
    print('SETUP', el(), flush=True)
    for spec in cfg['settings']:
        conv_rule, check_rule = cfg['conv_rule'], cfg['check_rule']
        rules = {'conv': conv_rule, 'check': check_rule, 'ctrl': cfg['control_rule']}
        st = CC.Setting(bank, spec, rules=rules)
        S_rep = rep['setup'][spec['name']] = dict(Rp=st.Rp, M=st.M, rule=spec['rule'], m=st.m, arms=spec['arms'],
                                                   conv_rule=conv_rule, check_rule=check_rule, successor_used=False)
        runs = budget_runs(spec['arms'])
        conv_runs = [r for r in runs if r[0] == 'conv']
        Wset = {}
        w0set = {}

        def do(r, ci, role_override=None):
            coh, c, ph = cases[ci]
            u0, nu, n0r = U0[(coh, c)]
            role = role_override or r[0]
            t1 = time.perf_counter()
            v = st.call(role, r[1], r[2], r[3], r[4], u0, nu)
            jax.block_until_ready(v)
            secs = time.perf_counter() - t1
            W, w0 = np.asarray(v['W']), np.asarray(v['w0'])
            if ci not in w0set:
                w0set[ci] = w0
            assert np.array_equal(w0, w0set[ci]), 'initial coefficients differ across budget runs (A1.4)'
            stt = {k: np.asarray(x) for k, x in v['stats'].items()}
            fr = np.asarray(v['fields'][:, ::s256, ::s256])
            kk = key((role,) + tuple(r[1:]))
            rep['rows'].append(dict(setting=spec['name'], run=kk, cohort=coh, case=c, seconds_first=secs, stats=stt,
                                    verified=bool(int(stt['nfail']) == 0 and np.isfinite(W).all()),
                                    **CC.errs(fr, refs, coh, c, n0r)))
            Wset.setdefault(kk, np.full((len(cases), 6, st.Rp), np.nan))[ci] = W
            return W, fr, n0r

        def field_dist(A_, B_, n0r):
            return max(float(np.linalg.norm(A_[j] - B_[j])) / n0r for j in range(1, 6))

        # ---- converged-rule check on the check set first (A2.4), successor decided in-job (codex-code-4 item 4):
        # acceptance needs every pair finite AND both rows verified; otherwise the check fails
        def conv_check():
            worst, ok = 0., True
            for ci in check_set:
                for r in conv_runs:
                    Wa, fa, n0r = do(r, ci)
                    Wb, fb, _ = do(r, ci, role_override='check')
                    d_ = field_dist(fa, fb, n0r)
                    rv = rep['rows'][-2:]
                    ok = ok and bool(np.isfinite(d_) and all(x['verified'] for x in rv))
                    worst = max(worst, d_) if np.isfinite(d_) else float('inf')
            return worst, ok

        worst, ok = conv_check()
        S_rep['conv_check'] = dict(pair=[conv_rule, check_rule], worst=worst, bar=cfg['conv_bar'], cases=len(check_set),
                                   all_verified=ok, passed=bool(ok and worst <= cfg['conv_bar']))
        save()
        if S_rep['conv_check']['passed']:
            S_rep['converged_rule_status'] = 'verified'
        else:
            # successor: r_inf = 640^2 checked against 768^2
            del st
            gc.collect()
            Wset.clear()
            w0set.clear()
            rep['rows'] = [x for x in rep['rows'] if x['setting'] != spec['name']]
            conv_rule, check_rule = cfg['check_rule'], cfg['successor_check']
            st = CC.Setting(bank, spec, rules={'conv': conv_rule, 'check': check_rule, 'ctrl': cfg['control_rule']})
            worst, ok = conv_check()
            S_rep['conv_check_successor'] = dict(pair=[conv_rule, check_rule], worst=worst, bar=cfg['conv_bar'],
                                                 all_verified=ok, passed=bool(ok and worst <= cfg['conv_bar']))
            if S_rep['conv_check_successor']['passed']:
                S_rep.update(successor_used=True, conv_rule=conv_rule, check_rule=check_rule,
                             converged_rule_status='verified (successor)')
            else:
                # A1.4: report against Gauss 320^2 labelled 'converged rule unverified' (no quadrature attribution)
                del st
                gc.collect()
                Wset.clear()
                w0set.clear()
                rep['rows'] = [x for x in rep['rows'] if x['setting'] != spec['name']]
                conv_rule, check_rule = cfg['conv_rule'], cfg['check_rule']
                st = CC.Setting(bank, spec, rules={'conv': conv_rule, 'check': check_rule, 'ctrl': cfg['control_rule']})
                S_rep.update(successor_used=False, conv_rule=conv_rule, check_rule=check_rule,
                             converged_rule_status='converged rule unverified')
            save()
        # ---- B1 sham (bitwise re-run of the baseline arm) and B2 known effect (control rule) on the check set (A1.10)
        b1, b2 = [], []
        base_r = ('main', 'LSPG', 'BE', DT0, 'prod')
        for ci in check_set:
            coh, c, ph = cases[ci]
            u0, nu, n0r = U0[(coh, c)]
            v1 = st.call(*base_r, u0, nu)
            v2 = st.call(*base_r, u0, nu)
            b1.append(bool(np.array_equal(np.asarray(v1['W']), np.asarray(v2['W']))))
            _, fc_, _ = do(base_r, ci, role_override='ctrl')
            fm_ = np.asarray(v1['fields'][:, ::s256, ::s256])
            okv = bool(rep['rows'][-1]['verified'] and int(np.asarray(v1['stats']['nfail'])) == 0)
            b2.append(field_dist(fm_, fc_, n0r) if okv else float('nan'))
        S_rep['B1_sham_bitwise'] = dict(all_equal=all(b1), cases=len(b1))
        if not all(b1):                         # B1 is a stopping gate (codex-code-4 item 5)
            save()
            print('B1 FAILED: a re-run of the same compiled query is not bitwise identical; stopping', flush=True)
            raise SystemExit(3)
        S_rep['B2_control_rule'] = dict(rule=cfg['control_rule'], per_case=b2, exceeds_tau=sum(x > 2.5e-4 for x in b2),
                                        complete=bool(np.isfinite(b2).all()),
                                        discriminating=bool(np.isfinite(b2).all() and sum(x > 2.5e-4 for x in b2) > len(b2) / 2))
        # ---- every run on every case (check-set conv runs were already done; repeats are skipped)
        done = {(r_['run'], r_['cohort'], r_['case']) for r_ in rep['rows'] if r_['setting'] == spec['name']}
        for ci in order:
            coh, c, _ = cases[ci]
            for r in runs:
                if (key(r), coh, c) not in done:
                    do(r, ci)
            print('CASE', spec['name'], ci, coh, c, el(), flush=True)
            if ci % 8 == 7:
                save()
        try:
            S_rep['peak_bytes'] = int(jax.devices()[0].memory_stats().get('peak_bytes_in_use', -1))
        except Exception:  # noqa: BLE001
            S_rep['peak_bytes'] = -1
        np.savez_compressed(out / f"WB_{spec['name']}.npz", **Wset, w0=np.stack([w0set[i] for i in range(len(cases))]),
                            cases=np.array([f'{coh}|{c}' for coh, c, _ in cases]))
        save()
        del st, Wset
        gc.collect()
    rep['elapsed_seconds'] = el()
    rep['complete'] = True
    save()
    (out / 'COMPLETE').write_text('complete\n')
    print('C2BUD COMPLETE', flush=True)


if __name__ == '__main__':
    main()
