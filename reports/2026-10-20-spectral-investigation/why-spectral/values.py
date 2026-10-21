"""Every number quoted in report.md, computed from this folder's analysis JSONs or from lane outputs, with its source.

values() -> {key: (formatted string, source description)}. Used by make_report.py; run standalone to print them.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import common as C

A = C.OUT


def J(name):
    return json.loads((A / name).read_text())


def g(x, d=3):
    if x is None:
        return '–'
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    x = float(x)
    if x != 0 and (abs(x) >= 1e5 or abs(x) < 1e-3):
        m, e = f'{x:.{d - 1}e}'.split('e')
        return f'{m}e{int(e)}'
    return f'{x:.{d}g}'.replace('e+0', 'e').replace('e+', 'e').replace('e-0', 'e-')


def pct(x, d=3):
    return '–' if x is None else g(100 * float(x), d) + ' %'


def ms(x, d=3):
    return g(1e3 * float(x), d) + ' ms'


def fit_pow(s, n):
    s, n = np.log(np.asarray(s, float)), np.log(np.asarray(n, float))
    p = np.polyfit(s, n, 1)
    return float(p[0]), float(math.exp(p[1]))


def first_N(es, target):
    for N in sorted(es, key=int):
        v = es[N]
        if v is not None and v <= target:
            return int(N)
    return None


def values():
    V = {}
    aux = {}

    def put(k, v, src):
        assert k not in V, k
        V[k] = (v, src)

    # ------------------------------------------------------------------ A1 spectra
    a1 = J('a1_spectra.json')['cohorts']
    src1 = 'analysis/a1_spectra.json'
    names = dict(E0_2d='E0 2D dev6+val32', E0_3d='E0 3D dev4+val64', reg_N0='regimes N0', reg_Am='regimes A−',
                 reg_C1='regimes C1', reg_C2='regimes C2')
    for key, rows in a1.items():
        lab = key
        s = np.array([r['s'] for r in rows])
        put(f'{lab}_n', g(len(rows)), f'{src1} cohorts.{key} (rows)')
        put(f'{lab}_smin', g(s.min()), f'{src1} cohorts.{key}[*].s min')
        put(f'{lab}_smax', g(s.max()), f'{src1} cohorts.{key}[*].s max')
        for e in ('0.01', '0.001'):
            Ns = [r['N_proj'][e] for r in rows]
            cens = sum(1 for x in Ns if x is None or x >= 240)
            Nv = np.array([x if x is not None else np.nan for x in Ns], float)
            el = e.replace('0.', 'p')
            put(f'{lab}_Nproj{el}_med', g(np.nanmedian(Nv)), f'{src1} cohorts.{key}[*].N_proj["{e}"] median')
            put(f'{lab}_Nproj{el}_max', g(np.nanmax(Nv)), f'{src1} cohorts.{key}[*].N_proj["{e}"] max over uncensored')
            put(f'{lab}_Nproj{el}_cens', g(cens), f'{src1} number of cases with N_proj["{e}"] >= 240 or beyond the lattice')
        for ti, tl in ((1, 't1'), (5, 't5')):
            sg = [r['sigma'][ti] for r in rows if r['sigma'][ti] is not None]
            put(f'{lab}_sigma_{tl}_med', g(np.median(sg)), f'{src1} cohorts.{key}[*].sigma[{ti}] median')
    # time dependence and the 2D IC
    e2 = a1['E0_2d']
    put('E0_2d_sigma_t0_med', g(np.median([r['sigma'][0] for r in e2 if r['sigma'][0] is not None])),
        f'{src1} cohorts.E0_2d[*].sigma[0] median (t = 0, the nodal IC with zero wall nodes)')
    # algebraic slope of tail at t0 between N = 16 and 128
    sl = []
    for r in e2:
        t0 = np.array(r['tail_t'][0])
        sl.append(math.log(t0[127] / t0[15]) / math.log(128 / 16))
    put('E0_2d_t0_slope', g(np.median(sl), 2), f'{src1} median over cases of log(tail_t0(128)/tail_t0(16)) / log 8')
    # N_proj(1e-2) per output time (median over E0 2D)
    for ti in range(1, 6):
        Nt = []
        for r in e2:
            tl = np.array(r['tail_t'][ti]); idx = np.nonzero(tl <= 1e-2)[0]
            Nt.append(idx[0] + 1)
        put(f'E0_2d_Nt{ti}_med', g(np.median(Nt)), f'{src1} median over cases of smallest N with tail_t{ti}(N) <= 1e-2')
        put(f'E0_2d_Nt{ti}_max', g(np.max(Nt)), f'{src1} max over cases, same')
    # steepness fits (2D, all cohorts, uncensored)
    allrows = [r for k in ('E0_2d', 'reg_N0', 'reg_Am', 'reg_C1', 'reg_C2') for r in a1[k]]
    for e in ('0.01', '0.001'):
        rr = [r for r in allrows if r['N_proj'][e] is not None and r['N_proj'][e] < 240]
        s = [r['s'] for r in rr]; n = [r['N_proj'][e] for r in rr]
        p, c = fit_pow(s, n)
        el = e.replace('0.', 'p')
        put(f'fit_Nproj{el}_alpha', g(p, 2), f'{src1} least squares log N_proj("{e}") = log c + alpha log s, 2D, {len(rr)} cases')
        put(f'fit_Nproj{el}_c', g(c, 2), 'same fit, prefactor c')
        put(f'fit_Nproj{el}_n', g(len(rr)), 'same fit, cases used')
        put(f'sp_Nproj{el}_s', g(spearmanr(s, n)[0], 2), 'Spearman(N_proj, s), same cases')
        put(f'sp_Nproj{el}_nu', g(spearmanr([r['nu'] for r in rr], n)[0], 2), 'Spearman(N_proj, nu)')
        put(f'sp_Nproj{el}_a', g(spearmanr([r['a'] for r in rr], n)[0], 2), 'Spearman(N_proj, a)')
        put(f'sp_Nproj{el}_w', g(spearmanr([r['w'] for r in rr], n)[0], 2), 'Spearman(N_proj, w)')
        X = np.c_[np.ones(len(rr)), np.log([r['a'] for r in rr]), np.log([r['nu'] for r in rr]), np.log([r['w'] for r in rr])]
        cf = np.linalg.lstsq(X, np.log(n), rcond=None)[0]
        put(f'mr_Nproj{el}_a', g(cf[1], 2), 'multiple regression log N_proj on log a, log nu, log w: coefficient of log a')
        put(f'mr_Nproj{el}_nu', g(cf[2], 2), 'same, coefficient of log nu')
        put(f'mr_Nproj{el}_w', g(cf[3], 2), 'same, coefficient of log w')
    # measured front width: N_proj(1e-2) x min_t delta
    prod = [r['N_proj']['0.01'] * min(x for x in r['delta'][1:] if x) for r in allrows if r['N_proj']['0.01'] and r['N_proj']['0.01'] < 240]
    put('Ndelta_med', g(np.median(prod), 2), f'{src1} median over 2D cases of N_proj(1e-2) x min_(t>0) delta(t)')
    put('Ndelta_iqr', f'{g(np.percentile(prod, 25), 2)}–{g(np.percentile(prod, 75), 2)}', 'same, interquartile range')
    sd = [r['sigma'][5] for r in allrows if r['sigma'][5]]
    ss = [r['s'] for r in allrows if r['sigma'][5]]
    put('sp_sigma_s', g(spearmanr(ss, sd)[0], 2), f'{src1} Spearman(sigma at t5, s), 2D all cohorts')
    e3 = a1['E0_3d']
    p3, c3 = fit_pow([r['s'] for r in e3], [r['N_proj']['0.01'] for r in e3])
    put('fit3_Nproj_alpha', g(p3, 2), f'{src1} 3D fit log N_proj(1e-2) vs log s, {len(e3)} cases (narrow s range)')
    # ------------------------------------------------------------------ A2 spectral solver
    gt = J('a2_gate2d.json')['rows']
    put('gate2d_maxrel', g(max(max(v['rel_diff_median'], v['rel_diff_worst']) for v in gt.values()), 2),
        'analysis/a2_gate2d.json: max relative difference of CPU median/worst vs jcp-fom2d in-job (H200) scores')
    a3d = J('a2_e03d.json')['rows']
    f1 = C.J(C.S3D / 'runs/f1/archive/output/result.json')['fom']['ps']['ps|N16|0.025']['cases']
    d3 = [abs(r['gate_N16_0.025'] - f1[k.split('|')[1]]['e']) / f1[k.split('|')[1]]['e'] for k, r in a3d.items() if k.startswith('val64')]
    put('gate3d_maxrel', g(max(d3), 2), 'analysis/a2_e03d.json gate_N16_0.025 vs jcp-speed3d f1 per-case e, max relative difference (64 cases)')
    a4 = J('a4_match.json')
    for N in ('16', '24', '32', '48', '64', '96', '128'):
        v = a4['ps2_summary'][N]
        put(f'ps2_N{N}_med', pct(v['median']), f'analysis/a4_match.json ps2_summary["{N}"].median (CPU, dt 0.0025, 38 cases)')
        put(f'ps2_N{N}_worst', pct(v['worst']), f'analysis/a4_match.json ps2_summary["{N}"].worst')
    for N in ('8', '10', '12', '14', '16', '24', '32'):
        v = a4['ps3_summary'][N]
        put(f'ps3_N{N}_med', pct(v['median']), f'analysis/a4_match.json ps3_summary["{N}"].median (CPU, dt 0.01, val64)')
        put(f'ps3_N{N}_worst', pct(v['worst']), f'analysis/a4_match.json ps3_summary["{N}"].worst')
    sel = ['1024|R128|LSPG|BE|1', '1024|R128|LSPG|CNR|1', '1024|R256|LSPG|BE|1', '1024|R384|LSPG|BE|1',
           '1024|R384|LSPG|CNR|1', '1024|R384|GAL|CN|1', '1024|R512|LSPG|CNR|1', '1024|R512|GAL|CN|1', '1024|R128|LSPG|BE|2.5']
    rows = []
    for k in sel:
        v = a4['rom2d'][k]
        rows.append(f"| {k.split('|', 1)[1].replace('|', ' ')} | {pct(v['median'])} | {pct(v['worst'])} | {v['N_joint']} | {v['N_joint_charged']} | "
                    f"{(v['N_joint'] - 1) ** 2} | {g(v['N_case_median'])} | {g(v['N_case_max'])} |")
    put('tab_match2d', '\n'.join(rows), 'analysis/a4_match.json rom2d[...] (FRONT-2D per-case errors at 1024^2 vs A2)')
    Nj = [v['N_joint'] for v in a4['rom2d'].values() if v['mesh'] == 1024]
    put('Njoint_min', g(min(Nj)), 'analysis/a4_match.json min N_joint over all qualified 1024^2 configurations')
    put('Njoint_max', g(max(Nj)), 'same, max')
    put('Njoint_n', g(len(Nj)), 'same, number of configurations')
    put('Njoint_max_modes', g((max(Nj) - 1) ** 2), '(Njoint_max - 1)^2')
    Njc = [v['N_joint_charged'] for v in a4['rom2d'].values() if v['mesh'] == 1024]
    put('Njointc_min', g(min(Njc)), 'analysis/a4_match.json rom2d[*].N_joint_charged min (spectral at dt 0.0125, 20 steps)')
    put('Njointc_max', g(max(Njc)), 'same, max')
    put('Njointc_max_modes', g((max(Njc) - 1) ** 2), '(Njointc_max - 1)^2')
    N3c = sorted({v['N_joint_charged'] for v in a4['rom3d'].values()})
    put('N3c_joint', '–'.join(str(x) for x in (N3c[0], N3c[-1])) if len(N3c) > 1 else str(N3c[0]), 'analysis/a4_match.json rom3d[*].N_joint_charged (dt 0.025, 10 steps)')
    N3s = sorted({v['N_joint'] for v in a4['rom3d'].values()})
    assert len(N3s) == 1
    put('N3_joint', g(N3s[0]), 'analysis/a4_match.json rom3d[*].N_joint (identical for all four)')
    put('N3_modes', g((N3s[0] - 1) ** 3), '(N3_joint - 1)^3')
    Nc = [v['N_case_median'] for v in a4['rom2d'].values() if v['mesh'] == 1024]
    put('Ncase_med_lo', g(min(Nc)), 'min over configurations of the per-case median N*')
    put('Ncase_med_hi', g(max(Nc)), 'max over configurations of the per-case median N*')
    rows = []
    for k, v in a4['rom3d'].items():
        rows.append(f"| {k} | {pct(v['median'])} | {pct(v['worst'])} | {v['N_joint']} | {v['N_joint_charged']} | {(v['N_joint'] - 1) ** 3} |")
    put('tab_match3d', '\n'.join(rows), 'analysis/a4_match.json rom3d (speed3d m1 L=384 median/worst vs A2 3D)')
    # per-case N* vs s for R384 CN-R, and spectral N*(1e-2)
    ps2 = J('a2_e02d.json')['rows']
    pc = a4['rom2d']['1024|R384|LSPG|CNR|1']['N_case']
    s_ = [ps2[c]['s'] for c in pc]; n_ = [pc[c] for c in pc]
    put('sp_Ncase_s', g(spearmanr(s_, n_)[0], 2), 'Spearman(N*_case for R384 LSPG-CN-R 1024^2, s), 38 cases')
    worst_case = max(pc, key=lambda c: pc[c])
    put('Ncase_worst_case', worst_case, 'case with the largest N*_case for R384 LSPG-CN-R')
    put('Ncase_worst_s', g(ps2[worst_case]['s']), 'its s')
    # regimes spectral solver
    try:
        reg = J('a2_reg.json')['rows']
        put('reg_ok', 'yes', 'analysis/a2_reg.json present')
        regpc = C.J(C.REG / 'checks/analysis.json')['configs']['R384|LSPG_CNR_1']['per_case']
        dts_reg = J('a2_reg.json')['dt']
        rows = []
        Ns_e = {}

        def ncell(Nl, Nmat):
            ok = [x for x in Nl if x]
            nc = len(Nl) - len(ok)
            med = g(np.median([x if x else 10 ** 9 for x in Nl])) if 2 * nc < len(Nl) else f'>{Nmat[-1]}'
            mx = g(max(ok)) if nc == 0 else f'>{Nmat[-1]} ({nc} cases)'
            return f'{med} / {mx}'
        for gname in C.REGIMES:
            cs_all = {k: r for k, r in reg.items() if k.startswith(gname + '|')}
            elig = {k for k in cs_all if regpc.get(k, {}).get('eligible') and regpc[k].get('valid')}
            cs = {k: cs_all[k] for k in sorted(elig)}
            Nmat = sorted(int(n) for n in next(iter(cs.values()))['e'])
            med = {N: np.median([r['e'][str(N)] if r['e'][str(N)] is not None else np.inf for r in cs.values()]) for N in Nmat}
            wst = {N: np.max([r['e'][str(N)] if r['e'][str(N)] is not None else np.inf for r in cs.values()]) for N in Nmat}
            ev = [regpc[k]['e'] for k in cs]
            rm, rw = float(np.median(ev)), float(np.max(ev))
            Nj = next((N for N in Nmat if rm is not None and med[N] <= rm and wst[N] <= rw), None)
            N1 = [first_N(r['e'], 1e-2) for r in cs.values()]
            N3 = [first_N(r['e'], 1e-3) for r in cs.values()]
            Ns_e[gname] = (N1, N3, [r['s'] for r in cs.values()])
            rows.append(f"| {gname.replace('Am', 'A−')} | {len(cs)} of {len(cs_all)} | {g(min(r['s'] for r in cs.values()))}–{g(max(r['s'] for r in cs.values()))} | "
                        f"{pct(rm)} / {pct(rw)} | {Nj if Nj else '–'} | "
                        f"{ncell(N1, Nmat)} | {ncell(N3, Nmat)} |")
        put('tab_reg', '\n'.join(rows), f'analysis/a2_reg.json (CPU, dt per regime {dts_reg}) and jcp-regimes checks/analysis.json configs["R384|LSPG_CNR_1"].per_case (ROM e; same case IDs, eligible and valid only; C1/C2 unqualified)')
        # PS N*(1e-2) vs s fit, E0 + regimes
        S, N1 = [], []
        for r in J('a2_stab.json')['rows'].values():
            n1 = first_N(r['e'], 1e-2)
            if n1:
                S.append(r['s']); N1.append(n1)
        p, c = fit_pow(S, N1)
        put('fitPS_alpha', g(p, 2), f'log N*_PS(1e-2) = log c + alpha log s over E0 2D + regimes at the charged step (analysis/a2_stab.json), {len(S)} resolved cases')
        put('fitPS_c', g(c, 2), 'same fit, prefactor')
        put('fitPS_n', g(len(S)), 'same fit, cases')
        aux['fitPS'] = (p, c)
    except FileNotFoundError:
        put('reg_ok', 'no', 'analysis/a2_reg.json missing')
        for k_ in ('tab_reg', 'fitPS_alpha', 'fitPS_c', 'fitPS_n'):
            put(k_, 'PENDING', 'analysis/a2_reg.json missing')
    # ------------------------------------------------------------------ A2b stability
    st = J('a2b_stab.json')
    prods, dtmins = [], {}
    for k, r in st['rows'].items():
        m = {N: max([dt for dt in st['DTS'] if r['res'][f'{N}|{dt}']['stable']] or [0]) for N in st['NS']}
        mn = min(m.values())
        gname = k.split('|')[0]
        dtmins.setdefault(gname, []).append(mn)
        if mn > 0 and mn < max(st['DTS']):
            prods.append(mn * r['s'])
    brk = []
    for k, r_ in st['rows'].items():
        m = {N: max([dt for dt in st['DTS'] if r_['res'][f'{N}|{dt}']['stable']] or [0]) for N in st['NS']}
        if min(m.values()) == 0:
            brk.append(f"{k} (s = {g(r_['s'])}: c < {g(min(st['DTS']) * r_['s'], 3)})")
    put('stab_excluded', '; '.join(brk), 'a2b cases with no stable tested step at some N: upper limit c < min(dt) * s')
    put('stab_cmin', g(min(prods), 2), 'analysis/a2b_stab.json min over cases (largest stable dt < 0.0125) of dt_max x s')
    put('stab_cmed', g(np.median(prods), 2), 'same, median')
    for gname, v in dtmins.items():
        put(f'stab_{gname}_n0125', g(sum(1 for x in v if x >= 0.0125)), f'a2b {gname}: cases stable at dt 0.0125 for every N tested')
        put(f'stab_{gname}_nnone', g(sum(1 for x in v if x == 0)), f'a2b {gname}: cases with no stable tested dt at some N')
        put(f'stab_{gname}_n', g(len(v)), f'a2b {gname}: cases')
    aux['stab_c'] = min(prods)
    # ------------------------------------------------------------------ A3 bank vs sines
    b3 = J('a3_bank.json')['rows']
    rows = []
    for src_ in ('E0', 'N0', 'Am', 'C1', 'C2'):
        rr = [r for r in b3 if r['src'] == src_]
        def st_(key, r_):
            v = np.array([x['K_match_over_r'][r_] for x in rr if x['K_match_over_r'][r_] is not None])
            return f'{g(np.median(v), 2)} ({g(v.max(), 2)})'
        fb = np.array([x['floor_bank']['384'] for x in rr])
        rows.append(f"| {src_.replace('Am', 'A−')} | {len(rr)} | {g(min(x['s'] for x in rr))}–{g(max(x['s'] for x in rr))} | "
                    f"{pct(np.median(fb))} | {st_('K', '128')} | {st_('K', '384')} | {st_('K', '512')} |")
    put('tab_bank', '\n'.join(rows), 'analysis/a3_bank.json rows[*].K_match_over_r (median (max)) and floor_bank["384"] median')
    ks = [(x['s'], x['K_match_over_r']['384']) for x in b3 if x['K_match_over_r']['384']]
    put('sp_bank_s', g(spearmanr(*zip(*ks))[0], 2), 'Spearman(K_match/r at r = 384, s), all 2D cases of A3')
    e0b = [x for x in b3 if x['src'] == 'E0']
    put('bank_e0_384_med', g(np.median([x['K_match_over_r']['384'] for x in e0b]), 2), 'A3 E0 median K_match/r at 384')
    put('bank_e0_384_max', g(np.max([x['K_match_over_r']['384'] for x in e0b]), 2), 'A3 E0 max K_match/r at 384')
    idea = C.J(C.IDEA / 'analysis/bank_vs_sine.json')['summary']['ratio_sineK_over_r_384']['median']
    put('bank_idea_384_med', g(idea, 2), 'improvement-ideation analysis/bank_vs_sine.json ratio_sineK_over_r_384.median (first-order refdv fields)')
    rhos = [x['rho']['0.01'] for x in e0b if x['rho']['0.01']]
    put('rho_e0_1e2_med', g(np.median(rhos), 2), 'A3 E0 median of K_lap(1e-2)/r_bank(1e-2)')
    put('rho_e0_1e2_max', g(np.max(rhos), 2), 'A3 E0 max of the same')
    put('rho_e0_1e2_n', g(len(rhos)), 'A3 E0 cases where the bank reaches 1e-2 within 512')
    c12 = [x for x in b3 if x['src'] in ('C1', 'C2')]
    put('c12_bank512_reach3', g(sum(1 for x in c12 if x['r_bank']['0.03'])), 'A3 C1+C2 cases where the bank (<= 512) reaches a 3 % floor')
    put('c12_n', g(len(c12)), 'A3 C1+C2 cases')
    put('c12_Klap3_med', g(np.median([x['K_lap']['0.03'] for x in c12])), 'A3 C1+C2 median Laplacian-ordered sine count for a 3 % floor')
    # ------------------------------------------------------------------ A5 synthetic
    s5 = J('a5_synthetic.json')
    rows = []
    for fam, lab in (('F1', 'front, 1 parameter'), ('F2', 'front, 2 parameters'), ('F3', 'bump, 2 parameters'), ('F4', 'bump width, 1 parameter')):
        cells = []
        for d in s5['deltas']:
            r = [x for x in s5['rows'] if x['family'] == fam and x['delta'] == d][0]
            rp, kb = r['r_pod']['0.01'], r['K_block']['0.01']
            ka = r['K_aniso']['0.01']
            cells.append(f"{rp if rp else '>400'} / {kb if kb else '>lattice'} / {ka if ka else '>lattice'}"
                         + (f" ({g(r['rho']['0.01'], 2)}×, {g(r['rho_aniso']['0.01'], 2)}×)" if r['rho']['0.01'] else ''))
        rows.append(f"| {fam} {lab} | " + ' | '.join(cells) + ' |')
    put('tab_syn', '\n'.join(rows), 'analysis/a5_synthetic.json rows[*] r_pod / K_block at eps = 1e-2 (rho)')
    put('syn_deltas', ' | '.join(f'δ = {d}' for d in s5['deltas']), 'analysis/a5_synthetic.json deltas')
    for fam in ('F1', 'F3', 'F4'):
        rr = [x for x in s5['rows'] if x['family'] == fam and x['rho']['0.01']]
        put(f'syn_{fam}_rho_lo', g(min(x['rho']['0.01'] for x in rr), 2), f'analysis/a5_synthetic.json {fam} rho (square blocks, 1 %) min over resolved delta')
        put(f'syn_{fam}_rho_hi', g(max(x['rho']['0.01'] for x in rr), 2), 'same, max')
        put(f'syn_{fam}_an_lo', g(min(x['rho_aniso']['0.01'] for x in rr), 2), f'analysis/a5_synthetic.json {fam} rho_aniso (1 %) min')
        put(f'syn_{fam}_an_hi', g(max(x['rho_aniso']['0.01'] for x in rr), 2), 'same, max')
        lo_, hi_ = g(min(x['rho']['0.01'] for x in rr), 2), g(max(x['rho']['0.01'] for x in rr), 2)
        put(f'syn_{fam}_rho_rng', lo_ if lo_ == hi_ else f'{lo_}–{hi_}', 'collapsed range of the same')
    put('syn_delim', '|'.join(['---'] * (1 + len(s5['deltas']))), 'table delimiter row for tab_syn')
    # ------------------------------------------------------------------ A6 cost
    c6 = J('a6_cost.json')
    lat = c6['ps2_lattice']; full = c6['ps2_full']
    for N in ('16', '24', '48', '96', '256'):
        put(f'ps2_tau_{N}', ms(lat[N]['tau']), f'analysis/a6_cost.json ps2_lattice["{N}"].tau (H200, fom2d match-final costs regressed on steps)')
        put(f'ps2_alpha_{N}', ms(lat[N]['alpha']), f'analysis/a6_cost.json ps2_lattice["{N}"].alpha')
        put(f'ps2_alpha4k_{N}', ms(full['4096'][N]['alpha']), f'analysis/a6_cost.json ps2_full["4096"]["{N}"].alpha (full-field 4096^2 output)')
        put(f'ps2_dtmax_{N}', g(lat[N]['dt_max_admissible']), f'analysis/a6_cost.json ps2_lattice["{N}"].dt_max_admissible (38 cases)')
    p3 = c6['ps3_lattice']
    for N in ('16', '32', '64'):
        put(f'ps3_tau_{N}', ms(p3[N]['tau']), f'analysis/a6_cost.json ps3_lattice["{N}"].tau (H200, speed3d f1 warm medians)')
        put(f'ps3_alpha_{N}', ms(p3[N]['alpha']), f'analysis/a6_cost.json ps3_lattice["{N}"].alpha')
    r2 = c6['rom2']
    for key, lab in (('R128|BE|frozen', 'R128'), ('R256|BE|frozen', 'R256'), ('R384|CNR|frozen', 'R384')):
        v = r2[key]
        put(f'rom_tau_{lab}', ms(v['tau']), f'analysis/a6_cost.json rom2["{key}"].tau = P1 t_evolve(1024)/50')
        put(f'rom_W_{lab}', g(v['W_step']), f'analysis/a6_cost.json rom2["{key}"].W_step (model: n_J x 2 m M r)')
        put(f'rom_rate_{lab}', g(v['W_step'] / v['tau'] / 1e12, 3), 'W_step / tau / 1e12 (TFLOP/s, model flops over measured time)')
        put(f'rom_bytesJ_{lab}', g(v['bytes_per_jac'] / 1e6, 3), 'bytes_per_jac / 1e6 (MB of f64 tables read per Jacobian, model)')
        put(f'rom_init_{lab}', ms(v['t_init']['1024']), f'P1 t_init["1024"] for {key}')
        put(f'rom_ev_{lab}', ms(v['t_evolve']['1024']), f'P1 t_evolve["1024"] for {key}')
        put(f'rom_dec256_{lab}', ms(v['t_decode']['256']), f'P1 t_decode["256"] for {key}')
        put(f'rom_dec4k_{lab}', ms(v['t_decode']['4096']), f'P1 t_decode["4096"] for {key}')
        put(f'rom_perunk_{lab}', g(1e6 * v['tau'] / v['r'], 3), 'tau / r in microseconds')
        put(f'rom_flopunk_{lab}', g(v['W_step'] / v['r'], 3), 'W_step / r (flops per unknown per step)')
    put('rom_nj', g(r2['R384|CNR|frozen']['nj'], 3), 'analysis/a6_cost.json rom2[...].nj = 1 + mean LM iterations (P1 component run, R384 CN-R)')
    from a6_cost import w_ps
    for N, d in ((24, 2), (48, 2), (256, 2), (14, 3), (16, 3), (64, 3)):
        put(f'psW_{d}d_{N}', g(w_ps(N, d)), f'a6_cost.w_ps({N}, {d}) (operation-count model)')
        put(f'psflopunk_{d}d_{N}', g(w_ps(N, d) / (N - 1) ** d, 3), f'w_ps({N},{d}) / (N-1)^{d}')
    put('ps_perunk_48', g(1e6 * lat['48']['tau'] / 47 ** 2, 3), 'ps2_lattice["48"].tau / 47^2 in microseconds')
    put('ps_rate_48', g(w_ps(48, 2) / lat['48']['tau'] / 1e12, 2), 'w_ps(48,2) / tau(48) / 1e12 TFLOP/s')
    put('ps_rate_256', g(w_ps(256, 2) / lat['256']['tau'] / 1e12, 2), 'w_ps(256,2) / tau(256) / 1e12 TFLOP/s')
    for k_, lab in (('fit_ps2', 'ps2'), ('fit_ps3', 'ps3'), ('fit_rom2', 'rom')):
        f = c6[k_]
        put(f'{lab}_tau0', ms(f['tau0']), f'analysis/a6_cost.json {k_}.tau0 (relative least squares)')
        put(f'{lab}_P', g(f['P'] / 1e12, 3), f'analysis/a6_cost.json {k_}.P / 1e12 (TFLOP/s)')
        put(f'{lab}_resid', pct(f['max_rel_resid'], 2), f'analysis/a6_cost.json {k_}.max_rel_resid')
    for k_, lab in (('fit_ps2_asym', 'ps2a'), ('fit_ps3_asym', 'ps3a')):
        put(f'{lab}_P', g(c6[k_]['P'] / 1e12, 3), f'analysis/a6_cost.json {k_}.P / 1e12')
    r3 = c6['rom3']
    for lab in ('W_C_W768', 'C_old', 'C_W1024'):
        v = r3[lab]
        put(f'r3_{lab}_fused', ms(v['fused']), f'analysis/a6_cost.json rom3["{lab}"].fused (speed3d p1 H31)')
        put(f'r3_{lab}_evolve', ms(v['evolve']), f'rom3["{lab}"].evolve')
        put(f'r3_{lab}_proj', ms(v['project']), f'rom3["{lab}"].project')
        put(f'r3_{lab}_dec', ms(v['decode']), f'rom3["{lab}"].decode')
    put('r3_tau', ms(r3['W_C_W768']['tau']), 'rom3["W_C_W768"].evolve / 15 steps')
    put('r3_contraction', ms(r3['W_C_W768']['contraction']), 'rom3["W_C_W768"].contraction (one test-space contraction)')
    q4 = c6['c2_queries']['4096']
    for lab, key in (('R384', 'R384|gauss128|frozen|CNR|0.005|LSPG|prod'), ('R128', 'R128|fib4181|frozen|BE|0.005|LSPG|prod')):
        put(f'q4_{lab}_rom', ms(q4[key]['rom_s']), f'analysis/a6_cost.json c2_queries["4096"]["{key}"].rom_s (fom2d stage 2)')
        put(f'q4_{lab}_fom', ms(q4[key]['fom_s']), f'same .fom_s ({q4[key]["fom"]})')
        put(f'q4_{lab}_S', g(q4[key]['S'], 3), 'same .S (registered union speed-up, dev6 workload)')
    # GB10 shape check
    gg = json.loads((A / 'g1_gpu.json').read_text())['rows']
    rat = gg['2|1024']['tau'] / gg['2|256']['tau']
    put('g1_ratio_2d', g(rat, 3), 'analysis/g1_gpu.json tau(2D N=1024)/tau(2D N=256) on the GB10')
    put('g1_Wratio_2d', g(w_ps(1024, 2) / w_ps(256, 2), 3), 'w_ps(1024,2)/w_ps(256,2)')
    rat3 = gg['3|128']['tau'] / gg['3|64']['tau']
    put('g1_ratio_3d', g(rat3, 3), 'analysis/g1_gpu.json tau(3D N=128)/tau(3D N=64) on the GB10')
    put('g1_Wratio_3d', g(w_ps(128, 3) / w_ps(64, 3), 3), 'w_ps(128,3)/w_ps(64,3)')
    gc = json.loads((A / 'g1_cpu.json').read_text())['rows']
    dev = max(np.max(np.abs(np.array(gg[k]['check']) - np.array(gc[k]['check']))) / np.max(np.abs(gc[k]['check'])) for k in gg)
    put('g1_cpu_dev', g(dev, 2), 'max relative deviation of the GB10 fingerprints from the CPU run (analysis/g1_*.json)')
    # ------------------------------------------------------------------ A7 model
    m7 = J('a7_model.json')
    ch = m7['check']
    put('model_check_lo', g(min(v['ratio'] for v in ch.values()), 2), 'analysis/a7_model.json check[*].ratio min (model / measured query time)')
    put('model_check_hi', g(max(v['ratio'] for v in ch.values()), 2), 'same, max')
    rows = [f"| {k.replace('|', ' ')} | {ms(v['model'])} | {ms(v['model_own_kappa']) if 'model_own_kappa' in v else '–'} | {ms(v['meas'])} | {g(v['ratio'], 3)} |" for k, v in ch.items()]
    put('tab_check', '\n'.join(rows), 'analysis/a7_model.json check')
    cr = m7['crossover_floor']
    rows = []
    for k, v in cr.items():
        d, nps, nrom, S = k.split('|')
        lo = min(v['N_c_fit'], v['N_c_asym']) if v['N_c_fit'] and v['N_c_asym'] else None
        hi = max(v['N_c_fit'], v['N_c_asym']) if v['N_c_fit'] and v['N_c_asym'] else None
        val = '≤ 8 (any N)' if v['below_grid_fit'] else (f'{lo}–{hi}' if lo != hi else f'{lo}')
        rows.append(f"| {d[1]}D | {nps[3:]} | {nrom[4:]} | {S[1:]}× | {ms(v['rom_floor_s'])} | {val} |")
    put('tab_cross', '\n'.join(rows), 'analysis/a7_model.json crossover_floor (N_c over the two throughput fits)')
    for k, lab in (('d2|nps20|nrom50|S2', 'c2_50_2'), ('d2|nps20|nrom20|S2', 'c2_20_2'), ('d3|nps10|nrom15|S2', None),
                   ('d3|nps10|nrom50|S2', 'c3_50_2'), ('d3|nps10|nrom20|S2', 'c3_20_2'), ('d2|nps20|nrom50|S1', 'c2_50_1'),
                   ('d3|nps10|nrom20|S1', 'c3_20_1')):
        if lab and k in cr:
            v = cr[k]
            put(f'Nc_{lab}', f"{min(v['N_c_fit'], v['N_c_asym'])}–{max(v['N_c_fit'], v['N_c_asym'])}" if v['N_c_fit'] != v['N_c_asym'] else str(v['N_c_fit']),
                f'analysis/a7_model.json crossover_floor["{k}"]')
            aux[lab] = (v['N_c_fit'], v['N_c_asym'])
    rn = m7['rho_needed']
    rows = []
    for k, v in rn.items():
        d, nps, nrom, N, var = k.split('|')
        if var != 'fit' or int(N[1:]) not in ((256, 512, 768, 1024, 2048) if d == 'd2' else (16, 32, 48, 64, 96, 128, 256)):
            continue
        va = rn[k.replace('|fit', '|asym')]
        def rr_(x, y):
            if x is None and y is None:
                return 'none (ROM floor too slow)'
            xs = [z for z in (x, y) if z is not None]
            return f'{g(min(xs), 2)}–{g(max(xs), 2)}' if len(xs) == 2 and abs(xs[0] - xs[1]) > 0.005 * xs[0] else g(xs[0], 2)
        rows.append(f"| {d[1]}D | {N[1:]} | {v['K']} | {nps[3:]} | {nrom[4:]} | {ms(v['T_ps'])} | "
                    f"{rr_(v['rho_S1'], va['rho_S1'])} | {rr_(v['rho_S2'], va['rho_S2'])} | {g(v['r_max_S1'], 3) if v['r_max_S1'] else '–'} |")
    put('tab_rho', '\n'.join(rows), 'analysis/a7_model.json rho_needed (both throughput fits)')
    for d_, key_, lab_, nps_, nrom_ in ((2, 'd2|nps20|nrom50', 'rho_need2', 20, 50), (3, 'd3|nps10|nrom15', 'rho_need3', 10, 15)):
        cand = sorted((int(k.split('|')[3][1:]), v['rho_S2']) for k, v in rn.items()
                      if k.startswith(key_ + '|') and k.endswith('|fit') and v['rho_S2'])
        put(lab_, f'{g(cand[0][1], 2)} (at $N={cand[0][0]}$, {nps_} spectral and {nrom_} ROM steps)', f'analysis/a7_model.json rho_needed["{key_}|N{cand[0][0]}|fit"].rho_S2 (smallest tabulated N with any S=2 solution)')
    sc = m7['s_crossover']
    cv = sc['c_variants']
    put('c_dt', g(cv['c_min'], 3), 'analysis/a7_model.json s_crossover.c_variants.c_min: dt_max ~ c / s')
    put('c_dt_med', g(cv['c_med'], 3), 'analysis/a7_model.json s_crossover.c_variants.c_med')
    rows = []
    keys_de = [k_ for k_ in sc if k_.startswith('d')]
    for key in keys_de:
        v = sc[key]
        d, e = key.split('|')
        put(f'sfit_{d}_{e}_alpha', g(v['alpha'], 2), f'analysis/a7_model.json s_crossover["{key}"].alpha (N*_PS(eps) = c s^alpha, resolved cases)')
        put(f'sfit_{d}_{e}_c', g(v['c'], 2), f'analysis/a7_model.json s_crossover["{key}"].c')
        put(f'sfit_{d}_{e}_env', g(v['envelope'], 2), f'analysis/a7_model.json s_crossover["{key}"].envelope (max over resolved cases of N / fit)')
        put(f'sfit_{d}_{e}_n', g(v['n_cases']), f'analysis/a7_model.json s_crossover["{key}"].n_cases')
        put(f'sfit_{d}_{e}_cens', g(v['n_censored']), f'analysis/a7_model.json s_crossover["{key}"].n_censored (no tested N reached eps)')
        put(f'sfit_{d}_{e}_smax', g(v['s_max_data'], 3), f'analysis/a7_model.json s_crossover["{key}"].s_max_data (largest resolved s)')
        for nmod, nlab in (('typ', 'fit'), ('env', 'envelope')):
            for scen, lab in (('fixed50', 'ROM 50 steps'), ('fixed20', 'ROM 20 steps'), ('equal', 'ROM steps = spectral steps')):
                cells = []
                for rho_ in ('inf', '1000.0', '100.0', '5.0'):
                    x = v[f'{nmod}|c_min|{scen}|rho{rho_}']
                    cells.append('none' if x is None else f"{g(x['s'], 2)} (N = {x['N']}, {x['n_ps']} steps)")
                rows.append(f"| {d[1]}D | {e[3:]} | {nlab} | {lab} | " + ' | '.join(cells) + ' |')
    put('tab_scross', '\n'.join(rows), 'analysis/a7_model.json s_crossover[...]["<typ|env>|c_min|<scenario>|rho<...>"] (2x bar; first s on a geometric grid of 600 points in [5, 1e6])')
    def pick(pred):
        return [x['s'] for k_ in keys_de for kk, x in sc[k_].items() if isinstance(x, dict) and pred(kk.split('|'))]
    fixed = pick(lambda t: t[1] == 'c_min' and t[2] in ('fixed50', 'fixed20') and t[3] in ('rhoinf', 'rho1000.0'))
    put('scross_fixed_lo', g(min(fixed), 2), 'min s over fit/envelope, c_min, fixed-ROM-step scenarios with rho = inf or 1000')
    put('scross_fixed_hi', g(max(fixed), 2), 'max s over the same')
    fixedm = pick(lambda t: t[1] == 'c_med' and t[2] in ('fixed50', 'fixed20') and t[3] in ('rhoinf', 'rho1000.0'))
    put('scross_fixed_med_lo', g(min(fixedm), 2), 'same with c_med')
    put('scross_fixed_med_hi', g(max(fixedm), 2), 'same with c_med')
    eq = pick(lambda t: t[2] == 'equal' and t[3] == 'rhoinf' and t[1] == 'c_min')
    put('scross_eq_lo', g(min(eq), 2), 'min s over equal-step scenarios with free unknowns (fit and envelope, c_min)')
    put('scross_eq_hi', g(max(eq), 2), 'max s over the same')
    eqr = [x for k_ in keys_de for kk, x in sc[k_].items() if kk.split('|')[2:3] and kk.split('|')[2] == 'equal' and kk.split('|')[3] != 'rhoinf']
    put('scross_eq_rho_none', 'none of the tested finite-$\\rho$ equal-step scenarios crosses over in the scanned range' if all(x is None for x in eqr) else 'some finite-$\\rho$ equal-step scenarios cross over (table)', 'equal-step scenarios with finite rho (both N models, both c)')
    ss = m7['sensitivity']
    rows = []
    for k_, v in ss.items():
        d, nps, nrom = k_.split('|')
        rows.append(f"| {d[1]}D | {nps[3:]} | {nrom[4:]} | {v['base']} | {v['tau0_rom_half']} | {v['ps_throughput_x2']} | {v['ps_alpha_grows']} |")
    put('tab_sens', '\n'.join(rows), 'analysis/a7_model.json sensitivity')
    ours_ = m7['ours']
    put('rho_ours_lo', g(min(v['rho'] for v in ours_.values()), 2), 'analysis/a7_model.json ours[*].rho min')
    put('rho_ours_hi', g(max(v['rho'] for v in ours_.values()), 2), 'analysis/a7_model.json ours[*].rho max')
    put('flop_ratio', g(r2['R384|CNR|frozen']['W_step'] / 384 / (w_ps(48, 2) / 47 ** 2), 2), 'rom_flopunk_R384 / psflopunk_2d_48')
    put('ps_modes_ratio_256_48', g((255 / 47) ** 2, 2), '(255/47)^2: modes of N = 256 over N = 48')
    put('ps_tau_ratio_256_48', g(lat['256']['tau'] / lat['48']['tau'], 2), 'ps2_lattice tau(256)/tau(48)')
    from a6_cost import bytes_ps
    put('ps_bytes48_kB', g(bytes_ps(48, 2) / 1e3, 2), 'a6_cost.bytes_ps(48, 2) / 1e3 (one padded f64 field)')
    put('ev_share_R384', pct(r2['R384|CNR|frozen']['t_evolve']['4096'] / (r2['R384|CNR|frozen']['t_init']['4096'] + r2['R384|CNR|frozen']['t_evolve']['4096'] + r2['R384|CNR|frozen']['t_decode']['4096']), 2), 'P1 evolve / (init + evolve + decode) at 4096^2, R384 CN-R')
    put('ev_share_R128', pct(r2['R128|BE|frozen']['t_evolve']['4096'] / (r2['R128|BE|frozen']['t_init']['4096'] + r2['R128|BE|frozen']['t_evolve']['4096'] + r2['R128|BE|frozen']['t_decode']['4096']), 2), 'same, R128 BE')
    kap = sorted(v['m'] / v['r'] for v in r2.values())
    put('kappa_lo', g(kap[0], 2), 'min m/r over the P1 configurations')
    put('kappa_hi', g(kap[-1], 2), 'max m/r over the P1 configurations')
    put('kappa_384', g(16384 / 384, 3), 'm/r for R384 gauss128 (model constant kappa)')
    ours = m7['ours']
    rows = [f"| {k.replace('|', ' ')} | {v['N_joint']} | {v['K']} | {v['r']} | {g(v['rho'], 2)} |" for k, v in ours.items()]
    put('tab_ours', '\n'.join(rows), 'analysis/a7_model.json ours')
    pr = m7['pred_S']
    put('predS_R384', g(pr['1024|R384|LSPG|CNR|1']['S_model'], 2), 'analysis/a7_model.json pred_S["1024|R384|LSPG|CNR|1"].S_model (lattice, N_joint)')
    put('predS_R128', g(pr['1024|R128|LSPG|BE|1']['S_model'], 2), 'same for R128 LSPG-BE dt0')
    # final campaign K2 numbers (as published)
    fin = (C.FINAL / 'report.md').read_text()
    assert 'union speed-up 0.0493–0.706' in fin and '3D 0.00786–0.158' in fin
    put('k2_2d', '0.0493–0.706', 'reports/2026-10-17-full-experiment-plan/final/report.md (text "union speed-up 0.0493–0.706")')
    put('k2_3d', '0.00786–0.158', 'reports/2026-10-17-full-experiment-plan/final/report.md (text "3D 0.00786–0.158")')
    # s mapping of the crossover N (2D, eps 1e-2 fit on the spectral solver's own error)
    if 'fitPS' in aux:
        p, c = aux['fitPS']
        for lab in ('c2_50_2', 'c2_20_2', 'c3_50_2', 'c3_20_2', 'c2_50_1', 'c3_20_1'):
            if lab in aux:
                Nlo = min(aux[lab])
                put(f's_{lab}', g((Nlo / c) ** (1 / p), 2), f'(N_c / c)^(1/alpha) with the 2D spectral-solver fit (fitPS), N_c lower end of {lab}; extrapolated')
    return V


if __name__ == '__main__':
    for k, (v, s) in values().items():
        if '\n' not in v:
            print(f'{k:28s} {v:24s} {s[:90]}')
