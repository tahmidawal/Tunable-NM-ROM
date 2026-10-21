"""Every number used in report.md: read from a lane output, a committed report, or this folder's analysis JSONs.

values() returns {key: (formatted string, source description)}. Derived estimates (cost models, crossovers) are computed
here from the read values; each source string gives the formula and labels assumptions. Run make_report.py to fill
report.src.md -> report.md. Run analysis/*.py first (CPU, about two minutes in total).
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude')
W = ROOT / 'worktrees'
HERE = Path(__file__).resolve().parent
AN = HERE / 'analysis'
IDEA = ROOT / 'reports/2026-10-20-improvement-ideation/VALUES.json'
FINAL = ROOT / 'reports/2026-10-17-full-experiment-plan/final'
P1 = W / '2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling'
FOM = W / '2026-10-18-jcp-fom2d/experiments/jcp-fom2d'
S3 = W / '2026-10-14-jcp-speed3d/experiments/jcp-speed3d'
DIMS = W / '2026-10-18-jcp-dims/experiments/jcp-dims'
NS3 = W / '2026-10-01-ns3d-coordnet-bank'
LSH_BRANCH = 'exp/2026-09-25-poisson-lshape3d-test'
LSH_FILE = 'experiments/poisson-lshape3d-test/reports/2026-09-25-poisson-lshape3d-test-rows.md'


def J(p):
    return json.loads(Path(p).read_text())


def rel(p):
    return str(Path(p).relative_to(ROOT))


def g(x, d=3):
    return f'{x:.{d}g}'


def ig(x):
    """Two significant figures, written as a plain integer with thousands separators."""
    return f'{float(f"{x:.2g}"):,.0f}'


def pct(x, d=3):
    return f'{100 * x:.{d}g} %'


def ms(x, d=3):
    return f'{1e3 * x:.{d}g} ms'


def num(s):
    """Parse the leading number of a formatted VALUES.json string ('1.77 ms' -> 1.77e-3 s, '0.198 %' -> 0.00198)."""
    m = re.match(r'\s*([-+0-9.eE]+)\s*(ms|%|h|×)?', s)
    v = float(m.group(1))
    return {'ms': 1e-3, '%': 1e-2}.get(m.group(2), 1.0) * v


def values():
    V = {}

    def put(k, v, src):
        assert k not in V, k
        V[k] = (v, src)

    # ------------------------------------------------ inherited, traced values (ideation VALUES.json) ----------
    iv = J(IDEA)
    isrc = rel(IDEA)
    keep = ['step_R128', 'step_R384', 't4k_R128', 't4k_R384', 't4k_ps24', 't4k_ps48', 't_dec_R128_4k', 't_dec_R384_4k',
            'ps_N48_cost', 'ps_N48_med', 'ps_N48_worst', 'ps_N24_med', 'ps_N24_worst', 'hrb_R384_base', 'hrb_R384_ours',
            'hrb_R384_base_step', 'hrb_R384_ours_step', 'tq_e128k8', 'tq_e384k8', 'hrb_build_2d', 'bank_fit_h',
            'p1_R128BE_med', 'p1_R128BE_worst', 'p1_R384CNR_med', 'p1_R384CNR_worst', 'nom_J_384', 'flops_J_384',
            'floor_N32', 'floor_C1_384', 'floor_C2_384', 'reg_med_C1', 'reg_med_C2', 'hrb_pick_worst_R384',
            'batch_W768', 'k2_3d_range', 'ns3d_pod', 'est_S_etd_quad', 'est_S_etd_gal', 'smax_R128', 'smax_R384',
            'worst_ratio_384', 'ratio_med_384', 'pts_e384k8']
    for k in keep:
        put(k, iv[k]['value'], f'{isrc}[{k}] <- {iv[k]["source"]}')
    f = {k: num(iv[k]['value']) for k in keep if k not in ('k2_3d_range', 'ns3d_pod', 'batch_W768')}

    # ------------------------------------------------ 2D FOM families against the qualified LSPG ROMs ----------
    c2 = J(FOM / 'checks/c2-analysis.json')
    c2s = rel(FOM / 'checks/c2-analysis.json')
    fam = {}
    for k, v in c2['hcomp'].items():
        if not k.startswith('4096|') or v.get('S') is None or not str(v.get('scope', '')).startswith('primary'):
            continue
        cfg = '|'.join(k.split('|')[2:8])
        fam.setdefault(v['family'], {})[cfg] = v['S']
    for F in ('UB', 'CC', 'PS', 'union'):
        Ss = list(fam[F].values())
        put(f'S_{F}_lo', g(min(Ss)), f'{c2s} hcomp["4096|...|{F}|joint"].S with scope "primary (LSPG)", min over {len(Ss)} configurations')
        put(f'S_{F}_hi', g(max(Ss)), f'{c2s} same, max')
        put(f'S_{F}_n', str(len(Ss)), f'{c2s} count of 4096^2 {F} rows with an S value')
    # geometry multiplier needed to reach 1x and 2x against the second-order FD family (CC), which uses a DST
    # preconditioner that a non-box domain loses; f_geo_needed = bar / S_CC (cost of geometry FOM / cost of box CC)
    cc = np.array(list(fam['CC'].values()))
    put('fgeo1_lo', g(1 / cc.max(), 2), '1 / max S_CC: FOM slowdown factor on a non-box domain at which the best ROM ties the CC family (ROM cost unchanged)')
    put('fgeo1_hi', g(1 / cc.min(), 2), '1 / min S_CC')
    put('fgeo2_lo', g(2 / cc.max(), 2), '2 / max S_CC: slowdown factor needed for a 2x win')
    put('fgeo2_hi', g(2 / cc.min(), 2), '2 / min S_CC')
    ps = np.array(list(fam['PS'].values()))
    put('fps2_lo', g(2 / ps.max(), 2), '2 / max S_PS: slowdown of the spectral FOM needed for a 2x win')
    put('fps2_hi', g(2 / ps.min(), 2), '2 / min S_PS')
    mf = J(FOM / 'checks/match-final.json')
    ccp = {p['key']: p for p in mf['pareto']['CC']['points']}
    for lab, key in (('L64', 'fd|CC|L64|0.0125|t2'), ('L128', 'fd|CC|L128|0.01|t2')):
        p = ccp[key]
        put(f'cc2_{lab}_cost', ms(p['cost']), f'{rel(FOM / "checks/match-final.json")} pareto.CC[{key}].cost (screening, lattice output)')
        put(f'cc2_{lab}_med', pct(p['median']), f'... pareto.CC[{key}].median')
        put(f'cc2_{lab}_worst', pct(p['worst']), f'... pareto.CC[{key}].worst')

    # ------------------------------------------------ 2D ROM step-cost model -----------------------------------
    m128, M128, r128 = 4181, 512, 128
    m384, M384, r384 = 16384, 1536, 384
    x1, x2 = m128 * M128 * r128, m384 * M384 * r384
    b = (f['step_R384'] - f['step_R128']) / (x2 - x1)
    a = f['step_R128'] - b * x1
    put('stepmodel_a', ms(a), 'two-point fit t_step = a + b m M r through (R128: m 4181, M 512) and (R384: m 16384, M 1536); a = fixed per-step cost')
    put('stepmodel_b', f'{b * 1e3 * 1e9:.3g} ms per 1e9', 'slope b of the same fit (ms per 1e9 units of m M r)')
    el = J(DIMS / 'checks/analysis-2d.json')['elasticity']['beta']
    put('el_r', g(el[1], 2), f'{rel(DIMS / "checks/analysis-2d.json")} elasticity.beta[1] (descriptive full-query elasticity in r)')
    put('el_M', g(el[2], 2), '... elasticity.beta[2] (in M)')
    put('el_m', g(el[3], 2), '... elasticity.beta[3] (in m)')
    thr = f['flops_J_384'] / f['nom_J_384']
    put('thr_J', f'{thr / 1e12:.2g} TFLOP/s', 'flops_J_384 / nom_J_384: effective throughput of the dense Jacobian kernel (H200)')

    # ------------------------------------------------ 3D: ROM, FOM ladders ------------------------------------
    am = J(S3 / 'checks/analysis-pm.json')['m1']['per_L']
    s3 = rel(S3 / 'checks/analysis-pm.json')
    tA, tB, R = {}, {}, {}
    for L in ('64', '128', '256', '384'):
        for lab in ('W_C_W768', 'C_old', 'C_W1024'):
            d = am[L]['panels'][lab]['same_mesh'][f'fd|CC|L{L}|0.01|t1']
            tA[(lab, L)], tB[(lab, L)], R[(lab, L)] = d['t_A'], d['t_B'], d['ratio']
    # C_old is the Q2-qualified 3D ROM in this panel (final report: "C_old pass, ... W_C_W768 fail")
    fin = (FINAL / 'report.md').read_text()
    assert 'C_old pass, C_W1024 fail, W_C_old fail, W_C_W768 fail' in fin
    put('q2_note', 'C_old pass; W_C_W768, W_C_old, C_W1024 fail', f'{rel(FINAL / "report.md")} (text "C_old pass, C_W1024 fail, W_C_old fail, W_C_W768 fail")')
    put('t3_Cold', ms(tA[('C_old', '128')]), f'{s3} m1.per_L["128"].panels.C_old.same_mesh["fd|CC|L128|0.01|t1"].t_A (qualified ROM, H200, scoring-lattice output)')
    put('t3_W768', ms(tA[('W_C_W768', '128')]), f'{s3} ... panels.W_C_W768 ... t_A (fails Q2: unqualified, optimistic)')
    cold = [tA[('C_old', L)] for L in ('64', '128', '256', '384')]
    put('t3_Cold_rng', f'{1e3 * min(cold):.3g}–{1e3 * max(cold):.3g} ms', f'{s3} C_old t_A over L = 64, 128, 256, 384 (approximately mesh-independent)')
    put('r3_W1024', g(tA[('C_W1024', '128')] / tA[('C_old', '128')], 2) + '×', f'{s3} t_A(C_W1024) / t_A(C_old) at L=128 (wider W1024 bank)')
    for L in ('128', '256', '384'):
        tb = tB[('C_old', L)]
        put(f't3_CC_L{L}', ms(tb) if tb < 1 else f'{tb:.3g} s', f'{s3} m1.per_L["{L}"].panels.C_old.same_mesh["fd|CC|L{L}|0.01|t1"].t_B (DST-preconditioned central CN FD, H200, full L^3 field)')
        put(f'S3_same_L{L}', g(R[('C_old', L)]), f'{s3} m1.per_L["{L}"].panels.C_old.same_mesh["fd|CC|L{L}|0.01|t1"].ratio (registered paired statistic; same mesh, not accuracy-matched)')
        put(f'S3W_same_L{L}', g(R[('W_C_W768', L)]), f'{s3} ... panels.W_C_W768 ... ratio (unqualified ROM)')
    fm = J(S3 / 'checks/fom-match.json')['matches']
    s3m = rel(S3 / 'checks/fom-match.json')
    w = fm['C_old']
    put('t3_PS', ms(w['PS']['cost_screen']), f'{s3m} matches.C_old.PS.cost_screen ({w["PS"]["matched"]}; screening cost, cheapest admissible = boundary match)')
    put('t3_CC64', ms(w['CC']['cost_screen']), f'{s3m} matches.C_old.CC.cost_screen ({w["CC"]["matched"]}; screening cost)')
    put('t3_CC64_worst', pct(w['CC']['worst']), f'{s3m} matches.C_old.CC.worst')
    rs = J(S3 / 'checks/rom-scores.json')
    for lab, key in (('C_old', 'Cold'), ('W_C_W768', 'W768')):
        e = np.array(rs[lab]['e'])
        put(f'e3_{key}_med', pct(float(np.median(e))), f'{rel(S3 / "checks/rom-scores.json")} {lab}.e median over {len(e)} cases (conditional E0)')
        put(f'e3_{key}_worst', pct(float(e.max())), f'... {lab}.e max')
    p3 = np.log(tB[('C_old', '384')] / tB[('C_old', '128')]) / np.log(3)
    put('p3_CC', g(p3, 2), 'log(t_CC(384)/t_CC(128)) / log 3 (C_old panel): growth exponent of the 3D FD FOM in L at fixed dt = 0.01')

    # ------------------------------------------------ CPU analyses (this folder) --------------------------------
    ls = J(AN / 'lshape_fd.json')
    put('ls_order_max', g(ls['singular']['order_max'], 2), f'{rel(AN / "lshape_fd.json")} singular.order_max (last three meshes)')
    put('ls_order_l2', g(ls['singular']['order_l2'], 2), '... singular.order_l2')
    put('ls_order_smooth', g(ls['smooth']['order_l2'], 2), '... smooth.order_l2')
    put('ls_N_2e3', ig(ls['singular']['N_for_rel_l2']['0.002']), '... singular.N_for_rel_l2["0.002"] (intervals on the side-2 domain)')
    put('ls_N_1e4', ig(ls['singular']['N_for_rel_l2']['0.0001']), '... singular.N_for_rel_l2["0.0001"]')
    r64 = [r for r in ls['singular']['rows'] if r['N'] == 64][0]
    put('ls64_err', pct(r64['rel_l2']), '... singular.rows[N=64].rel_l2')
    put('ls64_dof', str(r64['dof']), '... singular.rows[N=64].dof')
    put('ls64_t', ms(r64['solve_s']), '... singular.rows[N=64].solve_s (CPU sparse LU)')
    r1k = [r for r in ls['singular']['rows'] if r['N'] == 1024][0]
    put('ls1k_max', pct(r1k['max_err']), '... singular.rows[N=1024].max_err (absolute; |u| <= 1)')
    pe = J(AN / 'penalized_spectral.json')
    put('pen_order', g(pe['order'], 2), f'{rel(AN / "penalized_spectral.json")} order (last three N)')
    rr = {r['N']: r for r in pe['penalized']}
    put('pen_err_1024', pct(rr[1024]['rel_l2']), '... penalized[N=1024].rel_l2 (periodic cell of side 2, 512 points per unit length)')
    put('pen_err_256', pct(rr[256]['rel_l2']), '... penalized[N=256].rel_l2')
    put('pen_N_1e2', ig(pe['N_for_rel_l2_periodic_cell']['0.01'] / 2), '... N_for_rel_l2_periodic_cell["0.01"] / 2 (points per unit length)')
    put('pen_N_2e3', ig(pe['N_for_rel_l2_periodic_cell']['0.002'] / 2), '... ["0.002"] / 2')
    vc = J(AN / 'varcoef_pcg.json')['rows']
    assert all(r['converged'] for r in vc)
    it = {(r['case'], r['N'], round(r['contrast']), r['dt']): r['pcg_iters'] for r in vc}
    itl = {(r['case'], r['N'], round(r['contrast']), r['dt']): r['pcg_iters_1e4'] for r in vc}
    vsrc = rel(AN / 'varcoef_pcg.json')
    for c in (10, 100, 1000):
        for case in ('noisy', 'smooth', 'warm'):
            put(f'pcg_{case}_{c}', f'{it[(case, 48, c, 0.005)]}–{it[(case, 48, c, 0.0125)]}', f'{vsrc} rows[case={case}, N=48, contrast {c}, dt 0.005-0.0125].pcg_iters (tol 1e-10)')
            put(f'pcgl_{case}_{c}', f'{itl[(case, 48, c, 0.005)]}–{itl[(case, 48, c, 0.0125)]}', f'{vsrc} same rows, pcg_iters_1e4 (tol 1e-4)')
    put('pcg128_100', f'{it[("noisy", 128, 100, 0.005)]}–{it[("noisy", 128, 100, 0.0125)]}', f'{vsrc} case noisy, N=128, contrast 100, tol 1e-10')
    put('pcg48_1', str(it[('noisy', 48, 1, 0.005)]), f'{vsrc} N=48, contrast 1.0001 (nearly constant coefficient; the script never runs exactly 1)')
    dte = {(r['N'], round(r['contrast'])): r['dt_explicit_max'] for r in vc}
    put('dtexp48_100', g(dte[(48, 100)], 2), '... dt_explicit_max at N=48, contrast 100 (h^2 / 4 kappa_max, nu = 0.01 assumed)')

    # ------------------------------------------------ derived cost models (illustrative) -----------------------
    # (A) variable coefficient: spectral FOM step = measured PS step + n_it * t_it; t_it assumed = 3 transform-pair
    # latencies, the transform-pair latency estimated as PS N48 evolution / (20 steps x 12 transform pairs)
    t_ps_step = f['ps_N48_cost'] / 20
    put('ps_step', ms(t_ps_step), 'ps_N48_cost / 20 ETDRK4 steps (screening query incl. lattice synthesis; used only as a scale)')
    # per-configuration ROM and union-comparator latencies at 4096^2 (fom2d stage-2 table)
    rep2 = (FOM / 'report.md').read_text().splitlines()
    i0 = [i for i, l in enumerate(rep2) if l.startswith('**Absolute prepared-input latencies')][0]
    lat = []
    for l in rep2[i0 + 3:i0 + 40]:
        m_ = re.match(r'\| (.+?) \| ([\d.]+) \| PS \$N=(\d+)\$, \$\\Delta t=([\d.]+)\$ \| ([\d.]+) \|', l)
        if m_:
            lat.append((m_.group(1), float(m_.group(2)), int(m_.group(3)), float(m_.group(4)), float(m_.group(5))))
    assert len(lat) == 21, len(lat)
    put('lat_n', str(len(lat)), f'{rel(FOM / "report.md")} "Absolute prepared-input latencies at 4096^2" table rows parsed (stage-2, dev6, H200)')
    # Sensitivity, contrast 100: S = (t_FOM + steps * n_it * t_it) / t_ROM, ROM cost/accuracy unchanged (assumption).
    # n_it scenarios (CPU FD-operator counts, indicative only); t_it hypothetical per-iteration GPU costs.
    scen = {'warm': itl[('warm', 48, 100, 0.0125)], 'smooth': it[('smooth', 48, 100, 0.0125)], 'noisy': it[('noisy', 48, 100, 0.0125)]}
    for case, n in scen.items():
        put(f'nit_{case}', str(n), f'{vsrc} contrast 100, N=48, dt 0.0125: ' + ('warm start, tol 1e-4' if case == 'warm' else f'{case}, tol 1e-10'))
    for tit in (0.02e-3, 0.065e-3, 0.2e-3):
        tg = f'{int(round(tit * 1e6))}'
        put(f'tit_{tg}', ms(tit, 2), 'hypothetical GPU cost per PCG iteration (assumption, not measured)')
        for case, n in scen.items():
            S = np.array([(tf + round(0.25 / dt) * n * tit) / tr for _, tr, N, dt, tf in lat])
            put(f'Svc_{case}_{tg}', f'{S.max():.2g} / {np.median(S):.2g} / {int((S >= 2).sum())}', f'best / median / count >= 2 over the 21 configurations of (t_FOM + steps x n_it x t_it) / t_ROM, n_it = nit_{case}, t_it = {tit*1e3} ms; ROM unchanged (assumption)')
    best = max(lat, key=lambda z: z[4] / z[1])
    put('vc_best', best[0].replace('$', '').replace('\\Delta t', 'dt').replace(' @ 4096^2', ''), 'configuration with the largest measured S_PS in the latency table')
    # (B) 3D scenario: a UNIFORM-grid FD FOM whose L is set by geometry, against the qualified C_old ROM with extra
    # quadrature points per hole and optionally a wider bank. S_new = S_box * f_FOM / f_ROM with f_FOM = 1 here.
    tC = tA[('C_old', '128')]
    m0 = 32 ** 3  # C_old uses Gauss 32^3
    elm, elr, elM = el[3], el[1], el[2]
    for H in (1, 8, 27):
        for mh, tag in ((4096, 'a'), (16384, 'b')):
            gq = ((m0 + H * mh) / m0) ** elm
            put(f'g3_H{H}{tag}', g(gq, 2), f'ROM slowdown ((32768 + {H} x {mh}) / 32768)^el_m: {mh} extra shell-rule points per hole (assumption), 2D m-elasticity transported to 3D (assumption)')
            for wide, wtag in ((1, ''), (2, 'w')):
                gf = gq * (wide ** elr) * (wide ** elM)
                for L in ('128', '256'):
                    S = R[('C_old', L)] / gf
                    put(f'S3geo_H{H}{tag}{wtag}_L{L}', g(S, 2), f'paired ratio C_old vs CC L={L} / (ROM slowdown {gf:.3g}; bank width x{wide}, tests x{wide} by el_r, el_M): uniform-grid scenario, f_FOM = 1')
    # (C) many-query break-even: Q* = C_off / (t_FOM - t_ROM), with the borrowed 2D bank-fit time as an
    # ILLUSTRATIVE offline cost (not a lower bound for a new 3D pipeline); scenario H = 8, 4096 points per hole
    off = f['bank_fit_h'] * 3600
    gq8 = ((m0 + 8 * 4096) / m0) ** elm
    for L in ('128', '256'):
        sav = tB[('C_old', L)] - tA[('C_old', L)] * gq8
        put(f'Qstar_3d{L}', ig(off / sav) if sav > 0 else 'no break-even', f'bank_fit_h / (t_CC(L={L}) - t_C_old(L={L}) x g3_H8a), both from the L={L} paired panel: illustrative partial-cost crossover (borrowed 2D fit time; data, qualification, FOM setup and output-contract difference excluded)')
    for case in ('warm', 'noisy'):
        tf_vc = best[4] + round(0.25 / best[3]) * scen[case] * 0.065e-3
        sav = tf_vc - best[1]
        put(f'Qstar_vc_{case}', ig(off / sav) if sav > 0 else 'no break-even (ROM not faster)', f'bank_fit_h / ((FOM latency + steps x nit_{case} x 0.065 ms) - ROM latency) for vc_best: illustrative')
    off_pod = f['hrb_build_2d'] * 3600
    put('Q_pod_vs_ours', f'{off_pod / 3600:.3g} h vs {off / 3600:.3g} h', 'hrb_build_2d vs bank_fit_h (construction cost, 2D)')
    rep3 = (S3 / 'report.md').read_text().splitlines()
    ln = [l for l in rep3 if l.startswith('| W_C_W768 | 64 |')][0]
    t64 = float(ln.split('|')[3])
    put('t3_batch64', ms(t64 / 64, 2), f'{rel(S3 / "report.md")} row "| W_C_W768 | 64 |": {t64} s for a batch of 64 -> per query')

    # ------------------------------------------------ L-shape Poisson lane (git branch, no worktree) -----------
    txt = subprocess.run(['git', '-C', str(ROOT), 'show', f'{LSH_BRANCH}:{LSH_FILE}'], capture_output=True, text=True, check=True).stdout
    for mesh in ('256²', '2048²'):
        row = [l for l in txt.splitlines() if l.startswith(f'| L-shape | {mesh} | test')][0].split('|')
        put(f'lsh_{mesh[:-1]}_worst', row[4].strip() + ' %', f'{LSH_BRANCH}:{LSH_FILE} row L-shape {mesh} test, "accurate %" (worst, vs same-grid FD)')
        put(f'lsh_{mesh[:-1]}_x', row[5].strip() + '×', f'... "accurate ×" (vs fastest CG tolerance meeting the worst error, same grid)')
        put(f'lsh_{mesh[:-1]}_fom', row[9].strip(), '... FOM setting')
    # ------------------------------------------------ NS 3D coordinate-network bank (report text) -------------
    nsr = NS3 / 'reports/2026-10-01-ns3d-coordnet-bank.md'
    put('ns3d_src', rel(nsr) if nsr.exists() else 'missing', 'path of the NS 3D bank report')
    nsj = NS3 / 'experiments/ns3d-coordnet/runs/dev96/output/mesh/summary.json'
    c = J(nsj)['comparators']['coordnet_head_k8']
    put('ns96_x', g(c['speedup'], 3) + '×', f'{rel(nsj)} comparators.coordnet_head_k8.speedup (vs {c["comparator"]}, mesh ROM, A100)')
    put('ns96_worst', pct(c['arm_worst']), f'{rel(nsj)} comparators.coordnet_head_k8.arm_worst')
    put('ns96_cmp_worst', pct(c['comparator_worst']), f'{rel(nsj)} comparators.coordnet_head_k8.comparator_worst')
    # ------------------------------------------------ final-report facts ---------------------------------------
    # literature figures, as reported by the cited papers (abstracts checked by the literature pass, 2026-10-20);
    # these are quotations, not project measurements
    for k, v, src in (('lit_ecsw', 'up to four orders of magnitude', 'Farhat, Chapman & Avery, IJNME 2015, doi:10.1002/nme.4820, abstract'),
                      ('lit_gnat', 'more than two orders of magnitude', 'Carlberg et al., JCP 2013, arXiv:1207.1349, abstract (computational resources)'),
                      ('lit_kim1', '2.6×', 'Kim, Choi, Widemann & Zohdi, JCP 2022, arXiv:2009.11990, abstract (1D Burgers)'),
                      ('lit_kim2', '11.7×', 'same, abstract (2D Burgers)'),
                      ('lit_crom_cpu', '109×', 'Chen et al., ICLR 2023, arXiv:2206.02607, abstract (CPU wall-clock)'),
                      ('lit_crom_gpu', '89×', 'same, abstract (GPU wall-clock)'),
                      ('lit_snf', '199×', 'Puri et al., JCP 2025, arXiv:2405.14890, abstract'),
                      ('lit_mcg', '79 %', 'McGreivy & Hakim, Nat. Mach. Intell. 2024, arXiv:2407.07218, abstract (60 of 76 papers)'),
                      ('lit_koch', '40–80×', 'Kochkov et al., PNAS 2021, doi:10.1073/pnas.2101784118, abstract'),
                      ('lit_opinf', 'about $10^6$×', 'McQuarrie, Huang & Willcox, J. R. Soc. NZ 2021, doi:10.1080/03036758.2020.1863237')):
        put(k, v, 'literature quotation: ' + src)
    ll = (ROOT / 'LAB-LOG.md').read_text()
    assert 'Total 168.8 GPU-h over ten lanes' in ll
    put('campaign_gpuh', '168.8 GPU-h', 'LAB-LOG.md (text "Total 168.8 GPU-h over ten lanes")')
    pr = J(FINAL / 'PROVENANCE.json')
    put('p1_R384CNR_1024_worst', pct(pr['p1.kn.1024|R384|CNR.worst']['value']), f'{rel(FINAL / "PROVENANCE.json")}["p1.kn.1024|R384|CNR.worst"]')
    return V


if __name__ == '__main__':
    for k, (v, s) in values().items():
        print(f'{k:22s} {v:28s} {s[:110]}')
