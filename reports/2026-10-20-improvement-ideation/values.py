"""Every number used in report.md, read from a lane output or from this folder's analysis JSONs (no hand-typed values).

values() returns {key: (formatted string, source description)}. Quoted values from generated markdown reports are
checked by string match at the cited file (and line range). Back-of-envelope estimates are computed here from the
read values, with the formula in the source description. Run `python make_report.py` to fill report.src.md.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude')
W = ROOT / 'worktrees'
HERE = Path(__file__).resolve().parent
P1 = W / '2026-10-17-jcp-p1-scaling/experiments/jcp-p1-scaling'
FOM = W / '2026-10-18-jcp-fom2d/experiments/jcp-fom2d'
HRB = W / '2026-10-06-jcp-hr-baselines/experiments/jcp-hr-baselines'
BUD = W / '2026-10-18-jcp-budget/experiments/jcp-budget'
REG = W / '2026-10-19-jcp-regimes/experiments/jcp-regimes'
HEADS = W / '2026-10-18-jcp-heads/experiments/jcp-heads'
DIMS = W / '2026-10-18-jcp-dims/experiments/jcp-dims'
FINAL = ROOT / 'reports/2026-10-17-full-experiment-plan/final'
H200_BW = 4.8e12   # bytes/s, H200 HBM3e spec-sheet bandwidth: an assumption, not a measurement

def J(p):
    return json.loads(Path(p).read_text())

def rel(p):
    return str(Path(p).relative_to(ROOT))

def pct(x, d=3):
    return f'{100 * x:.{d}g} %'

def g(x, d=3):
    return f'{x:.{d}g}'

def ms(x, d=3):
    return f'{1e3 * x:.{d}g} ms'

def quote(path, needle, lines=None):
    txt = Path(path).read_text().splitlines()
    rng = range(len(txt)) if lines is None else range(lines[0] - 1, min(lines[1], len(txt)))
    assert any(needle in txt[i] for i in rng), (path, needle, lines)
    return f'{rel(path)}' + (f':{lines[0]}-{lines[1]}' if lines else '') + f' (text "{needle}")'


def values():
    V = {}
    def put(k, v, src):
        assert k not in V, k
        V[k] = (v, src)

    # ---------------- P1 timing (2D, H200, 1024^2 unless stated) ----------------
    a = J(P1 / 'checks/analysis.json')
    src = rel(P1 / 'checks/analysis.json')
    hc = a['timing']['h_cost_n']
    tev128 = hc['R128|BE|frozen']['descriptive_seconds']['t_evolve']['1024']
    tev384 = hc['R384|CNR|frozen']['descriptive_seconds']['t_evolve']['1024']
    tdec128 = hc['R128|BE|frozen']['descriptive_seconds']['t_decode']['4096']
    tdec384 = hc['R384|CNR|frozen']['descriptive_seconds']['t_decode']['4096']
    tdec384_1k = hc['R384|CNR|frozen']['descriptive_seconds']['t_decode']['1024']
    tinit384 = hc['R384|CNR|frozen']['descriptive_seconds']['t_init']['1024']
    put('t_ev_R128', ms(tev128), f'{src} timing.h_cost_n["R128|BE|frozen"].descriptive_seconds.t_evolve["1024"]')
    put('t_ev_R384', ms(tev384), f'{src} timing.h_cost_n["R384|CNR|frozen"].descriptive_seconds.t_evolve["1024"]')
    put('step_R128', ms(tev128 / 50), 't_ev_R128 / 50 steps (dt 0.005, T 0.25)')
    put('step_R384', ms(tev384 / 50), 't_ev_R384 / 50 steps')
    put('t_dec_R128_4k', ms(tdec128), f'{src} ...["R128|BE|frozen"].descriptive_seconds.t_decode["4096"]')
    put('t_dec_R384_4k', ms(tdec384), f'{src} ...["R384|CNR|frozen"].descriptive_seconds.t_decode["4096"]')
    put('t_dec_R384_1k', ms(tdec384_1k), f'{src} ...["R384|CNR|frozen"].descriptive_seconds.t_decode["1024"]')
    comp = [c for c in a['timing']['components'] if c['subject'] == '1024|R384|gauss128|frozen|CNR'][0]
    cc = comp['components']
    it_mean = float(np.mean(comp['iterations']))
    for k in ('hist_cn', 'pred', 'init', 'iter'):
        put(f'comp_{k}', ms(cc[k]), f'{src} timing.components[R384 CNR].components.{k}')
    put('comp_it_mean', g(it_mean), f'{src} timing.components[R384 CNR].iterations (mean over the 50 steps of dev6|0)')
    nom = a['kernels_2d']['descriptive']['nominal']
    fix = a['kernels_2d']['descriptive']['fixedm']
    for k in ('N', 'J', 'JTJ', 'chol', 'res'):
        for r in ('128', '384'):
            put(f'nom_{k}_{r}', ms(nom[k][r]), f'{src} kernels_2d.descriptive.nominal.{k}["{r}"] (M = 4r, own rule)')
    put('floor_N32', ms(fix['N']['32']), f'{src} kernels_2d.descriptive.fixedm.N["32"] (smallest timed kernel; latency proxy)')
    put('floor_JTJ32', ms(fix['JTJ']['32']), f'{src} kernels_2d.descriptive.fixedm.JTJ["32"] (latency proxy for an r x r op)')
    model_cur = cc['hist_cn'] + cc['pred'] + cc['init'] + it_mean * cc['iter']
    put('model_cur_step', ms(model_cur), 'hist_cn + pred + init + mean_it * iter (P1 component model; the measured step is t_ev_R384/50)')
    onepass = cc['hist_cn'] + nom['res']['384'] + nom['J']['384'] + nom['JTJ']['384'] + nom['chol']['384']
    put('est_onepass_step', ms(onepass), 'hist_cn + res + J + JTJ + chol (nominal R384): one Jacobian, one solve, no predictor, no acceptance Jacobian')
    put('est_onepass_gain', g(model_cur / onepass, 2) + '×', 'model_cur_step / est_onepass_step')
    put('est_onepass_gain_meas', g((tev384 / 50) / onepass, 2) + '×', 'step_R384 (measured) / est_onepass_step')
    etd_q = 4 * (nom['N']['384'] + fix['JTJ']['32'])
    etd_g = 4 * (fix['N']['32'] + fix['JTJ']['32'])
    put('est_etd_step_quad', ms(etd_q), '4 stages x (nominal tested-advection kernel N at R384 + one small r x r op at the latency floor)')
    put('est_etd_step_gal', ms(etd_g), '4 stages x (latency-floor N kernel + latency-floor r x r op): bank-tested Galerkin, m r work per stage')
    put('est_etd_ev_quad', ms(20 * etd_q), '20 steps (dt 0.0125) x est_etd_step_quad')
    put('est_etd_ev_gal', ms(20 * etd_g), '20 steps x est_etd_step_gal')
    put('est_etd_gain_quad', g(tev384 / (20 * etd_q), 2) + '×', 't_ev_R384 (dt 0.005) / est_etd_ev_quad')
    put('est_etd_gain_gal', g(tev384 / (20 * etd_g), 2) + '×', 't_ev_R384 (dt 0.005) / est_etd_ev_gal')

    # ---------------- fom2d (K2 2D) ----------------
    c2 = J(FOM / 'checks/c2-analysis.json')
    k2 = {m.split('|', 1)[1]: v['S'] for m, v in c2['k2']['members'].items()}
    Ss = list(k2.values())
    put('k2_S_min', g(min(Ss)), f'{rel(FOM / "checks/c2-analysis.json")} k2.members[*].S (min over 21)')
    put('k2_S_max', g(max(Ss)), f'{rel(FOM / "checks/c2-analysis.json")} k2.members[*].S (max over 21)')
    def S(key):
        return [v for kk, v in k2.items() if kk.startswith(key)][0]
    for lab, key in (('R384CNR005', 'R384|gauss128|frozen|CNR|0.005|'), ('R384CNR0125', 'R384|gauss128|frozen|CNR|0.0125|'),
                     ('R384CNR001', 'R384|gauss128|frozen|CNR|0.01|'), ('R128BE005', 'R128|fib4181|frozen|BE|0.005|'),
                     ('R128BE0125', 'R128|fib4181|frozen|BE|0.0125|'), ('R512BE005', 'R512|gauss192|frozen|BE|0.005|')):
        put(f'k2_S_{lab}', g(S(key)), f'{rel(FOM / "checks/c2-analysis.json")} k2.members[...{key}...].S')
    gal = [v['S'] for v in c2['k2_diagnostic_gal']['members'].values()]
    put('k2gal_S_max', g(max(gal)), f'{rel(FOM / "checks/c2-analysis.json")} k2_diagnostic_gal.members[*].S (max)')
    mf = J(FOM / 'checks/match-final.json')
    ps = {p['key']: p for p in mf['pareto']['PS']['points']}
    ub = {p['key']: p for p in mf['pareto']['UB']['points']}
    for lab, key in (('N48', 'ps|N48|0.0125'), ('N24', 'ps|N24|0.025')):
        put(f'ps_{lab}_med', pct(ps[key]['median']), f'{rel(FOM / "checks/match-final.json")} pareto.PS[{key}].median')
        put(f'ps_{lab}_worst', pct(ps[key]['worst']), f'{rel(FOM / "checks/match-final.json")} pareto.PS[{key}].worst')
        put(f'ps_{lab}_cost', ms(ps[key]['cost']), f'{rel(FOM / "checks/match-final.json")} pareto.PS[{key}].cost (screening, lattice output)')
    u = ub['fd|UB|L256|0.005|t2']
    put('ub256_med', pct(u['median']), f'{rel(FOM / "checks/match-final.json")} pareto.UB[fd|UB|L256|0.005|t2].median')
    put('ub256_worst', pct(u['worst']), f'{rel(FOM / "checks/match-final.json")} pareto.UB[fd|UB|L256|0.005|t2].worst')
    put('ps_N48_modes', str(47 ** 2), 'fom2d glossary: N keeps sine modes 1..N-1 per axis -> (N-1)^2; ' + quote(FOM / 'report.md', 'sine', (460, 475)))
    put('ps_N24_modes', str(23 ** 2), '(24-1)^2, same definition')
    # 4096^2 stage-2 times from the generated fom2d report table
    rep = (FOM / 'report.md').read_text().splitlines()
    t4 = {}
    for ln in rep[369:395]:
        m_ = re.match(r'\| (R\d+) (\w+) (LSPG-\w+) \$\\Delta t=([\d.]+)\$ @ \$4096\^2\$ \| ([\d.]+) \| PS \$N=(\d+)\$.*\| ([\d.]+) \|', ln)
        if m_:
            t4[(m_.group(1), m_.group(3), m_.group(4))] = (float(m_.group(5)), int(m_.group(6)), float(m_.group(7)))
    assert len(t4) == 21, len(t4)
    rom, nps, tps = t4[('R384', 'LSPG-CNR', '0.005')]
    put('t4k_R384', ms(rom), f'{rel(FOM / "report.md")}:370-393 (4096^2 stage-2 table, R384 LSPG-CNR dt 0.005)')
    put('t4k_ps48', ms(tps), f'{rel(FOM / "report.md")}:370-393 (PS N=48 dt 0.0125, full-field 4096^2)')
    rom128, _, tps128 = t4[('R128', 'LSPG-BE', '0.005')]
    put('t4k_R128', ms(rom128), f'{rel(FOM / "report.md")}:370-393 (R128 LSPG-BE dt 0.005)')
    rom128b, n128b, tps24 = t4[('R128', 'LSPG-BE', '0.0125')]
    put('t4k_ps24', ms(tps24), f'{rel(FOM / "report.md")}:370-393 (PS N=24 dt 0.025)')
    ti384_4k = hc['R384|CNR|frozen']['descriptive_seconds']['t_init']['4096']
    ti128_4k = hc['R128|BE|frozen']['descriptive_seconds']['t_init']['4096']
    put('smax_R384', g(tps / (tdec384 + ti384_4k), 3), 'illustrative K2 ceiling with zero evolution cost and the current dense decode: t4k_ps48 / (P1 t_decode + t_init at 4096^2, R384 CN-R); cross-job ratio')
    put('smax_R128', g(tps / (tdec128 + ti128_4k), 3), 'same for R128 BE vs PS N48: t4k_ps48 / (t_dec_R128_4k + t_init R128 at 4096^2)')
    put('smax_R128_N24', g(tps24 / (tdec128 + ti128_4k), 3), 'R128 dt 0.0125 vs its N24 comparator: t4k_ps24 / (t_dec_R128_4k + t_init R128 at 4096^2)')
    put('est_S_etd_quad', g(tps / (20 * etd_q + ti384_4k + tdec384), 2), 'PS N48 time / (est_etd_ev_quad + init + decode at 4096^2), R384')
    put('est_S_etd_gal', g(tps / (20 * etd_g + ti384_4k + tdec384), 2), 'PS N48 time / (est_etd_ev_gal + init + decode), R384')

    tdec384_256 = hc['R384|CNR|frozen']['descriptive_seconds']['t_decode']['256']
    put('t_dec_R384_256', ms(tdec384_256), f'{src} ...["R384|CNR|frozen"].descriptive_seconds.t_decode["256"] (proxy for a 257^2 lattice decode)')
    pscost = ps['ps|N48|0.0125']['cost']
    put('est_S_lat_quad', g(pscost / (20 * etd_q + tinit384 + tdec384_256), 2), 'ps_N48_cost (lattice contract) / (est_etd_ev_quad + init + 256^2 decode)')
    put('est_S_lat_gal', g(pscost / (20 * etd_g + tinit384 + tdec384_256), 2), 'ps_N48_cost / (est_etd_ev_gal + init + 256^2 decode)')
    # ---------------- hr-baselines (H-base) ----------------
    sb = J(HRB / 'runs/s2d1024b/archive/output/stageb.json')['timing']
    def med(key, which):
        if which == 'tA':
            xs = [0.5 * (t['tA1'] + t['tA2']) for c in sb[key]['cases'].values() for t in c['triples'] if t['valid']]
        else:
            xs = [t[which] for c in sb[key]['cases'].values() for t in c['triples'] if t['valid']]
        return float(np.median(xs))
    for lab, key in (('R384', '1024|R384|CNR||ecsw|384|k8|CNR|0.005|common|central'),
                     ('R128', '1024|R128|BE||qdeim|256|512|CNR|0.005|common|central'),
                     ('R512', '1024|R512|BE||qdeim|256|512|CNR|0.005|common|central')):
        o, b = med(key, 'tA'), med(key, 'tB')
        put(f'hrb_{lab}_ours', ms(o), f'{rel(HRB / "runs/s2d1024b/archive/output/stageb.json")} timing["{key}"] median of (tA1+tA2)/2')
        put(f'hrb_{lab}_base', ms(b), f'... same key, median tB')
        put(f'hrb_{lab}_ratio', g(b / o, 2), 'hrb_base / hrb_ours (own recomputation; the lane reports the paired ratio)')
        if lab == 'R384':
            put('hrb_R384_base_step', ms(b / 50), 'hrb_R384_base / 50 steps')
            put('hrb_R384_ours_step', ms(o / 50), 'hrb_R384_ours / 50 steps')
            put('est_etd_vs_ecsw_quad', g(b / (20 * etd_q + tinit384 + tdec384_1k), 2) + '×', 'hrb_R384_base / (est_etd_ev_quad + init + decode at 1024^2)')
            put('est_etd_vs_ecsw_gal', g(b / (20 * etd_g + tinit384 + tdec384_1k), 2) + '×', 'hrb_R384_base / (est_etd_ev_gal + init + decode at 1024^2)')
    hr_rep = HRB / 'report.md'
    fz = J(HRB / 'checks/FROZEN-STAGEB-c2d1024.json')['member_info']
    fsrc = rel(HRB / 'checks/FROZEN-STAGEB-c2d1024.json')
    for lab, key in (('e384k8', 'ecsw|384|k8|CNR|0.005|common|central'), ('e256k8', 'ecsw|256|k8|CNR|0.005|common|central'),
                     ('e384k4', 'ecsw|384|k4|CNR|0.005|common|central'), ('q256', 'qdeim|256|512|CNR|0.005|common|central'),
                     ('e128k8', 'ecsw|128|k8|CNR|0.005|common|central'), ('e512k8', 'ecsw|512|k8|CNR|0.005|common|central')):
        put(f'pts_{lab}', str(fz[key]['hr_points']), f'{fsrc} member_info["{key}"].hr_points')
        put(f'tq_{lab}', ms(fz[key]['t']), f'{fsrc} member_info["{key}"].t (screening query time, 1024^2)')
    put('est_F3_pts', f"{fz['ecsw|384|k4|CNR|0.005|common|central']['hr_points']}–{fz['ecsw|384|k8|CNR|0.005|common|central']['hr_points']}", 'kappa r for kappa 4, 8 at r = 384 (pts_e384k4, pts_e384k8)')
    put('est_F3_ratio', f"{16384 / fz['ecsw|384|k8|CNR|0.005|common|central']['hr_points']:.2g}–{16384 / fz['ecsw|384|k4|CNR|0.005|common|central']['hr_points']:.2g}×", '16384 / pts_e384k8 to 16384 / pts_e384k4')
    put('tq_native_e384k8', ms(fz['ecsw|384|k8|CNR|0.005|native|central']['t']), f'{fsrc} member_info["ecsw|384|k8|CNR|0.005|native|central"].t')
    put('hrb_pick_worst_R384', '1.35 %', quote(FINAL / 'report.md', '`1024\\|R384\\|CNR` | 1.78 % | `ecsw\\|384\\|k8\\|CNR\\|0.005\\|common\\|central` | 1.35 %'))
    put('ours_worst_R384CNR', '1.78 %', 'same row')

    # ---------------- budget (2D 256^2, 38 cases) ----------------
    bd = J(BUD / 'checks/budget2d-b256-256.json')['members']
    bsrc = rel(BUD / 'checks/budget2d-b256-256.json')
    for lab, mem in (('LCNR384', 'R384|LSPG|CNR|1'), ('GAL384', 'R384|GAL|CN|1'), ('LBE128', 'R128|LSPG|BE|1')):
        s_ = bd[mem]['summary']
        put(f'bud_{lab}_T', pct(s_['median_main']['T']), f'{bsrc} members["{mem}"].summary.median_main.T')
        put(f'bud_{lab}_eP', pct(s_['median_e_P']), f'{bsrc} members["{mem}"].summary.median_e_P')
        put(f'bud_{lab}_eP_worst', pct(s_['worst_e_P']), f'{bsrc} members["{mem}"].summary.worst_e_P')
        put(f'bud_{lab}_dyn_worst', pct(s_['worst_e_dyn_star']), f'{bsrc} members["{mem}"].summary.worst_e_dyn_star')
        put(f'bud_{lab}_tot', pct(s_['median_e_tot']), f'{bsrc} members["{mem}"].summary.median_e_tot')
        put(f'bud_{lab}_tot_worst', pct(s_['worst_e_tot']), f'{bsrc} members["{mem}"].summary.worst_e_tot')
        put(f'bud_{lab}_dom', s_['dominant'], f'{bsrc} members["{mem}"].summary.dominant')
    put('bud_T_ratio', g(bd['R384|LSPG|CNR|1']['summary']['median_main']['T'] / bd['R384|GAL|CN|1']['summary']['median_main']['T'], 2) + '×',
        'bud_LCNR384_T / bud_GAL384_T')

    # ---------------- regimes ----------------
    ra = J(REG / 'checks/analysis.json')
    pf = ra['projection_floor']
    rsrc = rel(REG / 'checks/analysis.json')
    for coh in ('val32', 'N0s', 'C1', 'C2', 'Am'):
        for r in ('384', '512', '128'):
            put(f'floor_{coh}_{r}', pct(pf[f'{coh}|R{r}']['median']), f'{rsrc} projection_floor["{coh}|R{r}"].median')
            put(f'floormax_{coh}_{r}', pct(pf[f'{coh}|R{r}']['max']), f'{rsrc} projection_floor["{coh}|R{r}"].max')
    cells = ra['configs']['R384|LSPG_CNR_1']['cells']
    for coh in ('C1', 'C2', 'N0s'):
        put(f'reg_med_{coh}', pct(cells[coh]['median']), f'{rsrc} configs["R384|LSPG_CNR_1"].cells["{coh}"].median')
        put(f'reg_ratio_{coh}', g(cells[coh]['median'] / pf[f'{coh}|R384']['median'], 2), f'reg_med_{coh} / floor_{coh}_384 (ratio of medians)')
    put('C1_floor_r_gain', g(pf['C1|R384']['median'] / pf['C1|R512']['median'], 2) + '×', 'floor_C1_384 / floor_C1_512')
    put('N0s_over_val32', g(pf['N0s|R384']['median'] / pf['val32|R384']['median'], 2) + '×', 'floor_N0s_384 / floor_val32_384')
    put('reg_qual_C1', quote(FINAL / 'report.md', 'C1 (low ν, high steepness s; paired with A−) | 16/16 | 0/8'), 'final report §3.12 table')

    # ---------------- heads ----------------
    put('head_K16_med', '1.44 %', quote(FINAL / 'report.md', 'the K = 16 head (median 1.44 %)'))
    put('lin_r128_med', '1.1 %', quote(FINAL / 'report.md', 'r = 128 reaches 1.1 % at 0.591×'))
    put('lin_r128_cost', '0.591×', 'same sentence')

    # ---------------- 3D (final report rows) ----------------
    put('k2_3d_range', '0.00786–0.158', quote(FINAL / 'report.md', 'union speed-up 0.00786–0.158 (4 ROMs)'))
    put('h2q_W768', '2.95', quote(FINAL / 'report.md', '| C_W768 | 5.37 | 2.95 | pass | pass |'))
    put('dims_mstar_2d', '9,216–36,864', quote(W / '2026-10-18-jcp-dims/experiments/jcp-dims/report.md', 'm* ranges over 9216 … 36864 points', (40, 50)))
    put('b6_square', '0/108', quote(DIMS / 'report.md', '0 of 108 square (M = r) configurations pass Q1'))
    put('lmf_gauss', '65,536 vs 16,384', quote(DIMS / 'report.md', 'gauss: LMF m* = 65536 (stable) vs default m* = 16384 (stable)'))

    # ---------------- P1 errors, iterations, quoted facts ----------------
    cf = a['configurations']
    for lab, key in (('R128BE', '1024|R128|BE'), ('R384CNR', '1024|R384|CNR')):
        put(f'p1_{lab}_med', pct(cf[key]['e0_median']), f'{src} configurations["{key}"].e0_median')
        put(f'p1_{lab}_worst', pct(cf[key]['e0_worst']), f'{src} configurations["{key}"].e0_worst')
    sbm = J(HRB / 'runs/s2d1024b/archive/output/stageb.json')['members']['ecsw|384|k8|CNR|0.005|common|central']['certs']
    its = [c['it_sum'] / c['steps'] for c in sbm.values() if c.get('valid')]
    put('base_it_step', f'{min(its):.2f}–{max(its):.2f}', f'{rel(HRB / "runs/s2d1024b/archive/output/stageb.json")} members["ecsw|384|k8|CNR|0.005|common|central"].certs[*].it_sum/steps (range over cases)')
    LL = ROOT / 'LAB-LOG.md'
    put('batch_W768', '3.94×', quote(LL, 'batching ≥ 2× only for W_C_W768 (3.94×)'))
    put('qn_h14', '2 pass, 20 fail, 3 rescue, 5 unavailable', quote(LL, 'H1.4 (QN ≤ 0.6× FS-K2): 2 pass, 20 fail, 3 rescue, 5 unavailable'))
    put('ns3d_pod', '8–10×', quote(LL, 'about 8–10×', (88, 92)))
    put('tensor_34gb', '34 GB', quote(LL, 'against 34 GB for the tensor'))
    put('tensor_bw_3d', 'Jacobian 0.66 vs 1.04 ms despite 29 vs 1.1 GFLOP', quote(LL, 'Jacobian 0.66 vs 1.04 ms despite 29 vs 1.1 GFLOP (GEMM vs bandwidth-bound contraction)'))
    put('hrb_build_2d', '1.45 h', quote(LL, 'Construction cost: baseline 1.45 h (2D)'))
    put('bank_fit_h', '2.97 h', quote(LL, 'ours 2.97 h (R512 bank)'))
    put('smooth_floor', '38.2%', quote(W / '2026-10-08-jcp-smooth-bank/experiments/jcp-smooth-bank/report.md', 'the median projection floor by 38.2% in `acc`'))
    put('smooth_roll', '+1.7%', 'same line: "rollout error vs S changes by +1.7% (`acc`)"')
    put('refdv_off_2d', '0.35 %', quote(LL, 'Old first-order references were off by:** 3D median about 0.7 %, maximum 2.81 %; 2D maximum 0.35 %'))
    put('wide_floor', '3.23 → 1.29 %', quote(LL, 'the 3D projection floor falls 3.23 → 1.29 %'))

    # ---------------- this folder's analyses ----------------
    sf = J(HERE / 'analysis/sine_floor.json')['summary']
    for N in ('24', '32', '48', '64'):
        put(f'sine_block{N}_med', pct(sf['block'][N]['median']), f'analysis/sine_floor.json summary.block["{N}"].median')
        put(f'sine_block{N}_worst', pct(sf['block'][N]['worst']), f'analysis/sine_floor.json summary.block["{N}"].worst')
    for K in ('128', '384', '512', '1536', '2304'):
        put(f'sine_lap{K}_med', pct(sf['lap'][K]['median']), f'analysis/sine_floor.json summary.lap["{K}"].median')
        put(f'sine_lap{K}_worst', pct(sf['lap'][K]['worst']), f'analysis/sine_floor.json summary.lap["{K}"].worst')
    bv = J(HERE / 'analysis/bank_vs_sine.json')['summary']
    for r in ('128', '384', '512'):
        put(f'bank{r}_med', pct(bv[f'bank_{r}']['median']), f'analysis/bank_vs_sine.json summary.bank_{r}.median')
        put(f'bank{r}_worst', pct(bv[f'bank_{r}']['max']), f'analysis/bank_vs_sine.json summary.bank_{r}.max')
        put(f'ratio_med_{r}', g(bv[f'ratio_sineK_over_r_{r}']['median'], 2), f'analysis/bank_vs_sine.json summary.ratio_sineK_over_r_{r}.median')
        put(f'ratio_max_{r}', g(bv[f'ratio_sineK_over_r_{r}']['max'], 2), f'analysis/bank_vs_sine.json summary.ratio_sineK_over_r_{r}.max')
        put(f'sineK_worst_{r}', str(bv[f'sineK_worstcase_match_bank_{r}']), f'analysis/bank_vs_sine.json summary.sineK_worstcase_match_bank_{r}')
        put(f'worst_ratio_{r}', g(bv[f'sineK_worstcase_match_bank_{r}'] / int(r), 2), f'sineK_worst_{r} / {r}')
    for K in ('128', '512'):
        put(f'hyb{K}_med', pct(bv[f'bank384_plus_sine{K}']['median']), f'analysis/bank_vs_sine.json summary.bank384_plus_sine{K}.median')
        put(f'hyb{K}_worst', pct(bv[f'bank384_plus_sine{K}']['max']), f'analysis/bank_vs_sine.json summary.bank384_plus_sine{K}.max')
    es = J(HERE / 'analysis/explicit_stability.json')['settings']
    for lab, key in (('128', 'R128_M512_gauss96'), ('384', 'R384_M1536_gauss128')):
        s_ = es[key]
        put(f'dta_min_{lab}', g(s_['dt_a']['min']), f'analysis/explicit_stability.json settings.{key}.dt_a.min')
        put(f'dta_med_{lab}', g(s_['dt_a']['median']), f'analysis/explicit_stability.json settings.{key}.dt_a.median')
        put(f'dtd_min_{lab}', g(s_['dt_d_explicit']['min']), f'analysis/explicit_stability.json settings.{key}.dt_d_explicit.min')
        put(f'frac0125_{lab}', g(s_['frac_states_dt_a_ge_0p0125']), f'analysis/explicit_stability.json settings.{key}.frac_states_dt_a_ge_0p0125')
        put(f'frac025_{lab}', g(s_['frac_states_dt_a_ge_0p025'], 2), f'analysis/explicit_stability.json settings.{key}.frac_states_dt_a_ge_0p025')
        put(f'maxre_{lab}', g(s_['max_re']['max']), f'analysis/explicit_stability.json settings.{key}.max_re.max')
    od = J(HERE / 'analysis/ood_indicator.json')['configs']
    put('ood_n', str(next(iter(od.values()))['n']), 'analysis/ood_indicator.json configs["R128|LSPG_BE_1"].n (rows per configuration, invalid runs included)')
    fu = J(HERE / 'analysis/floor_vs_upwind.json')
    put('floor_s_spearman', g(fu['spearman_floor384_vs_s'], 2), 'analysis/floor_vs_upwind.json spearman_floor384_vs_s')
    put('q_worst', g(fu['worst5_by_floor384'][0]['q'], 2), 'analysis/floor_vs_upwind.json worst5_by_floor384[0].q')
    put('q_med', g(fu['median_q_all'], 2), 'analysis/floor_vs_upwind.json median_q_all')
    ds = J(HERE / 'analysis/decode_spectral.json')['settings']
    put('dec_tail_384_4096_med', pct(ds['R384']['4096']['median']), 'analysis/decode_spectral.json settings.R384["4096"].median')
    put('dec_tail_384_4096_max', pct(ds['R384']['4096']['max']), 'analysis/decode_spectral.json settings.R384["4096"].max')
    put('sine_lap4096_worst', pct(J(HERE / 'analysis/sine_floor.json')['summary']['lap']['4096']['worst']), 'analysis/sine_floor.json summary.lap["4096"].worst')
    put('sine_lap4096_med', pct(J(HERE / 'analysis/sine_floor.json')['summary']['lap']['4096']['median']), 'analysis/sine_floor.json summary.lap["4096"].median')

    # ---------------- operation counts (design constants r, M = 4r, m from P1 configs) ----------------
    for r, m in ((128, 4181), (384, 16384)):
        M = 4 * r
        put(f'flops_J_{r}', f'{2 * m * M * r:.2g}', f'2 m M r with m = {m}, M = {M} (dense Psi^T [m x r] product per Jacobian)')
        put(f'flops_Jgal_{r}', f'{2 * m * r * r:.2g}', f'2 m r^2 (bank-tested Galerkin Jacobian)')
        put(f'bytes_T_lspg_{r}', f'{8 * M * r * r / 1e9:.2g} GB', f'8 M r^2 bytes (LSPG tensor)')
        put(f'bytes_T_gal_{r}', f'{8 * r ** 3 / 1e9:.2g} GB', f'8 r^3 bytes (Galerkin tensor)')
        put(f'tread_T_lspg_{r}', ms(8 * M * r * r / H200_BW), f'bytes_T_lspg_{r} / {H200_BW:.2g} B/s (assumed H200 bandwidth)')
        put(f'tread_T_gal_{r}', ms(8 * r ** 3 / H200_BW), f'bytes_T_gal_{r} / {H200_BW:.2g} B/s')
        put(f'm_over_M_{r}', g(m / M, 2), f'm / M')
        put(f'm_over_r_{r}', g(m / r, 2), f'm / r')
    m_, r_, M_ = 16384, 384, 1536
    put('flops_N_sine_384', f'{2 * m_ * (2 * r_ + M_):.2g}', '2 m (2r + M): two m x r products and the m x M test projection (r = 384, M = 1536, m = 16384)')
    put('flops_N_bank_384', f'{6 * m_ * r_:.2g}', '6 m r: two m x r products and an m x r bank-test projection')
    put('flops_N_pre_384', f'{6 * m_ * r_:.2g}', 'precontracted sine-tested Galerkin: P~ = Psi Q R_m^{-T} (m x r) built offline; 6 m r online')
    put('m_over_ps48', g(16384 / (71 * 71), 2), 'm (Gauss 128^2 = 16384) / PS N48 padded-grid interior points (72 intervals -> 71^2 = 5041; fom2d pscore.py:4-8, 3/2 padding)')
    pc_ = J(HERE / 'analysis/pod_control.json')
    for r in ('64', '128', '200'):
        for b in ('pod', 'bank', 'sine'):
            put(f'loo_{b}_{r}_med', pct(pc_['loo_pod'][r][b]['median']), f'analysis/pod_control.json loo_pod["{r}"].{b}.median')
            put(f'loo_{b}_{r}_max', pct(pc_['loo_pod'][r][b]['max']), f'analysis/pod_control.json loo_pod["{r}"].{b}.max')
    put('sp_bank_s', g(pc_['steepness_control']['spearman_bank384_vs_s'], 2), 'analysis/pod_control.json steepness_control.spearman_bank384_vs_s')
    put('sp_sine_s', g(pc_['steepness_control']['spearman_sine384_vs_s'], 2), 'analysis/pod_control.json steepness_control.spearman_sine384_vs_s')
    for r in ('128', '384', '512'):
        put(f'condM_{r}', g(pc_['galerkin_mass'][r]['cond_M'], 3), f'analysis/pod_control.json galerkin_mass["{r}"].cond_M')
        put(f'condA2_{r}', g(pc_['galerkin_mass'][r]['cond_A_squared'], 3), f'analysis/pod_control.json galerkin_mass["{r}"].cond_A_squared')
    od2 = J(HERE / 'analysis/ood_indicator.json')['configs']
    for tgt, lab in (('A_e_gt_2x_inrange_median', 'A'), ('B_e_gt_1pct', 'B')):
        for ind in ('dist_qp', 's', 'd_box', 'rho_max'):
            for met in ('auc', 'detection', 'false_alarm'):
                xs = [c['indicators'][ind][tgt][met] for c in od2.values()]
                put(f'ood{lab}_{ind}_{met}', f'{min(xs):.2f}–{max(xs):.2f}', f'analysis/ood_indicator.json configs[*].indicators.{ind}.{tgt}.{met} (range over 9 configurations)')
    put('ood_n_invalid', str(sum(c['n_invalid'] for c in od2.values())), 'analysis/ood_indicator.json sum of configs[*].n_invalid')
    idc = J(HERE / 'analysis/inspan_decomp.json')['regimes']
    for coh_ in ('C1', 'C2', 'N0s'):
        put(f'inspan_{coh_}', pct(idc[coh_]['median_inspan']), f'analysis/inspan_decomp.json regimes.{coh_}.median_inspan (per-case max over t of ||v - Pu|| / ||u0||, then median)')
    put('a4_384_1152', '16384', quote(DIMS / 'report.md', '| 384 | 1152 | m* = 16384 (stable)'))
    put('mM_1152_vs_1536', f'{1 - (16384 * 1152) / (16384 * 1536):.0%}', '1 - (m* M at M = 1152) / (m* M at M = 1536), r = 384, both m* = 16384')
    for M__, mm in (('512', '36864'), ('768', '36864'), ('1536', '16384'), ('2048', '9216')):
        put(f'a4_384_{M__}', mm, quote(DIMS / 'report.md', f'| 384 | {M__} | m* = {mm} (stable)'))
    put('a4_384_384', 'none', quote(DIMS / 'report.md', '| 384 | 384 | none (within the candidate list)'))
    rr = J(HERE / 'analysis/ood_indicator.json')['configs']['R384|LSPG_CNR_1']
    for c_ in ('dev6', 'val32', 'N0', 'N0s', 'Am', 'Ap', 'B1', 'B2', 'C1', 'C2'):
        put(f'alarm_{c_}', f"{rr['per_regime_dist_qp_alarm_rate'][c_]:.2f}", f'analysis/ood_indicator.json configs["R384|LSPG_CNR_1"].per_regime_dist_qp_alarm_rate["{c_}"]')
    put('ood_excl', ', '.join(rr['excluded_ineligible_reference']), 'analysis/ood_indicator.json configs[*].excluded_ineligible_reference')
    put('bytes_T_gal_384_sym', f'{8 * 384 * 384 * 385 / 2 / 1e9:.2g} GB', '8 r * r(r+1)/2 bytes (Galerkin tensor, symmetric in the quadratic pair)')
    put('bytes_T_lspg_384_sym', f'{8 * 1536 * 384 * 385 / 2 / 1e9:.2g} GB', '8 M r(r+1)/2 bytes (LSPG tensor, symmetric pair)')
    put('flops_N_bank_a4worst', f'{6 * 36864 * 384:.2g}', '6 m r at m = 36864 (largest A4 m* at r = 384)')
    put('wide_n_cases', '96', quote(W / '2026-10-08-jcp-wide-bank/experiments/jcp-wide-bank/report.md', '96 cases × six saved times', (113, 133)))
    return V


if __name__ == '__main__':
    for k, (v, s) in values().items():
        print(f'{k:28s} {v:>16s}   {s[:110]}')
