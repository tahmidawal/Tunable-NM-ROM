"""Shared data access, formatting and styling for make_final_page.py and make_deck.py.

Every number shown on the page and the deck is read here from FINAL-EXPORT.json (verdicts, stop-rule outcomes,
GPU-hours) or PROVENANCE.json (the registered lane values behind report.md). The only literals in the
generators are the plan's registered bars (2x, 3x, 0.5, 1.5x, ...), kept in BARS below.
"""
import base64, html, json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
X = json.load(open(os.path.join(HERE, 'FINAL-EXPORT.json')))
PROV = json.load(open(os.path.join(HERE, 'PROVENANCE.json')))

# Registered constants of the plan (bars, targets), not measurements.
BARS = dict(win_S=2, stop_S=0.5, gate_G=2, e5_speed=3, e6_gain=1.5, e6_kmax=16, e6_mmax=16384,
            e8_a=3, e8_b=2, e8_c_m=2, e8_c_tail=3, e8_guard=2, e9_good=30, e9_bad=20)
LANES = ['imp-box', 'imp-geometry', 'imp-varcoef', 'imp-faststep', 'imp-hrcost', 'imp-multimesh', 'imp-fronts']
LANE_EXP = {'imp-box': 'E1', 'imp-geometry': 'E2', 'imp-varcoef': 'E3', 'imp-faststep': 'E5', 'imp-hrcost': 'E6',
            'imp-multimesh': 'E7', 'imp-fronts': 'E8, E9, E10'}


def p(key):
    """Registered value from PROVENANCE.json."""
    v = PROV[key]
    return v['value'] if isinstance(v, dict) and 'value' in v else v


def esc(s):
    return html.escape(str(s))


def r3(x):
    """Ratio to 3 significant figures."""
    return f'{x:.3g}'


def pct(x, d=2):
    return f'{100 * x:.{d}f} %'


def pct3(x):
    return f'{100 * x:.3g} %'


def gh(x):
    return f'{x:.2f}'


def img(name, cap, cls=''):
    b = base64.b64encode(open(os.path.join(HERE, 'figs', name), 'rb').read()).decode()
    return (f'<figure class="{cls}"><div class="frame"><img src="data:image/png;base64,{b}" alt="{esc(cap)}" loading="lazy">'
            f'</div><figcaption>{esc(cap)}</figcaption></figure>')


def branches():
    out = []
    for lane in LANES:
        h = PROV[f'git.head.{lane}']
        br = h['command'].split()[-1]
        out.append(dict(lane=lane, exp=LANE_EXP[lane], branch=br, head=h['value']['commit'][:9],
                        date=h['value']['date'][:10], mirror=p(f'git.mirror.{lane}')[:9]))
    return out


def verdict_class(v):
    v = v.lower()
    if v.startswith('partly'):
        return 'part'
    if v.startswith(('not run',)):
        return 'skip'
    if v.startswith(('inconclusive', 'descriptive')):
        return 'inc'
    return 'fail'


# Derived values used by both outputs (all from the JSONs).
def facts():
    f = {}
    f['v1'], f['v2'], f['v3'] = (X['problem_verdicts'][k] for k in ('1', '2', '3'))
    f['e1_maxS'] = p('e1.stop_rule.max_S')
    f['e1_nmatch'] = p('e1.stop_rule')['n_match']
    f['e1_lspg_min'], f['e1_lspg_max'], f['e1_lspg_n'] = p('e1.bank_lspg.S.min'), p('e1.bank_lspg.S.max'), p('e1.bank_lspg.n')
    f['e1_e4_wide_max'] = p('e1.bank_e4.wide.S.max')
    f['e1_podg'] = p('e1.S.W08_A2.podg|0.01|worst')
    f['e1_ecsw'] = p('e1.S.W08_A2.ecsw|0.01|worst')
    f['e1_bank'] = p('e1.S.W08_A2.bank_e4|0.01|worst')
    f['e1_bank_test'] = p('e1.S_test.W08_A2.bank_e4|0.01|worst')
    f['e1_trans'] = p('e1.S.W40_A3.trans|0.03|median')
    f['e1_trans_test'] = p('e1.S_test.W40_A3.trans|0.03|median')
    f['e1_trans_inc'] = p('e1.S.W40_A3.trans|0.01|median')
    f['e1_a16_min'], f['e1_a16_max'] = p('e1.a16.fom_dense.min'), p('e1.a16.fom_dense.max')
    scope = p('e1.scope')
    f['e1_cells'] = [k for k, v in scope.items() if v == 'collected']
    f['e1_deferred'] = [k for k, v in scope.items() if v != 'collected']
    f['e2_L_min'], f['e2_L_max'] = p('e2.L|wide|S1.S_val.min'), p('e2.L|wide|S1.S_val.max')
    f['e2_H_min'], f['e2_H_max'] = p('e2.H|wide|S1.S_val.min'), p('e2.H|wide|S1.S_val.max')
    f['e2_max'] = p('e2.S.max_all')
    f['e2_labels'] = p('e2.geometry_labels')
    pfL, pfH = p('e2.L|wide|S1.per_family_val'), p('e2.H|wide|S1.per_family_val')
    f['e2_fdmg'] = {g: (pf['0.03|worst']['fdmg'], pf['0.1|worst']['fdmg']) for g, pf in (('L', pfL), ('H', pfH))}
    f['e2_sem'] = {g: (p(f'e2.{g}|wide|S1.sem_worst.min'), p(f'e2.{g}|wide|S1.sem_worst.max')) for g in ('L', 'H')}
    f['e3_G03'], f['e3_G10'] = p('e3.0.03.G'), p('e3.0.1.G')
    f['e3_ci03'], f['e3_ci10'] = p('e3.0.03.G_ci95'), p('e3.0.1.G_ci95')
    f['e3_rom03'], f['e3_rom10'] = p('e3.0.03.rom_worst'), p('e3.0.1.rom_worst')
    f['e3_C'] = p('e3.contrast')
    f['e3_gate'], f['e3_gate_reg'] = p('e3.gate'), p('e3.gate_registered')
    f['e3_t10'], f['e3_t3'] = p('e3.test.etd10.worst'), p('e3.test.etd3.worst')
    f['e3_t_elig'], f['e3_t_n'] = p('e3.test.etd10.n_eligible'), p('e3.test.n_cases')
    f['e5_min'], f['e5_max'] = p('e5.etdrk4.speedup.min'), p('e5.etdrk4.speedup.max')
    f['e5_rows'] = []
    names = {'G-CN': 'Galerkin CN (precontracted)', 'G-E4a': 'Galerkin ETDRK4, small step',
             'G-E4b': 'Galerkin ETDRK4, large step', 'G-R1': 'one-pass linearised Galerkin CN'}
    for mesh in ('1024', '4096'):
        for arm in ('G-CN', 'G-E4a', 'G-E4b', 'G-R1'):
            k = f'e5.{mesh}.{arm}'
            fired = [c.replace('_', ' ') for c, b in p(k + '.fired').items() if b]
            f['e5_rows'].append(dict(mesh=mesh, arm=names[arm], worst=p(k + '.worst384'), ref=p(k + '.arm1worst384'),
                                     better=p(k + '.better384'), worse=p(k + '.worse384'),
                                     speed=p(k + '.speedup384'), fired=fired))
    f['e5_ref1024'] = p('e5.1024.G-CN.arm1worst384')
    f['e5_e4a_worst'] = p('e5.1024.G-E4a.worst384')
    f['e5_pod_a'], f['e5_pod_b'], f['e5_pod_lspg'] = p('e5.pod_ratio.G-E4a'), p('e5.pod_ratio.G-E4b'), p('e5.pod_ratio.L-CNR')
    f['e6_F2'], f['e6_F3'] = X['experiments']['E6']['F2'], X['experiments']['E6']['F3']
    f['e6_R_worst'], f['e6_R_m'] = p('e6.R.worst'), p('e6.R.m')
    f['e6_F2_m'], f['e6_F2_worst'], f['e6_F2_e2e'] = p('e6.F2.m'), p('e6.F2.worst'), p('e6.F2.e2e')
    f['e6_best'] = p('e6.F3.best_e2e')
    f['e6_npass'] = p('e6.F3.n_q2_pass')
    f['e6_n_f3'] = sum(1 for k in PROV if re.fullmatch(r'e6\.F3[ab]_k\d+\.q2', k))
    f['e6_jac2'], f['e6_jac4'] = p('e6.jac.dup'), p('e6.jac.quad')
    f['e6_f2f3'] = p('e6.F2F3_k16.e2e')
    f['e6_rows'] = [dict(arm=a, m=p(f'e6.{a}.m'), worst=p(f'e6.{a}.worst'), e2e=p(f'e6.{a}.e2e'), q2=p(f'e6.{a}.q2'))
                    for a in ('F3a_k4', 'F3a_k8', 'F3a_k16', 'F3b_k4', 'F3b_k8', 'F3b_k16')]
    c = p('e7.counts.worst')
    f['e7_match'], f['e7_n'] = c['MATCH'], p('e7.n_pairs')
    f['e7_min'], f['e7_med'], f['e7_max'] = p('e7.ratio.min'), p('e7.ratio.median'), p('e7.ratio.max')
    f['e7_case'] = p('e7.counts.casewise')
    f['e7_roll'] = p('e7.counts.rollout')
    f['e7_noadm_case'], f['e7_noadm_roll'] = p('e7.noadm.casewise'), p('e7.noadm.rollout')
    f['e8_b_ratio'], f['e8_b_med'], f['e8_K1'] = p('e8.b.ratio'), p('e8.b.fc1_median'), p('e8.K1.fc1_median')
    f['e8_b_q'], f['e8_b_n'] = p('e8.b.fc1_qualified'), p('e8.b.fc1_n')
    f['e8_b_guard'], f['e8_c_guard'] = p('e8.b.guard_ratio'), p('e8.c.guard_ratio')
    f['e8_c_nodes'], f['e8_c_bar'], f['e8_c_rule'] = p('e8.c.nodes'), p('e8.c.nodes_bar'), p('e8.c.rule')
    f['e8_c_tail'], f['e8_c_med'] = p('e8.c.tail_ratio'), p('e8.c.fc1_median')
    f['e8_phaseb'] = p('e8.phaseb.verdicts')
    f['e8_A'] = {k: p(f'e8.{k}.verdict') for k in ('A1', 'A2', 'A3', 'A4')}
    f['e9_tau'], f['e9_n'], f['e9_flag'], f['e9_fi'] = p('e9.tau_s'), p('e9.n'), p('e9.flagged'), p('e9.FI_alarm')
    f['e9_A'] = {k: p(f'e9.A.{k}') for k in ('auc', 'detection', 'false_alarm', 'n_bad', 'n_good', 'threshold')}
    f['e9_B'] = {k: p(f'e9.B.{k}') for k in ('auc', 'false_alarm', 'n_bad', 'n_good', 'threshold')}
    f['e9_label'] = X['experiments']['E9']
    s = p('e10.s')
    f['e10s_med'], f['e10s_bar'], f['e10s_n'], f['e10s_conv'] = s['fc1_median'], s['bar'], s['n_fc1'], s['fc1_step_converged']
    f['e10s_q'] = p('e10.s.fc1_q')
    q = p('e10.q')['evidence']
    f['e10q_rho'], f['e10q_states'], f['e10q_over'] = q['rho_max_all'], q['accepted_step_states'], q['states_over_bar']
    g = X['gpu_hours']
    f['gpu'] = {k: g[k] for k in LANES}
    f['gpu_total'] = g['total']
    f['gpu_cap'] = p('ledger.cap_total')
    f['e8_gpu'], f['e10_gpu'] = p('e8.gpu'), p('e10.gpu')
    return f


CSS = r'''
:root{--bg:#f7f7f4;--surface:#ffffff;--ink:#1d2129;--mute:#5b6270;--line:#dcdfe4;--accent:#2f5d8a;
--pass:#1f6f43;--pass-bg:#dff1e6;--fail:#9a2a22;--fail-bg:#fbe3e0;--inc:#8a5a00;--inc-bg:#fbefd4;
--skip:#5b6270;--skip-bg:#eceef1;--code:#eef0f3;--frame:#ffffff;}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#14161a;--surface:#1d2026;--ink:#e6e8eb;
--mute:#a3a9b4;--line:#343943;--accent:#8fb6e0;--pass:#8fd6aa;--pass-bg:#1d3a2a;--fail:#f2a59d;--fail-bg:#45211e;
--inc:#f0cc7a;--inc-bg:#3f3218;--skip:#a3a9b4;--skip-bg:#2a2e35;--code:#2a2e35;--frame:#ffffff;}}
:root[data-theme="dark"]{--bg:#14161a;--surface:#1d2026;--ink:#e6e8eb;--mute:#a3a9b4;--line:#343943;--accent:#8fb6e0;
--pass:#8fd6aa;--pass-bg:#1d3a2a;--fail:#f2a59d;--fail-bg:#45211e;--inc:#f0cc7a;--inc-bg:#3f3218;--skip:#a3a9b4;
--skip-bg:#2a2e35;--code:#2a2e35;--frame:#ffffff;}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 "Source Sans 3","Segoe UI",system-ui,-apple-system,sans-serif;
font-variant-numeric:tabular-nums}
h1,h2,h3{font-family:"Source Serif 4",Georgia,"Times New Roman",serif;line-height:1.25;margin:1.4em 0 .5em}
h1{font-size:1.9rem;margin-top:.4em}h2{font-size:1.4rem;border-bottom:1px solid var(--line);padding-bottom:.25em}
h3{font-size:1.12rem}
a{color:var(--accent)}
code{font-family:"JetBrains Mono",ui-monospace,Menlo,Consolas,monospace;font-size:.86em;background:var(--code);
padding:.05em .3em;border-radius:4px;word-break:break-all}
.mute{color:var(--mute)}
.pill{display:inline-block;padding:.05em .55em;border-radius:999px;font-size:.82em;font-weight:600;white-space:nowrap}
.pill.part{background:var(--pass-bg);color:var(--pass)}.pill.fail{background:var(--fail-bg);color:var(--fail)}
.pill.inc{background:var(--inc-bg);color:var(--inc)}.pill.skip{background:var(--skip-bg);color:var(--skip)}
figure{margin:0}
.frame{background:var(--frame);border:1px solid var(--line);border-radius:8px;padding:8px}
.frame img{display:block;max-width:100%;height:auto;margin:0 auto}
figcaption{font-size:.9rem;color:var(--mute);margin-top:.35em}
.tbl{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:.6em 0;border:1px solid var(--line);border-radius:8px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:.92rem}
th,td{text-align:left;padding:.45em .6em;border-bottom:1px solid var(--line);vertical-align:top}
th{background:var(--code);font-weight:600}
tr:last-child td{border-bottom:none}
td.n{text-align:right;white-space:nowrap}
'''

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Source+Sans+3:wght@400;600;700'
         '&family=Source+Serif+4:wght@600;700&display=swap" rel="stylesheet">')
