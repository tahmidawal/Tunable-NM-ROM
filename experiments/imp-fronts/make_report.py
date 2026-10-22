"""Generate report.md, EXPORT.json and plots/ for lane imp-fronts from checks/*.json (no hand-typed numbers).

Phase A (interim): A1 smearing, A2 wall/tail, A3 F7 truncation screen, A4 F15 hybrid floors.
Phase B sections are added when their result JSONs exist (checks/phaseb_*.json); until then the report says so.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CK = HERE / 'checks'
PL = HERE / 'plots'


def load(name):
    p = CK / f'{name}.json'
    return json.loads(p.read_text()) if p.exists() else None


def pct(x, d=3):
    if x is None:
        return 'n/a'
    if isinstance(x, float) and np.isnan(x):
        return 'n/a (NaN)'
    if not np.isfinite(x):
        return '∞'
    v = 100 * x
    return f'{v:.{d}g} %'


def g(x, d=3):
    if x is None:
        return 'n/a'
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, (int, float)):
        if np.isnan(x):
            return 'n/a (NaN)'
        return f'{x:.{d}g}' if np.isfinite(x) else '∞'
    return str(x)


def esc(x):
    return str(x).replace('|', '\\|')


def ok(d, *keys):
    """True iff the nested keys exist and are not None (unavailable sections render as such)."""
    for k in keys:
        if not isinstance(d, dict) or d.get(k) is None:
            return False
        d = d[k]
    return True


def commit():
    try:
        return subprocess.check_output(['git', '-C', str(HERE), 'rev-parse', '--short', 'HEAD'], text=True).strip()
    except Exception:
        return 'unknown'


def plots(a1, a2, a3, a4):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PL.mkdir(exist_ok=True)
    out = []
    if a4:
        regs = [c for c in ('val32', 'N0s', 'Am', 'C1', 'C2') if a4['summary'].get(c, {}).get('n')]
        fig, ax = plt.subplots(figsize=(7, 3.6))
        w = 0.25
        for k, (b, lab) in enumerate((('B512', 'bank 512'), ('S1024', '1024 sines'), ('H', 'bank 512 + 512 sines'))):
            ax.bar(np.arange(len(regs)) + (k - 1) * w, [100 * a4['summary'][c][f'{b}_worst'] for c in regs], w, label=lab)
        ax.set_xticks(range(len(regs)), regs)
        ax.set_yscale('log')
        ax.set_ylabel('worst floor (%)')
        ax.set_title('A4: worst floor per cohort (conditional E0 references)')
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(PL / 'a4_hybrid_floors.png', dpi=150)
        plt.close(fig)
        out.append('plots/a4_hybrid_floors.png')
    if a1:
        rows = list(a1['rows'].values())
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.scatter([r['q'] for r in rows], [r['D_bank384'] for r in rows], s=14, label='bank 384')
        ax.scatter([r['q'] for r in rows], [r['D_pod_upw'] for r in rows], s=10, marker='x', label='POD of upwind (control +)')
        ax.scatter([r['q'] for r in rows], [r['D_pod_ref'] for r in rows], s=10, marker='+', label='POD of E0 (control −)')
        ax.axhline(1, color='k', lw=0.6)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('q = a h / (2 ν)')
        ax.set_ylabel('smearing excess D')
        ax.set_title(f"A1 (X, {len(rows)} cases): {a1['verdict']}", fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(PL / 'a1_smearing.png', dpi=150)
        plt.close(fig)
        out.append('plots/a1_smearing.png')
    if a3 and a3['primary'] in a3.get('configs', {}) and a3['configs'][a3['primary']]['scopes']['in_range']['n']:
        c = a3['configs'][a3['primary']]
        Ks = [k for k in (2304, 4096, 9216, 65025)]
        fig, ax = plt.subplots(figsize=(5.5, 3.4))
        for sc, lab in (('in_range', 'dev6 + val32 (decision)'), ('all_regimes', 'regime cohorts')):
            if not c['scopes'][sc]['n']:
                continue
            ax.plot(Ks, [100 * c['scopes'][sc][f'dworst_{K}'] for K in Ks], 'o-', label=lab)
        ax.axhline(0.1, color='r', lw=0.8, ls='--', label='stop bar +0.1 pp')
        ax.set_xscale('log')
        ax.set_xlabel('K (sines kept)')
        ax.set_ylabel('change of worst error (pp)')
        ax.set_title(f'A3: truncation, {a3["primary"]} (conditional E0 references)', fontsize=9)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(PL / 'a3_truncation.png', dpi=150)
        plt.close(fig)
        out.append('plots/a3_truncation.png')
    sb, sk = load('phaseb/score_b'), load('phaseb/score_K1')
    if sb and sk:
        cohs = list(sk['summary'])
        fig, ax = plt.subplots(figsize=(6, 3.4))
        sets = [('K1 (control retrain)', sk), ('arm (b) coverage', sb)] + ([('arm (c) boundary-compatible', load('phaseb/score_c'))]
                                                                           if load('phaseb/score_c') else [])
        wd = .8 / len(sets)
        for k, (nm, sc) in enumerate(sets):
            ax.bar(np.arange(len(cohs)) + (k - (len(sets) - 1) / 2) * wd, [100 * sc['summary'][c]['median_e_diagnostic'] for c in cohs], wd, label=nm)
        ax.set_xticks(range(len(cohs)), cohs)
        ax.set_yscale('log')
        ax.set_ylabel('median ROM error (%)')
        ax.set_title('Phase B: fresh test cohorts, diagnostic medians (conditional refs)', fontsize=9)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(PL / 'phaseb_medians.png', dpi=150)
        plt.close(fig)
        out.append('plots/phaseb_medians.png')
    return out


def floors_summary(name):
    f = load(f'phaseb/floors_{name}')
    if not f:
        return None
    out = {}
    for coh, d in f['cohorts'].items():
        v = [x['f384'] for x in d['per_case'].values() if x.get('eligible') and x.get('f384') is not None]
        out[coh] = dict(median=float(np.median(v)) if v else None, worst=float(max(v)) if v else None, n=len(v))
    return out


def phase_b(L, pb, plots_done=()):
    led = json.loads((HERE / 'runs/GPU-LEDGER.json').read_text())
    L.append('')
    L.append('## Phase B: single-change retrains (coverage arm (b), boundary-compatible arm (c)) against the control retrain K1')
    L.append('')
    L.append(f"Schedule {pb['schedule']} (coordinator releases: S3, then arm (c) after (b) stopped; arm (a) deferred). Fresh test cohorts: {esc(json.dumps(pb['ledger_counts']))} reference outcomes. "
             f"GPU-h spent (ledger): {g(led['cumulative_gpu_hours'], 4)}. {pb['note']}.")
    L.append('')
    L.append('| job | item | GPU | GPU-h charged | state |')
    L.append('|---|---|---|---|---|')
    for j in led['jobs']:
        L.append(f"| {j['name']} ({j['job_id']}) | {j['item']} | {j['gpu']} × {j.get('ngpu', 1)} | {g(j.get('charged_gpu_h'), 3)} | {j['state']} |")
    L.append('')
    banks = [('K1', load('phaseb/score_K1'))] + [(nm, load(f'phaseb/score_{nm}')) for nm in ('b', 'c', 'e10s')
                                                  if load(f'phaseb/score_{nm}')]
    L.append('Error table: **diagnostic** errors of each case\'s terminal policy run, unqualified runs included (conditional E0-style references).')
    L.append('')
    L.append('| bank | cohort | n | diagnostic median e | diagnostic worst e | numerically qualified | terminal states |')
    L.append('|---|---|---|---|---|---|---|')
    for nm, sc in banks:
        for coh, d in sc['summary'].items():
            L.append(f"| {nm} | {coh} | {d['n_eligible']} | {pct(d['median_e_diagnostic'])} | {pct(d['worst_e'])} | {d['qualified']}/{d['n_eligible']} | "
                     f"{esc(', '.join(f'{k} {v}' for k, v in d['terminals'].items()))} |")
    L.append('')
    L.append('Representation floors on the fresh test cohorts (conditional E0-style references):')
    L.append('')
    L.append('| bank | FI median / worst floor $f_{384}$ | FN0s median floor | FC1 median floor | FC2 median floor |')
    L.append('|---|---|---|---|---|')
    for nm in ('K0', 'K1', 'K2', 'K3', 'b', 'c'):
        fs = floors_summary(nm)
        if fs:
            L.append(f"| {nm} | {pct(fs['FI']['median'])} / {pct(fs['FI']['worst'])} | {pct(fs['FN0s']['median'])} | "
                     f"{pct(fs['FC1']['median'])} | {pct(fs['FC2']['median'])} |")
    L.append('')
    L.append('Verdict (the guard compares worst FI floors; the ROM route compares diagnostic FC1 medians and requires qualification; '
             '"wins/losses" count cases where the arm\'s error is lower/higher than the control\'s by more than '
             '$2\\varepsilon_{\\rm ref}$; errors and floors inside the route dictionaries are fractions, not percent; K2/K3 have no ROM runs, so their ROM comparisons are unavailable, not failures):')
    L.append('')
    L.append('| arm | outcome (pre-registered precedence) | guard | routes | passes against all of K1–K3 |')
    L.append('|---|---|---|---|---|')
    for arm, d in pb['verdict']['arms'].items():
        L.append(f"| ({arm}) | {d['outcome']} | {d['guard']} | {esc(json.dumps(d['routes']))} | {d['passes_against_all_K1_K3']} |")
    L.append('')
    L.append('Stop rules (verbatim from the brief): (a) data fidelity — stop if the N0s floor is not ≥ 3× below the control\'s; '
             '(b) steep / low-ν coverage — stop if the C1 ROM median is not ≥ 2× lower; (c) boundary-compatible bank — stop if '
             '$m^*$ is not ≥ 2× smaller **and** the tail not ≥ 3× smaller; every arm also stops if the in-range worst floor is '
             'more than 2× the control\'s. Arm (b) can pass its ROM route only if it is numerically qualified (Q1, Q2 and step '
             'convergence) on every eligible FC1 test case (Amendments 2–5).')
    L.append('')
    e10 = pb.get('e10') or {}
    if e10:
        L.append('### E10 (numerics-only single changes on the K1 bank; re-used Phase B fresh test cases)')
        L.append('')
        L.append('| arm | outcome | FC1 diagnostic median (conditional) | K1 FC1 diagnostic median | FC1 terminal states | reason |')
        L.append('|---|---|---|---|---|---|')
        for arm, d in e10.items():
            terms = d.get('terminals') or {}
            cnt = {}
            for t in terms.values():
                cnt[t] = cnt.get(t, 0) + 1
            L.append(f"| {arm} | {d['outcome']} | {pct(d.get('fc1_median'))} | {pct(d.get('k1_fc1_diagnostic_median'))} | "
                     f"{esc(', '.join(f'{k} {v}' for k, v in cnt.items())) or 'n/a'} | {esc(d.get('why', ''))} |")
        L.append('')
        L.append('E10-s = factor set extended to an eighth of the deployed step (E10-DESIGN §§8–9); E10-q = state-triggered '
                 'quadrature escalation, resolved without a job because its trigger never fires on any accepted state of '
                 'K1\'s runs (§10, `checks/phaseb/e10q_every_step.json`).')
        L.append('')
    e9 = load('phaseb/e9')
    if e9:
        L.append(f"**E9 (descriptive, K1):** threshold $s>{g(e9['tau_s'], 5)}$ (frozen on dev) flags {pct(e9['flagged_fraction'])} of the "
                 f"{e9['n']} fresh test cases, all by the known input $s$.")
        for t, d in e9['targets'].items():
            L.append(f"- target {t} (conditional error > {pct(d['threshold'])}): {d['n_bad']} bad / {d['n_good']} good, AUC {g(d['auc'])}, detection "
                     f"{g(d['detection'])} (one-sided 95 % lower bound {g(d['detection_cp95_lower'])}), false alarm {g(d['false_alarm'])} "
                     f"(upper bound {g(d['false_alarm_cp95_upper'])}); {d['label']}.")
        L.append('- FOM-fallback cost: unmeasured (paired timing deferred under the cap).')
        L.append('')
    sb, sk = load('phaseb/score_b'), load('phaseb/score_K1')
    if sb and sk:
        a, c = sb['summary']['FC1']['median_e_diagnostic'], sk['summary']['FC1']['median_e_diagnostic']
        L.append(f"Diagnostic FC1 medians (qualification ignored): arm (b) {pct(a)}, K1 {pct(c)}, ratio {g(c / a)} (the bar needs ≥ 2). "
                 f"Rule selection: arm (b) {load('phaseb/selection_b')['rule']} ({load('phaseb/selection_b')['nodes']} nodes), "
                 f"K1 {pb['k1']['selection']['rule']} ({pb['k1']['selection']['nodes']} nodes).")
        L.append('')
    if pb['verdict']['arms']['b']['outcome'].startswith('STOP'):
        L.append('**E8 fell short (arm (b) stopped), so E10 applied:** the coordinator released arm (c) under E8\'s cap and the '
                 'two numerics-only E10 arms under a separate 10 GPU-h cap (`E10-DESIGN.md`).')
        L.append('')
    cap = 15.0
    e8 = sum(j.get('charged_gpu_h') or 0 for j in led['jobs'] if j.get('bucket', 'E8') == 'E8')
    e10b = sum(j.get('charged_gpu_h') or 0 for j in led['jobs'] if j.get('bucket') == 'E10')
    L.append(f"GPU-h by cap bucket: E8 {g(e8, 4)} of 15; E10 {g(e10b, 3)} of 10.")
    L.append('')
    if 'plots/phaseb_medians.png' in plots_done:
        L.append('![Phase B](plots/phaseb_medians.png)')
        L.append('')
    if load('phaseb/score_c'):
        sc_ = load('phaseb/score_c')
        L.append(f"**Arm (c)** (boundary-compatible bank, released by the coordinator after (b) stopped; DESIGN Amendments 19–22): "
                 f"FC1 diagnostic median {pct(sc_['summary']['FC1']['median_e_diagnostic'])} against K1's "
                 f"{pct(sk['summary']['FC1']['median_e_diagnostic'])}; rule {load('phaseb/selection_c')['rule']} "
                 f"({load('phaseb/selection_c')['nodes']} nodes, bar ≤ 12 800).")
        L.append('')
    devs = [j for j in led['jobs'] if j.get('note')]
    for j in devs:
        L.append(f"- Disclosed deviation, {j['name']} ({j['job_id']}): {j['note']}")
    if devs:
        L.append('')
    aw = load('phaseb/armc_wall')
    if aw:
        L.append(f"- Arm (c) identity: the registered wall-value bar ({g(aw['registered_bar'])} relative) was tested on the initial bank; "
                 f"on the trained arm-(c) columns the raw wall residual is up to {g(aw['max_rel_wall_rms'])} (median "
                 f"{g(aw['median_rel_wall_rms'])}) of the column RMS ({aw['note']}), found by the results audit; scoring zeroes "
                 'the wall nodes as for every bank.')
        L.append('')
    L.append('**Deferred:** arm (a) fidelity (the coordinator released S3, then arm (c), not (a)).')
    L.append('')


def main():
    a1, a2, a3, a4 = (load(n) for n in ('a1_smear', 'a2_wall', 'a3_trunc', 'a4_hybrid'))
    figs = plots(a1, a2, a3, a4)
    L = []
    L.append('# imp-fronts: diagnostics of the off-mesh bank ROM failure on steep fronts — Phase A diagnostics, Phase B retrains (coverage, boundary-compatible) and E10')
    L.append('')
    L.append('**Status: final for Phase A (CPU diagnostics), Phase B (coverage arm (b) and boundary-compatible arm (c), each '
             'against the control retrain K1; arm (a) deferred) and E10 (E10-s run; E10-q resolved from data).** Every accuracy number is measured against E0 ST '
             '(dev6, val32) or E0-style ST (jcp-regimes cohorts) references whose uncertainty $\\varepsilon_{\\rm ref}$ is a **conditional '
             'Richardson estimate**; all are therefore conditional. '
             f'Generated by `make_report.py` from `checks/*.json` at lane commit `{commit()}`.')
    L.append('')
    L.append('```mermaid')
    L.append('flowchart LR')
    L.append('  A1["A1 smearing test<br/>(is the bank biased to upwind data?)"]:::diag --> B["Phase B arm (a)<br/>fidelity retrain (deferred)"]:::arm')
    L.append('  A2["A2 wall curvature / tail<br/>(boundary incompatibility?)"]:::diag --> C["Phase B arm (c)<br/>boundary-compatible bank<br/>(eligible by A2; run, STOP)"]:::arm')
    L.append('  A3["A3 F7 truncation screen"]:::diag --> F7["F7 fast decode<br/>(GPU timing deferred)"]:::other')
    L.append('  A4["A4 F15 hybrid floors"]:::diag --> F15["F15 sine-enriched bank<br/>(STOP)"]:::other')
    L.append('  classDef diag fill:#dbe9f7,stroke:#2b6cb0')
    L.append('  classDef arm fill:#fde2c8,stroke:#c46a1b')
    L.append('  classDef other fill:#e2f3e1,stroke:#2f855a')
    L.append('```')
    L.append('')
    L.append('## Outcomes at a glance')
    L.append('')
    L.append('| diagnostic | pre-registered outcome | controls |')
    L.append('|---|---|---|')
    if a1:
        L.append(f"| A1 smearing (bank 384, T6) | {a1['verdict']} | {'pass' if ok(a1, 'controls') and a1['controls']['controls_pass'] else 'FAILED / unavailable'} |")
    if a2:
        L.append(f"| A2 arm-(c) decision | {a2['decision_arm_c']} | {'pass' if a2['controls']['controls_pass'] else 'FAILED'} |")
    if a3:
        L.append(f"| A3 F7 truncation screen | {a3['verdict']} | {'pass' if a3['controls']['controls_pass'] else 'FAILED'} |")
    if a4:
        L.append(f"| A4 F15 hybrid floors | {a4['verdict']} | {'pass' if a4['controls']['controls_pass'] else 'FAILED'} |")
    L.append('')
    # ---------------- A1
    if a1 and not ok(a1, 'summary'):
        L.append(f"## A1. Smearing test\n\n**{a1.get('verdict', 'UNAVAILABLE')}.**\n")
    if a1 and ok(a1, 'summary'):
        s, c = a1['summary'], a1['controls']
        L.append('## A1. Smearing test with a sine control')
        L.append('')
        L.append('The incumbent upwind + backward-Euler generator (byte-identical copy) was rerun on every eligible dev6 + val32 '
                 'case and every N0s case (set X) at 257 nodes, and on the six target cases T6 at its own 256-node grid. For a '
                 'basis $B$, $\\rho_B=f_B(u_{\\rm ref})/f_B(u_{\\rm upw})$ and the smearing excess $D_B=\\rho_B/\\rho_S$ with $S$ '
                 'the same number of Laplacian sines; $D>1$ means $B$ prefers the smeared field more than a data-agnostic basis does.')
        L.append('')
        L.append('| quantity | value |')
        L.append('|---|---|')
        L.append(f"| T6 | {', '.join(esc(t) for t in a1['T6'])} |")
        L.append(f"| median $D$, bank 384, T6 (cases with $D>1$) | {g(s['D_bank384_med_T6'])} ({s['D_bank384_gt1_T6']}/6) |")
        L.append(f"| median $D$, bank 128, T6 | {g(s['D_bank128_med_T6'])} |")
        L.append(f"| median $D$, bank 384, X ({len(a1['X'])} cases) | {g(s['D_bank384_med_X'])} |")
        L.append(f"| Spearman($D_{{384}}$, $q$) on X | {g(s['spearman_D384_q_X'])} |")
        L.append(f"| median $D$, bank 384, T6 at the 256-node training grid (sensitivity) | {g(s['D_bank384_med_T6_n256'])} |")
        L.append(f"| control +: median $D$ of LOO POD-128 of upwind fields, T6 (needs ≥ 2) | {g(c['Dpos_med_T6'])} |")
        L.append(f"| control −: median $D$ of LOO POD-128 of E0 fields, T6 (needs ≤ 1.25) | {g(c['Dneg_med_T6'])} |")
        L.append(f"| control resolution: median $D_+/D_-$, T6 (needs ≥ 2) | {g(c['resolution_med_T6'])} |")
        fx = c['exact_D_fixture']
        L.append(f"| exact-$D$ fixture (must hold) | {'pass' if fx['ok'] else 'FAIL'}" + ('' if fx['ok'] else
                 f" (registered prerequisites met: {fx.get('prerequisites')}; largest relative error of the algebraic check "
                 f"{g(max(fx['rel_err'].values()))})") + ' |')
        L.append(f"| upwind generator vs E0, median / worst error over X (conditional) | {pct(s['e_upw_vs_ref_med_X'])} / {pct(s['e_upw_vs_ref_max_X'])} |")
        L.append(f"| gates G1 (IC identity) / G2 (Newton residual, finite) | {a1['gates']['G1_ic_pass']} / {a1['gates']['G2_newton_pass']} |")
        L.append('')
        why = []
        if not c.get('C_A1plus_fires'):
            why.append('the positive POD control did not fire (median $D_+$ below 2)')
        if not c.get('resolution_ok'):
            why.append('the controls did not resolve each other (median $D_+/D_-$ below 2)')
        if not c.get('C_A1minus_quiet'):
            why.append('the negative POD control fired')
        if not c['exact_D_fixture'].get('prerequisites', True):
            why.append('the exact-$D$ fixture prerequisite $c\\ge10^{-3}$ was not met (the bank nearly contains the chosen sine '
                       'at r = 384), which fails the fixture by the registered rule although its algebra matched')
        if why:
            L.append('Failed controls: ' + '; '.join(why) + '.')
            L.append('')
        L.append(f"**Outcome: {a1['verdict']}.** The two POD comparators are empirical controls (Amendment 1): they test whether "
                 'the statistic can tell a basis fitted to smeared data from one fitted to sharp data at this effect size. '
                 'If they fail, the bank number above cannot be read either way; the Phase B fidelity arm is the actual test.')
        L.append('')
        if 'plots/a1_smearing.png' in figs:
            L.append('![A1](plots/a1_smearing.png)')
            L.append('')
    # ---------------- A2
    if a2:
        gi = a2['groups']['in_range']
        L.append('## A2. Wall curvature and sine tail of decoded fields')
        L.append('')
        L.append('The true solution satisfies $\\partial_n^2u=0$ on the walls ($u=0$ there gives $u_t=0$ and $u(u_x+u_y)=0$, '
                 'hence $\\Delta u=0$). The bank multiplies an MLP by the envelope $16x(1-x)y(1-y)$, whose normal second '
                 'derivative is not zero. Decoded fields are the saved r = 384 LSPG-CN-R runs of jcp-regimes.')
        L.append('')
        L.append('| quantity (in range: dev6 + val32; conditional E0 references) | decoded ROM | bank projection of E0 | E0 reference |')
        L.append('|---|---|---|---|')
        km = gi['kappa_a_med']
        if isinstance(km, dict):
            L.append(f"| median wall curvature $\\kappa_a$ (relative to interior Laplacian) | {g(km['rom'])} | {g(km['proj'])} | 0 (continuum compatibility identity, not measured) |")
        else:
            L.append(f'| median wall curvature $\\kappa_a$ | {km} | | |')
        L.append(f"| median sine tail $\\tau_{{4096}}$ | {g(gi['tau4096_med']['rom'])} | {g(gi['tau4096_med']['proj'])} | {g(gi['tau4096_med']['ref'])} |")
        L.append(f"| median tail decay slope $\\beta$ | {g(gi['beta_med']['rom'])} | {g(gi['beta_med']['proj'])} | {g(gi['beta_med']['ref'])} |")
        L.append(f"| median wall share $\\omega$ of the tail | {g(gi['omega_med']['rom'])} | {g(gi['omega_med']['proj'])} | {g(gi['omega_med']['ref'])} |")
        L.append('')
        L.append(f"Median $\\tau_{{4096}}$ ratio decoded/reference: {g(gi['tau4096_ratio_rom_ref_med'])} (RUN needs ≥ 3); "
                 f"median $\\omega$ of decoded fields: {g(gi['omega_med']['rom'])} (RUN needs ≥ 0.5). Controls: envelope bump "
                 f"$\\omega$ = {g(gi['ctrl_omega_plus_med'])} (needs ≥ 0.5), interior bump $\\omega$ = {g(gi['ctrl_omega_minus_med'])} "
                 f"(needs ≤ 0.25). Eligible in-range cases {gi['n_eligible']}, valid ROM runs {gi['n_rom_valid']}.")
        L.append('')
        L.append(f"**Outcome for arm (c): {a2['decision_arm_c']}** (a budget-prioritisation heuristic, Amendment 1).")
        L.append('')
        L.append('| cohort (conditional E0-style references) | median $\\tau_{4096}$ decoded / reference | median $\\omega$ decoded |')
        L.append('|---|---|---|')
        for coh in ('in_range', 'N0s', 'C1', 'C2'):
            gg = a2['groups'][coh]
            L.append(f"| {coh} | {g(gg['tau4096_ratio_rom_ref_med'])} | {g(gg['omega_med']['rom'])} |")
        L.append('')
    # ---------------- A3
    if a3 and a3['primary'] not in a3.get('configs', {}):
        L.append(f"## A3. F7 truncation screen\n\n**{a3['verdict']}** (primary configuration missing).\n")
    if a3 and a3['primary'] in a3.get('configs', {}):
        c = a3['configs'][a3['primary']]
        L.append('## A3. F7 truncation screen (CPU part)')
        L.append('')
        L.append('Decoded fields of every saved jcp-regimes ROM output were truncated to their first $K$ Laplacian sines and '
                 'rescored. Stop rule (ideation F7, verbatim): stop if the worst error rises by more than 0.1 percentage points '
                 'at K = 9216; decision population `R384|LSPG_CNR_1` on dev6 + val32 (Amendment 1).')
        L.append('')
        L.append('| configuration | worst error dev6+val32 (conditional) | change at K = 2304 | 4096 | 9216 | worst error, regimes (conditional) | change at 9216, regimes |')
        L.append('|---|---|---|---|---|---|---|')
        for k, cc in a3['configs'].items():
            ir, rg = cc['scopes']['in_range'], cc['scopes']['all_regimes']
            pp = lambda v: 'n/a' if v is None else f'{100 * v:+.3g} pp'
            L.append(f"| {esc(k)} | {pct(ir.get('worst'))} | {pp(ir['dworst_2304'])} | {pp(ir['dworst_4096'])} | {pp(ir['dworst_9216'])} | "
                     f"{pct(rg.get('worst'))} | {pp(rg['dworst_9216'])} |")
        L.append('')
        L.append(f"**Outcome: {a3['verdict']}** (regime scope, descriptive: {a3.get('verdict_regime_scope', 'n/a')}). Controls: full-sine "
                 f"identity {a3['controls']['identity_all']}, constructed fixture trips the rule {a3['controls']['fixture_trips']}, "
                 f"reproduction of jcp-regimes errors {a3['controls']['repro_all']}. A negative change means truncation removes "
                 'high-frequency ROM error; it is not an accuracy gain of the method. The GPU timing gate of F7 is deferred '
                 '(not in this lane\'s brief).')
        L.append('')
        if 'plots/a3_truncation.png' in figs:
            L.append('![A3](plots/a3_truncation.png)')
            L.append('')
    # ---------------- A4
    if a4:
        L.append('## A4. F15 hybrid (sine-enriched) floors')
        L.append('')
        L.append('Floors $f_B(u_{\\rm ref})$ of every eligible reference for the hybrid $H$ = bank 512 + 512 Laplacian sines '
                 '(1024 unknowns), the equal-size control of 1024 sines, and bank 512. Stop rule (ideation F15, verbatim): stop if '
                 "the hybrid's C1 worst floor is not ≥ 2× below the sine control and below the bank-512 floor.")
        L.append('')
        L.append('| cohort | n | bank 512 median / worst (conditional) | 1024 sines median / worst (conditional) | hybrid median / worst (conditional) |')
        L.append('|---|---|---|---|---|')
        for coh, s in a4['summary'].items():
            if not s.get('n'):
                continue
            L.append(f"| {coh} | {s['n']} | {pct(s['B512_median'])} / {pct(s['B512_worst'])} | {pct(s['S1024_median'])} / "
                     f"{pct(s['S1024_worst'])} | {pct(s['H_median'])} / {pct(s['H_worst'])} |")
        L.append('')
        w = a4['C1_worst'] or {}
        if not w:
            L.append(f"C1 unavailable. **Outcome: {a4['verdict']}.**")
        else:
          L.append(f"C1 worst floors: hybrid {pct(w['H'])}, 1024 sines {pct(w['S1024'])}, bank 512 {pct(w['B512'])}; the rule needs the "
                 f"hybrid at ≤ {pct(0.5 * w['S1024'])}. **Outcome: {a4['verdict']}.** Controls: containment violation "
                 f"{g(a4['controls']['containment_max_violation'])} (≤ 1e-12), duplicate-column null enrichment stopped: "
                 f"{a4['controls']['null_must_be_stopped']}." + (' The 1024-sine control alone has a lower C1 worst floor than '
                 'bank 512.' if w['S1024'] < w['B512'] else ''))
        L.append('')
        if 'plots/a4_hybrid_floors.png' in figs:
            L.append('![A4](plots/a4_hybrid_floors.png)')
            L.append('')
    L.append('## What was wrong or changed during Phase A (disclosed)')
    L.append('')
    if a2 and ok(a2, 'retracted_control_replay'):
        rr = a2['retracted_control_replay']
        L.append(f"- The curvature condition of the retracted original A2 control (finite-difference stencil, must fire on every "
                 f"in-range case) fires on {rr['n_fired']} of {rr['n_in_range']} in-range cases when replayed; the control was "
                 'dropped before any A2 decision (Amendment 1), as design audit 1 had found.')
    if a2 and ok(a2, 'controls', 'identity'):
        idn = a2['controls']['identity']
        if 'original_dv_1em3_gap' in idn:
            L.append(f"- The registered bank-values check at spacing $10^{{-3}}$ gave a gap of {g(idn['original_dv_1em3_gap'])} against "
                     f"a bound of {g(idn['bank_path_bound'])} (value-path truncation); it was respecified (Amendments 6, 7): "
                     f"Richardson gap {g(idn['bank_path_gap'])}, pass {idn['bank_path_ok']}.")
    L.append('- Before Amendment 1, draft A3 and A4 code was run once as debugging; those outputs (including the A4 C1 worst '
             'floors) were seen, discarded and regenerated; Amendment 1 only added controls to A4.')
    L.append('- Demoted to descriptive status by amendment (not gating): the A1 own-field self-fit checks, the A3 $K=16$ stress '
             'test and the A4 high-frequency enrichment comparator.')
    if a3:
        gal = (a3.get('configs', {}).get('R384|GAL_CN_1') or {}).get('invalid', [])
        if gal:
            L.append(f"- A3 counts {len(gal)} `R384\\|GAL_CN_1` runs ({', '.join(esc(x) for x in gal)}) as invalid because jcp-regimes "
                     'marks them invalid (budget exits) although the saved archive flag is true (Amendment 13).')
    L.append('- Phase-A code audits found and the lane fixed: tail round-off from cumulative subtraction, NaN-tolerant A1 gates, '
             'unverified A1 inputs, A3 secondary-scope and reproduction handling, and report robustness.')
    if a1 and ok(a1, 'inputs') and a1['inputs']['ok'] and a1['inputs'].get('settings_unpersisted') == 0:
        L.append(f"- The A1 upwind data were regenerated with persisted generator settings and descriptors; all "
                 f"{a1['inputs']['n_files']} files verified (manifest, generator hash, settings, descriptors, IC).")
    for nm, d in (('A1', a1), ('A2', a2), ('A3', a3), ('A4', a4)):
        if d and d.get('run_commit'):
            rc = d['run_commit']
            L.append(f"- {nm} output generated at lane commit `{str(rc['commit'])[:9]}` (uncommitted code changes: {rc['code_dirty']}).")
    L.append('')
    L.append('## Phase B status')
    L.append('')
    pb = load('phaseb_verdicts')
    L.append('No Phase B result exists yet.' if pb is None else 'See the Phase B section below.')
    if pb is not None:
        phase_b(L, pb, figs)
    import sys
    sys.path.insert(0, str(HERE / 'rom'))
    import schedule as SCH
    L.append('')
    L.append(f"Budget context (rom/schedule.py ceilings; E8 cap {SCH.CAP} GPU-h): both retrain arms (S1) would need "
             f"{g(SCH.total('S1'), 4)} GPU-h of ceilings, over the cap; the coordinator released S3 (arm (b)), then arm (c) "
             '(schedule S3C, certified on measured spending). Arm (a) stayed deferred by release decision.')
    L.append('')
    L.append('## Glossary')
    L.append('')
    for t, d in (
        ('bank', 'the learned set of 512 ordered coordinate-network functions whose first r span the reduced model'),
        ('floor $f_B$', 'relative error of the best lattice fit of a reference field by basis $B$: '
                        '$\\max_{k=1..5}\\lVert u(t_k)-P_Bu(t_k)\\rVert/\\lVert u(t_0)\\rVert$'),
        ('E0 / E0-style ST reference', 'second-order space-time references on fine meshes; $\\varepsilon_{\\rm ref}$ is their '
                                        'conditional Richardson uncertainty estimate'),
        ('dev6, val32', 'the development and validation cases of the training box ($\\nu\\in[0.01,0.1]$, $a\\in[0.5,2]$)'),
        ('N0s', 'the stiff corner inside the training box (lowest $\\nu$, largest $a$)'),
        ('C1, C2', 'cohorts with fronts steeper than training: $s=a/\\nu$ in $[250,500]$ and $[500,1000]$'),
        ('$s$, $q$', '$s=a/\\nu$ nominal steepness; $q=ah/(2\\nu)$ numerical-viscosity ratio of first-order upwinding'),
        ('N0, Ap, Am, B1, B2', 'other jcp-regimes cohorts: N0 in range; Ap ν above range; Am ν below range; B1 two same-sign bumps; '
                               'B2 a signed pair of bumps'),
        ('$\\beta$', 'least-squares slope of $\\log\\tau_K$ against $\\log K$ over $K\\in[2304,16384]$ (more negative = faster decay)'),
        ('Spearman', 'rank correlation coefficient between two per-case quantities'),
        ('containment violation', 'largest amount by which the hybrid floor exceeds the smaller of its two constituent floors; '
                                  'must be zero up to round-off because the hybrid span contains both'),
        ('upwind generator', 'the incumbent training-data solver: first-order sign-upwind advection, backward Euler, '
                             '$\\Delta t=0.005$, 256 nodes per axis'),
        ('smearing excess D', 'how much more a basis prefers smeared (upwind) over sharp (reference) fields than the same number of '
                              'sines does'),
        ('LOO POD', 'leave-one-case-out proper orthogonal decomposition basis built from the other cases'),
        ('T6, X', 'the six target cases of A1 and its extended case set'),
        ('Laplacian sines $S_K$', 'the first $K$ products $\\sin(\\pi k_1x)\\sin(\\pi k_2y)$ ordered by $k_1^2+k_2^2$'),
        ('sine tail $\\tau_K$', 'relative norm of the part of a field beyond its first $K$ Laplacian sines'),
        ('wall share $\\omega$', 'fraction of the beyond-4096 tail energy within 8 lattice nodes of the walls, pooled over '
                                   'the five output times'),
        ('wall curvature $\\kappa_a$', 'RMS normal second derivative of a decoded field on the walls (gradient-difference '
                                         'estimate), divided by the RMS interior 5-point Laplacian of the reference field'),
        ('LSPG-CN-R', 'least-squares Petrov–Galerkin ROM with Crank–Nicolson time stepping and Rannacher start'),
        ('arm (a), (b), (c)', 'Phase B single-change retrains: data fidelity, steep/low-ν coverage, boundary-compatible output'),
        ('INCONCLUSIVE', 'a pre-registered control did not behave as required, so the diagnostic is not read either way'),
        ('pp', 'percentage points'),
        ('$n$', 'number of eligible cases in a row'),
        ('$r$, $K$', '$r$ = number of bank functions in the prefix; $K$ = number of Laplacian sines kept'),
        ('ROM; GAL / LSPG; CN / BE', 'reduced-order model; Galerkin / least-squares Petrov–Galerkin projection; Crank–Nicolson / '
                                     'backward-Euler time stepping'),
        ('K1', 'the jcp-robust seed-1 retrain of the incumbent bank recipe, the Phase B control'),
        ('S1, S3', 'Phase B schedules: S1 both retrain arms; S3 the coverage arm only, against K1'),
        ('K0, K2, K3', 'K0 the archived incumbent bank (unmatched baseline); K2, K3 the jcp-robust seed-2 and seed-3 retrains'),
        ('FI, FN0s, FC1, FC2', 'fresh cohorts generated for Phase B: in range, stiff corner, $s\\in[250,500]$, $s\\in[500,1000]$; '
                               'their test cases were scored once in Phase B; E10 re-used them (disclosed)'),
        ('numerically qualified', 'a run passing Q1 (valid, fresh-Jacobian stationarity, distance to a tight solve), Q2 (agreement '
                                  'with the successor quadrature rule) and the step check (agreement with the half step)'),
        ('STEP-UNCONVERGED', 'policy terminal state: the step check failed down to the finest factor of the policy (a quarter '
                             'of the deployed step in Phase B, an eighth in E10-s), where no finer check exists; the finest run is '
                             'therefore uncertified and the case is not qualified'),
        ('E10-s, E10-q', 'E10 numerics-only single changes on the K1 bank: E10-s extends the step-refinement ladder to an eighth '
                         'of the deployed step; E10-q escalates the quadrature rule when the successor-rule $\\rho$ exceeds 0.116 at '
                         'an accepted state'),
        ('$m^*$', 'node count of the quadrature rule certified by rule selection at r = 384 (arm (c) needs ≤ 12 800, half of K1\'s)'),
        ('tail route', 'arm (c) route: median 4096-sine tail of decoded FI test fields ≥ 3× below K1\'s, with qualification on every '
                       'eligible FI test case'),
        ('S3C', 'schedule with arms (b) and (c) against K1 under the E8 cap'),
        ('diagnostic median', 'median error over cases counting every terminal run, qualified or not'),
        ('E9 targets, AUC, detection, false alarm', 'target A: error above twice the dev in-range median of the deployed configuration '
                                                     '(K0, `R384|LSPG_CNR_1`, over eligible valid dev6 + val32 + N0 cases, frozen before scoring); '
                                                     'target B: error above 1 %; AUC: area under the ROC curve of the continuous '
                                                     'steepness $s$; detection / false alarm: flagged fraction of bad / good cases'),
    ):
        L.append(f'- **{t}**: {d}.')
    (HERE / 'report.md').write_text('\n'.join(L) + '\n')
    exp = dict(lane='imp-fronts', phase='A + B (S3C: arms b, c) + E10', commit=commit(), reference_label='conditional E0 / E0-style ST',
               A1=None if not a1 else dict(verdict=a1.get('verdict'), controls_pass=(a1.get('controls') or {}).get('controls_pass'),
                                            summary=a1.get('summary'), source='checks/a1_smear.json'),
               A2=None if not a2 else dict(decision_arm_c=a2['decision_arm_c'], controls_pass=a2['controls']['controls_pass'],
                                            in_range=a2['groups']['in_range'], source='checks/a2_wall.json'),
               A3=None if not a3 else dict(verdict=a3.get('verdict'), verdict_regime_scope=a3.get('verdict_regime_scope'),
                                            primary=a3.get('primary'),
                                            primary_scopes=(a3.get('configs', {}).get(a3.get('primary')) or {}).get('scopes'),
                                            source='checks/a3_trunc.json'),
               A4=None if not a4 else dict(verdict=a4.get('verdict'), C1_worst=a4.get('C1_worst'), source='checks/a4_hybrid.json'),
               gpu_hours_phase_a=0.0,
               phase_b=None if not load('phaseb_verdicts') else dict(
                   verdicts={k: v['outcome'] for k, v in load('phaseb_verdicts')['verdict']['arms'].items()},
                   routes={k: v['routes'] for k, v in load('phaseb_verdicts')['verdict']['arms'].items()},
                   guards={k: v['guard'] for k, v in load('phaseb_verdicts')['verdict']['arms'].items()},
                   score_K1=(load('phaseb/score_K1') or {}).get('summary'), score_b=(load('phaseb/score_b') or {}).get('summary'),
                   score_c=(load('phaseb/score_c') or {}).get('summary'),
                   selection_b=load('phaseb/selection_b'), selection_c=load('phaseb/selection_c'),
                   e9=load('phaseb/e9'), armc_wall=load('phaseb/armc_wall'),
                   source='checks/phaseb_verdicts.json, checks/phaseb/*.json'),
               e10_results=dict(verdicts=(load('phaseb_verdicts') or {}).get('e10'),
                                score_e10s=(load('phaseb/score_e10s') or {}).get('summary'),
                                e10q_every_step='checks/phaseb/e10q_every_step.json'),
               gpu_hours_total=json.loads((HERE / 'runs/GPU-LEDGER.json').read_text())['cumulative_gpu_hours'],
               e10='E10-DESIGN.md (E10-s run, E10-q resolved from data)',
               plots=figs)
    (HERE / 'EXPORT.json').write_text(json.dumps(exp, indent=1, default=float))
    print('report.md written;', figs)


if __name__ == '__main__':
    main()
