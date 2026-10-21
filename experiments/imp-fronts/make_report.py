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
        ax.set_title('A4: worst representation floor per cohort')
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
        ax.set_title('A1: smearing excess per case')
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
        ax.set_title(f'A3: truncation of decoded ROM fields ({a3["primary"]})')
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(PL / 'a3_truncation.png', dpi=150)
        plt.close(fig)
        out.append('plots/a3_truncation.png')
    return out


def main():
    a1, a2, a3, a4 = (load(n) for n in ('a1_smear', 'a2_wall', 'a3_trunc', 'a4_hybrid'))
    figs = plots(a1, a2, a3, a4)
    L = []
    L.append('# imp-fronts: why the off-mesh bank ROM fails on steep fronts — Phase A CPU diagnostics (interim)')
    L.append('')
    L.append('**Status: interim, Phase A only (CPU diagnostics, 0 GPU-h).** Every accuracy number is measured against E0 ST '
             '(dev6, val32) or E0-style ST (jcp-regimes cohorts) references whose uncertainty ε_ref is a **conditional '
             'Richardson estimate**; all are therefore conditional, and nothing here is a Phase B (retrain) result. '
             f'Generated by `make_report.py` from `checks/*.json` at lane commit `{commit()}`.')
    L.append('')
    L.append('```mermaid')
    L.append('flowchart LR')
    L.append('  A1["A1 smearing test<br/>(is the bank biased to upwind data?)"]:::diag --> B["Phase B arm (a)<br/>fidelity retrain"]:::arm')
    L.append('  A2["A2 wall curvature / tail<br/>(boundary incompatibility?)"]:::diag --> C["Phase B arm (c)<br/>boundary-compatible bank"]:::arm')
    L.append('  A3["A3 F7 truncation screen"]:::diag --> F7["F7 fast decode<br/>(GPU timing deferred)"]:::other')
    L.append('  A4["A4 F15 hybrid floors"]:::diag --> F15["F15 sine-enriched bank"]:::other')
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
        L.append(f"| T6 | {', '.join(a1['T6'])} |")
        L.append(f"| median $D$, bank 384, T6 (cases with $D>1$) | {g(s['D_bank384_med_T6'])} ({s['D_bank384_gt1_T6']}/6) |")
        L.append(f"| median $D$, bank 128, T6 | {g(s['D_bank128_med_T6'])} |")
        L.append(f"| median $D$, bank 384, X ({len(a1['X'])} cases) | {g(s['D_bank384_med_X'])} |")
        L.append(f"| Spearman($D_{{384}}$, $q$) on X | {g(s['spearman_D384_q_X'])} |")
        L.append(f"| median $D$, bank 384, T6 at the 256-node training grid (sensitivity) | {g(s['D_bank384_med_T6_n256'])} |")
        L.append(f"| control +: median $D$ of LOO POD-128 of upwind fields, T6 (needs ≥ 2) | {g(c['Dpos_med_T6'])} |")
        L.append(f"| control −: median $D$ of LOO POD-128 of E0 fields, T6 (needs ≤ 1.25) | {g(c['Dneg_med_T6'])} |")
        L.append(f"| control resolution: median $D_+/D_-$, T6 (needs ≥ 2) | {g(c['resolution_med_T6'])} |")
        L.append(f"| exact-$D$ fixture (must hold) | {'pass' if c['exact_D_fixture']['ok'] else 'FAIL'} |")
        L.append(f"| upwind generator vs E0, median / worst error over X (conditional) | {pct(s['e_upw_vs_ref_med_X'])} / {pct(s['e_upw_vs_ref_max_X'])} |")
        L.append(f"| gates G1 (IC identity) / G2 (Newton residual, finite) | {a1['gates']['G1_ic_pass']} / {a1['gates']['G2_newton_pass']} |")
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
            L.append(f"| median wall curvature $\\kappa_a$ (relative to interior Laplacian) | {g(km['rom'])} | {g(km['proj'])} | 0 (exact) |")
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
        L.append('| cohort | median $\\tau_{4096}$ decoded / reference | median $\\omega$ decoded |')
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
            L.append(f"| {k} | {pct(ir['worst'])} | {pp(ir['dworst_2304'])} | {pp(ir['dworst_4096'])} | {pp(ir['dworst_9216'])} | "
                     f"{pct(rg['worst'])} | {pp(rg['dworst_9216'])} |")
        L.append('')
        L.append(f"**Outcome: {a3['verdict']}** (regime scope, descriptive: {a3['verdict_regime_scope']}). Controls: full-sine "
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
        L.append(f"- The original A2 finite-difference wall-curvature control (must fire on every in-range case) fires on "
                 f"{rr['n_fired']} of {rr['n_in_range']} in-range cases when replayed; it was dropped before any A2 decision "
                 '(Amendment 1), as design audit 1 had found.')
    if a2 and ok(a2, 'controls', 'identity'):
        idn = a2['controls']['identity']
        if 'original_dv_1em3_gap' in idn:
            L.append(f"- The registered bank-values check at spacing $10^{{-3}}$ gave a gap of {g(idn['original_dv_1em3_gap'])} against "
                     f"a bound of {g(idn['bank_path_bound'])} (value-path truncation); it was respecified (Amendments 6, 7): "
                     f"Richardson gap {g(idn['bank_path_gap'])}, pass {idn['bank_path_ok']}.")
    L.append('- Code audits 1–3 found and the lane fixed: tail round-off from cumulative subtraction, NaN-tolerant A1 gates, '
             'unverified A1 inputs (the upwind data were regenerated with persisted settings), A3 secondary-scope and '
             'reproduction handling, and report robustness.')
    for nm, d in (('A1', a1), ('A2', a2), ('A3', a3), ('A4', a4)):
        if d and d.get('run_commit'):
            rc = d['run_commit']
            L.append(f"- {nm} output generated at lane commit `{str(rc['commit'])[:9]}` (uncommitted code changes: {rc['code_dirty']}).")
    L.append('')
    L.append('## Phase B status')
    L.append('')
    pb = load('phaseb_verdicts')
    L.append('No Phase B result exists yet.' if pb is None else 'See the Phase B section.')
    L.append('')
    L.append('## Glossary')
    L.append('')
    for t, d in (
        ('bank', 'the learned set of 512 ordered coordinate-network functions whose first r span the reduced model'),
        ('floor $f_B$', 'relative error of the best lattice fit of a reference field by basis $B$: '
                        '$\\max_{k=1..5}\\lVert u(t_k)-P_Bu(t_k)\\rVert/\\lVert u(t_0)\\rVert$'),
        ('E0 / E0-style ST reference', 'second-order space-time references on fine meshes; ε_ref is their conditional Richardson '
                                        'uncertainty estimate'),
        ('dev6, val32', 'the development and validation cases of the training box (ν ∈ [0.01, 0.1], a ∈ [0.5, 2])'),
        ('N0s', 'the stiff corner inside the training box (lowest ν, largest a)'),
        ('C1, C2', 'cohorts with fronts steeper than training: s = a/ν in [250, 500] and [500, 1000]'),
        ('$s$, $q$', '$s=a/\\nu$ nominal steepness; $q=ah/(2\\nu)$ numerical-viscosity ratio of first-order upwinding'),
        ('N0, Ap, Am, B1, B2', 'other jcp-regimes cohorts: N0 in range; Ap ν above range; Am ν below range; B1 two same-sign bumps; '
                               'B2 a signed pair of bumps'),
        ('$\\beta$', 'least-squares slope of $\\log\\tau_K$ against $\\log K$ over $K\\in[2304,16384]$ (more negative = faster decay)'),
        ('Spearman', 'rank correlation coefficient between two per-case quantities'),
        ('containment violation', 'largest amount by which the hybrid floor exceeds the smaller of its two constituent floors; '
                                  'must be zero up to round-off because the hybrid span contains both'),
        ('upwind generator', 'the incumbent training-data solver: first-order sign-upwind advection, backward Euler, Δt = 0.005, '
                             '256 nodes per axis'),
        ('smearing excess D', 'how much more a basis prefers smeared (upwind) over sharp (reference) fields than the same number of '
                              'sines does'),
        ('LOO POD', 'leave-one-case-out proper orthogonal decomposition basis built from the other cases'),
        ('T6, X', 'the six target cases of A1 and its extended case set'),
        ('Laplacian sines $S_K$', 'the first $K$ products $\\sin(\\pi k_1x)\\sin(\\pi k_2y)$ ordered by $k_1^2+k_2^2$'),
        ('sine tail $\\tau_K$', 'relative norm of the part of a field beyond its first $K$ Laplacian sines'),
        ('wall share ω', 'fraction of the beyond-4096 tail energy within 8 lattice nodes of the walls'),
        ('wall curvature κ_a', 'RMS normal second derivative of a decoded field on the walls, relative to the interior Laplacian'),
        ('LSPG-CN-R', 'least-squares Petrov–Galerkin ROM with Crank–Nicolson time stepping and Rannacher start'),
        ('arm (a), (b), (c)', 'Phase B single-change retrains: data fidelity, steep/low-ν coverage, boundary-compatible output'),
        ('INCONCLUSIVE', 'a pre-registered control did not behave as required, so the diagnostic is not read either way'),
        ('pp', 'percentage points'),
    ):
        L.append(f'- **{t}**: {d}.')
    (HERE / 'report.md').write_text('\n'.join(L) + '\n')
    exp = dict(lane='imp-fronts', phase='A (interim)', commit=commit(), reference_label='conditional E0 / E0-style ST',
               A1=None if not a1 else dict(verdict=a1['verdict'], controls_pass=a1['controls']['controls_pass'], **a1['summary'],
                                            source='checks/a1_smear.json'),
               A2=None if not a2 else dict(decision_arm_c=a2['decision_arm_c'], controls_pass=a2['controls']['controls_pass'],
                                            in_range=a2['groups']['in_range'], source='checks/a2_wall.json'),
               A3=None if not a3 else dict(verdict=a3['verdict'], verdict_regime_scope=a3['verdict_regime_scope'],
                                            primary=a3['primary'], primary_scopes=a3['configs'][a3['primary']]['scopes'],
                                            source='checks/a3_trunc.json'),
               A4=None if not a4 else dict(verdict=a4['verdict'], C1_worst=a4['C1_worst'], source='checks/a4_hybrid.json'),
               plots=figs)
    (HERE / 'EXPORT.json').write_text(json.dumps(exp, indent=1, default=float))
    print('report.md written;', figs)


if __name__ == '__main__':
    main()
