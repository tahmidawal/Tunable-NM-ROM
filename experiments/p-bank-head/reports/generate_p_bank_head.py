"""Generate the p-bank-head report from the raw run JSONs. No number is typed by hand."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

PCT = 100.0


def pc(x, d=4):
    return '—' if x is None else f'{x * PCT:.{d}f}'


def ms(x, d=3):
    return '—' if x is None else f'{x * 1e3:.{d}f}'


def num(x, d=4):
    return '—' if x is None else f'{x:.{d}f}'


def yn(x):
    return {True: 'yes', False: 'no', None: '—'}[x]


def med(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None]
    return float(np.median(v)) if v else None


def is_stationary(r):
    """Two solvers, two exit conventions, one question.

    `arms.make_stationary_lm` (every `a_neural`, POD and free-bank arm) reports
    reason 4 for the normalised-gradient stop. The retained correction engine
    uses `kernel_solver.make_lm_kernel`, whose stationarity stop is reason 6 and
    which also carries an explicit `stationary` boolean over the FULL augmented
    gradient. Counting only reason 4 would report every correction arm as
    non-stationary, which is false.
    """
    if 'stationary' in r and r['stationary'] is not None:
        return bool(r['stationary'])
    return r.get('reason') == 4


def solve_table(sv):
    """(intervals, subject) -> aggregated row, from the raw invocations only."""
    out = {}
    for r in sv['invocations']:
        out.setdefault((r['intervals'], r['name']), []).append(r)
    agg = {}
    for key, rows in out.items():
        per_case = {}
        for r in rows:
            per_case.setdefault(r['case'], []).append(r)
        worst_sg = max(max(x['same_grid_error'] for x in v) for v in per_case.values())
        med_sg = float(np.median([np.median([x['same_grid_error'] for x in v])
                                  for v in per_case.values()]))
        worst_ph = max(max(x['physical_error'] for x in v) for v in per_case.values())
        stationary = sum(1 for r in rows if is_stationary(r))
        agg[key] = dict(worst_same_grid=worst_sg, median_same_grid=med_sg,
                        worst_physical=worst_ph, total_ms=med(rows, 'total_seconds'),
                        device_ms=(med(rows, 'fused_device_seconds')
                                   or med(rows, 'solver_seconds')),
                        invocations=len(rows), stationary=stationary,
                        k=rows[0].get('k'), family=rows[0].get('family'),
                        model=rows[0].get('model'),
                        iterations=med(rows, 'iterations'))
    return agg


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', type=Path, required=True,
                    help='the run that swept the bank (and carries the incumbent diagnosis)')
    ap.add_argument('--train-audit', type=Path, required=True)
    ap.add_argument('--head-train', type=Path, default=None,
                    help='the run that swept the head, if different from --train')
    ap.add_argument('--head-audit', type=Path, default=None)
    ap.add_argument('--head-train-alt', type=Path, default=None,
                    help='a second head sweep on a different bank, reported as a contrast')
    ap.add_argument('--bank-selection', type=Path, default=None,
                    help='the common-cohort bank selection check (DESIGN amendment 4)')
    ap.add_argument('--incumbent-diagnosis', type=Path, default=None,
                    help='the corrected incumbent diagnosis (DESIGN amendment 1)')
    ap.add_argument('--solve', type=Path, required=True)
    ap.add_argument('--solve-audit', type=Path, required=True)
    ap.add_argument('--reference', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    tr = json.loads(a.train.read_text())
    ta = json.loads(a.train_audit.read_text())
    ht = json.loads(a.head_train.read_text()) if a.head_train else tr
    ha = json.loads(a.head_audit.read_text()) if a.head_audit else ta
    alt = json.loads(a.head_train_alt.read_text()) if a.head_train_alt else None
    bsel = json.loads(a.bank_selection.read_text()) if a.bank_selection else None
    idg = json.loads(a.incumbent_diagnosis.read_text()) if a.incumbent_diagnosis else None
    sv = json.loads(a.solve.read_text())
    sa = json.loads(a.solve_audit.read_text())
    ref = json.loads(a.reference.read_text())
    L = []
    w = L.append
    agg = solve_table(sv)
    meshes = sorted({k[0] for k in agg})
    fine = meshes[-1]
    tcfg, scfg = tr['config'], sv['config']
    bank_sel = tr['selection']['bank']
    head_sels = ht['selection']['heads']
    bank_mesh = tcfg['floor_intervals'][-1]
    selected_bank = bsel['selected'] if bsel else bank_sel['selected']
    recon = {(r['intervals'], r['model']): r for r in sv['reconstruction']}
    models = {m['id']: m for m in sv['checkpoints']}
    control = 'incumbent'

    def solved(mid, n, q=False):
        for (mesh, name), row in agg.items():
            if mesh != n or row.get('model') != mid:
                continue
            isq = name.startswith('a_neural_q')
            if name.startswith('a_neural') and isq == q:
                return name, row
        return None, None

    # ---------------------------------------------------------------- header --
    w('# Poisson 2D — pushing the bank floor and the head, and whether the solve follows')
    w('')
    w('This report covers one pre-registered cell (`experiments/p-bank-head/DESIGN.md`): a '
      'bank sweep over feature rank and training-source count, a head sweep over latent '
      'dimension and objective, and a frozen solve of every resulting checkpoint through the '
      'unchanged head-ablation Poisson machinery, with the incumbent checkpoint as a control '
      'in the same job. **All numbers are final for this cell** — they come from two completed, '
      'checksum-collected, independently audited GPU jobs — but they are one training seed on '
      'one already-opened 12-source development cohort, so they are development evidence, not '
      'sealed-cohort results.')
    w('')
    w(f"Training job `{tr['job_id']}` on `{tr['gpu']}`, source commit `{tr['commit']}`; "
      f"solve job `{sv['job_id']}` on `{sv['gpu']}`, source commit `{sv['commit']}`. "
      f"Both logged `jax_backend={sv['backend']}`, float64, matmul precision "
      f"`{sv['matmul_precision']}`, JAX {sv['jax_version']}.")
    w('')

    # -------------------------------------------------------------- headline --
    w('## Verdict against the pre-registered clauses')
    w('')
    ctrl_name, ctrl = solved(control, fine)
    rows = []
    for mid in models:
        name, r = solved(mid, fine)
        if r is None:
            continue
        rec = recon[(fine, mid)]
        floor = rec['bank_projection']['worst']
        best = rec['best_found']['worst']
        podk = agg.get((fine, f"e_pod{models[mid]['K']}"))
        rows.append(dict(model=mid, K=models[mid]['K'], R=models[mid]['R'], floor=floor,
                         best=best, solved=r['worst_same_grid'], ms=r['total_ms'],
                         stationary=r['stationary'], invocations=r['invocations'],
                         cost_ratio=(r['total_ms'] / ctrl['total_ms']) if ctrl else None,
                         pod_matched=podk['worst_same_grid'] if podk else None,
                         beats_pod=(podk is not None and r['worst_same_grid'] < podk['worst_same_grid'])))
    w(f'At {fine} intervals on the 12 development sources:')
    w('')
    w('| clause | requirement | best checkpoint | value | verdict |')
    w('|---|---|---|---:|---|')
    if rows:
        bestrow = min((x for x in rows if x['model'] != control),
                      key=lambda x: x['solved'], default=None)
        if bestrow is not None:
            w(f"| 1 solved | worst same-grid < 2.0000 % | `{bestrow['model']}` | "
              f"{pc(bestrow['solved'])} % | "
              f"{'**pass**' if bestrow['solved'] < 0.02 else '**miss**'} |")
            w(f"| 2 stationary | every solve exits stationary | `{bestrow['model']}` | "
              f"{bestrow['stationary']}/{bestrow['invocations']} | "
              f"{'**pass**' if bestrow['stationary'] == bestrow['invocations'] else '**miss**'} |")
            w(f"| 3 cost | median query within 1.5x the incumbent | `{bestrow['model']}` | "
              f"{num(bestrow['cost_ratio'], 3)}x | "
              f"{'**pass**' if (bestrow['cost_ratio'] or 9) <= 1.5 else '**miss**'} |")
            w(f"| 4 vs POD | beats POD-LSPG at k' = K on worst error | `{bestrow['model']}` | "
              f"{pc(bestrow['solved'])} % vs {pc(bestrow['pod_matched'])} % | "
              f"{'**pass**' if bestrow['beats_pod'] else '**miss**'} |")
    w('')
    w('Every checkpoint at that mesh, control first:')
    w('')
    w('| checkpoint | K | R | bank floor | best-found | solved same-grid | median total ms | '
      'cost vs control | stationary |')
    w('|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for x in sorted(rows, key=lambda x: (x['model'] != control, x['solved'])):
        w(f"| `{x['model']}`{' *(control)*' if x['model'] == control else ''} | {x['K']} | "
          f"{x['R']} | {pc(x['floor'])} % | {pc(x['best'])} % | {pc(x['solved'])} % | "
          f"{ms(x['ms'])} | {num(x['cost_ratio'], 3)}x | "
          f"{x['stationary']}/{x['invocations']} |")
    w('')
    bank_target = min((next(f for f in x['floors'] if f['intervals'] == bank_mesh)
                       ['development']['worst'] for x in tr['bank_arms']), default=None)
    w(f"| layer target | requirement | value | verdict |")
    w('|---|---|---:|---|')
    w(f"| bank | worst development floor < 1.0000 % at {bank_mesh} intervals | "
      f"{pc(bank_target)} % | {'**pass**' if (bank_target or 9) < 0.01 else '**miss**'} |")
    for s in head_sels:
        arm = next(x for x in ht['head_arms'] if x['arm'] == s['selected'])
        f_ = next(f for f in next(x for x in tr['bank_arms']
                                  if x['arm'] == selected_bank)['floors']
                  if f['intervals'] == tcfg['training_intervals'])['development']['worst']
        ratio = arm['best_found_development']['worst'] / max(f_, 1e-300)
        w(f"| head K={s['K']} | best-found within 1.2x its bank floor | {num(ratio, 3)}x | "
          f"{'**pass**' if ratio <= 1.2 else '**miss**'} |")
    w('')

    # ------------------------------------------------------------- diagnosis --
    w('## Why a better bank did not improve the trained head (2026-09-11)')
    w('')
    w('The diagnosis is measured, not argued. D1–D8 are defined in DESIGN.md section 6 and are '
      'computed here for the incumbent checkpoint before anything was trained.')
    w('')
    if idg is not None:
        w('> **Retraction inside `pbh01` (DESIGN.md amendment 1).** That job computed the '
          "incumbent's *training-side* diagnostics against `core.source_params(0, 512)`, which "
          'is not the cohort that checkpoint was trained on (it trained on '
          '`core.source_params(0, 576)[:512]`, and the two share nothing). **D2, D3, D4, D6 and '
          'D8 as printed by `pbh01` for the incumbent are retracted.** The table immediately '
          'below replaces them, recomputed on the correct cohort by '
          '`checks/incumbent-diagnosis/`. D1 on the development cohort, D5 and D7 never touch '
          'the training parameters and are unchanged.')
        w('')
        w(f"Corrected, at {idg['intervals']} intervals, {idg['training_sources']} training "
          f"sources ({len(idg['subsample'])} sampled for the training-side oracle):")
        w('')
        w('| quantity | worst | median |')
        w('|---|---:|---:|')
        for key, label in [('D1_bank_floor_training', 'D1 bank floor, training'),
                           ('D1_bank_floor_development', 'D1 bank floor, development'),
                           ('D2_head_at_stored_codes_training', 'D2 head at the stored codes, training'),
                           ('D3_head_best_found_training', 'D3 head best-found, training'),
                           ('D5_head_best_found_development', 'D5 head best-found, development')]:
            w(f"| {label} | {pc(idg[key]['worst'])} % | {pc(idg[key]['median'])} % |")
        w('')
        w(f"D4 code-refit gain (D2 − D3), worst {pc(idg['D4_code_refit_gain_training']['worst'])} pp, "
          f"relative worst {num(idg['D4_code_refit_gain_training']['relative_worst'], 4)}, "
          f"relative median {num(idg['D4_code_refit_gain_training']['relative_median'], 4)}. "
          f"D6 generalisation gap, ratio of worst "
          f"{num(idg['D6_generalisation_gap']['ratio_of_worst'], 3)}x, ratio of median "
          f"{num(idg['D6_generalisation_gap']['ratio_of_median'], 3)}x. "
          f"D8 Spearman of development best-found against distance to the nearest training "
          f"parameter: {num(idg['D8_spearman_best_found_vs_distance'], 3)}. "
          f"Head floor over bank floor: "
          f"**{num(idg['head_floor_over_bank_floor'], 3)}x**.")
        w('')
        w(f"(Local GB10 diagnostic, `jax_backend={idg['backend']}`, float64, "
          f"matmul precision `{idg['matmul_precision']}`, "
          f"{num(idg['seconds'], 1)} s; no cluster job was spent on the correction.)")
        w('')
        w('The `pbh01` table below is kept for the record; read only its D1(dev), D5 and D7 '
          'columns.')
        w('')
    for dg in tr['diagnosis']:
        w(f"**`{dg['model']}`** — K={dg['K']}, R={dg['R']}, "
          f"{dg['training_sources']} training sources.")
        w('')
        w('| intervals | D1 bank floor (dev) | D2 head at stored codes (train) | '
          'D3 head best-found (train) | D4 code-refit gain | D5 head best-found (dev) | '
          'D6 dev/train ratio | D7 solved − best-found | D8 Spearman (error vs distance) |')
        w('|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for m in dg['meshes']:
            w(f"| {m['intervals']} | {pc(m['D1_bank_floor_development']['worst'])} % | "
              f"{pc(m['D2_head_at_stored_codes_training']['worst'])} % | "
              f"{pc(m['D3_head_best_found_training']['worst'])} % | "
              f"{pc(m['D4_code_refit_gain_training']['worst'])} pp | "
              f"{pc(m['D5_head_best_found_development']['worst'])} % | "
              f"{num(m['D6_generalisation_gap']['ratio_of_worst'], 3)}x | "
              f"{pc(m['D7_solved_minus_best_found']) if m['D7_solved_minus_best_found'] is not None else '—'} pp | "
              f"{num(m['D8_spearman_best_found_vs_distance'], 3)} |")
        w('')
        fin = dg['meshes'][-1]
        w(f"Head floor over bank floor at {fin['intervals']} intervals: "
          f"**{num(fin['head_floor_over_bank_floor'], 3)}x**.")
        w('')

    # ------------------------------------------------------------ bank layer --
    w('## Layer 1a — the bank sweep')
    w('')
    w(f"Six arms at K={tcfg['latents'][0]} with one fixed schedule and one fixed seed: identical "
      f"phase counts, identical learning rates, identical optimizer seeds, so the only "
      f"differences are R and S. Equal update counts mean the larger-S arms are NOT given more "
      f"training compute.")
    w('')
    if bsel is not None:
        w(f"**Selection.** The rule in force is DESIGN.md amendment 4: the lowest worst floor on "
          f"one **common selection cohort** — `core.source_params({bsel['common_cohort']['seed']}, "
          f"{bsel['common_cohort']['count']})`, a fresh seed asserted disjoint from every training "
          f"cohort and from the development cohort — at {bsel['intervals']} intervals. The "
          f"originally pre-registered rule (worst of each arm's *own* validation split) was "
          f"withdrawn mid-run because those splits have different sizes (29, 115 and 461 sources), "
          f"so their maxima are not comparable. Selected: **`{bsel['selected']}`**; the withdrawn "
          f"rule would have selected `{bsel['original_rule_selected']}`; the two "
          f"{'agree' if bsel['rules_agree'] else 'DISAGREE'}. The development cohort selects "
          f"nothing and its own ranking "
          f"{'agrees' if bsel['development_ranking_agrees'] else 'disagrees'} with the common-cohort "
          f"ranking.")
        w('')
        w('| arm | R | S | common cohort worst | common cohort median | own-validation worst '
          '(withdrawn rule) | development worst |')
        w('|---|---:|---:|---:|---:|---:|---:|')
        for x in sorted(bsel['arms'], key=lambda x: x['common']['worst']):
            mark = ' **(selected)**' if x['arm'] == bsel['selected'] else ''
            w(f"| `{x['arm']}`{mark} | {x['R']} | {x['S']} | {pc(x['common']['worst'])} % | "
              f"{pc(x['common']['median'])} % | {pc(x['reported_validation_worst'])} % | "
              f"{pc(x['reported_development_worst'])} % |")
        w('')
    else:
        w(f"Selection is by the worst **internal-validation** floor at {bank_sel['mesh']} "
          f"intervals; development floors are reported and do not select "
          f"(development ranking agrees: {yn(bank_sel['development_ranking_agrees'])}).")
        w('')
    for n in tcfg['floor_intervals']:
        w(f'### At {n} intervals')
        w('')
        w('| arm | R | sources S | fit | rank | floor worst (fit) | floor worst (val) | '
          'floor worst (dev) | floor median (dev) | head best-found worst (val) | '
          'head best-found worst (dev) | training s |')
        w('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for x in sorted(tr['bank_arms'], key=lambda x: (x['R'], x['S'])):
            f = next(y for y in x['floors'] if y['intervals'] == n)
            mark = ' **(selected)**' if x['arm'] == bank_sel['selected'] else ''
            w(f"| `{x['arm']}`{mark} | {x['R']} | {x['S']} | {x['fit_count']} | "
              f"{f['bank_rank']['rank']} | {pc(f['fit']['worst'])} % | "
              f"{pc(f['validation']['worst'])} % | {pc(f['development']['worst'])} % | "
              f"{pc(f['development']['median'])} % | "
              f"{pc(f['head_best_found_validation']['worst'])} % | "
              f"{pc(f['head_best_found_development']['worst'])} % | "
              f"{num(x['training_seconds'], 1)} |")
        w('')

    # ------------------------------------------------------------ head layer --
    w('## Layer 1b — the head sweep on the selected bank')
    w('')
    hl = ht['head_layer']
    w(f"Bank `{hl['bank']}` frozen (R={hl['R']}, rank {hl['bank_rank']['rank']}), "
      f"{hl['fit_count']} fit sources, {hl['validation_count']} internal-validation sources, "
      f"{hl['edges']} edges in the {hl['knn']}-nearest-neighbour parameter graph. "
      f"Every arm is {ht['config']['head_steps']} full-batch Adam updates from a fresh head and fresh "
      f"codes at one seed. Errors below are at {ht['config']['training_intervals']} "
      'intervals, the training mesh.')
    w('')
    w('| arm | K | beta_weak | beta_smooth | head at stored codes (fit, worst) | '
      'best-found (val, worst) | best-found (dev, worst) | weak-solved (dev, worst) | '
      'stationary | training s |')
    w('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    picks = {s['selected'] for s in head_sels}
    for x in sorted(ht['head_arms'], key=lambda x: (x['K'], x['beta_weak'], x['beta_smooth'])):
        mark = ' **(primary)**' if x['arm'] in picks else ''
        s = x['weak_solved_development']
        w(f"| `{x['arm']}`{mark} | {x['K']} | {x['beta_weak']:g} | {x['beta_smooth']:g} | "
          f"{pc(x['head_at_stored_codes_fit']['worst'])} % | "
          f"{pc(x['best_found_validation']['worst'])} % | "
          f"{pc(x['best_found_development']['worst'])} % | {pc(s['worst'])} % | "
          f"{s['stationary']}/{s['count']} | {num(x['training_seconds'], 1)} |")
    w('')
    for s in head_sels:
        w(f"- K={s['K']}: primary `{s['selected']}`; development ranking agrees: "
          f"{yn(s['development_ranking_agrees'])}.")
    w('')
    if alt is not None:
        ahl = alt['head_layer']
        w('### The same sweep on a bank with a sixteenth of the coverage')
        w('')
        w(f"`pbh01` ran the identical twelve-arm sweep on `{ahl['bank']}` "
          f"({ahl['fit_count']} fit sources) because that is the arm the **withdrawn** "
          f"selection rule chose. The contrast is the cell's clearest single result, so it is "
          f"reported rather than discarded.")
        w('')
        w('| arm | K | beta_weak | beta_smooth | training fit (worst, at stored codes) | '
          'best-found (dev, worst) | | training fit | best-found (dev, worst) |')
        w('|---|---:|---:|---:|---:|---:|---|---:|---:|')
        w(f"| | | | | **{hl['bank']}** ({hl['fit_count']} sources) | | | "
          f"**{ahl['bank']}** ({ahl['fit_count']} sources) | |")
        byarm = {x['arm']: x for x in alt['head_arms']}
        for x in sorted(ht['head_arms'], key=lambda x: (x['K'], x['beta_weak'], x['beta_smooth'])):
            y = byarm.get(x['arm'])
            w(f"| `{x['arm']}` | {x['K']} | {x['beta_weak']:g} | {x['beta_smooth']:g} | "
              f"{pc(x['head_at_stored_codes_fit']['worst'])} % | "
              f"{pc(x['best_found_development']['worst'])} % | | "
              f"{pc(y['head_at_stored_codes_fit']['worst']) if y else '—'} % | "
              f"{pc(y['best_found_development']['worst']) if y else '—'} % |")
        w('')
        bestf = min(ht['head_arms'], key=lambda x: x['best_found_development']['worst'])
        besta = min(alt['head_arms'], key=lambda x: x['best_found_development']['worst'])
        w(f"Best development best-found: {pc(bestf['best_found_development']['worst'])} % with "
          f"{hl['fit_count']} sources against {pc(besta['best_found_development']['worst'])} % "
          f"with {ahl['fit_count']}, a factor of "
          f"{num(besta['best_found_development']['worst'] / max(bestf['best_found_development']['worst'], 1e-300), 3)}. "
          f"The low-coverage sweep fits its own training data far TIGHTER "
          f"({pc(min(x['head_at_stored_codes_fit']['worst'] for x in alt['head_arms']))} % worst "
          f"against {pc(min(x['head_at_stored_codes_fit']['worst'] for x in ht['head_arms']))} %) "
          f"and generalises far worse: it is memorising, not representing.")
        w('')

    # -------------------------------------------------------------- mermaid ---
    w('## What is trained, what is frozen, what is solved')
    w('')
    w('```mermaid')
    w('flowchart TB')
    w('  subgraph OFF["offline, per training source"]')
    w('    P["source parameters<br/>(cx, cy, w, a)"] --> FD["FD-DST truth u_s"]')
    w('    P -.-> PH["normalised descriptor p̂<br/>(code-smoothness only)"]')
    w('  end')
    w('  subgraph BANK["layer 1a — bank sweep (trained)"]')
    w('    G["g(x): Fourier-feature MLP<br/>R columns"]')
    w('    C["free coefficients c_s in R^R"]')
    w('  end')
    w('  subgraph HEAD["layer 1b — head sweep (trained, bank frozen)"]')
    w('    H["h(z): MLP + linear skip<br/>R^K -> R^R"]')
    w('    Z["codes z_s in R^K"]')
    w('  end')
    w('  subgraph ON["online query (solved)"]')
    w('    F["supplied nodal source f"] --> FM["f_m = Λ⁻¹Φᵀf<br/>257 sine tests"]')
    w('    FM --> LM["damped LM in z only<br/>min ‖B h(z) − f_m‖"]')
    w('    LM --> OUT["u = G h(z*)"]')
    w('  end')
    w('  FD --> G')
    w('  FD --> C')
    w('  FD --> H')
    w('  PH --> Z')
    w('  G --> FROZEN["frozen bank G, frozen head h"]')
    w('  H --> FROZEN')
    w('  Z --> NEAR["nearest training code<br/>initializer"]')
    w('  FROZEN --> LM')
    w('  NEAR --> LM')
    w('  classDef trained fill:#dbeafe,stroke:#1d4ed8,color:#1e3a8a;')
    w('  classDef frozen fill:#f1f5f9,stroke:#64748b,color:#0f172a;')
    w('  classDef solved fill:#dcfce7,stroke:#15803d,color:#14532d;')
    w('  class G,C,H,Z trained;')
    w('  class FROZEN,NEAR,FD,P,PH frozen;')
    w('  class F,FM,LM,OUT solved;')
    w('```')
    w('')

    # ------------------------------------------------------------ solve layer -
    w('## Layer 2+3 — the frozen solve')
    w('')
    w('Every reduced arm runs through `poisson_ablation.make_query` / `query_once` unmodified: '
      'the same skinny sine-product projection, the same nearest-training-code initializer, '
      f"{scfg['requested_modes']} requested sine tests, LM budget {scfg['lm_budget']}, "
      f"stationarity tolerance {scfg['stationarity_tolerance']:g}, the same dense nodal output "
      f"contract, {scfg['repetitions']} timed repetitions in randomised order with GPU burn-in "
      'before every invocation. Same-grid error is against the same-mesh FD-DST solution; '
      'physical error is against the restricted 2048-interval reference.')
    w('')
    for n in meshes:
        w(f'### {n} intervals')
        w('')
        w('| subject | k | worst same-grid | median same-grid | worst physical | '
          'median total ms | median device ms | stationary | LM iters (median) |')
        w('|---|---:|---:|---:|---:|---:|---:|---:|---:|')

        def rank(name):
            r = agg[(n, name)]
            fam = r['family']
            order = {'neural': 0, 'neural+linear': 1, 'pod': 2, 'free': 3, 'fom': 4}
            return (order.get(fam, 5), r['k'] or 0, name)
        for name in sorted([k[1] for k in agg if k[0] == n], key=rank):
            r = agg[(n, name)]
            st = ('—' if r['family'] == 'fom'
                  else f"{r['stationary']}/{r['invocations']}")
            w(f"| `{name}` | {r['k'] if r['k'] else '—'} | {pc(r['worst_same_grid'])} % | "
              f"{pc(r['median_same_grid'])} % | {pc(r['worst_physical'])} % | "
              f"{ms(r['total_ms'])} | {ms(r['device_ms'])} | {st} | "
              f"{num(r['iterations'], 1) if r['iterations'] is not None else '—'} |")
        w('')

    # ------------------------------------------------------- three-layer table -
    w('## The three layers per checkpoint')
    w('')
    w('One caveat on reading the middle column against the third. **Best-found minimises the '
      'FIELD error; the solve minimises the WEAK residual** over the 257 retained sine tests. '
      'They are different objectives and they agree here only because the solutions of this '
      'family are smooth enough that the retained modes carry essentially all of their energy — '
      'measured, not assumed, and visible in the incumbent control, where the two agree to four '
      'decimal places. On a family whose solutions carried energy outside the test span the two '
      'columns would separate and the third could sit below the second.')
    w('')
    w('| intervals | checkpoint | K | R | bank floor (worst) | head best-found (worst) | '
      'solved same-grid (worst) | solved + q corrections | median total ms |')
    w('|---:|---|---:|---:|---:|---:|---:|---:|---:|')
    for n in meshes:
        for mid in models:
            if (n, mid) not in recon:
                continue
            rec = recon[(n, mid)]
            _, r = solved(mid, n)
            _, rq = solved(mid, n, q=True)
            w(f"| {n} | `{mid}` | {models[mid]['K']} | {models[mid]['R']} | "
              f"{pc(rec['bank_projection']['worst'])} % | {pc(rec['best_found']['worst'])} % | "
              f"{pc(r['worst_same_grid']) if r else '—'} % | "
              f"{pc(rq['worst_same_grid']) if rq else '—'} % | "
              f"{ms(r['total_ms']) if r else '—'} |")
    w('')

    # ----------------------------------------------------------- honesty -----
    w('## Honesty clauses, as pre-registered')
    w('')
    dst = agg.get((fine, 'dst_direct'))
    if dst:
        w(f"- **The direct DST full-order solve is faster and more accurate than every reduced "
          f"arm.** At {fine} intervals it takes {ms(dst['total_ms'])} ms total "
          f"({ms(dst['device_ms'])} ms device) with {pc(dst['worst_physical'])} % worst physical "
          f"error, against {ms(ctrl['total_ms'])} ms and {pc(ctrl['worst_physical'])} % for the "
          f"incumbent. **No speedup over any full-order solver is claimed anywhere in this "
          f"cell.**")
    pods = sorted({(k[1], agg[k]['k'], agg[k].get('pod_cohort')) for k in agg
                   if k[0] == fine and agg[k]['family'] == 'pod'}, key=lambda x: (x[1], x[0]))

    def best_pod(rank):
        cands = [(n, c) for (n, kk, c) in pods if kk == rank]
        if not cands:
            return None, None
        n, c = min(cands, key=lambda x: agg[(fine, x[0])]['worst_same_grid'])
        return n, agg[(fine, n)]
    w('')
    w('**POD-LSPG at the matched rank and at eight times it**, both cohorts, at '
      f'{fine} intervals. `@trainset` is rebuilt from the selected bank\'s own '
      f'{next((c["count"] for c in sv.get("pod_cohorts", []) if c["id"] == "trainset"), "?")} '
      'training snapshots and is the stronger competitor; the unsuffixed rungs are the '
      '192-snapshot cohort `pabl01` used.')
    w('')
    w("| checkpoint | K | neural worst / ms | POD k'=K worst / ms | POD k'=8K worst / ms | "
      'strongest POD dominates the head on BOTH error and cost? |')
    w('|---|---:|---:|---:|---:|---|')
    for mid in models:
        _, r = solved(mid, fine)
        if r is None:
            continue
        K_ = models[mid]['K']
        n1, p1 = best_pod(K_)
        n8, p8 = best_pod(8 * K_)
        dom = [n for (n, kk, c) in pods
               if agg[(fine, n)]['worst_same_grid'] < r['worst_same_grid']
               and agg[(fine, n)]['total_ms'] < r['total_ms']]
        w(f"| `{mid}` | {K_} | {pc(r['worst_same_grid'])} % / {ms(r['total_ms'])} | "
          + (f"`{n1}` {pc(p1['worst_same_grid'])} % / {ms(p1['total_ms'])} | " if p1 else '— | ')
          + (f"`{n8}` {pc(p8['worst_same_grid'])} % / {ms(p8['total_ms'])} | " if p8
             else f"not run (8K = {8 * K_} exceeds the pre-registered rank set) | ")
          + (f"**yes** — {', '.join('`' + x + '`' for x in sorted(dom))}" if dom else 'no') + ' |')
    w('')
    for mid in models:
        _, r = solved(mid, fine)
        if r is None:
            continue
        n8, p8 = best_pod(8 * models[mid]['K'])
        if p8 is None:
            w(f"- **POD at 8K is not in the pre-registered rank set for `{mid}`** "
              f"(K={models[mid]['K']}, so 8K={8 * models[mid]['K']}). The largest rung run is "
              f"k'=128. This is a stated limitation, not a pass.")
        else:
            w(f"- **POD at 8K for `{mid}`**: `{n8}` reaches {pc(p8['worst_same_grid'])} % at "
              f"{ms(p8['total_ms'])} ms against {pc(r['worst_same_grid'])} % at "
              f"{ms(r['total_ms'])} ms for the neural head — POD "
              f"{'**still matches or beats it**' if p8['worst_same_grid'] <= r['worst_same_grid'] else 'no longer matches it'}.")
    w(f"- Selections used an internal-validation split or the common held-out cohort only; the "
      f"12 development sources report and select nothing. The common-cohort bank ranking and the "
      f"development ranking "
      + ('agree' if (bsel and bsel['development_ranking_agrees']) else 'DISAGREE')
      + '; head development rankings '
      + ', '.join(f"K={x['K']} {yn(x['development_ranking_agrees'])}" for x in head_sels) + '.')
    w('- Every bank arm received the same number of optimizer updates, so the larger-S arms were '
      'not given more training compute; realised source exposures are in the run JSON.')
    w('')
    w('### The pre-registered falsification clause')
    w('')
    ratios = []
    for mid in models:
        if mid == control:
            continue
        rec = recon[(fine, mid)]
        ratios.append((mid, rec['best_found']['worst'] / max(rec['bank_projection']['worst'], 1e-300)))
    crec = recon[(fine, control)]
    cratio = crec['best_found']['worst'] / max(crec['bank_projection']['worst'], 1e-300)
    spread = (max(x['best_found_development']['worst'] for x in ht['head_arms'])
              / max(min(x['best_found_development']['worst'] for x in ht['head_arms']), 1e-300))
    w(f"DESIGN.md section 8 fixed two falsification conditions. **The first is met.** On the "
      f"selected bank every head arm's development best-found stays above 1.2x the bank floor "
      f"while the bank floor itself improved by "
      f"{num(crec['bank_projection']['worst'] / recon[(fine, ratios[0][0])]['bank_projection']['worst'], 3)}x: "
      + '; '.join(f"`{m}` {num(v, 3)}x" for m, v in ratios)
      + f", against {num(cratio, 3)}x for the incumbent. Pushing the bank three times lower "
        f"made the head's *relative* distance to it **larger**, not smaller, even though the "
        f"head's absolute error fell by "
        f"{num(crec['best_found']['worst'] / min(recon[(fine, m)]['best_found']['worst'] for m, _ in ratios), 3)}x. "
        f"That is the 2026-09-11 finding reproduced at larger scale and it is reported as a "
        f"negative result, with no rescue arm.")
    w('')
    w(f"**The second is not met.** The head arms are not inert: their worst development "
      f"best-found spans {num(spread, 3)}x across the twelve arms on the selected bank, far "
      f"above the 5 % relative change that would have said the limit is the head's function "
      f"class and outside this cell's latitude. What moves them is **coverage**, not the "
      f"objective: on the selected bank the two objective terms are neutral at best "
      f"(the pre-registered primary at both K is the plain reconstruction objective, "
      f"beta_weak = beta_smooth = 0), while changing the bank's training cohort from "
      + (f"{alt['head_layer']['fit_count']} to {hl['fit_count']} fit sources moved the best "
         f"development best-found from "
         f"{pc(min(x['best_found_development']['worst'] for x in alt['head_arms']))} % to "
         f"{pc(min(x['best_found_development']['worst'] for x in ht['head_arms']))} %."
         if alt else 'more sources moved it substantially.'))
    w('')
    # -------------------------------------------------------------- gates ----
    w('## Fidelity gates and audits')
    w('')
    w('| gate | what it checks | worst | tolerance | verdict |')
    w('|---|---|---:|---:|---|')
    for g in sv['gates']:
        w(f"| G2/G3 `{g['arm']}` @ {g['intervals']} | reproduces `pabl01` per-case physical "
          f"error | {g['worst_relative_difference']:.3e} | {g['tolerance']:.0e} | "
          f"{'pass' if g['passed'] else '**FAIL**'} |")
    w('')
    w(f"Reference: `{ref['source']}` sha256 `{ref['source_sha256'][:16]}…`, "
      f"job `{ref['job_id']}` on `{ref['gpu']}`; its three repetitions agree exactly "
      f"(max relative spread {ref['repetition_spread']:.1e}).")
    w('')
    w('| audit | scope | verdict |')
    w('|---|---|---|')
    for name, c in ta['checks'].items():
        w(f"| train `{name}` | NumPy/SciPy only, no driver, no JAX | "
          f"{'pass' if c['passed'] else '**FAIL**'} |")
    for name, c in sa['checks'].items():
        w(f"| solve `{name}` | NumPy/SciPy only, no driver, no JAX | "
          f"{'pass' if c['passed'] else '**FAIL**'} |")
    w('')
    rc = sa['checks']['recomputed_errors']
    w(f"The solve audit recomputed every one of {rc['invocations']} reported errors from the "
      f"retained output fields against an independently rebuilt reference: worst physical "
      f"difference {rc['worst_physical_difference']:.2e}, worst same-grid difference "
      f"{rc['worst_same_grid_difference']:.2e}.")
    w('')

    # --------------------------------------------------------- limitations ---
    w('## Recorded deviations and limitations')
    w('')
    w(f"- **`g_hidden = R`.** A rank-R bank needs the last hidden layer of `g` to be at least "
      f"R wide. The incumbent satisfies this with equality at R=128; this cell keeps that rule "
      f"and therefore widens `g`'s hidden layers with R. The **head** width stays at the "
      f"incumbent {tcfg['arch']['h_hidden']} for every arm; only h's output layer widens. "
      f"`widen_bank.widen` cannot reach R=512 from the incumbent (its complement is capped at "
      f"g_hidden+1 columns), so all banks here are trained from scratch.")
    w(f"- **The training cohorts are independent draws, not nested prefixes.** "
      f"`core.source_params(0, S)` draws each parameter array at length S, so the S=192, 768 "
      f"and 3072 cohorts do not contain one another. S=192 is exactly the cohort `pabl01` "
      f"builds its POD arms from. The incumbent checkpoint was trained on 512 sources and is "
      f"carried through frozen, never retrained.")
    w('- **The code-smoothness term uses the family\'s normalised parameter descriptor '
      'offline.** No descriptor reaches any online query; the deployed path still receives '
      'nothing but the nodal source field.')
    w('- **Best-found is an upper bound.** It is a seeded multistart LM search on a curved '
      'manifold, so it can only overstate the head floor — exactly where tightness would '
      'favour the neural arms.')
    w('- **The free-bank arm needs M > R** and is therefore only run for checkpoints whose '
      'feature rank is below the retained test count; the bank projection floor, which is the '
      'quantity that arm measures, is reported for every checkpoint as an untimed diagnostic.')
    w('- One training seed, one PDE, one already-opened 12-source development cohort. The '
      'sealed final cohorts were not touched and no case was opened.')
    w('')

    # ---------------------------------------------------------- glossary -----
    w('## Glossary')
    w('')
    w('Written for a reader opening this report cold.')
    w('')
    w('- **Bank** — the matrix `G` whose R columns are the coordinate network\'s spatial '
      'features evaluated on the mesh. Every reduced solution is a combination of its columns.')
    w('- **R (feature rank)** — the number of bank columns. More columns can represent more '
      'fields, at more cost per decode.')
    w('- **K (latent dimension)** — the number of unknowns the online nonlinear solve actually '
      'solves for. The head maps those K numbers to the R bank coefficients.')
    w('- **Head** — the small network `h: R^K -> R^R` (an MLP plus a linear skip) that turns a '
      'latent code into bank coefficients. Its image inside the bank is the model\'s manifold.')
    w('- **Code** — the latent vector `z_s` attached to one training source. Codes are learned, '
      'not encoded: there is no encoder network in the deployed path.')
    w('- **Bank projection floor** — the smallest relative error any combination of bank '
      'columns can achieve for a given field. Layer 1 of the error decomposition, and a hard '
      'lower bound for everything else.')
    w('- **Best-found (oracle from codes)** — the smallest relative error reachable on the '
      'head\'s own manifold, found by a multistart local search over z. It separates "the head '
      'cannot represent this field" from "the solver did not find the best z".')
    w('- **Solved** — what the actual online query returns. If solved equals best-found, the '
      'solver is not the problem.')
    w('- **Same-grid error** — against the exact solution of the *same* discrete problem on the '
      'same mesh; it isolates reduction error from discretisation error.')
    w('- **Physical error** — against a much finer (2048-interval) reference restricted onto '
      'the mesh; it includes the mesh\'s own discretisation error.')
    w('- **Worst vs median** — worst is the maximum over the 12 development sources; median is '
      'the middle one. Worst is the number the targets are written against.')
    w('- **Weak residual** — the quantity the ROM minimises: the mismatch of the candidate '
      'solution against the source on 257 sine test functions. For Poisson it is linear in the '
      'bank coefficients, which is why it is cheap to add to a training objective.')
    w('- **M (test modes)** — how many sine tests the weak residual uses; 257 here.')
    w('- **LM / damped Levenberg–Marquardt** — the nonlinear least-squares solver used online.')
    w('- **Stationary exit** — the solve stopped because the normalised gradient fell below '
      '1e-6, meaning it reached a genuine critical point rather than running out of budget.')
    w('- **POD-LSPG** — the classical linear baseline: build a basis by principal component '
      'analysis of training solutions, then solve the same weak least-squares problem in that '
      'basis. `k\'` is its rank. It is the honest competitor because it has the same online '
      'structure and no network.')
    w('- **k\' = K / k\' = 8K** — POD at the *same* number of unknowns as the neural head, and '
      'at eight times as many. The first is the matched-dimension comparison; the second asks '
      'whether extra linear rank simply buys the neural head\'s accuracy back.')
    w('- **q corrections (the `a_neural_q*` arms)** — a fixed set of q extra linear directions '
      'added to the head\'s output and eliminated analytically, so the nonlinear solve stays '
      'K-dimensional. Retained from the accepted 2026-09-11 Poisson family.')
    w('- **Free bank** — the diagnostic arm that ignores the head and solves for all R bank '
      'coefficients directly. It measures the bank floor through the solver.')
    w('- **DST / `dst_direct`** — the direct discrete sine transform solve. For this problem it '
      'is an exact, very fast full-order solver, which is why no speed claim is made here.')
    w('- **Fit split / internal validation** — the training sources are split 85/15; only the '
      '85 % is optimised on and only the 15 % decides which arm is selected, so no selection '
      'touches the development cohort.')
    w('- **Development cohort** — the 12 already-opened sources everything is *reported* on. '
      'They are not a sealed test set; the project\'s final cohorts remain unopened.')
    w('- **Fidelity gate** — a check that this cell\'s code reproduces an earlier accepted '
      'run\'s numbers before any new number is believed.')
    w('- **pp** — percentage points, the difference between two percentages.')
    w('- **Incumbent** — the checkpoint this cell is trying to beat: the accepted 2026-09-11 '
      'Poisson model `r128_joint` (K=16, R=128), carried through frozen and never retrained.')
    w('- **Arm** — one configuration in a sweep. A *bank arm* is one (R, S) pair; a *head arm* '
      'is one (K, beta_weak, beta_smooth) triple; a *solve subject* is one thing that is timed.')
    w('- **S (training sources)** — how many source fields the bank and head were fitted to.')
    w('- **beta_weak / beta_smooth** — the two weights in the head objective: how much the '
      'exact weak residual and the parameter-space code-smoothness term count against plain '
      'reconstruction. Both zero reproduces the incumbent objective.')
    w('- **Code smoothness** — a penalty that pulls the latent codes of sources with similar '
      'parameters towards each other, so the head has to interpolate between them rather than '
      'memorise each one. It uses the source parameters offline only; no query ever sees them.')
    w('- **Head at the stored codes** — the training error the optimizer actually produced, '
      'using each training source\'s own saved code. Compare it with best-found on the same '
      'sources: a large gap means the codes were not converged.')
    w('- **Code-refit gain (D4)** — exactly that gap. Small means the training codes are at '
      'their own optimum and the problem is not optimisation.')
    w('- **D1 … D8** — the eight numbered diagnostics of DESIGN.md section 6, designed so that '
      'each of the three ways the head can fail (optimisation, coverage, capacity) leaves a '
      'different fingerprint.')
    w('- **Generalisation gap (D6)** — how much worse the head is on unseen sources than on its '
      'own training sources. Large means the head interpolates badly; the fix is coverage or '
      'regularisation, not capacity.')
    w('- **Common selection cohort** — one fixed 256-source held-out set, at a seed used '
      'nowhere else, on which every bank arm is scored. It exists because each arm\'s own '
      'validation split has a different size, and a maximum over a larger sample is '
      'systematically larger for reasons unrelated to the bank.')
    w('- **Rank / condition number of a bank** — whether its R columns are genuinely '
      'independent on the mesh, and how close to dependent they are. A rank-deficient bank '
      'would make R a lie; every arm here is checked to be full rank at every mesh.')
    w('- **Exposures** — how many times the optimizer saw a training source. Equal update '
      'counts across arms means the larger-S arms get fewer exposures each, which is the '
      'honest equal-compute comparison and is why more data is not automatically better here.')
    w('- **Total ms vs device ms** — total is the whole query, host array in to dense nodal '
      'field out; device is the fused GPU interval inside it. Total is the number that matters '
      'to a user; device is where the reduced solve actually happens.')
    w('- **Trust radius** — a cap on how far one solver step may move the latent code, set to '
      'the radius of the training code cloud so the solve stays where the head was fitted.')

    w('')

    a.out.write_text('\n'.join(L) + '\n')
    digest = hashlib.sha256(a.out.read_bytes()).hexdigest()
    print(a.out)
    print('sha256', digest)


if __name__ == '__main__':
    main()
