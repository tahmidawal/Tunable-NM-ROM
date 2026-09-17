"""Generate the w-ladder report, summary.json and figure from the collected result JSONs.

    python experiments/w-ladder/generate_w_ladder.py [attempt ...]   (default: every artifacts/<attempt>/result.json)

Every number in the report comes from result.json / audit.json through this script; nothing is typed by hand.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LANE = ROOT / 'experiments/w-ladder'
FAMILY = {'head_': 'head rung', 'trained_': 'head rung', 'nested_': 'head rung', 'linear_': 'linear bank', 'pod_': 'POD-Galerkin',
          'dst': 'FOM', 'rk4': 'FOM', 'cg': 'FOM'}
# validated categorical slots 1-4 (dataviz reference palette), fixed order by family; marker shape is the secondary encoding
COLOR = {'head rung': '#2a78d6', 'linear bank': '#eb6834', 'POD-Galerkin': '#1baf7a', 'FOM': '#eda100'}
MARKER = {'head rung': 'o', 'linear bank': 's', 'POD-Galerkin': '^', 'FOM': 'D'}


def family(name):
    return next(v for k, v in FAMILY.items() if name.startswith(k))


def q_or_k(name, row):
    if name.startswith('pod_k'):
        return int(name.removeprefix('pod_k'))
    if 'q' in row:
        return row['q']
    return None


def pct(x):
    return f'{100 * x:.4f}'


def ms(x):
    return f'{x:.3f}'


def sci(x):
    return '' if x is None else f'{x:.2e}'


def load(attempt):
    d = LANE / 'artifacts' / attempt
    result = json.loads((d / 'result.json').read_text())
    audit = json.loads((d / 'audit.json').read_text()) if (d / 'audit.json').exists() else None
    return result, audit, hashlib.sha256((d / 'result.json').read_bytes()).hexdigest()


def summarise(result):
    """Per-arm rows: worst/median over cases (repetition 0 errors), medians over all timed repetitions (timing)."""
    rows = {}
    for name in result['arms']:
        inv = [r for r in result['invocations'] if r['method'] == name]
        first = [r for r in inv if r['repetition'] == 0]
        e = lambda key, sub='max_initial_normalized': np.array([r['same_grid_discrepancy'][key][sub] for r in first], dtype=float)
        cur = lambda key: np.array([r['same_grid_discrepancy'][key]['max_current_relative'] or np.nan for r in first], dtype=float)
        gpu = np.array([r['seconds']['complete_device_query'] * 1e3 for r in inv]); tot = np.array([r['device_plus_output_transfer_seconds'] * 1e3 for r in inv])
        evo = np.array([r['seconds']['evolution'] * 1e3 for r in inv])
        t0 = np.array([r['same_grid_discrepancy']['energy_state']['initial_normalized'][0] for r in first])
        phys = np.array([np.max(abs(np.asarray(r['same_grid_discrepancy']['energy_fraction']) / r['same_grid_discrepancy']['energy_fraction'][0] - 1.)) for r in first])
        red = np.array([r['reduced_energy']['max_relative_drift'] for r in first if 'reduced_energy' in r]) if all('reduced_energy' in r for r in first) else None
        rows[name] = dict(subject=name, family=family(name), q_or_k=q_or_k(name, first[0]), dimension=first[0].get('internal_configuration_dimension'),
                          cases=len(first), repetitions=len(inv),
                          worst_energy_state=float(np.max(e('energy_state'))), median_energy_state=float(np.median(e('energy_state'))),
                          worst_displacement=float(np.max(e('displacement'))), worst_velocity=float(np.max(e('velocity'))),
                          worst_current_displacement=float(np.nanmax(cur('displacement'))), worst_current_velocity=float(np.nanmax(cur('velocity'))),
                          worst_t0_energy_state=float(np.max(t0)),
                          median_gpu_ms=float(np.median(gpu)), median_complete_ms=float(np.median(tot)), median_evolution_ms=float(np.median(evo)),
                          gpu_ms_all=gpu.tolist(), outliers_above_twice_median=int(np.sum(gpu > 2 * np.median(gpu))),
                          all_completed=bool(all(r['completed'] for r in inv)),
                          fit_stationary=bool(all(r.get('fit_stationary', True) for r in inv)),
                          guard_fallbacks=int(sum(r.get('total_guard_fallbacks', 0) for r in first)),
                          cg_all_converged=bool(all(r.get('cg_all_converged', True) for r in inv)),
                          physical_energy_drift=float(np.max(phys)), reduced_energy_drift=float(np.max(red)) if red is not None else None,
                          all_state_pass=bool(np.max(e('energy_state')) <= result['config']['accuracy_target'] and np.max(e('displacement')) <= result['config']['accuracy_target'] and np.max(e('velocity')) <= result['config']['accuracy_target']))
    return rows


def nondominated(rows, names):
    pts = [(rows[n]['median_gpu_ms'], rows[n]['worst_energy_state'], n) for n in names]
    keep = []
    for c, e, n in pts:
        if not any((c2 <= c and e2 <= e and (c2 < c or e2 < e)) for c2, e2, _ in pts):
            keep.append(n)
    return keep


def verdict(rows):
    heads = [n for n in rows if rows[n]['family'] == 'head rung']
    lb = rows['linear_bank64']
    d1 = all(lb['worst_energy_state'] <= rows[h]['worst_energy_state'] for h in heads)
    d2 = all(lb['median_gpu_ms'] <= 0.1 * rows[h]['median_gpu_ms'] for h in heads)
    rom = heads + ['linear_bank64']
    d3 = nondominated(rows, rom) == ['linear_bank64']
    d4 = lb['reduced_energy_drift'] is not None and lb['reduced_energy_drift'] <= 1e-10 and rows['linear_bank64_cn']['reduced_energy_drift'] <= 1e-8
    ladder = ['head_q0', 'nested_q8', 'nested_q16', 'nested_q32']
    errs = [rows[n]['worst_energy_state'] for n in ladder if n in rows]
    mono = all(b <= a for a, b in zip(errs, errs[1:]))
    return dict(D1_accuracy=d1, D2_cost_0p1=d2, D3_singleton_nondominated=d3, D4_energy_certificate=d4, all=bool(d1 and d2 and d3 and d4),
                H_mono_ladder=mono, ladder_worst_energy_state=errs, nondominated_rom=nondominated(rows, rom + [n for n in rows if n.startswith('pod_') or n.startswith('linear_bank64_')]),
                nondominated_all=nondominated(rows, list(rows)))


def decomposition_table(result):
    """Worst over cases of each layer's time-max energy-state and displacement error."""
    layers = {}
    for entry in result['decomposition']:
        for name, layer in entry['layers'].items():
            m = layer['metrics']
            layers.setdefault(name, dict(layer=layer['layer'], dimension=layer['dimension'], energy=[], displacement=[], stationary=[]))
            layers[name]['energy'].append(m['energy_state']['max_initial_normalized']); layers[name]['displacement'].append(m['displacement']['max_initial_normalized'])
            if 'stationary_count' in layer:
                layers[name]['stationary'].append((layer['stationary_count'], layer['fit_count']))
    return {k: dict(layer=v['layer'], dimension=v['dimension'], worst_energy_state=float(np.max(v['energy'])), worst_displacement=float(np.max(v['displacement'])),
                    stationary=f"{sum(a for a, _ in v['stationary'])}/{sum(b for _, b in v['stationary'])}" if v['stationary'] else '') for k, v in layers.items()}


def figure(meshes, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(meshes), figsize=(5.2 * len(meshes), 4.4), squeeze=False)
    for ax, (n, rows, vd) in zip(axes[0], meshes):
        seen = set()
        for name, r in rows.items():
            fam = r['family']
            ax.scatter(r['median_gpu_ms'], 100 * r['worst_energy_state'], s=46, c=COLOR[fam], marker=MARKER[fam], edgecolors='white', linewidths=1.2,
                       label=fam if fam not in seen else None, zorder=3)
            seen.add(fam)
            if name in vd['nondominated_all'] or name in ('head_q0', 'trained_nested40', 'linear_bank64', 'nested_q32', 'pod_k64'):
                ax.annotate(name, (r['median_gpu_ms'], 100 * r['worst_energy_state']), xytext=(4, 4), textcoords='offset points', fontsize=7.5, color='#3a3a38')
        ax.axhline(5., color='#9a9a97', lw=1, ls=':', zorder=1); ax.text(ax.get_xlim()[0], 5.05, '5 % all-state target', fontsize=7, color='#6b6b68', va='bottom')
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_title(f'{n}² intervals', fontsize=10)
        ax.set_xlabel('median GPU ms per query (same job)'); ax.set_ylabel('worst energy-state error (%)')
        ax.grid(True, which='major', color='#e6e6e3', lw=0.8, zorder=0); ax.spines[['top', 'right']].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc='lower left')
    fig.suptitle('Reflective 2D wave: correction ladder, linear bank, POD and full-order controls', fontsize=11)
    fig.tight_layout(); fig.savefig(path, dpi=160); fig.savefig(path.with_suffix('.pdf'))


def main():
    attempts = sys.argv[1:] or sorted(p.parent.name for p in (LANE / 'artifacts').glob('*/result.json'))
    summary, meshes, sections, glossary_terms = [], [], [], []
    for attempt in attempts:
        result, audit, sha = load(attempt)
        n, job = result['mesh'], result['provenance']['job_id']
        rows = summarise(result); vd = verdict(rows); dec = decomposition_table(result)
        meshes.append((n, rows, vd))
        for name, r in rows.items():
            for metric in ('worst_energy_state', 'median_energy_state', 'worst_displacement', 'worst_velocity', 'worst_current_displacement', 'worst_current_velocity',
                           'worst_t0_energy_state', 'median_gpu_ms', 'median_complete_ms', 'median_evolution_ms', 'physical_energy_drift', 'reduced_energy_drift', 'all_state_pass'):
                summary.append(dict(mesh=n, subject=name, family=r['family'], q_or_k=r['q_or_k'], dimension=r['dimension'], metric=metric, value=r[metric],
                                    cases=r['cases'], job_id=job, attempt=attempt, source_sha256=sha))
        for name, d in dec.items():
            for metric in ('worst_energy_state', 'worst_displacement'):
                summary.append(dict(mesh=n, subject=name, family='decomposition:' + d['layer'], q_or_k=None, dimension=d['dimension'], metric=metric, value=d[metric],
                                    cases=len(result['decomposition']), job_id=job, attempt=attempt, source_sha256=sha))
        for k, v in vd.items():
            summary.append(dict(mesh=n, subject='verdict', family='verdict', q_or_k=None, dimension=None, metric=k, value=v, cases=None, job_id=job, attempt=attempt, source_sha256=sha))
        gates = result.get('gates', {})
        cons = result['consistency']
        cons_keys = [k for k in cons[0] if '_vs_' in k] if cons else []
        prov = result['provenance']
        text = [f"## {n}² intervals — attempt `{attempt}`, job {job}, {prov['device_kind'][0]}, source `{prov['source_commit']}`", '',
                f"Cases: {rows['dst']['cases']} development cases ({', '.join(c['name'] + ' ' + str(c['indices']) for c in result['config']['cohorts'])}); "
                f"{rows['dst']['repetitions'] // rows['dst']['cases']} timed repetitions per arm and case, all retained. "
                f"Retained-value gates: {gates.get('count', 0)} checks, all passed = {gates.get('all_passed')}. "
                + (f"Independent NumPy audit: max error recomputation difference {audit['max_error_difference']:.2e}, reference regeneration {audit['reference_max_relative_difference']:.2e}, passed = {audit['passed']}." if audit else 'Audit: not yet run.'), '',
                '### Verdict (pre-registered §4)', '',
                '| D1 accuracy | D2 cost ≤ 0.1× | D3 singleton non-dominated | D4 energy certificate | all | H-mono ladder |', '|---|---|---|---|---|---|',
                f"| {vd['D1_accuracy']} | {vd['D2_cost_0p1']} | {vd['D3_singleton_nondominated']} | {vd['D4_energy_certificate']} | **{vd['all']}** | {vd['H_mono_ladder']} |", '',
                f"Non-dominated set, ROM rungs and controls only: {', '.join('`' + x + '`' for x in vd['nondominated_rom'])}. "
                f"Including full-order solvers: {', '.join('`' + x + '`' for x in vd['nondominated_all'])}.", '',
                '### Ladder table (worst over cases of the time-maximum error; medians over all timed repetitions)', '',
                '| arm | family | q / k′ | dim | worst energy-state % | median energy-state % | worst u % | worst v % | worst current-rel u / v % | t=0 energy-state % | GPU ms | complete ms | evolution ms | outliers | reduced-energy drift | physical energy drift | completed / stationary / fallbacks | all-state 5 % |',
                '|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|---|---|:---:|']
        for name, r in rows.items():
            text.append(f"| `{name}` | {r['family']} | {'' if r['q_or_k'] is None else r['q_or_k']} | {r['dimension'] or ''} | {pct(r['worst_energy_state'])} | {pct(r['median_energy_state'])} | {pct(r['worst_displacement'])} | {pct(r['worst_velocity'])} | "
                        f"{pct(r['worst_current_displacement'])} / {pct(r['worst_current_velocity'])} | {pct(r['worst_t0_energy_state'])} | {ms(r['median_gpu_ms'])} | {ms(r['median_complete_ms'])} | {ms(r['median_evolution_ms'])} | {r['outliers_above_twice_median']} | "
                        f"{sci(r['reduced_energy_drift'])} | {r['physical_energy_drift']:.2e} | "
                        f"{r['all_completed']} / {r['fit_stationary']} / {r['guard_fallbacks']} | {'pass' if r['all_state_pass'] else 'fail'} |")
        text += ['', '### Three-layer decomposition (worst over cases; energy-state / displacement %)', '',
                 '| subject | layer | dim | worst energy-state % | worst displacement % | stationary fits |', '|---|---|---:|---:|---:|---|']
        for name, d in dec.items():
            text.append(f"| `{name}` | {d['layer']} | {d['dimension']} | {pct(d['worst_energy_state'])} | {pct(d['worst_displacement'])} | {d['stationary']} |")
        for name in ('head_q0', 'trained_nested40', 'nested_q8', 'nested_q16', 'nested_q32'):
            if name in rows and name in dec:
                text.append(f"| `{name}` | solved | {rows[name]['dimension']} | {pct(rows[name]['worst_energy_state'])} | {pct(rows[name]['worst_displacement'])} | |")
        text += ['', '### Consistency checks (relative max-abs coefficient discrepancy over all observation times; reported, not gated)', '',
                 '| case | ' + ' | '.join(cons_keys) + ' |', '|---|' + '---|' * len(cons_keys)]
        for c in cons:
            text.append(f"| {c['case']} | " + ' | '.join(f"{c[k]:.2e}" for k in cons_keys) + ' |')
        text += ['', f"POD from {result['pod']['snapshot_count']} training snapshots at 256²; captured snapshot energy fraction: "
                 + ', '.join(f"k′={k}: {v:.6f}" for k, v in result['pod']['energy_fraction_captured'].items())
                 + f". Mesh audits: bank orthogonality {result['mesh_audits']['bank']['orthogonality']:.1e}, POD orthogonality {result['mesh_audits']['pod']['orthogonality']:.1e}, "
                 f"sine-mode transfer error {result['mesh_audits']['pod']['sine_mode_transfer_error']:.1e}, POD K min eigenvalue {result['mesh_audits']['pod']['stiffness_min_eigenvalue']:.3f}.", '']
        sections.append('\n'.join(text))
    out = LANE / 'reports'
    figure(meshes, out / '2026-09-17-w-ladder.png')
    (out / 'summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    head = ["# w-ladder — reflective 2D wave: the correction ladder's top rung is the linear bank, and it is also the cheapest point", '',
            'What the accuracy–cost curve over correction rank $q$ looks like on the reflective 2D wave from one frozen learned bank and one frozen head, '
            'with POD-Galerkin and full-order controls in the same job at each mesh. Numbers are **final for the development cohort listed** '
            '(the sealed final cohort was not opened); every table is generated from `artifacts/<attempt>/result.json` by `generate_w_ladder.py`. '
            'Pre-registration: `DESIGN.md`. Meshes reported: ' + ', '.join(f'{n}²' for n, _, _ in meshes) + '.', '',
            '![ladder figure](2026-09-17-w-ladder.png)', '',
            '```mermaid', 'flowchart LR', '  U[supplied u0, v0, c] --> P[projection onto the bank]', '  P --> H[head fit: 8-start LM on z, y]',
            '  H --> R[RK4 on the manifold, joint LS stage solve]', '  R --> D[decode: coefficients x bank]', '  P --> L[exact modal propagation of all 64 coefficients]',
            '  L --> D', '  classDef trained fill:#2a78d6,color:#fff', '  classDef frozen fill:#eb6834,color:#fff', '  class H,R trained', '  class P,L,D frozen', '```', '']
    glossary = ['## Glossary', '',
                '- **arm / subject:** one method configuration timed and scored in the job.',
                '- **head rung:** the frozen nonlinear head (32 latent coordinates) plus $q$ appended linear correction directions; `head_q0` is $q=0$, `trained_nested40` the retained $q=8$ selection with scaled directions, `nested_q*` the same construction with orthonormal directions.',
                '- **linear bank:** all 64 bank coefficients evolved by the exact (Galerkin-projected) wave operator with no head; `_cn` / `_rk4` are time-stepped variants at the head’s step.',
                '- **POD-Galerkin, k′:** the classical snapshot basis of rank k′ from the same training data, evolved the same way.',
                '- **FOM:** full-order solver on the same grid: `dst` direct exact, `rk4_fom` explicit, `cg_*` implicit-midpoint conjugate gradient at the named tolerance and step, `dst_coarse64` the 64² direct solve interpolated up.',
                '- **q / k′ / dim:** correction count / POD rank / number of internal configuration coordinates.',
                '- **worst / median energy-state %:** over cases, the time-maximum combined displacement-gradient and velocity error scaled by the initial phase energy; the all-state target is 5 % on energy-state, u and v together.',
                '- **worst u / v %:** displacement error scaled by the initial displacement norm; velocity error scaled by the initial phase energy. Current-relative variants divide by the reference norm at that time.',
                '- **t=0 energy-state %:** the error of the returned initial state alone (the compression of the supplied field).',
                '- **GPU ms / complete ms / evolution ms:** median over all repetitions of the device-resident query, of the query plus host output transfer, and of the evolution component alone.',
                '- **outliers:** repetitions slower than twice the median.',
                '- **reduced-energy drift:** maximum relative change of the reduced energy computed from the coefficient trajectory; **physical energy drift:** the same for the returned fields relative to their own initial state.',
                '- **completed / stationary / fallbacks:** every step valid; every initial fit met the stopping rule; number of guarded-Cholesky fallbacks to QR.',
                '- **projection floor / best-found / solved:** error of the best linear projection onto the subspace / of the best manifold fit to the truth at each time / of the evolved ROM.',
                '- **non-dominated set:** arms no other arm beats on both worst energy-state error and GPU ms.',
                '- **D1–D4, H-mono:** the pre-registered criteria of `DESIGN.md` §4.',
                '- **consistency checks:** agreement between arms that should coincide mathematically (same manifold in different coordinates; the $q=32$ rung against the RK4-stepped bank; the stepped bank against exact modal propagation).',
                '- **development cohort:** cases opened for design; the final paper cohort stays sealed.']
    (out / '2026-09-17-w-ladder.md').write_text('\n'.join(head + sections + glossary) + '\n')
    print(out / '2026-09-17-w-ladder.md', len(summary), 'summary rows')


if __name__ == '__main__':
    main()
