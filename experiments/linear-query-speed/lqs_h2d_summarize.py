"""Tables for heat 2D from the saved result JSON of `lqs_h2d.py`.  Usage: lqs_h2d_summarize.py <result.json> <out.md>"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def f(x, nd=3):
    return '-' if x is None else f'{x:.{nd}f}'


def main():
    r = json.loads(Path(sys.argv[1]).read_text())
    key = 'h200' if 'H200' in r['nvidia_smi'] else 'a100-80G'
    cmp_ = json.loads((HERE / 'comparators-heat2d.json').read_text())[key]
    fz = r['frozen']; fn = fz['F']['variant']
    L = [f"# Heat 2D query time, {key} (stored paper values are H200) (job {r['job_id']}, node {r['node']}, `{r['nvidia_smi']}`)", '',
         f"Complete: {r['complete']}.  Frozen from the development cohort only at {fz['selection_mesh']}^2: "
         f"F = `{fn}` (frozen file SHA256 {r['frozen_sha256'][:16]}..., tied candidates {fz['F']['tied']}).", '',
         'Times: median GPU-query in ms over 16 test cases x %d repetitions; initial field on the device -> six output '
         'fields on the device, blocked (the paper\'s heat scope).  Errors: worst over the test cohort of the largest '
         'same-grid relative L2 error over the six output times.  "Coefficients only" returns the six reduced '
         'coefficient vectors and no field; it is a separate measurement, not the headline.  cuDSS / AmgX columns are '
         'the sparse-solvers lane\'s arithmetic estimates for 20 Crank-Nicolson solves (quoted, not measured '
         'trajectories).' % r['repetitions'], '',
         '## Headline (full six-field output)', '',
         '| mesh | arm | worst err, paper path | worst err, F | paper stored | paper path, this job (A1/A2) | F, this job | '
         'F vs paper path | cuDSS 20 solves, factor reused (est.) | AmgX setup + 20 solves (est.) | cuDSS est. / F | '
         'coefficients only (no field) |', '|' + '---|' * 12]
    meshes = sorted(r['meshes'], key=lambda m: m['intervals'])
    for m in meshes:
        n = m['intervals']; c = cmp_.get(str(n), {})
        for arm in r['arms']:
            rows = [s for s in m['summary'] if s['arm'] == arm]
            p = next(s for s in rows if s['kind'] == 'paper'); F = next(s for s in rows if s['kind'] == 'frozen')
            co = next(s for s in rows if s['kind'] == 'coefficients_only')
            cu = c.get('cudss_20_solves_factor_reused'); am = c.get('amgx_setup_plus_20_solves_1e-6')
            L.append(f"| {n}^2 | {arm} (`{p['name']}`) | {p['worst_test_error']:.6f} | {F['worst_test_error']:.6f} | "
                     f"{f(p.get('stored_test_median_ms'))} | {f(p['median_ms'])} ({f(p['A1_ms'])}/{f(p['A2_ms'])}) | "
                     f"**{f(F['median_ms'])}** | {p['median_ms'] / F['median_ms']:.1f}x | {f(cu, 0)} | {f(am, 0)} | "
                     f"{'-' if cu is None else f'{cu / F['median_ms']:.0f}x'} | {f(co['median_ms'])} |")
    L += ['', '## All subjects', '', '| mesh | arm | subject | ranks p, q | worst test err | max test field dev. from paper path | ms |',
          '|' + '---|' * 7]
    for m in meshes:
        for s in m['summary']:
            tag = {'paper': 'paper path', 'frozen': f"{s['name']} [F]", 'ablation': f"{s['name']} (ablation)",
                   'coefficients_only': 'COEFFICIENTS ONLY, no field'}[s['kind']]
            pq = f"{s['p']}, {s['q']}" if 'p' in s else '-'
            dv = s.get('max_test_deviation_from_paper')
            L.append(f"| {m['intervals']}^2 | {s['arm']} | {tag} | {pq} | "
                     f"{'-' if s.get('worst_test_error') is None else f'{s['worst_test_error']:.6f}'} | "
                     f"{'-' if dv is None else f'{dv:.1e}'} | {f(s['median_ms'])} |")
    L += ['', '## Stage medians (ms, first test case)', '',
          '| mesh | arm | paper: moments (transform) | paper: evolve | paper: six-field bank product | paper: whole | '
          'F: moments + coefficients | F: core contraction | F: field products | F: whole |', '|' + '---|' * 10]
    for m in meshes:
        for arm, pr in m['profile_median_ms'].items():
            p, q = pr['paper'], pr['frozen']
            L.append(f"| {m['intervals']}^2 | {arm} | {f(p['encode'])} | {f(p['evolve'])} | {f(p['decode'])} | {f(p['query'])} | "
                     f"{f(q['moments_and_coefficients'])} | {f(q['core_contraction'])} | {f(q['field_products'])} | {f(q['query'])} |")
    L += ['', '## Gates and offline cost', '',
          '| mesh | reproduction gate (max abs diff of paper-path errors vs stored, tol %g) | development gate (max dev., F, both arms; limit %g) | '
          'moment identity | compression s | range-finder residual | peak device GB |' % (r['repro_tolerance'], r['gate']), '|' + '---|' * 7]
    for m in meshes:
        g = m['repro_gate']; dg = [d for d in m['dev_gate'] if d['variant'] == fn]
        L.append(f"| {m['intervals']}^2 | {g['passed']} ({g['max_abs_difference'] if g['max_abs_difference'] is None else format(g['max_abs_difference'], '.1e')}) | "
                 f"{all(d['passes'] for d in dg)} ({max(d['max_deviation_from_paper'] for d in dg):.1e}) | "
                 f"{m['moment_identity']:.1e} | {m['compression']['seconds']:.1f} | "
                 f"{list(m['compression']['range_finder_relative_residual'].values())[0]:.1e} | "
                 f"{m.get('peak_device_bytes', 0) / 1e9:.1f} |")
    L += ['', '## Selection candidates (development cohort, %d^2)' % fz['selection_mesh'], '',
          '| variant | passes gate | max dev. | accurate ms | fast ms |', '|---|---|---|---|---|']
    for c in fz['candidates']:
        L.append(f"| {c['variant']} | {c['passes']} | {c['max_deviation']:.1e} | {f(c['accurate_ms'])} | {f(c['fast_ms'])} |")
    Path(sys.argv[2]).write_text('\n'.join(L) + '\n')


if __name__ == '__main__':
    main()
