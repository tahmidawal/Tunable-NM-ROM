"""Generate reports/summary.json and the Table-3 test-vs-development report from the audited summaries (DESIGN.md §8).

    python reports/make_report.py --mesh 1024=checks/t31024-summary.json --mesh 2048=checks/t32048h-summary.json \
        [--unused 2048=checks/t32048a-summary.json] --out-md reports/2026-09-25-t3-burgers-test.md \
        --out-json reports/summary.json

Every number in the report is read from those JSON files or from inputs/dev-reference.json; nothing is typed by hand.
"""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
ROWS = [('nmrom_accurate', "NM-ROM accurate (span R'=384)"), ('nmrom_head', 'NM-ROM head, k=16'),
        ('pod_lspg16', 'POD-LSPG, k=16'), ('qman16', 'quadratic manifold, r=16')]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def arm_of(key, s):
    if key == 'nmrom_accurate':
        return s['roles']['nmrom_accurate']
    if key == 'nmrom_head':
        return s['roles']['nmrom_fast']
    return {'pod_lspg16': 'pod16_M64_dense', 'qman16': 'qman16_quad_M64'}[key]


def reproduction(s, ref):
    """DESIGN §4.1: rebuilt bases against the parent's record."""
    pod = s.get('pod') or {}
    q = next((x for x in s['quadratic_manifold_fits'] if x['rank'] == 16), None)
    snap = (s.get('snapshots') or {}).get('snapshot_sha256')
    ev = pod.get('eigenvalues', [])[:16]
    ev_rel = max(abs(a - b) / abs(b) for a, b in zip(ev, ref['pod']['eigenvalues_top16'])) if len(ev) == 16 else None
    out = dict(snapshot_sha256_equal=(snap == ref['snapshots']['snapshot_sha256']),
               pod_modes_sha256_equal=(pod.get('modes_sha256') == ref['pod']['modes_sha256']),
               pod_kmax=pod.get('kmax'), pod_eigenvalue_max_relative_difference=ev_rel,
               qman16_ridge=(q or {}).get('ridge'), qman16_ridge_parent=ref['qman16']['ridge'],
               qman16_heldout=(q or {}).get('heldout'), qman16_heldout_parent=ref['qman16']['heldout_relative'])
    hq = (abs(out['qman16_heldout'] - out['qman16_heldout_parent']) / out['qman16_heldout_parent']) if q else None
    out['qman16_heldout_relative_difference'] = hq
    out['passed'] = bool(out['snapshot_sha256_equal'] or (
        ev_rel is not None and ev_rel <= 1e-8 and q is not None and out['qman16_ridge'] == out['qman16_ridge_parent']
        and hq <= 1e-6))
    return out


def fmt(x, nd=2):
    if x is None:
        return '—'
    if abs(x) >= 100:
        return f'{x:,.1f}'
    return f'{x:.{nd}f}' if abs(x) >= 1 else f'{x:.{nd + 1}g}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh', action='append', required=True)
    p.add_argument('--unused', action='append', default=[])
    p.add_argument('--out-md', required=True)
    p.add_argument('--out-json', required=True)
    a = p.parse_args()
    refp = HERE / 'inputs' / 'dev-reference.json'
    ref = json.loads(refp.read_text())
    res = dict(title='Paper Table 3 (tab:nmrom-baselines), Burgers 2D 1024^2/2048^2, on the 64 held-out test cases',
               design='experiments/t3-burgers-test/DESIGN.md', dev_reference=dict(file=str(refp.relative_to(HERE)),
                                                                                sha256=sha(refp)), meshes={}, unused=[])
    for m in a.mesh:
        L, f = m.split('=')
        fp = HERE / f
        s = json.loads(fp.read_text())
        assert int(s['intervals']) == int(L)
        r = ref[L]
        t3 = s['table3']
        rows = {x['arm']: x for x in t3['rows']}
        ev = {x['name']: sorted(x['evolved'], reverse=True) for x in s['arms']}
        bs = {x['arm']: x for x in (s.get('bank_span') or {}).get('arms', [])}
        mesh = dict(summary_file=f, summary_sha256=sha(fp), job_id=s['job_id'], gpu=s['gpu'], commit=s['commit'],
                    cohort=s['cohort'], cohort_sha256=s['cohort_sha256'], failed_gates=s['failed_gates'],
                    cell_fom=t3['fom'], basis_reproduction=reproduction(s, r),
                    dev=dict(job_id=r['job_id'], gpu=r['gpu'], summary_sha256=r['summary_sha256'], cell_fom=r['cell_fom']),
                    rows=[])
        for key, label in ROWS:
            arm = arm_of(key, s)
            x = rows.get(arm, {})
            d = r['rows'][key]
            mesh['rows'].append(dict(row=label, arm=arm, unknowns=x.get('unknowns'),
                                     test=dict(worst_percent=x.get('worst_evolved_percent'),
                                               median_percent=x.get('median_evolved_percent'), gpu_ms=x.get('gpu_ms'),
                                               speedup_vs_cell_fom=x.get('speedup_vs_cell_fom'),
                                               second_worst_percent=(100 * ev[arm][1] if arm in ev else None),
                                               worst_case=(next(a_['evolved'].index(max(a_['evolved'])) for a_ in s['arms']
                                                                if a_['name'] == arm) if arm in ev else None),
                                               cases=(len(ev[arm]) if arm in ev else None)),
                                     dev=dict(worst_percent=d['worst_evolved_percent'], median_percent=d['median_evolved_percent'],
                                              gpu_ms=d['gpu_ms'], speedup_vs_cell_fom=d['speedup_vs_cell_fom']),
                                     rho=(dict(rho_max=bs[arm]['rho_max'], exceeds_bar=bs[arm]['exceeds_bar'],
                                               bar=bs[arm]['bar']) if arm in bs else None)))
        res['meshes'][L] = mesh
    for m in a.unused:
        L, f = m.split('=')
        s = json.loads((HERE / f).read_text())
        res['unused'].append(dict(mesh=L, summary_file=f, summary_sha256=sha(HERE / f), job_id=s['job_id'], gpu=s['gpu'],
                                  failed_gates=s['failed_gates'], note='run and not used (DESIGN §7 decision rule)'))
    Path(a.out_json).write_text(json.dumps(res, indent=1, allow_nan=False) + '\n')

    md = ['# Table 3 (Burgers 2D, matched latent dimension) at 1024² and 2048² on the 64 held-out test cases', '',
          'The paper Table 3 methods at 1024² and 2048², frozen, re-measured on the 64 held-out Burgers 2D test cases '
          '(`params_draw(20260916, 64)`) instead of the six development cases. Same protocol and same job for every ratio. '
          'Status: **final** for the meshes listed; generated by `reports/make_report.py` from the audited summaries '
          '(SHA256 below). Development values are the paper\'s current cells, from `inputs/dev-reference.json`.', '']
    for L, mesh in res['meshes'].items():
        cf, dcf = mesh['cell_fom'], mesh['dev']['cell_fom']
        md += [f'## {L}²', '',
               f"Test job {mesh['job_id']} ({mesh['gpu']}, commit `{(mesh['commit'] or '?')[:10]}`), summary `{mesh['summary_file']}` "
               f"SHA256 `{mesh['summary_sha256']}`. Cohort SHA256 `{mesh['cohort_sha256'][:16]}…`. "
               f"Failed gates: {', '.join(mesh['failed_gates']) or 'none'}.",
               f"Development job {mesh['dev']['job_id']} ({mesh['dev']['gpu']}), summary SHA256 `{mesh['dev']['summary_sha256'][:16]}…`.", '',
               f"Cell FOM, test: `{cf['name']}` {fmt(cf['gpu_ms'])} ms, worst {fmt(cf['worst_evolved_percent'], 3)} % "
               f"(median {fmt(cf['median_evolved_percent'], 3)} %). Cell FOM, development: `{dcf['name']}` {fmt(dcf['gpu_ms'])} ms, "
               f"worst {fmt(dcf['worst_evolved_percent'], 3)} %." if cf else 'Cell FOM, test: none qualifies (no ratio printed).', '',
               '| row | unknowns | test worst % (case) | test 2nd-worst % | test median % | test GPU ms | test speedup | dev worst % | dev GPU ms | dev speedup |',
               '|---|---|---|---|---|---|---|---|---|---|']
        for x in mesh['rows']:
            t, d = x['test'], x['dev']
            sp = f"{t['speedup_vs_cell_fom']:.3g}×" if t['speedup_vs_cell_fom'] else '—'
            md.append(f"| {x['row']} | {x['unknowns']} | {fmt(t['worst_percent'], 3)} ({t['worst_case']}) | "
                      f"{fmt(t['second_worst_percent'], 3)} | {fmt(t['median_percent'], 3)} | "
                      f"{fmt(t['gpu_ms'])} | {sp} | {fmt(d['worst_percent'], 3)} | {fmt(d['gpu_ms'])} | {d['speedup_vs_cell_fom']:.3g}× |")
        acc = mesh['rows'][0]
        br = mesh['basis_reproduction']
        md += ['', f"- Bank-span ρ (8 held-out trajectories, bar {acc['rho']['bar'] if acc['rho'] else '—'}): "
               f"ρ_max {fmt(acc['rho']['rho_max'], 3) if acc['rho'] else '—'}"
               f"{' — EXCEEDS the bar' if acc['rho'] and acc['rho']['exceeds_bar'] else ' — within the bar'}.",
               f"- Basis reproduction (DESIGN §4.1): {'PASS' if br['passed'] else 'FAIL'} — snapshot SHA256 equal to the parent: "
               f"{br['snapshot_sha256_equal']}; POD mode SHA256 equal: {br['pod_modes_sha256_equal']}; top-16 POD eigenvalue max rel. "
               f"difference {br['pod_eigenvalue_max_relative_difference']:.2e}; QM ridge {br['qman16_ridge']} (parent "
               f"{br['qman16_ridge_parent']}); QM held-out {br['qman16_heldout']:.6f} (parent {br['qman16_heldout_parent']:.6f}).", '']
    if res['unused']:
        md += ['## Run and not used', '']
        for u in res['unused']:
            md.append(f"- {u['mesh']}²: job {u['job_id']} ({u['gpu']}), failed gates: {', '.join(u['failed_gates']) or 'none'}; "
                      f"summary SHA256 `{u['summary_sha256'][:16]}…`. {u['note']}.")
        md.append('')
    md += ['## Glossary', '',
           '- **test / dev**: the 64 held-out test cases (never used for any choice) / the six development cases the paper\'s current cells use.',
           '- **worst %**: over the cases, the largest of max over the five evolved output times of ‖u − u_ref‖/‖u₀‖, in percent; '
           'u_ref = the tightly converged Newton solve (`fft_tight`) on the same mesh in the same job. **median %**: the median over cases.',
           '- **2nd-worst %**: the second-largest per-case error (shows whether the worst is a single outlier case); **(case)** = index of the worst case in test64.',
           '- **GPU ms**: median over cases × 5 repetitions of the time from input field on the GPU to the six output fields on the GPU.',
           '- **cell FOM**: one full-order setting per mesh — the fastest of the 15 Newton–BiCGStab settings whose worst error is at '
           'most the NM-ROM accurate row\'s worst error, timed in the same job. **speedup** = cell FOM ms / row ms.',
           "- **NM-ROM accurate (span R'=384)**: the frozen decoder's bank restricted to its first 384 rotated columns, solved as a "
           'linear span with a lattice quadrature rule (`lat64`). **NM-ROM head, k=16**: the nonlinear 16-unknown head only.',
           '- **POD-LSPG, k=16**: least-squares Petrov–Galerkin on a 16-mode POD basis of training snapshots at the mesh. '
           '**quadratic manifold, r=16**: POD-16 plus a quadratic correction fitted on the same snapshots (ridge chosen on held-out training trajectories).',
           '- **ρ**: relative error of the quadrature rule\'s advection projection on held-out states; bar 0.116.',
           '- **basis reproduction**: the POD/QM bases were rebuilt in-job from training data by the parent code and compared with the parent job\'s record.']
    Path(a.out_md).write_text('\n'.join(md) + '\n')
    print(a.out_md, a.out_json)


if __name__ == '__main__':
    main()
