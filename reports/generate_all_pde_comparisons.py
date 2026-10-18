"""Build one Markdown containing the reported four-PDE comparisons.

New summary/control tables come from the audited run JSONs. Existing generated
reports are included in full, with their protocol and chronology preserved.
No simulation, fitting, timing or model selection occurs here.
"""
import hashlib
import json
from pathlib import Path
import re
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT/'reports'
DEST = HERE/'2026-09-10-all-pde-speed-and-accuracy.md'
REPORTS = {
    'historical': HERE/'2026-09-10-historical-burgers-poisson-replay.md',
    'waves': HERE/'2026-09-10-fresh-wave-device-comparison.md',
    'earlier': HERE/'2026-09-07-multiresolution-pilots.md',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(map(str, row))+' |' for row in rows]])


def close(actual, expected):
    assert abs(actual-expected) <= 2e-11*max(1., abs(expected)), (actual, expected)


def pct(value):
    return f'{100*value:.6g}'


def ms(value):
    return f'{value:.6g}'


def source_body(text, heading, glossary_title):
    """Preserve generated source content and move its glossary to the end."""
    body, glossary = text.rsplit('\n## '+glossary_title+'\n', 1)
    body = body.split('\n', 1)[1].lstrip()
    inside_fence = False
    lines = []
    for line in body.splitlines():
        if line.startswith('```'):
            inside_fence = not inside_fence
        if not inside_fence and re.match(r'^#{1,5} ', line):
            line = '#'+line
        lines.append(line)
    return f'## {heading}\n\n'+'\n'.join(lines), glossary.strip()


def main():
    inputs = {}
    def remember(path):
        inputs[str(path.relative_to(ROOT))] = dict(sha256=digest(path), bytes=path.stat().st_size)
    def read(path):
        remember(path)
        return json.loads(path.read_text())
    bp = read(REPORTS['historical'].with_suffix('.json'))
    wave = read(REPORTS['waves'].with_suffix('.json'))
    earlier = read(REPORTS['earlier'].with_suffix('.json'))
    source_text = {}
    for key, path in REPORTS.items():
        remember(path)
        source_text[key] = path.read_text()
    # These are run/owner/root-review JSONs already independently audited.
    heat = {}
    for name in ('heat_transfer', 'heat_transfer_audit', 'heat_transfer_review'):
        record = earlier['artifacts'][name]
        path = ROOT/record['path']
        heat[name] = read(path)
        assert digest(path) == record['sha256']
    hp, ha, hr = (heat[k] for k in ('heat_transfer', 'heat_transfer_audit', 'heat_transfer_review'))
    assert hp['complete'] and ha['passed'] and not hp['final_cohort_opened']
    assert ha['results_sha256'] == hr['source_json_sha256'] == earlier['artifacts']['heat_transfer']['sha256']
    assert wave['reference_acceptance'] and wave['audit']['timed_invocations'] == wave['config']['expected_timed_invocations']
    wave_native_path = next(ROOT/p for p in wave['source_sha256'] if p.endswith('/out/pilot/result.json'))
    wn = read(wave_native_path)
    assert digest(wave_native_path) == wave['source_sha256'][str(wave_native_path.relative_to(ROOT))]
    # JSON provenance is checked without repeating the expensive full-field audits.
    native_b = {}
    native_p = {}
    for rows, dest in ((bp['burgers'], native_b), (bp['poisson'], native_p)):
        for r in rows:
            if r['N'] in dest:
                continue
            path = ROOT/r['path']
            dest[r['N']] = read(path)
            assert digest(path) == bp['sources'][r['path']]['sha256']

    heat_rows = []
    target = hp['config']['accuracy_targets'][1]
    for selection in ha['selections']:
        if (selection['cohort'], selection['norm'], selection['target']) != ('union', 'full', target):
            continue
        n = selection['intervals']
        groups = {label: next(g for g in ha['groups'] if
            (g['cohort'], g['intervals'], g['method']) == ('union', n, selection[label]))
            for label in ('rom', 'fom')}
        for label, g in groups.items():
            reviewed = [r for r in hr['rows'] if r['intervals'] == n and r['method'] == selection[label]]
            assert {r['case'] for r in reviewed} == set(g['cases'])
            close(max(r['worst_full_grid_error'] for r in reviewed), g['full_error'])
            close(median(g['case_times'].values()), selection[label+'_seconds'])
        heat_rows.append(dict(intervals=n, selection=selection, rom=groups['rom'], fom=groups['fom']))
    heat_rows.sort(key=lambda r: r['intervals'])

    summary = []
    b = max(bp['burgers'], key=lambda r: r['N'])
    summary.append(['Burgers 2D', f"{b['N']} nodes", 'GPU / same grid', ms(b['rom_ms']), ms(b['fom_ms']),
                    f"{b['ratio']:.6g}", pct(b['worst_error']), pct(b['fom_worst_error']), 'Original Newton FOM'])
    p = max((r for r in bp['poisson'] if r['cohort'] == 'heldout_seed0' and r['tau'] == .001), key=lambda r: r['N'])
    cg = next(r for r in native_p[p['N']]['rows'] if r['cohort'] == p['cohort'] and
              r['method'] == 'cg' and r['fom_tol'] == p['cg_tol'])
    summary.append(['Poisson 2D', f"{p['N']} nodes", 'GPU / same grid', ms(p['rom_ms']), ms(p['cg_ms']),
                    f"{p['ratio']:.6g}", pct(p['worst_error']), pct(cg['err_rel_l2_max']), 'Original CG'])
    direct = next(r for r in native_p[p['N']]['rows'] if r['cohort'] == p['cohort'] and r['method'] == 'spectral_dense')
    summary.append(['Poisson 2D', f"{p['N']} nodes", 'GPU / same grid', ms(p['rom_ms']), ms(p['direct_ms']),
                    f"{p['direct_ms']/p['rom_ms']:.6g}", pct(p['worst_error']), pct(direct['err_rel_l2_max']), 'Direct spectral control'])
    for bc, name in (('dirichlet', 'Reflective wave 2D'), ('absorbing', 'Absorbing wave 2D')):
        r = max((r for r in wave['rows'] if r['boundary'] == bc and r['method'] == 'MLP32'), key=lambda r: r['intervals'])
        f = next(f for f in wave['rows'] if f['boundary'] == bc and f['intervals'] == r['intervals'] and f['method'].endswith('FOM'))
        summary.append([name, f"{r['intervals']} intervals", 'GPU / same grid', ms(r['query_ms']), ms(f['query_ms']),
                        f"{r['fom_over_rom']:.6g}", pct(r['errors']['displacement']['current_relative']['worst']),
                        pct(f['errors']['displacement']['current_relative']['worst']), f['method']])
    h = heat_rows[-1]; s = h['selection']
    summary.append(['Heat 2D', f"{h['intervals']} intervals", 'Host / FOM envelope', ms(1000*s['rom_seconds']),
                    ms(1000*s['fom_seconds']), f"{s['paired_fom_over_rom']:.6g}", pct(h['rom']['full_error']),
                    pct(h['fom']['full_error']), s['fom']])

    contents = [
        '- [Latest results at a glance](#latest-results-at-a-glance)',
        '- [How to compare the tables](#how-to-compare-the-tables)',
        '- [Latest heat resolution ladder](#latest-heat-resolution-ladder)',
        '- [Latest Burgers and Poisson GPU comparisons](#latest-burgers-and-poisson-gpu-comparisons)',
        '- [All paired Burgers replay controls](#all-paired-burgers-replay-controls)',
        '- [All Poisson replay configurations](#all-poisson-replay-configurations)',
        '- [Latest reflective and absorbing wave GPU comparisons](#latest-reflective-and-absorbing-wave-gpu-comparisons)',
        '- [Wave accuracy-only controls](#wave-accuracy-only-controls)',
        '- [Earlier multiresolution results and development controls](#earlier-multiresolution-results-and-development-controls)',
        '- [Evidence and reproduction](#evidence-and-reproduction)',
        '- [Glossary](#glossary)',
    ]
    lines = ['# All PDE results: ROM versus FOM speed and accuracy', '',
        'This file consolidates the reported comparisons for Burgers 2D, Poisson 2D, heat 2D, and fresh reflective/absorbing waves 2D, including resolution ladders, tested controls and earlier development results. Measurements are audited but remain provisional development evidence; independent final paper cohorts are unopened.', '',
        'The latest comparisons appear first. The earlier multiresolution study is reproduced later with its original protocol and limitations; its headings referring to “latest” apply only within that earlier study. Discarded pre-reset wave experiments and the older ViT + CP comparator are excluded.', '',
        *contents, '',
        '## Latest results at a glance', '',
        'Finest tested mesh for each latest comparison. Worst errors below are current-relative field errors; wave entries here use displacement only, with velocity and energy-state errors in the detailed section. Heat uses its complete-grid physical reference; the other rows use their stated same-grid reference. Rows are not a shared cross-PDE accuracy or hardware ranking.', '',
        table(['Problem', 'Mesh per axis', 'Timing / baseline scope', 'ROM ms', 'FOM ms', 'FOM/ROM',
               'ROM worst error (%)', 'FOM worst error (%)', 'Named FOM'], summary), '',
        'A ratio above one favors ROM runtime against the named FOM. Burgers and Poisson reproduce gains against their original named baselines; the direct Poisson control remains faster. Neither tested nonlinear wave head beats its same-grid FOM. Heat has not been replayed under the newer GPU-resident contract and retains the earlier charged host-output/FOM-envelope comparison.', '',
        '## How to compare the tables', '',
        '- **Timing boundary:** GPU rows start with ready GPU fields and finish with ready GPU fields. Initialization, projection, evolution/solution and full field reconstruction are charged; host transfers, compilation and offline setup are excluded. Heat charges host input handling, transfers, interpolation and full host output.',
        '- **Baseline:** same-grid rows use the named solver on the ROM mesh. The heat FOM envelope may solve on a coarser mesh and interpolate outputs, charging that work. Its selected solver name gives the solver mesh.',
        '- **Ratio:** latest Burgers/Poisson/wave ratios divide cohort median times. Heat reports the median of paired case ratios; it need not equal the ratio of the two displayed medians. The earlier-study ratios retain their declared definitions.',
        '- **Accuracy:** every table labels current/initial normalization, spatial reference and aggregation. A small error relative to the initial wave can coexist with poor prediction of the remaining absorbing wave. A low weak residual or a passed numerical check does not certify the physical-error target.',
        '- **Resolution:** Burgers/Poisson replay sizes count nodes including boundaries; wave/heat sizes count intervals. Historical checkpoints can differ with mesh, whereas the latest wave/heat studies transfer frozen learned weights.',
        '- **Evidence:** timing and accuracy use the same solver invocations. Repetition arrays, failed/stalled/censored solves, outliers, reference checks and checkpoint provenance remain visible in the included reports or their raw artifacts. Compare hardware timings within each allocation.', '',
        '## Latest heat resolution ladder', '',
        f"These are the existing union-cohort selections at the {pct(target)}% full-grid current-relative target, with an empirical reference allowance, across {len(hp['settings']['requested_intervals'])} requested meshes and {len(heat_rows[0]['rom']['cases'])} development inputs. Both ROM and selected FOM accuracy are shown here. This is the host-to-host comparison; the GPU-resident heat replay remains unmeasured.", '',
        table(['Output intervals', 'Selected ROM', 'Selected FOM', 'ROM / FOM ms', 'Paired FOM/ROM',
               'ROM / FOM worst error (%)', 'ROM / FOM timing outliers'],
              [[r['intervals'], r['selection']['rom'], r['selection']['fom'],
                f"{ms(1000*r['selection']['rom_seconds'])} / {ms(1000*r['selection']['fom_seconds'])}",
                f"{r['selection']['paired_fom_over_rom']:.6g}",
                f"{pct(r['rom']['full_error'])} / {pct(r['fom']['full_error'])}",
                f"{r['rom']['timing_outliers']} / {r['fom']['timing_outliers']}"] for r in heat_rows]), '',
        'The later heat section retains every endpoint/cohort result, initial-fit and evolution checks, transfer accounting and reference validation. These selected rows do not imply the same endpoint wins on every mesh.', '',
    ]
    body_bp, glossary_bp = source_body(source_text['historical'], 'Latest Burgers and Poisson GPU comparisons', 'Glossary')
    lines += [body_bp, '', '## All paired Burgers replay controls', '',
        'These rows include every retained ROM arm with its own paired, mean-accuracy-selected FOM. Error summaries below are recomputed from the captured paired outputs. The worst column takes the maximum over every trajectory and output time; it is not the native maximum of trajectory means. Stalled exits remain visible. These controls share the historical decoder family, including its sign-dependent upwinding limitation described above.', '']
    bcontrols = []
    for n, data in sorted(native_b.items()):
        for arm, match in data['matched']['arms'].items():
            pair = match['paired']; var = data['variants'][arm]
            a = [e for row in pair['per_traj'] for e in row['timed_output_errors']['a']['per_time']]
            b = [e for row in pair['per_traj'] for e in row['timed_output_errors']['b']['per_time']]
            close(pair['rom_ms'], median(median(r['a_raw_ms']) for r in pair['per_traj']))
            close(pair['fom_ms'], median(median(r['b_raw_ms']) for r in pair['per_traj']))
            close(mean(a), var['err_traj_rel_mean'])
            close(pair['speedup'], pair['fom_ms']/pair['rom_ms'])
            bcontrols.append([n, arm, f"{ms(pair['rom_ms'])} / {ms(pair['fom_ms'])}", f"{pair['speedup']:.6g}",
                f"{pct(mean(a))} / {pct(max(a))}", f"{pct(mean(b))} / {pct(max(b))}",
                ', '.join(f'{k}: {v}' for k,v in var['stop_reasons'].items()), var['n_blowups']])
    lines += [table(['Nodes', 'ROM arm', 'ROM / FOM ms', 'FOM/ROM', 'ROM mean / worst error (%)',
                     'FOM mean / worst error (%)', 'Step exits', 'Blowups'], bcontrols), '',
        '`full` evaluates the full-grid weak equation; `tensor` uses its preassembled polynomial contraction; `ex` uses fitted empirical quadrature; `ex_learned` uses the archived learned-node control. These are retained numerical controls for the same separable architecture. They do not reinstate the excluded older ViT + CP architecture.', '',
        '## All Poisson replay configurations', '',
        'Every recorded method, cohort, stopping threshold and CG tolerance from the completed replay is included below. Query times are medians across sources of their repetition medians. Error summaries are recomputed from the captured field-error arrays. The CG tolerance sweep remains visible alongside the selected rows above. Censored applies to nonlinear ROMs; it is not a physical-error guarantee. Full/QF methods differ in repeated weak-operator evaluation and share the frozen decoder.', '']
    pcontrols = []
    for n, data in sorted(native_p.items()):
        for r in data['rows']:
            times = [median(v)*1000 for v in r['time_raw_s'].values()]
            err = r['err_rel_l2_all']
            close(median(times), r['time_ms']); close(mean(err), r['err_rel_l2'])
            close(max(err), r['err_rel_l2_max']); close(median(err), r['err_rel_l2_median'])
            outliers = sum(v > 2*median(vv) for vv in r['time_raw_s'].values() for v in vv)
            threshold = '—' if r['tau'] is None else f"tau={r['tau']:g}"
            if r['fom_tol'] is not None:
                threshold = f"CG tol={r['fom_tol']:g}"
            pcontrols.append([n, r['cohort'], r['method'], threshold, ms(r['time_ms']),
                f"{pct(mean(err))} / {pct(median(err))} / {pct(max(err))}",
                '—' if r.get('censored_frac') is None else pct(r['censored_frac']), f"{outliers}/{r['total_repetitions']}"])
    lines += [table(['Nodes', 'Cohort', 'Method', 'Stop setting', 'Query ms', 'Mean / median / worst error (%)',
                     'Censored (%)', 'Timing outliers / repetitions'], pcontrols), '',
        'For this table, outliers exceed twice their own source repetition median; all repetitions remain included. `spectral_dense` is the direct spectral FOM, `cg` is the iterative FOM, `qf` is the preassembled ROM operator and `full` is its full-grid weak contraction.', '']
    body_wave, glossary_wave = source_body(source_text['waves'], 'Latest reflective and absorbing wave GPU comparisons', 'Glossary')
    lines += [body_wave, '', '## Wave accuracy-only controls', '',
        'The finer-step outputs below were produced as temporal checks. Their timing can include compilation and is excluded from speed comparisons. This table retains their actual physical errors in addition to the step-halving differences above. Errors are maxima across observation times for each individual case.', '']
    wave_controls = []
    for r in wn['accuracy_controls']:
        d = r['same_grid_discrepancy']
        refinement = next(x for x in wn['time_refinement'] if
            all(x[k] == r[k] for k in ('boundary', 'intervals', 'case', 'method')))
        wave_controls.append([r['boundary'], r['intervals'], r['case'], r['method'], r['setting'],
            pct(d['displacement']['max_initial_normalized']),
            ' / '.join(pct(d[k]['max_current_relative']) for k in ('displacement', 'velocity', 'energy_state')),
            r['completed'], refinement['passed']])
    lines += [table(['Boundary', 'Intervals', 'Case', 'Head', 'Step', 'Displacement initial worst (%)',
                     'Current displacement / velocity / energy worst (%)', 'Completed', 'Step check passed'], wave_controls), '',
        '## Earlier multiresolution results and development controls', '',
        '**Scope of this section:** this is the complete earlier generated development report, including its pilots, solver changes, training refinements, wider heat transfer and fresh-wave controls. Its timing protocols and model choices differ from the GPU replays above. Statements about the latest results within this section refer to that earlier study. All wave evidence here belongs to the verified post-reset lineage.', '']
    old_body, glossary_old = source_body(source_text['earlier'], 'Earlier-study report', 'Plain-language glossary')
    # Avoid a redundant section heading while retaining every paragraph/table.
    old_body = old_body.split('\n',1)[1].lstrip()
    old_body = old_body.replace('### Latest audited development results', '### Earlier-study final audited results', 1)
    lines += [old_body, '', '## Evidence and reproduction', '',
        'The source reports and their numerical provenance are preserved. This compilation checks the source JSON hashes, recomputes the additional control aggregates, and copies the existing generated comparison tables and qualifications. It does not rerun simulations or reinterpret a timing-only win as a physical-accuracy success.', '',
        table(['Source', 'Role', 'SHA256'], [[f'[{p.name}]({p.name})', role, inputs[str(p.relative_to(ROOT))]['sha256']]
            for role,p in REPORTS.items()]), '',
        f"Included additional configuration rows: {len(bcontrols)} paired Burgers controls, {len(pcontrols)} Poisson configurations and {len(wave_controls)} wave accuracy-only controls. The earlier report contributes {len(earlier['artifacts'])} source/audit artifacts and {len(earlier['followups'])} groups of follow-up findings. Complete per-case/time arrays and raw timing repetitions remain in the linked JSON/archive artifacts; the Markdown presents their reported aggregates and controls.", '',
        f"Regenerate from the repository root with `/home/tahmid/Dev/.venv/bin/python reports/{Path(__file__).name}`. The adjacent `{DEST.with_suffix('.manifest.json').name}` records every input hash and coverage checks. No new training, timing run or final-cohort evaluation was performed for this compilation.", '',
        '## Glossary', '',
        '### Shared reading conventions', '',
        '- **Latest / earlier:** the latest replay follows the requested GPU input/output contract; earlier results retain the protocol used when measured. They answer different comparison questions.',
        '- **Paired control / configuration:** a recorded method together with its numerical settings and comparison solver. Paired timings use the same physical case on the same allocation.',
        '- **Cohort / union:** a group of input cases / the combined original and fresh development groups. Repeated optimizer seeds are not independent physical cases.',
        '- **FOM envelope:** the cheapest tested full solver satisfying the declared accuracy selection, potentially using a coarser grid and charged interpolation.',
        '- **Worst error:** the maximum over the stated sources or trajectories and times. Median case and time-averaged means remain separate statistics.',
        '- **Timing outlier:** an unusually long repetition under the threshold stated beside its table. Counts do not remove samples from reported timings.',
        '- **Provisional:** audited development evidence with the stated limited cohorts and empirical reference checks, pending independent paper confirmation.', '',
        '### Burgers and Poisson terminology', '', glossary_bp, '',
        '### Wave terminology', '', glossary_wave, '',
        '### Earlier-study and heat terminology', '', glossary_old, '']
    content = '\n'.join(lines)
    # Check every source table survived the compilation, including all earlier findings.
    original_table_rows = sum(sum(line.startswith('|') for line in t.splitlines()) for t in source_text.values())
    combined_table_rows = sum(line.startswith('|') for line in content.splitlines())
    assert combined_table_rows >= original_table_rows+len(bcontrols)+len(pcontrols)+len(wave_controls)
    for text in source_text.values():
        for line in text.splitlines():
            if line.startswith('|'):
                assert line in content, line
    assert content.count('\n# ') == 0 and content.startswith('# ')
    # Resolve local links, including inherited plot/archive links, before publishing.
    link_count = 0
    for target in re.findall(r'\]\(([^)]+)\)', content):
        target = target.strip().strip('<>')
        if target.startswith(('http:', 'https:', '#')):
            continue
        target = target.split('#', 1)[0]
        assert (HERE/target).exists(), target
        link_count += 1
    headings = {re.sub(r'[^\w\s-]', '', m.group(1).lower()).replace(' ', '-')
                for m in re.finditer(r'^## (.+)$', content, re.MULTILINE)}
    for entry in contents:
        anchor = re.search(r'\(#([^)]*)\)', entry).group(1)
        assert anchor in headings, anchor
    remember(Path(__file__).resolve())
    DEST.write_text(content)
    manifest = dict(status='Generated compilation of audited provisional development results.',
        inputs=inputs, source_table_rows=original_table_rows, combined_table_rows=combined_table_rows,
        local_links_checked=link_count, summary_rows=len(summary), heat_selection_rows=len(heat_rows),
        burgers_paired_controls=len(bcontrols), poisson_configurations=len(pcontrols), wave_accuracy_controls=len(wave_controls),
        final_cohorts_opened=False, numerical_experiments_run=False, markdown_sha256=digest(DEST))
    DEST.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k != 'inputs'}, indent=2))


if __name__ == '__main__':
    main()
