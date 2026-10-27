"""Append a generated refinement section to the current user-edited Research Update.

Read the canonical saved slide sources afresh; preserve every existing slide byte and its
relative order. New slide numbers follow the old slides. Output stays in this worktree
until the independent publication audit and explicit copy-back step.
"""
from __future__ import annotations

import base64
import html
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from inventory import ROOT, read, sha, write

HERE = Path(__file__).resolve().parent
DEST = HERE.parents[0] / '2026-10-23-research-update'
SOURCE = ROOT / 'reports/2026-10-23-research-update'
GENERATED_IDS = ['b3d-refine-evidence', 'b3d-refine-solver', 'b3d-refine-time', 'b3d-refine-pod',
                 'b3d-refine-frontier', 'b3d-refine-mesh', 'b3d-refine-open']


def picture(name):
    return 'data:image/png;base64,' + base64.b64encode((HERE / 'figs' / name).read_bytes()).decode()


def slide(name, title, body, number):
    return f'''<section id="{name}" style="background:#fff;color:#172331;padding:72px 92px;font-family:Arial,sans-serif;position:relative">
<p style="font-size:24px;letter-spacing:2px;color:#476075;margin-bottom:20px">RESEARCH UPDATE · BURGERS 3D REFINEMENT</p>
<h1 style="font-size:58px;line-height:1.12;margin-bottom:36px">{title}</h1>
{body}
<p style="position:absolute;left:92px;right:92px;bottom:36px;font-size:22px;color:#526574;border-top:1px solid #d7dfe5;padding-top:14px">Conditional E0 ST estimates · Development/validation evidence · {number}</p>
</section>'''


def main():
    values = read(HERE / 'VALUES.json')
    frontier = values.get('full_validation_frontier')
    qualified_hr = [] if not frontier else [sid for sid, v in frontier['subjects'].items()
        if v['method'] == 'POD' and v['registered_qualified'] and
        v['qualification']['method_specific_extra_masks'].get('hyper_reduced') is True]
    DEST.mkdir(exist_ok=True)
    same_destination = DEST.resolve() == SOURCE.resolve()
    if not same_destination:
        shutil.copytree(SOURCE / 'project', DEST / 'project', dirs_exist_ok=True)
        for name in ['build_local.py', 'README.md']:
            shutil.copy2(SOURCE / name, DEST / name)
    original = {}
    original_pdf_backup = None
    for p in [SOURCE / 'project/deck.json', *sorted((SOURCE / 'project/slides').glob('*.html'))]:
        if p.stem not in GENERATED_IDS:
            original[str(p.relative_to(SOURCE))] = sha(p)
    if (SOURCE / 'Research-Update.pdf').is_file():
        snapshot = HERE / 'research-update-original'
        snapshot.mkdir(exist_ok=True)
        pdf_digest = sha(SOURCE / 'Research-Update.pdf')
        original_pdf_backup = snapshot / f'Research-Update-{pdf_digest[:12]}.pdf'
        if not original_pdf_backup.exists():
            shutil.copy2(SOURCE / 'Research-Update.pdf', original_pdf_backup)
        assert sha(original_pdf_backup) == pdf_digest
        original['Research-Update.pdf'] = pdf_digest
    deck = read(SOURCE / 'project/deck.json')
    # Repeatable from both the lane and canonical report: replace our generated section.
    # Other saved source slides and their order remain authoritative.
    original_order = [name for name in deck['order'] if name not in GENERATED_IDS]
    deck = json.loads(json.dumps(deck))
    deck['order'] = original_order.copy()
    deck['sections'].pop('b3d-refinement', None)
    count = len(deck['order'])
    bm, bw = values['best_median'], values['best_worst']
    text = f'''<div style="display:grid;grid-template-columns:1fr 1fr;gap:46px">
<div><p style="font-size:33px;line-height:1.5">The strongest median and worst-case settings differ.</p>
<table style="font-size:28px;margin-top:28px"><tr><th>Qualified historical role</th><th>Median</th><th>Worst</th></tr>
<tr><td>Best median</td><td>{bm['median']*100:.3g}%</td><td>{bm['worst']*100:.3g}%</td></tr>
<tr><td>Best worst case</td><td>{bw['median']*100:.3g}%</td><td>{bw['worst']*100:.3g}%</td></tr></table>
<p style="font-size:27px;line-height:1.5;margin-top:30px">The new inventory keeps references, exact cases, failed checks and later quadrature rounds separate.</p></div>
<div><img alt="Exact historical timed workload" src="{picture('timed_frontier.png')}" style="width:100%;margin-top:25px">
<p style="font-size:25px;line-height:1.45;margin-top:22px">The cost plot is re-scored on its exact saved timed cases {html.escape(str(values['timing_population']))}. It is a subset view; full-validation accuracy is reported separately.</p></div></div>'''
    output = [('b3d-refine-evidence', 'The recovered frontier needs explicit populations', text)]
    solver = read(HERE / 'data/solver-pilot-export.json')
    observations = [o for o in solver.get('observations', []) if o['source']['job_id'] == 6525183 and
                    o['key'].startswith('W1024|') and o['key'].endswith('|case30')]
    assert len(observations) == 5 and len({o['key'] for o in observations}) == 5
    pilot_bank, _, pilot_case = observations[0]['key'].split('|')
    pilot_case = pilot_case.removeprefix('case')
    pilot_job = observations[0]['source']['job_id']
    rows = ''.join(f"<tr><td>{html.escape(o['key'].split('|')[1])}</td><td>{o['fresh_nonstationary_steps']}</td><td>{o['score']['e']*100:.5g}%</td></tr>" for o in observations)
    text = f'''<p style="font-size:32px;line-height:1.45;margin-bottom:25px">{pilot_bank} · validation case {pilot_case} · pilot job {pilot_job}: a controlled replay repairs the nonlinear certificate while the worst physical error persists.</p>
<table style="font-size:31px"><tr><th>Policy on the diagnostic case</th><th>Uncertified steps</th><th>Maximum E0 error</th></tr>{rows}</table>
<p style="font-size:29px;line-height:1.5;margin-top:30px">Extra nonlinear work and more frequent matrix refresh address the demonstrated convergence gap. A larger trust radius alone does not. Fresh restarts are labelled separately from the original cached solver.</p>
<p style="font-size:27px;line-height:1.45;margin-top:24px;color:#6d4e2d">Selected difficult-case diagnostics. This table does not establish whole-population accuracy or speed gains.</p>'''
    output.append(('b3d-refine-solver', 'A convergence repair with unchanged physical error', text))
    time_diagnostic = values.get('time_refinement_diagnostic')
    if time_diagnostic:
        coarse, fine = time_diagnostic['rows'][0], time_diagnostic['rows'][-1]
        text = f'''<div style="display:grid;grid-template-columns:1.5fr 1fr;gap:45px">
<div><img alt="Time-step accuracy and nonlinear work tradeoff" src="{picture('time_refinement_tradeoff.png')}" style="width:100%;margin-top:30px">
<p style="font-size:24px;line-height:1.4;margin-top:15px">Fixed bank/test/quadrature setting; case ids {html.escape(str(coarse['cases']))}.</p></div>
<div style="font-size:31px;line-height:1.45"><p>On the fixed diagnostic cases, median error falls from {100*coarse['median']:.4g}% to {100*fine['median']:.4g}%, while worst error rises from {100*coarse['worst']:.4g}% to {100*fine['worst']:.4g}%.</p>
<p style="margin-top:25px">Both endpoints pass the subset convergence checks. A lower median does not guarantee a better worst case.</p>
<p style="margin-top:25px">The coarse and fine endpoints were selected after these diagnostics, before the full-validation sweep. The intermediate setting remains a valid measured tradeoff.</p></div></div>
<p style="font-size:26px;line-height:1.4;margin-top:30px;color:#6d4e2d">Error cancellation is a hypothesis. The inherited bank uses analytic advection and grid-dependent diffusion; it does not share POD's discrete residual.</p>'''
        output.append(('b3d-refine-time', 'Time refinement improves the median and worsens outliers', text))
    pod = read(HERE / 'data/pod-pilot-export.json')
    old = values['historical_POD_scope']
    q = pod.get('qualification', {}).get('whole_residual', {})
    pod_final = values.get('new_lanes', {}).get('pod', pod)
    failed_fit = pod_final.get('failed_support_extension')
    if failed_fit:
        extension_status = (f"The single larger-support constructor stopped at {failed_fit['terminal_support']} "
            f"of {failed_fit['support_target']} points and failed its optimality and banned-point checks. "
            'No larger-support rollout ran. Qualified unsampled POD remains available; the qualified HR comparison stays open.')
        if qualified_hr:
            extension_status = (f"The new support constructor failed at {failed_fit['terminal_support']} "
                f"of {failed_fit['support_target']} points; no rollout of that rejected rule ran. "
                'The separately tuned existing support qualifies on complete validation and is included in the matched frontier.')
    else:
        extension_status = ('The single larger support is being evaluated using training data only. '
                            'Unsampled POD is the fallback comparator; this is not evidence of a bank advantage.')
    rows = ''.join(f"<tr><td>{rule}</td><td>{v['delta_g_max']:.5g}</td><td>{v['full_nonstat']}/{v['steps']}</td><td>Fails new contract</td></tr>" for rule,v in q.items())
    a3 = pod_final.get('a3_replay')
    improved_status = ''
    if a3:
        improved_status = (f"The existing k16 rule also fails the field-agreement check under tuned "
            f"{a3['selected']['scheme']} (maximum gap {max(a3['common_field_distances'].values()):.5g}); "
            'its scalar reference-error changes remain within uncertainty. No full-validation HR release follows.')
    text = f'''<p style="font-size:34px;line-height:1.5;margin-bottom:28px">Under the historical integration policy, the saved POD supports solve their sampled equations but fail the new complete-objective check.</p>
<table style="font-size:29px"><tr><th>Frozen support</th><th>Gradient discrepancy</th><th>Full-objective uncertified steps</th><th>Outcome</th></tr>{rows}</table>
<p style="font-size:27px;line-height:1.45;margin-top:25px">The historical {old['dominated_all']}/{old['total']} domination headline pooled {old['diagnostic']} diagnostic and {old['qualified']} qualified bank rows. Within the qualified subset: {old['dominated_qualified']}/{old['qualified']} dominated, {old['withheld_qualified']} withheld. Old POD hyper-reduction convergence was not gated.</p>
<p style="font-size:26px;line-height:1.4;margin-top:20px">{extension_status} The diagnostic failures above use the new conservative rule and do not establish a bank advantage.</p>'''
    if improved_status:
        text += f'<p style="font-size:25px;line-height:1.4;margin-top:18px;color:#6d4e2d">{improved_status}</p>'
    output.append(('b3d-refine-pod', 'POD receives the same qualification scrutiny', text))
    if frontier:
        text = f'''<img alt="Same-allocation complete-validation accuracy versus descriptor-total query cost" src="{picture('full64_descriptor_frontier.png')}" style="display:block;width:94%;max-height:640px;object-fit:contain;margin:0 auto">
<p style="font-size:27px;line-height:1.4;margin-top:18px">Input preparation and synchronized device outputs are included; host output transfer and scoring are excluded. Filled points pass registered method-specific gates; open points remain visible. Joint accuracy matches and the finite tested set are documented in the report.</p>'''
        output.append(('b3d-refine-frontier', 'The complete validation comparison uses one GPU allocation', text))
    text = f'''<div style="display:grid;grid-template-columns:1.2fr 1fr;gap:40px">
<img alt="Quadrature successor convergence" src="{picture('quadrature_convergence.png')}" style="width:100%;margin-top:30px">
<div style="font-size:30px;line-height:1.5"><p>Completed quadrature trajectories are reused. The missing finer-rule check closes a specific evidence gap.</p>
<p style="margin-top:28px">A qualified refined frontier needs whole-population solver and quadrature checks, same-invocation timing, and matched output requirements.</p>
<p style="margin-top:28px">Reduced evolution, input preparation, output decoding, offline setup and memory remain separate costs.</p>
<p style="margin-top:28px;color:#6d4e2d">No new general advantage over POD or the fastest tested full solver is established by the diagnostic pilots.</p></div></div>'''
    open_title = 'Close missing evidence before extending the frontier'
    if frontier:
        hr_status = ('A tuned existing POD sampling rule passes complete-validation admission and is included.'
                     if qualified_hr else 'Qualified hyper-reduced POD remains open within the tested settings.')
        open_title = 'What is established, and what remains open'
        text = f'''<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:45px;font-size:31px;line-height:1.5">
<div><h2 style="font-size:37px;color:#27659a;margin-bottom:24px">Measured comparisons</h2>
<p>Earlier bank ROMs, tuned POD and full solvers have separate matched comparisons.</p>
<p style="margin-top:26px">Full query, initialisation, evolution, decoding, setup and memory are reported separately. Split phase calls do not replace the full-query timer.</p></div>
<div><h2 style="font-size:37px;color:#9a5b27;margin-bottom:24px">Failures retained</h2>
<p>The new support constructor failed its checks; no rollout of that rejected rule ran.</p>
<p style="margin-top:26px">A repaired convergence certificate does not guarantee lower physical error. Median and worst errors can move in opposite directions.</p></div>
<div><h2 style="font-size:37px;color:#3a7656;margin-bottom:24px">Scope and open work</h2>
<p>{hr_status}</p>
<p style="margin-top:26px">The finite exposed-validation set does not establish a universal fastest method. Sealed replication remains closed. A solver certificate does not establish spatial consistency.</p></div></div>'''
    output.append(('b3d-refine-open', open_title, text))
    for i, (name, title, body) in enumerate(output, start=count + 1):
        (DEST / 'project/slides' / (name + '.html')).write_text(slide(name, title, body, i))
        deck['order'].append(name)
    deck['sections']['b3d-refinement'] = dict(description='Burgers 3D refinement', start=output[0][0])
    write(DEST / 'project/deck.json', deck)
    marker = '## Burgers 3D refinement extension, 2026-10-26'
    if marker not in (DEST / 'README.md').read_text():
        with (DEST / 'README.md').open('a') as f:
            f.write(f'\n\n{marker}\n\n'
                f'The {count} pre-existing saved slides retain their source bytes and relative order. '
                f'{len(output)} generated refinement slides are appended. '
                'Rebuild with `../2026-10-26-burgers3d-refinement/update_research.py` after generating '
                'that report\'s VALUES.json. The preservation manifest records original hashes and the '
                'saved original PDF; new scientific numbers are taken from generated report inputs.\n')
    else:
        readme = (DEST / 'README.md').read_text()
        readme = re.sub(r'The \d+ pre-existing saved slides retain their source bytes and relative order\. '
                        r'\d+ generated refinement slides are appended\.',
                        f'The {count} pre-existing saved slides retain their source bytes and relative order. '
                        f'{len(output)} generated refinement slides are appended.', readme)
        (DEST / 'README.md').write_text(readme)
    # Execute the preserved local viewer builder against current canonical image assets.
    builder = (DEST / 'build_local.py').read_text().replace(
        'root=Path(__file__).resolve().parents[2]', f'root=Path({str(ROOT)!r})')
    exec(compile(builder, str(DEST / 'build_local.py'), 'exec'), {'__file__': str(DEST / 'build_local.py')})
    page = (DEST / 'research-update-local.html').read_text()
    page = re.sub(r'<script>.*?</script>', '', page, flags=re.S)
    css = '''<style>@page{size:1920px 1080px;margin:0}body{background:white}header{display:none!important}
#viewport{width:1920px!important;height:auto!important;overflow:visible!important}
#slides{transform:none!important;width:1920px!important;height:auto!important}
#slides>section,#slides>section[hidden]{display:block!important;break-after:page;page-break-after:always;width:1920px;height:1080px;overflow:hidden}
#slides>section:last-child{break-after:auto;page-break-after:auto}</style>'''
    page = page.replace('</head>', css + '</head>')
    print_path = DEST / 'research-update-print.html'
    print_path.write_text(page)
    with tempfile.TemporaryDirectory(prefix='b3d-pdf-') as tmp:
        subprocess.run(['/usr/bin/google-chrome', '--headless', '--no-sandbox', '--disable-gpu',
                        '--no-pdf-header-footer', '--user-data-dir=' + tmp,
                        '--print-to-pdf=' + str(DEST / 'Research-Update.pdf'), print_path.as_uri()],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=90)
    # Original slide bytes and relative order are the preservation contract.
    assert [k for k in deck['order'] if k in original_order] == original_order
    for rel, digest in original.items():
        if rel.startswith('project/slides/'):
            assert sha(DEST / rel) == digest
    write(HERE / 'RESEARCH-UPDATE-PRESERVATION.json', dict(source_root=str(SOURCE), original=original,
        preserved_slide_count=count, added_slide_count=len(output), generated_pdf_sha256=sha(DEST / 'Research-Update.pdf'),
        generated_pdf=str(DEST / 'Research-Update.pdf'), original_pdf_backup=None if original_pdf_backup is None else str(original_pdf_backup)))
    print(DEST / 'Research-Update.pdf')


if __name__ == '__main__':
    main()
