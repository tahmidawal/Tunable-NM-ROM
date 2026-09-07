"""Build the canonical fresh-wave report from immutable result JSONs.

Forward all table-builder arguments, including repeated --result/--refinement,
--verification-result, --reviewed, and --out. The adjacent publication manifest
records the exact inputs and source hashes used for the published artifact.
"""
from pathlib import Path
import json
import runpy
import sys


def main():
    builder = Path(__file__).with_name('fresh_wave_report_tables.py')
    runpy.run_path(str(builder), run_name='__main__')
    out = Path(sys.argv[sys.argv.index('--out')+1])
    text = out.read_text()
    field_path = out.parent/'figures/2026-09-07-fresh-wave-fields.json'
    field_section = ''
    if field_path.exists():
        fields = json.loads(field_path.read_text())
        rows = []
        for boundary, meta in fields['boundaries'].items():
            for i, t in enumerate(meta['times']):
                m = meta['metrics']['mlp']
                rows.append(f"| {boundary} | {meta['case']} | {meta['optimizer_seed']} | {t:g} | {100*m['displacement'][i]:.4f}% | {100*m['current_field_displacement'][i]:.4f}% | {100*m['energy_state'][i]:.4f}% |")
        field_section = r'''## Actual spatial fields and the normalization distinction

The earlier campaign tables use fixed initial-state scales. For displacement, $e_{u,0}(t)=\|u_{\mathrm{ROM}}(t)-u_{\mathrm{ref}}(t)\|_M/\|u_{\mathrm{ref}}(0)\|_M$. A different, instantaneous relative error is $e_{u,t}(t)=\|u_{\mathrm{ROM}}(t)-u_{\mathrm{ref}}(t)\|_M/\|u_{\mathrm{ref}}(t)\|_M$. The latter can be large when an absorbing reference has decayed, even though the absolute difference and initial-normalized error are small. A reference norm that vanishes makes instantaneous relative error undefined; the velocity figures label this explicitly.

These additional diagnostics are computed from the checked saved fields. They do not replace the predeclared campaign metric or its verdicts. In particular, passing the original full-bank linear ceiling does not establish accurate late-time relative reconstruction of a nearly vanished field.

The spatial comparison uses the same first validation case and first optimizer repeat for both boundaries, selected by index before inspecting errors. It is an illustration rather than a cohort summary. No PDE solve, fitting, training, or final-test evaluation was added. The table gives MLP errors at the saved snapshot times:

| Boundary | Case | Optimizer seed | Time | Displacement error / initial norm | Displacement error / current norm | State error / initial energy norm |
|---|---|---|---|---|---|---|
'''+ '\n'.join(rows)+r'''

![Reflective spatial displacement comparison](figures/2026-09-07-fresh-wave-reflective-displacement-fields.png)

![Absorbing spatial displacement comparison](figures/2026-09-07-fresh-wave-absorbing-displacement-fields.png)

Rows show reference, unrestricted learned-bank linear evolution, MLP evolution, and absolute MLP difference. Field colors share limits within each time column; limits can change across time. The error range is at least the field amplitude and is expanded when needed to avoid clipping, so small differences are not magnified by an independently stretched color map. The lower row is an absolute difference, never division by individual field values.

Corresponding velocity fields: [reflective](figures/2026-09-07-fresh-wave-reflective-velocity-fields.png), [absorbing](figures/2026-09-07-fresh-wave-absorbing-velocity-fields.png). The labels show both fixed initial-state scaling and normalization by the current velocity norm.

'''
    figures = '''## Figures

![Median and worst trajectory-maximum errors](figures/2026-09-07-fresh-wave-results.png)

Dots show medians of trajectory maxima and lines end at the worst case; they are not confidence intervals. Open dots mark original runs that fail the time-step check. Full-bank linear baselines have a larger state dimension than the nonlinear heads.

![Error evolution over time](figures/2026-09-07-fresh-wave-error-trajectories.png)

These curves pool the same validation trajectories across optimizer repeats. Shading shows descriptive interquartile spread, not uncertainty from independent data replications. Reflective quadratic primary curves retain their time-step qualification; the separately reported continuation does not replace them.

![Fresh reference wave fields](figures/2026-09-07-fresh-wave-reference.png)

These are independently verified reference fields for a predeclared off-center control. Each panel has its own amplitude scale; this figure does not show decoder predictions.

'''
    assert text.count('## Glossary')==1
    out.write_text(text.replace('## Glossary', field_section+figures+'## Glossary')+'\n- **Interquartile spread / confidence interval:** the middle half of plotted case/repeat values, and an interval describing statistical estimation uncertainty; the figures show only the former.\n- **Fixed-scale / instantaneous relative error:** error divided by an initial reference scale, or by the reference magnitude at the current time. The two answer different accuracy questions when the reference decays.\n')


if __name__=='__main__':
    main()
