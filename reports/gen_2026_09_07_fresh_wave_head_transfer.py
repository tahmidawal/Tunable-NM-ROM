"""Build the canonical fresh-wave report from immutable result JSONs.

Forward all table-builder arguments, including repeated --result/--refinement,
--verification-result, --reviewed, and --out. The adjacent publication manifest
records the exact inputs and source hashes used for the published artifact.
"""
from pathlib import Path
import runpy
import sys


def main():
    builder = Path(__file__).with_name('fresh_wave_report_tables.py')
    runpy.run_path(str(builder), run_name='__main__')
    out = Path(sys.argv[sys.argv.index('--out')+1])
    text = out.read_text()
    figures = '''## Figures

![Median and worst trajectory-maximum errors](figures/2026-09-07-fresh-wave-results.png)

Dots show medians of trajectory maxima and lines end at the worst case; they are not confidence intervals. Open dots mark original runs that fail the time-step check. Full-bank linear baselines have a larger state dimension than the nonlinear heads.

![Error evolution over time](figures/2026-09-07-fresh-wave-error-trajectories.png)

These curves pool the same validation trajectories across optimizer repeats. Shading shows descriptive interquartile spread, not uncertainty from independent data replications. Reflective quadratic primary curves retain their time-step qualification; the separately reported continuation does not replace them.

![Fresh reference wave fields](figures/2026-09-07-fresh-wave-reference.png)

These are independently verified reference fields for a predeclared off-center control. Each panel has its own amplitude scale; this figure does not show decoder predictions.

'''
    assert text.count('## Glossary')==1
    out.write_text(text.replace('## Glossary', figures+'## Glossary'))


if __name__=='__main__':
    main()
