"""Print a lab-log milestone paragraph for one attempt, numbers from reports/summary.json only."""
import json
import sys
from pathlib import Path

S = json.loads((Path(__file__).resolve().parent / 'summary.json').read_text())
attempt = sys.argv[1]
src = next(s for s in S['sources'] if s['attempt'] == attempt)
for v in [v for v in S['verdicts'] if v['attempt'] == attempt]:
    rows = {r['subject']: r for r in S['rows'] if r['attempt'] == attempt and r['mesh'] == v['mesh']}
    acc, floor = rows[v['arm']], next(b for b in S['bank_floor'] if b['attempt'] == attempt and b['mesh'] == v['mesh'])
    fast = rows[v['arm'].replace('q256', 'q0').replace('q96', 'q0')]
    sp = acc['speedups']

    def f(k):
        return '—' if sp[k] is None else f"{sp[k]['total']:.2f}x total / {sp[k]['device']:.1f}x device vs `{sp[k]['comparator']}`"
    cg = rows['cg_0.01']
    print(f"- **{v['mesh']}** (`{attempt}`, job {src['job_id']}, {src['gpu']}, source `{src['commit'][:12]}`, audit "
          f"{'passed' if src['audit_passed'] else 'FAILED'}, {src['error_checks']} recomputed errors): accurate arm `{v['arm']}` "
          f"worst same-grid {100 * acc['worst_same_grid']:.3f} % at {acc['median_total_ms']:.2f} ms total "
          f"({acc['median_device_ms']:.2f} ms device); fast arm `{fast['subject']}` {100 * fast['worst_same_grid']:.3f} % at "
          f"{fast['median_total_ms']:.2f} ms; bank floor {100 * floor['worst']:.3f} %. Named FOM `cg_0.01`: "
          f"{cg['median_total_ms']:.1f} ms, {cg['median_iterations']:g} iterations, {100 * cg['worst_physical']:.3f} % worst physical. "
          f"Speedups of the accurate arm: {f('named_cg_1e-2')}; fastest matched-accuracy CG: {f('fastest_cg_matched')}; "
          f"matched coarse grid: {f('fastest_coarse_matched')}; direct transform: {f('dst_direct')}. "
          f"Bar: **{'MET' if v['bar_met'] else 'MISSED'}** (<=1 %: {v['accuracy_bar_1pct']}, >=5x: {v['speed_bar_5x']}, "
          f"0.5 % stretch: {v['stretch_bar_0p5pct']}).")
