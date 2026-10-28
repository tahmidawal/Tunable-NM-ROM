"""Summarize a retained three-mesh DFG trend; never issue G2 admission."""
from pathlib import Path
import argparse
import hashlib
import json
import math


def assess(comparison_path, case_paths, campaign_path, out):
    sources = []
    def read(path):
        path = Path(path)
        raw = path.read_bytes()
        sources.append({'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest()})
        return json.loads(raw)
    comparison = read(comparison_path)
    if comparison.get('full_rule_preserved') is not True:
        raise ValueError('complete retained rule required')
    cases = [read(p) for p in case_paths]
    gates = read(campaign_path)['reference_gates']
    names = [Path(p).parent.name for p in case_paths]
    configs = [c['configuration'] for c in cases]
    if any(c.get('precision') != 'float64' for c in configs):
        raise ValueError('float64 evidence required')
    invariants = ('kind', 'U', 'D', 'nu', 'H', 'xmin', 'xmax', 'center', 'radius',
                  'outlet', 'velocity_order', 'pressure_order', 'geometry_order',
                  'assembly_bonus_order', 'diagnostic_order', 'nonlinear_relative_tolerance',
                  'linear_relative_tolerance')
    if any(c['kind'] != 'dfg' for c in configs):
        raise ValueError('DFG-only screen; manufactured controls have a separate role')
    if any(any(c[k] != configs[0][k] for k in invariants) for c in configs[1:]):
        raise ValueError('h trend must preserve physics, orders and solver/assembly policy')
    for key in ('maxh', 'obstacle_h'):
        if not configs[0][key] > configs[1][key] > configs[2][key]:
            raise ValueError('expected strictly refined mesh sizes')
    pairs = comparison['comparisons']
    if len(pairs) != 2 or comparison['rule_case'] != names[2]:
        raise ValueError('expected two successive differences on finest full rule')
    for i, pair in enumerate(pairs):
        if (pair['case'], pair['reference']) != (names[i], names[i+1]):
            raise ValueError('comparison cases do not match the ordered retained results')
        if pair['pressure_shift_fitted'] or not pair['unshifted_pressure_field_eligible_for_same_domain_gate']:
            raise ValueError('unshifted same-domain pressure required')
    rows = []
    def row(name, coarse, fine, threshold, raw):
        if not all(math.isfinite(x) and x >= 0 for x in (coarse, fine, threshold)):
            raise ValueError('invalid nonnegative metric')
        rows.append(dict(check=name, earlier=coarse, finest=fine, threshold=threshold,
                         decreasing=fine < coarse, finest_within_threshold=fine <= threshold,
                         raw=raw))
    for key, threshold in [('velocity_relative', 'h_velocity_relative'),
                           ('pressure_scaled', 'h_pressure_scaled')]:
        metrics = [pair[key] for pair in pairs]
        if any(not math.isfinite(m['denominator']) or m['denominator'] <= 0 for m in metrics):
            raise ValueError('positive normalization required')
        values = [m['numerator']/m['denominator'] for m in metrics]
        row(key, *values, gates[threshold], metrics)
    forces = [c['metrics']['forces']['physical_symmetric_order16']['coefficient'] for c in cases]
    if any(f[0] == 0 for f in forces):
        raise ValueError('nonzero drag normalization required')
    row('physical_drag_relative', abs(forces[0][0]-forces[1][0])/abs(forces[1][0]),
        abs(forces[1][0]-forces[2][0])/abs(forces[2][0]), gates['h_drag_relative'], forces)
    row('physical_lift_absolute', abs(forces[0][1]-forces[1][1]),
        abs(forces[1][1]-forces[2][1]), gates['h_transverse_absolute_2d'], forces)
    divergence = [c['metrics']['divergence']['value'] for c in cases]
    row('scaled_divergence', divergence[1], divergence[2], gates['divergence_scaled'], divergence)
    rows[-1]['decreasing'] = divergence[0] > divergence[1] > divergence[2]
    record = dict(scope='Numerical h-trend screen; raw-array review and resolved noise bounds still required',
                  physical_admission=False, training_released=False, cases=names, rows=rows,
                  numerical_screen_pass=all(r['decreasing'] and r['finest_within_threshold'] for r in rows),
                  sources=sources, generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  remaining=['raw full-rule/validity audit', 'resolved differences versus measured noise',
                             'p/geometry', 'solver', 'quadrature', 'domain', 'conditional uncertainty',
                             'per-case physical checks', 'common lift', 'all-parameter coverage'])
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    (out/'h-trend.json').write_text(json.dumps(record, indent=2)+'\n')
    lines = ['# Provisional spatial refinement screen', '', record['scope']+'. No reference admission.', '',
             '| Check | Earlier | Finest | Finest limit | Decreasing | Finest passes |',
             '|---|---:|---:|---:|---|---|']
    for r in rows:
        lines.append(f"| {r['check']} | {r['earlier']:.9g} | {r['finest']:.9g} | {r['threshold']:.9g} | {r['decreasing']} | {r['finest_within_threshold']} |")
    lines += ['', 'The threshold applies only to the finest pair. The earlier pair establishes the decreasing trend. Divergence is a per-mesh constraint, not a pair difference.', '',
              '## Glossary', '', 'Earlier/finest: successive mesh differences for fields and forces; second/last mesh values for divergence. Relative: divided by the finer result norm or drag. Scaled pressure: normalized by the frozen physical pressure scale. Lift: transverse force coefficient. Decreasing: numerical trend only; a measured noise floor is still required to resolve it.']
    (out/'h-trend.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--comparison', type=Path, required=True)
    p.add_argument('--cases', type=Path, nargs=3, required=True)
    p.add_argument('--campaign', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    assess(a.comparison, a.cases, a.campaign, a.out)
