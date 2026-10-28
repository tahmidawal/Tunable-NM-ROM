"""Generate provisional control tables exclusively from retained JSON evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def generate(repository: Path, destination: Path) -> None:
    campaign = repository / 'worktrees/2026-10-27-ns-obstacle/experiments/ns-obstacle'
    two = repository / 'worktrees/2026-10-27-ns-obstacle-2d/experiments/ns-obstacle-2d'
    three = repository / 'worktrees/2026-10-27-ns-obstacle-3d/experiments/ns-obstacle-3d'
    sources = []

    def read(path):
        data = path.read_bytes()
        sources.append({'path': str(path.resolve()), 'sha256': hashlib.sha256(data).hexdigest()})
        return json.loads(data)

    ledger = read(campaign / 'state/job-ledger.json')
    contract = read(campaign / 'campaign.json')
    channel = read(two / 'artifacts/g1-smoke06/results/channel/result.json')
    metrics = channel['metrics']
    gates = contract['reference_gates']
    rows = [
        ('Fluid area error', channel['geometry']['volume_error']['value'], gates['geometry_relative']),
        ('Velocity L2 error', metrics['velocity_error']['value'], gates['analytic_velocity_relative']),
        ('Pressure RMS error / U²', metrics['pressure_error']['value'], gates['analytic_pressure_scaled']),
        ('Scaled free residual', metrics['independent_residual']['value'], gates['free_dof_residual']),
        ('Inlet/outlet mass imbalance', metrics['mass']['value'], gates['mass_2d']),
        ('Scaled divergence', metrics['divergence']['value'], gates['divergence_scaled']),
        ('Sampled essential trace / U', metrics['boundary_trace']['value'], gates['essential_trace_over_U']),
    ]
    payload = {'scope': 'provisional controls only; no obstacle reference or method comparison',
               'channel': [{'metric': k, 'value': v, 'threshold': t, 'within_threshold': v <= t}
                           for k, v, t in rows], 'allocations': ledger['allocations']}
    lines = ['# Obstacle-flow runtime controls', '',
             'These are provisional implementation controls. No obstacle reference, learned-bank result, '
             'POD comparison or speed claim is admitted by these tables.', '',
             'The empty channel is an analytic Poiseuille control; its Stokes initialization already '
             'solves the steady problem, so it does not demonstrate nonlinear obstacle convergence.', '',
             '| Channel control | Saved value | Declared threshold |',
             '|---|---:|---:|']
    lines += [f'| {k} | {v:.9g} | {t:.9g} |' for k, v, t in rows]
    normal_path = two / 'artifacts/g1-normal01/results/geometry/result.json'
    if normal_path.exists():
        normal = read(normal_path)
        facet = normal['volume_facet_control']
        comparisons = [('Raw mapped boundary', normal['raw_totals']['divergence_theorem_relative_defect']),
                       ('Ordinary boundary integration', normal['Integrate_totals']['12']['divergence_theorem_relative_defect'])]
        if facet['status'] == 'completed_diagnostic_not_admission':
            comparisons.append(('Adjacent-volume facet integration', facet['divergence_theorem_relative_defect']))
        payload['normal_diagnostic'] = {'status': normal['status'], 'facet_status': facet['status'],
                                       'fresh_mesh_only': True, 'comparisons': comparisons}
        lines += ['', 'The normal diagnostic uses a fresh mesh generated with the unchanged geometry '
                  'recipe. It cannot reconstruct the missing mesh or vectors from the original failed run. '
                  'The table compares the relative defect in '
                  '$\\int_{\\partial\\Omega}x\\cdot n\\,ds=2|\\Omega|$.', '',
                  '| Normal evaluation path | Relative divergence-theorem defect |', '|---|---:|']
        lines += [f'| {name} | {value:.9g} |' for name, value in comparisons]
        lines += ['', f'Nested volume-facet status: `{facet["status"]}`. '
                  'Normal closure alone is insufficient to establish outward orientation.']
    three_path = three / 'artifacts/g1-normal01/output/result.json'
    if three_path.exists():
        result3 = read(three_path)
        facet3 = result3['volume_facet_control']
        ordinary3 = result3['ordinary_BND_moments']['12']
        rows3 = [
            ('Fluid-volume error against analytic geometry', result3['volume_relative_geometry_error']),
            ('Native exterior-facet divergence defect', facet3['divergence_theorem_relative_defect']),
            ('All-element-boundary divergence defect', facet3['all_element_divergence_theorem_relative_defect']),
            ('All-element minus exterior position-normal integral',
             facet3['all_element_boundaries']['position_dot_normal'] - facet3['boundary_position_dot_normal']),
            ('Wall facet minus ordinary boundary area', facet3['per_boundary']['wall']['measure'] - ordinary3['wall']['measure']),
            ('Wall facet minus ordinary position-normal integral',
             facet3['per_boundary']['wall']['position_dot_normal'] - ordinary3['wall']['position_dot_normal']),
        ]
        payload['3d_failed_geometry_controls'] = {'status': 'unresolved; no repair or reference admission',
                                                'metrics': rows3}
        lines += ['', 'The 3D geometry diagnostic completed, but its geometric identity checks fail. '
                  'These failures block a 3D repair/flow release. Positive Jacobians at the saved interior '
                  'samples do not rule out inversion elsewhere; curved-map folding remains an unmeasured '
                  'hypothesis. No correction inferred from the 2D case is admitted here.', '',
                  '| 3D geometric diagnostic | Saved or directly derived value |', '|---|---:|']
        lines += [f'| {name} | {value:.9g} |' for name, value in rows3]
    seconds = sum(a.get('elapsed_seconds', 0) * a['gpus'] for a in ledger['allocations'])
    payload['charged_gpu_seconds'] = seconds
    lines += ['', '| Allocation | Lane / attempt | Recorded state | Charged GPU-seconds |', '|---|---|---|---:|']
    for allocation in ledger['allocations']:
        charged = allocation.get('elapsed_seconds')
        value = 'pending accounting' if charged is None else str(charged * allocation['gpus'])
        lines.append(f'| {allocation["job_id"]} | {allocation["lane"]} / {allocation["attempt"]} | '
                     f'{allocation["state_last_observed"]} | {value} |')
    lines += ['', f'Total recorded allocation: {seconds} GPU-seconds ({seconds / 3600:.9g} GPU-hours). '
              'Finite-element calculations run on CPU within these GPU allocations; this is resource accounting, '
              'not a GPU-acceleration measurement.', '',
              '## Glossary', '',
              'L2 error measures spatially integrated field error. RMS means root mean square. '
              'U is the declared characteristic velocity. The free residual measures the equations on '
              'unconstrained finite-element unknowns. Mass imbalance compares inlet and outlet flux. '
              'Divergence measures local incompressibility error. Essential trace is prescribed boundary velocity; '
              'a sampled maximum is not a continuous maximum certificate. A facet is a volume element face '
              'or edge. An outward-fluid normal points out of the fluid, including into an excluded obstacle. '
              'POD is a weighted linear snapshot basis. A reference is an independently qualified numerical solution.']
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'controls.json').write_text(json.dumps(payload, indent=2) + '\n')
    (destination / 'controls.md').write_text('\n'.join(lines) + '\n')
    (destination / 'PROVENANCE.json').write_text(json.dumps({
        'inputs': sources, 'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'outputs': [{'path': name, 'sha256': hashlib.sha256((destination/name).read_bytes()).hexdigest()}
                    for name in ('controls.json', 'controls.md')]}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    generate(args.repository, args.destination)
