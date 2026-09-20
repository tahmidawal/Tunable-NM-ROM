#!/home/tahmid/Dev/.venv/bin/python
"""Generate the requested cross-dimensional inventory; never combine timing jobs."""
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from generate_3d_paper_results import aggregate, normalize

ROOT = Path(__file__).resolve().parents[1]
STEM = ROOT / 'reports/2026-09-20-2d-3d-error-speedup'
SOURCES = {}
ROWS = []


def read(path):
    data = (ROOT / path).read_bytes()
    SOURCES[path] = hashlib.sha256(data).hexdigest()
    return json.loads(data)


def pivot(data, **filters):
    if isinstance(data, dict):
        data = data['rows']
    groups = {}
    for row in data:
        if not all(row.get(k) == v for k, v in filters.items()):
            continue
        name = row.get('subject', row.get('arm'))
        group = groups.setdefault(name, dict(row))
        assert row['metric'] not in group or group[row['metric']] == row['value'], (name, row['metric'])
        group[row['metric']] = row['value']
    return groups


def simple(groups, error, timing, scale=100, time_scale=1, median=None):
    return {name: dict(error_pct=g[error]*scale, ms=g[timing]*time_scale,
                      median_error_pct=g[median]*scale if median and median in g else None,
                      source_metrics=g)
            for name, g in groups.items() if error in g and g.get(timing) is not None}


def emit(pde, mesh, group, methods, foms, source, norm, note='', clock='GPU', section='NM-ROM'):
    for name in methods:
        result = group[name]
        candidates = [f for f in foms if group[f]['error_pct'] <= result['error_pct'] + 1e-10]
        assert candidates, (pde, name)
        fom = min(candidates, key=lambda f: group[f]['ms'])
        baseline = group[fom]
        ROWS.append(dict(section=section, pde=pde, mesh=mesh, method=name,
                         error_pct=result['error_pct'], median_error_pct=result.get('median_error_pct'),
                         method_ms=result['ms'], fom=fom, fom_error_pct=baseline['error_pct'],
                         fom_ms=baseline['ms'], speedup=baseline['ms']/result['ms'],
                         clock=clock, norm=norm, note=note, source=source,
                         job_id=result.get('job_id', result.get('source_metrics', {}).get('job_id')),
                         method_evidence=result, fom_evidence=baseline))


def add_2d():
    path = 'worktrees/2026-09-17-b-panel/experiments/b-panel/reports/summary.json'
    data = read(path)
    for mesh in (256, 512, 1024):
        groups = pivot(data, mesh=mesh)
        values = simple(groups, 'worst_evolved_percent', 'median_gpu_ms', scale=1, median='median_evolved_percent')
        valid = [n for n, g in groups.items() if g['family'] in ('rom', 'fast') and g.get('admissible') and n in values]
        q0 = min((n for n in valid if groups[n]['q_or_k'] == 0), key=lambda n: values[n]['ms'])
        top = min(valid, key=lambda n: values[n]['error_pct'])
        foms = [n for n in values if groups[n]['family'] == 'fom']
        emit('Burgers 2D', f'{mesh}² intervals', values, [q0, top], foms, path,
             'initial-field L2; evolved times', 'Development. Evolved-time advantage at the largest mesh disappears when initial compression is included. EQ construction status is retained in source_metrics.')
        emit('Burgers 2D', f'{mesh}² intervals', values, ['fno-large'], foms, path,
             'initial-field L2; evolved times', 'Paired FNO from the Burgers panel. This cohort/reference differs from the separate U-Net/Transolver operator study.', section='Neural operators')

    path = 'worktrees/2026-09-17-p-linear/experiments/p-linear/reports/summary.json'
    groups = pivot(read(path), mesh=1024, job_id='3783813')
    values = simple(groups, 'worst_same_grid', 'median_device_ms', median='median_same_grid')
    for section, methods in [('NM-ROM', ['q0_m4@new_K32', 'q256_m4@new_K32']),
                             ('Linear endpoints', ['d_linear_qr_m4@new_K32'])]:
        emit('Poisson 2D', '1024² intervals', values, methods, ['dst_direct'], path,
             'current-field L2; steady state', 'Development. Iterative q=512 failed its stopping rule; the separately timed direct linear endpoint is listed instead.', section=section)

    path = 'worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/linear05/archive/outputs/results.json'
    data = read(path)
    groups = defaultdict(list)
    for row in data['rows']:
        if row['intervals'] == 1024:
            groups[row['method']] += row['repetitions']
    values = {}
    for name, calls in groups.items():
        times = [c['phases']['device_seconds']*1000 for c in calls]
        med = statistics.median(times)
        values[name] = dict(error_pct=100*max(max(c['vs_same_grid']['relative_current'][1:]) for c in calls),
                            all_times_error_pct=100*max(max(c['vs_same_grid']['relative_current']) for c in calls),
                            ms=med, repetitions_ms=times, timing_outliers=sum(t>1.5*med for t in times))
    for section, methods in [('NM-ROM', ['nmrom']), ('Linear endpoints', ['linear_weak_exact'])]:
        emit('Heat 2D', '1024² intervals', values, methods, ['fom_same_grid'], path,
             'current-field L2; evolved times', 'Earlier audited bank/head experiment. Same-grid error here; the prior paper table instead reported physical-reference all-times error.', section=section)

    path = 'worktrees/2026-09-17-w-ladder/experiments/w-ladder/reports/summary.json'
    groups = pivot(read(path), mesh=1024)
    # The DST is the numerical reference itself. Its current-relative summary is
    # NaN because of an unused denominator; stored displacement errors are zero.
    assert groups['dst']['worst_displacement'] == 0
    groups['dst']['worst_current_displacement'] = 0
    values = simple(groups, 'worst_current_displacement', 'median_gpu_ms')
    for section, methods in [('NM-ROM', ['head_q0', 'nested_q32']), ('Linear endpoints', ['linear_bank64'])]:
        emit('Reflective wave 2D', '1024² intervals', values, methods, ['dst'], path,
             'current displacement L2; all sampled times', 'Displacement only. Velocity and energy-state errors are separate source metrics; this does not establish an all-state accuracy pass.', section=section)

    path = 'worktrees/2026-09-17-lshape/experiments/lshape/reports/summary.json'
    data = read(path)
    for mesh, job in [(256, '3784663'), (512, '3789568')]:
        groups = pivot(data, mesh=mesh, job_id=job, test_modes=257)
        values = simple(groups, 'worst_same_grid', 'median_total_ms', median='median_same_grid')
        foms = [n for n in values if n.startswith('fom_')]
        emit('L-shaped Poisson 2D', f'{mesh}² intervals', values,
             ['neural_q0@head_sdf_R512_K16', 'neural_q64@head_sdf_R512_K16'], foms, path,
             'current-field L2; steady state', 'Complete-query clock includes transfers and the CPU sparse-direct control. Development.', clock='Complete query')

    path = 'worktrees/2026-09-17-ns2d/experiments/ns2d/artifacts/ns304/result.json'
    data = read(path)
    values = simple(data['aggregates'], 'worst_evolved', 'median_seconds', time_scale=1000, median='median_evolved')
    for name, v in values.items():
        v['job_id'] = data['job_id']
        times = data['aggregates'][name]['all_seconds']
        v['timing_outliers'] = sum(t>1.5*statistics.median(times) for t in times)
    emit('Navier–Stokes 2D', '256² periodic points', values, ['neural_q0', 'neural_q512'],
         [n for n in values if n.startswith('fom_')], path, 'initial vorticity L2; evolved times',
         'Exploratory after the representation gate failed; q=512 reaches full-bank capacity. Initial-fit convergence is not implied by zero evolution budget exits.')

    path = 'worktrees/2026-09-17-b-lowvisc/experiments/b-lowvisc/reports/summary.json'
    data = read(path)
    groups = pivot(data, job_id=data['panel_job'])
    values = simple(groups, 'worst_evolved_percent', 'median_gpu_ms', scale=1)
    emit('Low-viscosity Burgers 2D', '256² intervals', values,
         ['q0_M1088_dense_g1em06', 'q256_M1088_dense_g1em06'], ['nt1e-2_dt01'], path,
         'initial-field L2; evolved times', 'Fixed-test ladder. Under-resolved mesh: this is a comparison to the same discrete equation, not a resolved physical-accuracy claim.')


def add_3d():
    for lane, attempt, adapter, pde in [
        ('b', 'b3d005/collected/out/seed0', 'burgers', 'Burgers 3D'),
        ('h', 'extra03/archive/out', 'heat', 'Heat 3D'),
        ('p', 'seed04/archive/out', 'poisson', 'Poisson 3D'),
        ('ns', 'extra03/collected/output', 'ns_trajectory', 'Navier–Stokes 3D')]:
        experiment = 'ns3d' if lane == 'ns' else f'paper-{lane}3d'
        base = f'worktrees/2026-09-20-paper-{lane}3d/experiments/{experiment}/runs/{attempt}'
        path = base+'/result.json'
        data = read(path)
        raw = read(base+'/timing_rows.json') if lane == 'ns' else data['invocations'] + data.get('operator_invocations', [])
        values = {}
        for row in aggregate([normalize(r, adapter, data) for r in raw]):
            if row['mesh'] != (33 if lane == 'b' else 32):
                continue
            values[row['method']] = dict(error_pct=100*row['evolved_worst'], ms=row['gpu_ms_median'],
                                         median_error_pct=100*row['evolved_median'], job_id=data['job_id'], **row)
        choices = {
            'b': (['rom_q0', 'rom_q192'], [], [n for n in values if n.startswith('fom_')]),
            'h': (['nmrom_K32_q0_dense', 'nmrom_K32_q96_dense'], ['linear_bank_galerkin_exact'], ['dst_exact']),
            'p': (['nmrom_K16_q0_dense', 'nmrom_K16_q96_dense'], ['linear_bank_galerkin'], ['dst_exact']),
            'ns': (['nmrom_q0', 'nmrom_q128'], [], [n for n in values if n.startswith('fom_')]),
        }
        nm, linear, foms = choices[lane]
        mesh = '33³ nodes' if lane == 'b' else ('32³ periodic points' if lane == 'ns' else '32³ intervals')
        norm = 'current-field L2; steady state' if lane == 'p' else ('current-field L2; evolved times' if lane == 'h' else 'initial-field L2; evolved times')
        note = 'Provisional development results; final cohort unopened.'
        if lane == 'b':
            note += ' First of two independently trained checkpoints. Spatial refinement misses the physical-reference target.'
        if lane == 'ns':
            note += ' Representation target failed. New paired fields pass numerical auditing; the separately retained strict cross-run replay gate failed.'
        emit(pde, mesh, values, nm, foms, path, norm, note)
        if linear:
            emit(pde, mesh, values, linear, foms, path, norm, note, section='Linear endpoints')
        ops = [n for n in values if any(n.startswith(prefix) for prefix in ('fno', 'unet', 'deeponet', 'transolver')) and (lane != 'ns' or n.endswith('_projected'))]
        emit(pde, mesh, values, ops, foms, path, norm, note, section='Neural operators')

    # Newer augmented NS operators have their own paired FOM, but no NM-ROM
    # rollout passed the representation gate in this attempt.
    base = 'worktrees/2026-09-20-paper-ns3d/experiments/ns3d/runs/coverage04/collected/output'
    path = base+'/result.json'
    data = read(path)
    raw = read(base+'/timing_rows.json')
    values = {}
    for row in aggregate([normalize(r, 'ns_trajectory', data) for r in raw]):
        values[row['method']] = dict(error_pct=100*row['evolved_worst'], ms=row['gpu_ms_median'],
                                    median_error_pct=100*row['evolved_median'], job_id=data['job_id'], **row)
    ops = [n for n in values if any(n.startswith(prefix) for prefix in ('fno', 'unet', 'deeponet', 'transolver')) and n.endswith('_projected')]
    emit('Navier–Stokes 3D, augmented training', '32³ periodic points', values, ops,
         [n for n in values if n.startswith('fom_')], path, 'initial velocity L2; evolved times',
         'Provisional development. Later training-coverage experiment with its own paired FOM. No new NM-ROM rollout passed the representation gate in this attempt.', section='Neural operators')


def main():
    add_2d()
    add_3d()
    for row in ROWS:
        assert all(math.isfinite(row[k]) for k in ('error_pct', 'method_ms', 'fom_error_pct', 'fom_ms', 'speedup'))
        assert row['fom_error_pct'] <= row['error_pct'] + 1e-10
    output = dict(sources_sha256=SOURCES, rows=ROWS,
                  selection='Fastest listed same-job tested FOM whose reported error is no greater than the displayed method. Burgers includes time-step and coarse-grid controls. Separable linear PDEs use the direct same-grid FOM; L-shaped Poisson includes sparse-direct and iterative controls. This is a finite tested comparator set, not proof of an optimal FOM.',
                  snapshot='2026-09-20; completed data only, all 3D final cohorts unopened')
    def finite_json(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: finite_json(v) for k, v in value.items()}
        if isinstance(value, list):
            return [finite_json(v) for v in value]
        return value
    STEM.with_suffix('.json').write_text(json.dumps(finite_json(output), indent=2, allow_nan=False)+'\n')
    columns = [k for k in ROWS[0] if not k.endswith('_evidence')]
    with STEM.with_suffix('.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(ROWS)
    lines = ['# Relative L2 error and FOM speed: both dimensions',
             'This generated inventory covers completed NM-ROM comparisons in both dimensions and the matched three-dimensional operator panels. These included runs are development measurements, not final-cohort results. The user-selected iterative-CG comparison is available in [the companion table](2026-09-20-iterative-cg-comparisons.md); the named baselines in this inventory remain unchanged.',
             r'Errors are percentages. Each row uses its explicitly stated norm and reference convention; they must not be treated as one common cross-PDE metric. Speedup is $T_{\mathrm{FOM}}/T_{\mathrm{method}}$: above one means faster, below one means slower.',
             output['selection'],
             'Costs are median device times except L-shaped Poisson, which uses complete-query times to include its CPU direct solver. Every ratio is within a job. Source error medians, timing arrays/outlier counts where available, and full source metadata are preserved in the JSON. Summary-only sources do not supply a new outlier count here.',
             'Burgers and Navier–Stokes retain the paper’s initial-normalized evolved-time L2 metric. Heat uses current-normalized evolved-time error. Poisson is steady-state current-normalized L2. Wave rows use current-normalized displacement L2; the energy and velocity errors remain separate, and an all-state pass is not claimed. Except as explicitly disclosed, comparisons use the same numerical-grid reference.',
             'The 2D heat entry comes from the earlier audited bank/head experiment, not a new overnight retraining. For linear PDEs, unrestricted linear endpoints are listed separately from the nonlinear solves. Sealed Burgers2D accuracy from the independent final cohort is not attached to development timings.',
             'No current 3D reflective-wave, L-shaped-Poisson, or low-viscosity-Burgers counterpart is available. The wider matched 2D operator programme remains incomplete; the existing Burgers2D U-Net/Transolver accuracy study lacks same-allocation FOM timings and is not assigned a speedup here.']
    for section in ('NM-ROM', 'Linear endpoints', 'Neural operators'):
        lines += ['', f'**{section}**', '',
                  '| Problem | Mesh | Method | Worst L2 (%) | Method ms | FOM | FOM L2 (%) | FOM ms | Speedup |',
                  '|---|---|---|---:|---:|---|---:|---:|---:|']
        for row in ROWS:
            if row['section'] != section:
                continue
            lines.append(f"| {row['pde']} | {row['mesh']} | `{row['method']}` | {row['error_pct']:.4f} | {row['method_ms']:.4f} | `{row['fom']}` | {row['fom_error_pct']:.4f} | {row['fom_ms']:.4f} | {row['speedup']:.4f}× |")
    lines += ['', '**Row qualifications and provenance**', '']
    seen = set()
    for row in ROWS:
        key = row['pde'], row['note'], row['source']
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {row['pde']}: {row['note']} [Source JSON](../{row['source']}).")
    lines += ['', '**Glossary**', '',
              '- Relative L2: Euclidean field-difference norm divided by the named reference-field norm. Initial normalization divides by the initial state; current normalization divides by the reference at the evaluated time.',
              '- Worst: maximum over evaluated cases and the stated output times, retaining bad cases. Median error, where available in the JSON, summarizes cases instead.',
              '- Evolved times: requested output times after the supplied initial state. All times also scores initial-state compression.',
              '- FOM: full-order numerical solver. DST: direct discrete sine transform solver. CG: conjugate gradients; its tolerance appears in the method name. Sparse direct uses a matrix factorization.',
              '- NM-ROM: nonlinear-manifold reduced-order model. Head: compressed nonlinear map into a learned spatial bank. Linear endpoint: unrestricted solution in that bank.',
              '- q: number of correction directions. K: nonlinear latent dimension. M: weak test count. R: bank rank. EQ: empirical quadrature, a sampled weighted residual evaluation. Dense: evaluates all grid points.',
              '- Method/FOM ms: median milliseconds for the stated timing scope. Complete query includes transfers. GPU timing includes resident initialization, solution and requested dense outputs but excludes offline training.',
              '- Speedup: FOM milliseconds divided by method milliseconds; each row uses its displayed comparator. A value below one is a slowdown.',
              '- FNO, U-Net, DeepONet and Transolver: the four trained operator architectures tested. Projected: velocity output is projected to satisfy the discrete divergence-free constraint.',
              '- Nodes / intervals / periodic points: different mesh-size conventions, stated explicitly. Steady state: one solution rather than a trajectory.',
              '- Development: cases used while choosing or tuning configurations. Final cohort: separate cases reserved for evaluation after choices are frozen.',
              '- Representation gate: required accuracy check before promoting a compressed model. Exploratory: a diagnostic experiment conducted without passing that gate.',
              '- Same-grid reference: a converged solve of the discrete equation. Physical reference: a finer or analytic reference. Under-resolved: grid error remains too large for the stated physical target.',
              '- Timing outlier: retained repetition taking more than one and a half times its method median. Source hash: checksum identifying the exact JSON used to generate this inventory.']
    STEM.with_suffix('.md').write_text('\n\n'.join(lines[:8])+'\n'+'\n'.join(lines[8:])+'\n')
    print(f'Generated {len(ROWS)} same-job comparisons from {len(SOURCES)} source JSONs.')


if __name__ == '__main__':
    main()
