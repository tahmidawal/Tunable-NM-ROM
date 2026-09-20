#!/home/tahmid/Dev/.venv/bin/python
"""Render existing paired CG comparisons without borrowing cross-job timings."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEM = ROOT/'reports/2026-09-20-iterative-cg-comparisons'
SOURCES = {}
ROWS = []


def read(path):
    content = (ROOT/path).read_bytes()
    SOURCES[path] = hashlib.sha256(content).hexdigest()
    return json.loads(content)


def pivot(rows, mesh, job=None):
    result = {}
    for r in rows:
        if r['mesh'] == mesh and (job is None or r['job_id'] == job):
            g = result.setdefault(r['subject'], {'job_id': r['job_id']})
            assert g['job_id'] == r['job_id']
            assert r['metric'] not in g
            g[r['metric']] = r['value']
    return result


def emit(pde, mesh, method, a, candidates, error_key, time_key, source, norm, note):
    passing = [(n,b) for n,b in candidates.items() if b[error_key] <= a[error_key]
               and ('valid_count' not in a or b.get('valid_count') == a['valid_count'])]
    name,b = min(passing, key=lambda item:item[1][time_key])
    assert a['job_id'] == b['job_id']
    ROWS.append(dict(problem=pde, intervals=mesh, method=method,
                     error_pct=100*a[error_key], method_ms=a[time_key], cg=name,
                     cg_error_pct=100*b[error_key], cg_ms=b[time_key],
                     speedup=b[time_key]/a[time_key], job_id=a['job_id'], source=source,
                     error_definition=norm, note=note,
                     method_outliers=a.get('device_outliers'), cg_outliers=b.get('device_outliers'),
                     method_median_error_pct=100*a['median_same_grid'] if 'median_same_grid' in a else None))


def main():
    path='worktrees/2026-09-17-p-linear/experiments/p-linear/reports/summary.json'
    rows=read(path)
    for mesh,job in [(256,'3780692'),(1024,'3783813')]:
        g=pivot(rows,mesh,job)
        cg={n:v for n,v in g.items() if n.startswith('cg_')}
        for name in ['q0_m4@new_K32','q256_m4@new_K32','d_linear_qr_m4@new_K32']:
            assert g[name]['valid_count']==36
            emit('Poisson2D',mesh,name,g[name],cg,'worst_same_grid','median_device_ms',path,
                 'Worst current-relative L2 versus the same-grid reference, steady state.',
                 'Current bank/head development experiment. Unpreconditioned CG on the full finite-difference grid; constant-diagonal Jacobi is only a scalar rescaling. The direct linear endpoint is a separate method.')

    path='worktrees/2026-09-17-w-ladder/experiments/w-ladder/reports/summary.json'
    rows=read(path)
    for mesh in (64,256,1024):
        g=pivot(rows,mesh)
        cg={n:v for n,v in g.items() if n.startswith('cg') and v['all_state_pass']}
        for name in ['head_q0','nested_q32','linear_bank64']:
            emit('Reflective wave2D',mesh,name,g[name],cg,'worst_current_displacement','median_gpu_ms',path,
                 'Worst current-relative displacement L2. Velocity and energy are separate metrics.',
                 'Development. CG solves the full-grid implicit-midpoint system. Candidates include the retained tolerance/time-step ladder and must pass the full-state gate. These NM-ROM configurations still fail their full-state accuracy gate.')

    base='worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/runs/iterative_cg09'
    result=read(base+'/archive/outputs/results.json')
    audit=read(base+'/analysis/audit.json')
    assert result['complete'] and audit['passed']
    assert audit['result_sha256']==SOURCES[base+'/archive/outputs/results.json']
    config=result['settings']
    for mesh in config['requested_intervals']:
        g={r['method']:dict(r,job_id=result['metadata']['job_id']) for r in audit['summaries'] if r['intervals']==mesh}
        cg={name:g[name] for name,spec in config['cg_methods'].items() if spec['theta']==0.5 and g[name]['cg_nonconverged_steps']==0}
        for name in [config['primary_method'],'linear_weak_exact']:
            emit('Heat2D',mesh,name,g[name],cg,'worst_physical_error','device_median_ms',base+'/archive/outputs/results.json',
                 'Worst current-relative physical-reference L2 over all cases and all requested times.',
                 'Earlier audited frozen-checkpoint experiment, measured separately from the newest direct-FOM table. CG and NM-ROM use matching Crank–Nicolson steps. All displayed CG solves converged; the nonlinear model passed its stopping checks.')
    for r in ROWS:
        assert r['cg_error_pct'] <= r['error_pct'] and r['cg_ms']>0 and r['method_ms']>0
    out=dict(selection='Fastest retained same-job passing iterative CG whose displayed error is no greater than the method error; no cross-job ratio. This is selection from a finite measured tolerance/time-step set.',
             status='Collected development measurements. Heat is an earlier independently audited checkpoint; current 3D CG measurements are pending.',
             source_sha256=SOURCES,rows=ROWS)
    STEM.with_suffix('.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    with STEM.with_suffix('.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(ROWS[0]));writer.writeheader();writer.writerows(ROWS)
    text=['# NM-ROM versus iterative CG: collected comparisons', '',
          out['status']+' This generated table changes the named full-order comparator, not the trained models or the measured outputs.', '',
          out['selection'], '',
          r'Speedup is $T_{\mathrm{CG}}/T_{\mathrm{method}}$. Values above one mean the displayed method is faster than the named CG implementation. These are not speedups over every available full-order solver; the direct-transform comparison remains in the companion inventory.', '',
          '[Direct-FOM inventory](2026-09-20-2d-3d-error-speedup.md). Costs below are medians of synchronized GPU query measurements, including reduced-model initialization and requested dense device outputs. Offline training and host transfers are excluded.', '',
          '| Problem | Intervals/axis | Method | L2 error (%) | Method ms | CG setting | CG error (%) | CG ms | Speedup |',
          '|---|---:|---|---:|---:|---|---:|---:|---:|']
    for r in ROWS:
        text.append(f"| {r['problem']} | {r['intervals']} | `{r['method']}` | {r['error_pct']:.4f} | {r['method_ms']:.4f} | `{r['cg']}` | {r['cg_error_pct']:.4f} | {r['cg_ms']:.4f} | {r['speedup']:.4f}× |")
    text += ['', '**Definitions and qualifications**', '']
    for pde in dict.fromkeys(r['problem'] for r in ROWS):
        r=next(r for r in ROWS if r['problem']==pde)
        text.append(f"- {pde}: {r['error_definition']} {r['note']} [Source JSON](../{r['source']}).")
    text += ['', 'Heat timing outlier counts are retained in the CSV/JSON. The Poisson/wave source summaries do not provide outlier counts; the original repetition records remain upstream. No count is inferred from a median.', '',
             'The Heat3D and Poisson3D CG columns require new same-allocation measurements with the frozen NM-ROM and operators. Their direct-solver timings cannot be replaced by CG timings from another job. Burgers already uses iterative Newton–BiCGStab. For Navier–Stokes, plain CG does not apply to the full nonsymmetric nonlinear system; the existing solver families remain explicitly identified.', '',
             '**Glossary**', '',
             '- CG: conjugate gradients, an iterative method for symmetric positive-definite linear systems. A tolerance sets its residual stopping criterion; it is not the same as field error.',
             '- NM-ROM: nonlinear-manifold reduced-order model. Head: the compressed nonlinear decoder. Bank: learned spatial functions; its unrestricted linear endpoint has no nonlinear latent constraint.',
             '- q / K: correction rank / nonlinear latent dimension. The method identifiers name the actual stored configurations.',
             '- Relative L2: field-difference norm divided by the stated reference norm. Worst error is the maximum over the evaluated cases and specified times. Current-relative divides by the current reference field norm.',
             '- Same-grid reference: a converged solution of the discrete equation. Physical reference: the separately verified finer/continuum reference. Displacement is one part of the wave state; velocity error is separate.',
             '- Crank–Nicolson / implicit midpoint: second-order time discretizations that require a linear solve for these linear PDEs.',
             '- GPU ms: median milliseconds for the resident-device query. Speedup: named CG median divided by method median, within the same job.',
             '- Development: cases available during configuration selection, rather than untouched final cases. A passing gate is a recorded accuracy or solver check; it does not imply global optimality.',
             '- Source checksum: fingerprint of the input JSON used to generate the table. Outlier: a retained timing repetition classified by the upstream audit. Newton–BiCGStab: nonlinear Newton iterations with an iterative solver for their nonsymmetric linear systems.']
    STEM.with_suffix('.md').write_text('\n'.join(text)+'\n')
    print(f'Generated {len(ROWS)} paired CG comparisons.')
    for r in ROWS:
        if r['intervals']==1024:
            print(r['problem'],r['method'],f"error={r['error_pct']:.6f}%, CG={r['cg_error_pct']:.6f}%, speedup={r['speedup']:.6f}x")


if __name__=='__main__':
    main()
