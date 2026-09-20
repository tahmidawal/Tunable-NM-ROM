"""Generate auditable Heat3D table rows from closed, same-job measurements."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def passing(row):
    return row['nonfinite_cases'] == 0 and row['cases_with_nonstationary_solves'] == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('attempt')
    parser.add_argument('--pending-retention', action='store_true', help='Explicitly labeled numerically audited preview; does not accept or remove the remote archive.')
    args = parser.parse_args()
    assert args.attempt.isalnum()
    lane = Path(__file__).resolve().parent
    run = lane/'runs'/args.attempt
    collected = json.loads((run/'COLLECTED.json').read_text())
    assert collected['checksums_verified']
    if not args.pending_retention:
        assert collected['removed'] and collected['actual_git_blob_bytes_verified']
    for name in ['audit-local.json', 'audit-states-primary.json', 'audit-panel-primary.json',
                 'audit-field-seedB.json', 'audit-states-seedB.json', 'audit-panel-seedB.json']:
        assert json.loads((run/name).read_text())['passed'], name
    uuids = sorted(set(re.findall(r'GPU-[0-9a-fA-F-]+', (run/'archive/job.out').read_text())))
    assert len(uuids) == 1, uuids
    rows = []
    sources = {}
    for cohort, out in [('primary', run/'archive/out'), ('seedB', run/'archive/out/seedB')]:
        record = json.loads((out/'result.json').read_text())
        summary = json.loads((out/'summary.json').read_text())['rows']
        assert record['complete'] and record['backend'] == 'gpu' and record['x64']
        assert record['matmul_precision'] == 'highest'
        sources[cohort] = dict(result_sha256=sha(out/'result.json'), summary_sha256=sha(out/'summary.json'),
                               source_commit=record['source_commit'], job_id=record['job_id'])
        for row in summary:
            mesh = row['intervals']
            candidates = [r for r in summary if r['intervals'] == mesh
                          and r['method'].startswith('fom_cn_cg_') and passing(r)
                          and r['same_grid_current_evolved_worst'] <= row['same_grid_current_evolved_worst']]
            direct = next((r for r in summary if r['intervals'] == mesh and r['method'] == 'dst_exact'), None)
            is_fom = row['method'].startswith('fom_cn_cg_') or row['method'] == 'dst_exact'
            cg = min(candidates, key=lambda r: (r['device_ms_median'], r['method'])) if candidates and not is_fom and passing(row) else None
            rows.append(dict(cohort=cohort, evaluation_cohort=record['evaluation_cohort'], intervals=mesh,
                interior_unknowns=(mesh-1)**3, method=row['method'], cases=row['cases'], invocations=row['invocations'],
                median_rel_l2=row['same_grid_current_evolved_median'], worst_rel_l2=row['same_grid_current_evolved_worst'],
                worst_physical_rel_l2=row['physical_current_evolved_worst'], device_ms=row['device_ms_median'],
                total_ms=row['total_ms_median'], timing_outliers=row['time_outliers_above_1p5_median'],
                nonfinite_cases=row['nonfinite_cases'], nonstationary_cases=row['cases_with_nonstationary_solves'],
                matched_cg=cg['method'] if cg else None,
                matched_cg_worst_rel_l2=cg['same_grid_current_evolved_worst'] if cg else None,
                matched_cg_device_ms=cg['device_ms_median'] if cg else None,
                speedup_vs_matched_cg=cg['device_ms_median']/row['device_ms_median'] if cg else None,
                speedup_vs_dst=direct['device_ms_median']/row['device_ms_median'] if direct and not is_fom and passing(row) else None,
                gpu_uuid=uuids[0], job_id=record['job_id']))
    glossary = dict(
        cohort='Primary trained initialization or independent-seed accuracy confirmation; no timing ratio crosses cohorts.',
        evaluation_cohort='Development is used for selection; final is the prospectively reserved cohort.',
        intervals='Equal subdivisions per spatial axis; interior_unknowns=(intervals-1)^3 excludes Dirichlet boundaries.',
        interior_unknowns='Number of interior nodal values in one spatial field, excluding prescribed boundary values.',
        method='Exact measured solver/configuration identifier; CG names expose time step and relative residual tolerance.',
        cases='Number of distinct query initial fields.', invocations='All timed repetitions across those cases.',
        median_rel_l2='Median across cases of each case\u2019s maximum evolved-time current-relative field L2 error.',
        worst_rel_l2='Maximum evolved-time current-relative field L2 error over the cohort, against exact same-grid propagation; dimensionless.',
        worst_physical_rel_l2='Same worst-error metric against the refined continuum-spectral reference restricted to the requested grid; empirical refinement is separate.',
        device_ms='Median synchronized device query duration in milliseconds; input upload and host output copy are separate.',
        total_ms='Median complete host query duration including synchronized input upload and host output copy.',
        timing_outliers='Number of device query times exceeding 1.5 times the method median.',
        nonfinite_cases='Distinct cases with a nonfinite prediction.',
        nonstationary_cases='Distinct cases containing any solve that did not meet its declared stopping rule.',
        matched_cg='Fastest passing measured CG arm on the same job, cohort and mesh whose worst same-grid error is no larger than the method error. Blank means no eligible measured comparator or a failed/FOM row.',
        matched_cg_worst_rel_l2='Measured worst same-grid relative L2 error of that named CG comparator.',
        matched_cg_device_ms='Measured median synchronized device time of that named CG comparator.',
        speedup_vs_matched_cg='Named CG median device time divided by method median device time; values below one mean the method is slower.',
        speedup_vs_dst='Same-job/cohort/mesh exact sine-transform solver median device time divided by method median device time.',
        gpu_uuid='Physical GPU identifier recorded by the immutable allocation log.', job_id='Slurm allocation containing both the method and its comparator.',
        CG='Conjugate gradients, the iterative full-grid linear solver inside each Crank\u2013Nicolson time step.',
        DST='Discrete sine transform, used for exact propagation of the spatially discretized linear heat equation.',
        identity_preconditioning='No conditioning improvement is applied to the CG system; this is not a multigrid-preconditioned baseline.',
        warm_start='The preceding time-step field initializes the next CG solve.',
        true_residual='The linear-system residual independently recomputed from the seven-point stencil, distinct from field L2 error.',
        POD='Proper orthogonal decomposition: a linear basis fitted offline to the same training members.',
        NM_ROM='Nonlinear-manifold reduced-order model with a learned bank and latent head; q counts added bank correction directions.',
        evolved_times='The five requested output times after the supplied initial time; initial-field compression is separately retained in raw results.')
    result = dict(schema='heat3d-audited-paper-rows-v1', attempt=args.attempt, source=sources,
                  status='numerically_audited_pending_complete_git_retention' if args.pending_retention else 'accepted',
                  archive_commit=collected.get('complete_archive_commit'), rows=rows, glossary=glossary,
                  comparator_policy='Descriptive error-matched comparison within each completed cohort. Final ratios use only prospectively frozen CG arms; all rows remain visible and no model selection or retraining uses final fields. Direct DST remains visible. CG uses identity preconditioning and warm starts, with independent true-residual audits.')
    stem = 'paper-tables-pending-retention' if args.pending_retention else 'paper-tables'
    (run/(stem+'.json')).write_text(json.dumps(result, indent=2)+'\n')
    with (run/(stem+'.csv')).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print('PAPER_ROWS_GENERATED', len(rows), uuids[0])


if __name__ == '__main__':
    main()
