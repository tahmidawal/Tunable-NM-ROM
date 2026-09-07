"""Audit normalized paired-query records without running a PDE or training model.

This validates accounting, not the correctness of raw physical field metrics.
Each native experiment keeps its detailed JSON/arrays. An adapter supplies one
record per predeclared ROM/FOM configuration pair using those immutable outputs.
Do not fill absent evidence with a successful default.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def summary(values):
    finite = [v for v in values if finite_number(v)]
    return {
        'count': len(values), 'nonfinite': len(values)-len(finite),
        'mean_finite': statistics.mean(finite) if finite else None,
        'median_finite': statistics.median(finite) if finite else None,
        'worst_finite': max(finite) if finite else None,
    }


def audit(record):
    require(record.get('schema_version') == 1, 'unsupported schema version')
    require(record.get('stage') in {'development', 'validation', 'confirmation'}, 'missing cohort stage')
    require(record.get('reference_kind') in {'same_grid_discrete', 'refined_physical'}, 'missing reference kind')
    require(record.get('configuration_selection') in {'predeclared', 'validation_frozen'}, 'selection rule missing')
    require(bool(record.get('source_artifact_sha256')), 'native artifact hashes missing')
    require(all(isinstance(v, str) and re.fullmatch(r'[0-9a-f]{64}', v)
                for v in record['source_artifact_sha256'].values()), 'invalid native artifact hash')
    p = record['provenance']
    require(p.get('backend') == 'gpu' and p.get('x64') is True and
            p.get('matmul_precision') == 'highest', 'GPU/f64/highest evidence required')
    require(str(p.get('job_id', '')).isdigit() and bool(p.get('gpu_identity')), 'job/GPU identity missing')
    require(isinstance(p.get('source_commit'), str) and
            re.fullmatch(r'[0-9a-f]{40}', p['source_commit']), 'source commit missing')
    cases = record['case_ids']
    require(cases and len(set(cases)) == len(cases), 'empty or duplicate declared cases')
    repetitions = record['repetitions']
    require(isinstance(repetitions, int) and repetitions >= 3, 'at least three retained repetitions required')
    metrics = record['required_error_metrics']
    require(metrics and len(set(metrics)) == len(metrics), 'required metrics missing/duplicated')
    require(record.get('output_contract_id'), 'output contract missing')
    rows = {}
    invocation_ids = set()
    for row in record['invocations']:
        require(row['case_id'] in cases, 'undeclared case')
        require(row['method'] in {'nmrom', 'fom'}, 'invalid paired method')
        require(isinstance(row['repeat'], int) and 0 <= row['repeat'] < repetitions, 'invalid repeat index')
        key = (row['case_id'], row['repeat'], row['method'])
        require(key not in rows, 'duplicate paired observation')
        require(row['invocation_id'] not in invocation_ids, 'reused solver invocation')
        invocation_ids.add(row['invocation_id'])
        require(row.get('metric_invocation_id') == row['invocation_id'], 'cost/accuracy invocation mismatch')
        require(str(row['job_id']) == str(p['job_id']) and
                row['gpu_identity'] == p['gpu_identity'], 'cross-job or cross-GPU timing')
        require(row['output_contract_id'] == record['output_contract_id'], 'unequal output contracts')
        require(row.get('query_includes_input_and_output') is True, 'incomplete query timing')
        require(finite_number(row['query_seconds']) and row['query_seconds'] > 0, 'invalid elapsed time')
        require(row.get('warmup_and_burnin_complete') is True and
                row.get('device_synchronized') is True, 'timing warmup/synchronization missing')
        require(isinstance(row.get('completed'), bool) and isinstance(row.get('numerically_valid'), bool),
                'missing completion/numerical validity')
        require(set(metrics).issubset(row['errors']), 'physical metrics missing')
        for metric in metrics:
            v = row['errors'][metric]
            require(v is None or (finite_number(v) and v >= 0), 'errors must be nonnegative or null with a failure')
        if row['numerically_valid']:
            require(row['completed'] and all(finite_number(row['errors'][m]) for m in metrics),
                    'invalid/nonfinite result marked numerically valid')
        rows[key] = row
    require(len(rows) == len(cases)*repetitions*2, 'missing paired repetitions/cases')
    case_data = {}
    ratios = []
    for case in cases:
        entry = {}
        for method in ('nmrom', 'fom'):
            selected = [rows[case, rep, method] for rep in range(repetitions)]
            values = [row['errors'][m] for row in selected for m in metrics]
            valid = all(row['completed'] and row['numerically_valid'] for row in selected)
            valid = valid and all(finite_number(v) for v in values)
            entry[method] = {
                'query_median_seconds': statistics.median(row['query_seconds'] for row in selected),
                'valid': valid,
                'max_required_error': max(values) if valid else None,
            }
        entry['raw_speedup_ratio_of_case_medians'] = (
            entry['fom']['query_median_seconds']/entry['nmrom']['query_median_seconds'])
        ratios.append(entry['raw_speedup_ratio_of_case_medians'])
        case_data[case] = entry
    target_rows = []
    ref_bound = record.get('reference_uncertainty_bound')
    require(ref_bound is None or (finite_number(ref_bound) and 0 <= ref_bound < 1), 'invalid reference uncertainty')
    fraction = record['reference_uncertainty_fraction']
    require(finite_number(fraction) and 0 < fraction < 1, 'invalid reference allocation')
    for target in record['targets']:
        require(finite_number(target) and target > 0, 'invalid target')
        failure_counts = {
            method: sum(not entry[method]['valid'] or entry[method]['max_required_error'] > target
                        for entry in case_data.values())
            for method in ('nmrom', 'fom')
        }
        eligible = not any(failure_counts.values())
        physical = record['reference_kind'] == 'refined_physical' and ref_bound is not None and ref_bound <= fraction*target
        # The bound is relative to the reference norm. The triangle inequality
        # and reverse triangle inequality give (observed + delta)/(1-delta).
        # This also conservatively covers an exactly known fixed denominator.
        # Both error and normalization need margin for reference uncertainty;
        # merely making that uncertainty a small fraction of the target is not
        # sufficient when the observed error lies immediately below the target.
        bounded_failure_counts = {
            method: sum(not entry[method]['valid'] or
                        (entry[method]['max_required_error'] + ref_bound)/(1-ref_bound) > target
                        for entry in case_data.values())
            for method in ('nmrom', 'fom')
        } if physical else None
        qualified = physical and not any(bounded_failure_counts.values())
        target_rows.append({
            'target': target, 'failed_or_above_target_cases': failure_counts,
            'matched_reference_accuracy': eligible,
            'physical_reference_budget_met': physical,
            'failed_or_above_target_with_reference_margin': bounded_failure_counts,
            'qualified_complete_query_speedup': statistics.median(ratios) if qualified else None,
            'independent_confirmation': qualified and record['stage'] == 'confirmation'
                and record['configuration_selection'] == 'validation_frozen'
                and record.get('independent_final_cohort') is True,
        })
    return {
        'accounting_audit_passed': True,
        'scope': 'Accounting only; independently audit native operators and field errors before accepting science.',
        'case_results': case_data,
        'raw_speedup_median_of_case_ratios': statistics.median(ratios),
        'nmrom_error_summary': summary([v['nmrom']['max_required_error'] for v in case_data.values()]),
        'fom_error_summary': summary([v['fom']['max_required_error'] for v in case_data.values()]),
        'targets': target_rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    record = json.loads(args.input.read_text())
    result = audit(record)
    for name, digest in record['source_artifact_sha256'].items():
        path = args.input.parent / name
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, f'native artifact changed: {name}')
    result['native_artifact_hashes_verified'] = True
    result['input_sha256'] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'out': str(args.out), 'accounting_audit_passed': True}))


if __name__ == '__main__':
    main()
