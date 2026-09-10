"""Summarize sealed, frozen-config evaluation rows without selecting settings."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

from common.seal import verify_validation_seal


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory
    handoff = read(root / 'handoff.json')
    cfg = handoff['config']
    binding = verify_validation_seal(root, handoff['case_name'], cfg['seeds']['evaluation'])
    assert handoff['status'] == 'complete'
    assert all(handoff['provenance'][key] == value for key, value in binding.items())
    assert handoff['selections'] == read(root / 'selections.json')
    assert handoff['invocations'] == [json.loads(line) for line in (root / 'invocations.jsonl').read_text().splitlines()]
    owner = read(root / 'independent_owner_audit.json')
    assert owner['passed'] and owner['handoff_sha256'] == sha(root / 'handoff.json')
    assert not handoff['selection_timings']
    rows = handoff['invocations']
    chosen = {(s['intervals'], s['configuration']) for s in handoff['selections'] if s['configuration'] is not None}
    observed = {(r['intervals'], r['configuration']) for r in rows}
    assert observed == chosen
    expected = {(case, rep) for case in cfg['evaluation_case_ids'] for rep in range(cfg['evaluation_repetitions'])}
    summary = []
    for mesh, configuration in sorted(observed):
        group = [r for r in rows if (r['intervals'], r['configuration']) == (mesh, configuration)]
        assert len(group) == len(expected)
        assert {(r['case'], r['rep']) for r in group} == expected
        assert all(r['split'] == 'evaluation' and r['provenance'] == handoff['provenance'] for r in group)
        per_case = []
        for case in cfg['evaluation_case_ids']:
            repeats = sorted((r for r in group if r['case'] == case), key=lambda r: r['rep'])
            times = [r['seconds'] for r in repeats]
            median = statistics.median(times)
            per_case.append(dict(
                case=case, median_seconds=median, repetition_seconds=times,
                worst_repetition_error=max(r['errors']['displacement'] for r in repeats),
                initial_error=max(r['errors']['initial'] for r in repeats),
                finite=all(r['finite'] for r in repeats), completed=all(r['completed'] for r in repeats),
                stationary=all(r['stationary'] for r in repeats),
                high_timing_outliers=sum(t > 3 * median for t in times),
                distinct_output_hashes=len({r['field_sha256'] for r in repeats}),
                stop_reasons=dict(Counter(r['stop_reason'] for r in repeats))))
        selections = [s for s in handoff['selections'] if (s['intervals'], s['configuration']) == (mesh, configuration)]
        targets = []
        for target in cfg['targets']:
            passed = sum(c['finite'] and c['completed'] and c['worst_repetition_error'] <= target for c in per_case)
            selected = any(s['target'] == target and s['validation_passed'] for s in selections)
            targets.append(dict(target=target, cases_passing=passed,
                                validation_selected_for_target=selected,
                                whole_cohort_qualified=selected and passed == len(per_case)))
        summary.append(dict(
            intervals=mesh, method=group[0]['method'], configuration=configuration,
            median_case_median_seconds=statistics.median(c['median_seconds'] for c in per_case),
            median_case_error=statistics.median(c['worst_repetition_error'] for c in per_case),
            worst_case_error=max(c['worst_repetition_error'] for c in per_case),
            finite_cases=sum(c['finite'] for c in per_case),
            completed_cases=sum(c['completed'] for c in per_case),
            stationary_cases=sum(c['stationary'] for c in per_case),
            high_timing_outliers=sum(c['high_timing_outliers'] for c in per_case),
            targets=targets, cases=per_case))
    uncertainty = []
    profiles = []
    for mesh in cfg['meshes']:
        references = read(root / f'references/evaluation_L{mesh}_uncertainty.json')
        uncertainty.append(dict(intervals=mesh, cases=len(references), strict_continuum_bound=False,
            provisional_target_cases={str(t): sum(t in r['provisional_targets'] for r in references) for t in cfg['targets']}))
        for row in read(root / f'profiles_L{mesh}.json'):
            keys = [key for key in row['repetitions'][0] if key.endswith('_s')]
            profiles.append(dict(intervals=mesh, configuration=row['configuration'], case=row['case'],
                repetitions=len(row['repetitions']),
                median_seconds={key: statistics.median(rep[key] for rep in row['repetitions']) for key in keys}))
    result = dict(passed=True, case_name=handoff['case_name'], handoff_sha256=sha(root / 'handoff.json'),
        provenance=handoff['provenance'], global_seal_binding_verified=True,
        invocation_count=len(rows), unique_full_field_artifacts=len({r['field_artifact'] for r in rows}),
        coverage='All frozen non-null configurations, all evaluation cases, every recorded repetition; no evaluation selection.',
        metric='Maximum over output times of full-grid error L2 divided by true initial-field L2, against the fixed finer numerical reference.',
        timing='Median across cases of the seven paired invocation medians; high outlier means greater than three times its own case median.',
        interpretation='Targets qualify only a validation-selected configuration passing every evaluation case. Continuum accuracy is provisional where refinement flags apply.',
        summaries=summary, reference_uncertainty=uncertainty, component_profiles=profiles,
        component_profile_caveat='Separately compiled warmed case-zero components are diagnostics; their sum is not the paired end-to-end query time.')
    (root / 'evaluation_summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key not in ('summaries', 'component_profiles')}, indent=2))


if __name__ == '__main__':
    main()
