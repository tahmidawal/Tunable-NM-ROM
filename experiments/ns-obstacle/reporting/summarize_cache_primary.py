"""Extract cache-control diagnostics; independent audit is a separate requirement."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('input', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
d = json.loads(a.input.read_text())
if not d['passed'] or d['physical_admission'] or not d['one_new_producer_invocation']:
    raise ValueError('Expected passing diagnostic with no physical admission')
records = []
for record in d['records']:
    if not record['passed']:
        raise ValueError('Failed comparison: ' + record['tag'])
    records.append({
        'tag': record['tag'],
        'maximum_normalized_difference': max(x['value'] for x in record['checks'].values()),
        'checks': record['checks'],
    })
r = {
    'scope': 'geometry-cache and primary-state replay diagnostic only',
    'physical_admission': False,
    'training_released': False,
    'independent_audit': 'See separately retained independent review; this extraction does not audit raw arrays',
    'source_result_sha256': hashlib.sha256(a.input.read_bytes()).hexdigest(),
    'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'comparison_count': len(records),
    'maximum_normalized_difference': max(x['maximum_normalized_difference'] for x in records),
    'comparisons': records,
    'complete_producer_seconds': d['complete_producer_seconds'],
    'producer_phases_seconds': d['producer_result']['reference_phase_seconds'],
    'portable_fresh_process_seconds': d['portable_fresh_process_seconds'],
    'peak_rss_bytes': d['peak_rss_bytes'],
    'timing_scope': d['timings'],
    'ROM_speedup_claim': False,
}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(r, indent=2) + '\n')
