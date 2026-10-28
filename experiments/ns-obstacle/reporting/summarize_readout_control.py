"""Generate a resource-profile summary, never a ROM or whole-query speed claim."""
import argparse,hashlib,json,statistics,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
d=json.loads(a.input.read_text());assert d['passed'] and not d['physical_admission']
groups={}
for row in d['records']:
    assert row['passed'] and row['labels_match']
    assert math.isfinite(row['seconds_diagnostic_only']) and row['seconds_diagnostic_only']>=0
    assert all(math.isfinite(x['value']) and x['value']>=0 for x in row['checks'].values())
    key=row['kind']+'/'+row['mode'];groups.setdefault(key,[]).append(row)
rows=[]
for key,records in sorted(groups.items()):
    assert sorted(x['repetition'] for x in records)==list(range(d['repetitions']))
    seconds=[x['seconds_diagnostic_only'] for x in sorted(records,key=lambda x:x['repetition'])]
    rows.append({'readout':key,'repetition_seconds':seconds,'median_seconds':statistics.median(seconds),'max_normalized_gap':max(c['value'] for x in records for c in x['checks'].values())})
result={'status':'readout_control_summary_no_independent_review_inferred','source_control_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),'source_case_manifest_sha256':d['case_manifest_sha256'],'scope':d['timing_claim'],'physical_admission':False,'peak_rss_bytes':d['peak_rss_bytes'],'rows':rows,'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
