"""Copy the stored 256^2 test rows of lane 2026-09-25-t2-burgers-test (job 4327396: head k=16 and the paper's operator
rows) into inputs/rom_reference_t2.json, so the report reads them from a file and nothing is typed by hand."""
import hashlib, json
from pathlib import Path
SRC = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-09-25-t2-burgers-test/experiments/t2-burgers-test/reports/summary.json')
cells = json.loads(SRC.read_text())['cells']
out = dict(source='worktrees/2026-09-25-t2-burgers-test/experiments/t2-burgers-test/reports/summary.json',
           source_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(), meshes={})
for L in ('256', '1024'):
    t = cells[L]['test']
    out['meshes'][L] = dict(job=t['job_id'], gpu=t['gpu'], cases=t['cases'], cohort_sha256=t['cohort_sha256'],
        rows={k: {q: v[q] for q in ('arm', 'family', 'worst_percent', 'median_percent', 'gpu_ms', 'trained_at')} for k, v in t['rows'].items()})
(Path(__file__).resolve().parent.parent / 'inputs' / 'rom_reference_t2.json').write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps(out['meshes'], indent=1))
