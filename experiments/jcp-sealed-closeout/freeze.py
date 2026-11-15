"""Build the preopening manifest containing all decision-bearing bytes and selection evidence."""
import json
from pathlib import Path
import sys
from open_sealed import HERE,STORE,packed,sha
from roster import ROSTER_PATH,build
allow_dirs=['configs','receipts']
files={p for p in HERE.iterdir() if p.is_file() and p.name not in {'F1.json','F1-REVIEW.json'}}
for name in allow_dirs: files.update(p for p in (HERE/name).rglob('*') if p.is_file())
files=sorted(files)
if packed(build(json.loads(ROSTER_PATH.read_bytes())))!=ROSTER_PATH.read_bytes(): raise ValueError('Unverified roster')
up=json.loads((HERE/'upstream.json').read_bytes())
assert len(list((HERE/'configs').glob('d*m*.json')))==8
freeze={'store_version':STORE,'reference_status_sha256':up['files']['vendor/tree/experiments/jcp-references/store/STATUS.json'],
        'cohorts':{'2':'sealed128','3':'sealed64'},'case_counts':{'2':128,'3':64},
        'source_root':'experiments/jcp-sealed-closeout', 'roster':'roster.json',
        'upstream_source':up['commit'],'staged_upstream_source_tar_sha256':'dfe3b4c943f6e01c5abefdcba142105362bccf38134e119b1ec4ffedf73b4f60',
        'staged_pod_source_tar_sha256':'4b72647e6bb39d8a371b9b9fd2b0582ea3b8c94c4fcb56b11581188021afd806',
        'reference_group':'S and ST, original per-case outcomes and reference uncertainties',
        'sealed_scope':'Only quadrature sufficiency is prospective. Native POD and fixed full-order comparisons as scoped in DECISIONS.md.',
        'files':{p.relative_to(HERE).as_posix():sha(p.read_bytes()) for p in files}}
manifest=packed(freeze)
out=HERE/'F1.json'
if sys.argv[1:]==['--write']: out.write_bytes(manifest)
elif sys.argv[1:] or out.read_bytes()!=manifest: raise ValueError('F1 differs from verified working sources')
print(sha(manifest))
