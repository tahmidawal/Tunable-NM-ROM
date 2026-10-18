"""Record an edit of an inherited file in PROVENANCE.json: python prov_edit.py <file> '<description>'."""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
f, desc = sys.argv[1], sys.argv[2]
P = json.loads((HERE / 'PROVENANCE.json').read_text())
ent = [e for e in P if e['file'] == f]
assert len(ent) == 1, f
ent[0].setdefault('edits', []).append(dict(sha256_after=hashlib.sha256((HERE / f).read_bytes()).hexdigest(), description=desc))
(HERE / 'PROVENANCE.json').write_text(json.dumps(P, indent=1) + '\n')
print('recorded', f)
