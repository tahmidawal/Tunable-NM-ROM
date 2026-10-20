"""Fill report.src.md with the traced values of values.py and write report.md and VALUES.json.

Placeholders are «key»; «REVIEW_RECORD» is replaced by review_record.md. Fails if any placeholder is unknown or unused
keys would hide a typo. Run: JAX_PLATFORMS=cpu /home/tahmid/Dev/.venv/bin/python make_report.py
"""
import json
import re
from pathlib import Path

from values import values

HERE = Path(__file__).resolve().parent
V = values()
src = (HERE / 'report.src.md').read_text()
rec = (HERE / 'review_record.md').read_text().strip()
src = src.replace('«REVIEW_RECORD»', rec)
keys = re.findall(r'«([A-Za-z0-9_]+)»', src)
missing = sorted({k for k in keys if k not in V})
assert not missing, f'unknown placeholders: {missing}'
out = re.sub(r'«([A-Za-z0-9_]+)»', lambda m_: V[m_.group(1)][0], src)
assert '«' not in out and '»' not in out
(HERE / 'report.md').write_text(out)
used = sorted(set(keys))
(HERE / 'VALUES.json').write_text(json.dumps({k: dict(value=V[k][0], source=V[k][1]) for k in used}, indent=1, ensure_ascii=False))
print(f'report.md written: {len(keys)} placeholders, {len(used)} distinct values; {len(V) - len(used)} values unused')
