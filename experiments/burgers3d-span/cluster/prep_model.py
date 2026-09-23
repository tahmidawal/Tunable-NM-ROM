"""Copy a collected training output into inputs/model_R<R>/ (tracked): bank.pkl as is, heads without the
recomputable library (panel.py recomputes H = head(codes) exactly), training.json/log; writes SHA256SUMS."""
import hashlib
import pickle
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
att, R = sys.argv[1], sys.argv[2]
src = HERE / 'runs' / att / 'archive' / 'code' / 'output'
dst = HERE / 'inputs' / f'model_R{R}'
dst.mkdir(parents=True, exist_ok=False)
shutil.copy(src / 'bank.pkl', dst / 'bank.pkl')
for f in sorted(src.glob('head_K*.pkl')):
    h = pickle.loads(f.read_bytes())
    h.pop('library_H', None)
    (dst / f.name).write_bytes(pickle.dumps(h))
for f in ('training.json', 'training.log'):
    if (src / f).exists():
        shutil.copy(src / f, dst / f)
lines = [f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}' for p in sorted(dst.iterdir())]
(dst / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
