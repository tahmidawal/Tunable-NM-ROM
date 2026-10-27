"""Copy byte-identical source/checkpoint evidence; no model evaluation or cohort decryption."""
from pathlib import Path
import hashlib, json, shutil
ROOT = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude')
BASE = ROOT/'worktrees/2026-10-20-jcp-consolidated'
HERE = Path(__file__).resolve().parent
SOURCES = [
'experiments/jcp-smooth-bank/smoothtrain.py',
'experiments/jcp-smooth-bank/deps/burgers2d_film.py',
'experiments/jcp-smooth-bank/deps/sep_common.py',
'experiments/jcp-references/refcore.py',
'experiments/jcp-references/cohorts.py',
'experiments/jcp-references/refstore.py',
'experiments/jcp-references/verdict.py',
'experiments/mr-burgers2d/engines.py',
'experiments/separable-decoder/sep_common.py',
'experiments/jcp-references/vendor/quad3d/vendor/burgers3d-span/common.py',
'experiments/jcp-references/vendor/quad3d/vendor/paper-b3d/vendor/b3d_common.py',
]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    records=[]
    for rel in SOURCES:
        src=BASE/rel; dst=HERE/'vendor'/rel
        dst.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,dst)
        records.append(dict(path=str(dst.relative_to(HERE)),source=str(src),sha256=sha(dst),bytes=dst.stat().st_size))
    old=ROOT/'worktrees/2026-10-08-jcp-smooth-bank/experiments/jcp-smooth-bank/runs'
    for name,run in [('base','tr1b'),('coarse','tr2')]:
        for fn in ['bank.pkl','rotation.npz','train.json']:
            src=old/run/'archive/output'/name/fn; dst=HERE/'inputs'/name/fn
            dst.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,dst)
            records.append(dict(path=str(dst.relative_to(HERE)),source=str(src),sha256=sha(dst),bytes=dst.stat().st_size))
    ref=ROOT/'worktrees/2026-10-06-jcp-references/experiments/jcp-references/store'
    for coh in ['dev6','val32']:
        for src in sorted((ref/'2d'/coh/'ST').glob('c*')):
            if src.suffix not in ['.json','.npz']: continue
            dst=HERE/'inputs/refstore'/src.relative_to(ref)
            dst.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(src,dst)
            records.append(dict(path=str(dst.relative_to(HERE)),source=str(src),sha256=sha(dst),bytes=dst.stat().st_size))
    # The manifest is public hashes, not sealed field access; retain its identity as evidence.
    for fn in ['MANIFEST.sha256','STATUS.json']:
        src=ref/fn
        if src.exists():
            dst=HERE/'inputs/refstore'/fn; shutil.copyfile(src,dst)
            records.append(dict(path=str(dst.relative_to(HERE)),source=str(src),sha256=sha(dst),bytes=dst.stat().st_size))
    (HERE/'INPUTS.json').write_text(json.dumps(dict(base='93f399faad67f94ea2b56fe25f13003143e64694',records=records),indent=2)+'\n')
    print('Copied',len(records),'byte-verified sources/checkpoints/exposed references; no training data copied.')
if __name__=='__main__': main()
