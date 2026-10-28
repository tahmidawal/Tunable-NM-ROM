"""Stage committed sources and the immutable model; root owns jobs and ledger."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
LANE='experiments/burgers-shared-replay'


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dest',required=True); ap.add_argument('--model-dir',required=True)
    a=ap.parse_args(); dest=Path(a.dest); model=Path(a.model_dir)
    git=['git','-c','gc.auto=0','-C',str(ROOT)]
    commit=subprocess.check_output(git+['rev-parse','HEAD'],text=True).strip()
    files=[f'{LANE}/{n}' for n in ('pilot.py','replaycore.py','replay.py','analyze.py','DESIGN.md','README.md')]
    files += [f'experiments/jcp-heads/{n}.py' for n in ('p1core','p1common','hcore','hcommon','harm')]
    files += ['experiments/jcp-heads/vendor/quad2d/qcore.py','experiments/jcp-heads/vendor/quad2d/qstudy.py',
        'experiments/mr-burgers2d/engines.py','experiments/mr-burgers2d/iterative_paths.py',
        'experiments/separable-decoder/sep_common.py',
        'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl',
        'experiments/jcp-heads/vendor/quad2d/inputs/rotation_R512.npz']
    files += [f'experiments/jcp-heads/vendor/quad2d/vendor/{n}.py' for n in ('arms','hops','hfast','bkfast','hari_quadrature')]
    meta=json.loads((model/'MODEL.json').read_text())
    assert meta['schema']==1 and meta['complete'] and meta['training']['steps']==200000
    assert meta['training']['prefixes']==[16,32,64] and meta['training']['seed']==20261028
    for k in (16,32,64):
        desc=meta['heads'][str(k)]; assert desc['file']==f'h{k}.npz' and desc['k']==k
        assert hashlib.sha256((model/desc['file']).read_bytes()).hexdigest()==desc['sha256']
    dest.mkdir(parents=True,exist_ok=False); proof=[]
    for name in files:
        blob=subprocess.check_output(git+['show',f'{commit}:{name}'])
        assert blob==(ROOT/name).read_bytes(),f'dirty source: {name}'
        opath=dest/name; opath.parent.mkdir(parents=True,exist_ok=True); opath.write_bytes(blob)
        proof.append(dict(path=name,sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob),commit=commit))
    (dest/'model').mkdir()
    for name in ('MODEL.json','h16.npz','h32.npz','h64.npz'):
        blob=(model/name).read_bytes(); (dest/'model'/name).write_bytes(blob)
        proof.append(dict(path=f'model/{name}',sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob),source='shared immutable bundle'))
    (dest/'PROVENANCE.json').write_text(json.dumps(proof,indent=2)+'\n')
    (dest/'COMMIT.txt').write_text(commit+'\n')
    paths=sorted(p for p in dest.rglob('*') if p.is_file())
    (dest/'MANIFEST.sha256').write_text(''.join(f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(dest)}\n' for p in paths))
    print(dest)
    print('Runner: <absolute cluster venv>/python experiments/burgers-shared-replay/pilot.py --model-dir model --out output/dev6')


if __name__=='__main__': main()
