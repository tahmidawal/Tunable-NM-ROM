"""Content hashes for every prepared array consumed by the reconstruction screen."""
from pathlib import Path
import argparse,hashlib,json
COHORTS={'dev6':6,'val32':32}
MESHES=(32,64,128)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def required(root):
    root=Path(root)
    if not (root/'COMPLETE.json').is_file(): raise RuntimeError('preparation has no completion record')
    rows=json.loads((root/'coarse/summary.json').read_text())
    expected={f'{c}_c{j:03d}_L{L}' for c,n in COHORTS.items() for j in range(n) for L in MESHES}
    keys=[r['key'] for r in rows]
    if len(keys)!=len(expected) or set(keys)!=expected: raise RuntimeError('preparation cohort/mesh coverage is not exactly complete and unique')
    paths=['COMPLETE.json','runtime.json','bases/bases.json','coarse/summary.json']
    for k in sorted(expected): paths.extend([f'coarse/{k}.json',f'coarse/{k}.npz'])
    for bank in ['base','coarse']:
        for family in ['bank','pod']:
            for L in ([32,64,128,256] if bank=='base' else [128,256]): paths.append(f'bases/{family}_{bank}_L{L}.npy')
        paths.extend([f'bases/pod_{bank}.json',f'bases/training_{bank}.json'])
    if (root/'disjointness.json').is_file(): paths.append('disjointness.json')
    for rel in paths:
        if not (root/rel).is_file(): raise RuntimeError('missing prepared artifact '+rel)
    return sorted(paths)
def build(root):
    root=Path(root); paths=required(root)
    doc=dict(version=1,coverage='dev6+val32 x L32,L64,L128; ST only',records=[dict(path=r,bytes=(root/r).stat().st_size,sha256=sha(root/r)) for r in paths])
    (root/'CONSUMED.sha256.json').write_text(json.dumps(doc,indent=2)+'\n')
    return sha(root/'CONSUMED.sha256.json')
def verify(root):
    root=Path(root); p=root/'CONSUMED.sha256.json'; doc=json.loads(p.read_text())
    expected=required(root)
    if [r['path'] for r in doc['records']]!=expected: raise RuntimeError('manifest scope differs from consumed data contract')
    for r in doc['records']:
        f=root/r['path']
        if f.stat().st_size!=r['bytes'] or sha(f)!=r['sha256']: raise RuntimeError('prepared artifact checksum failed '+r['path'])
    return sha(p)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root'); ap.add_argument('--verify',action='store_true'); a=ap.parse_args(); print(verify(a.root) if a.verify else build(a.root))
if __name__=='__main__': main()
