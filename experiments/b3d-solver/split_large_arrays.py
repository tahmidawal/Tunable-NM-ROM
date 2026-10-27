"""Derived lossless per-array copies for mirroring; original collected NPZs are never modified."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('attempt');a=p.parse_args();root=HERE/'results'/a.attempt
    out=root/'derived/large-array-splits';out.mkdir(parents=True,exist_ok=False);records=[];large=[]
    for f in sorted((root/'raw/output').rglob('*.npz')):
        if f.stat().st_size<=50*1024**2:continue
        large.append(f'{sha(f)}  {f.relative_to(HERE.parents[1])}')
        with np.load(f,allow_pickle=False) as z:
            for key in z.files:
                v=z[key];target=out/(f.stem+'_'+key+'.npz');np.savez_compressed(target,array=v)
                assert target.stat().st_size<=50*1024**2,'requires additional chunking'
                assert np.array_equal(v,np.load(target,allow_pickle=False)['array'],equal_nan=True)
                records.append(dict(source=str(f.relative_to(HERE)),source_sha256=sha(f),key=key,shape=list(v.shape),dtype=str(v.dtype),
                    path=str(target.relative_to(HERE)),sha256=sha(target),bytes=target.stat().st_size))
    (out/'MANIFEST.json').write_text(json.dumps(dict(lossless=True,originals_unchanged=True,arrays=records),indent=1)+'\n')
    existing=(HERE/'LARGE_FILES.sha256').read_text().splitlines();(HERE/'LARGE_FILES.sha256').write_text('\n'.join(sorted(set(existing+large)))+'\n')
    print(json.dumps(dict(arrays=len(records),manifest=str(out/'MANIFEST.json'))))
if __name__=='__main__':main()
