"""Exercise the production metadata and scoring adapter against ordinary public references."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from panel import check_upstream,HERE
from published import Opened
from open_sealed import sha
ap=argparse.ArgumentParser()
ap.add_argument('base',type=Path); ap.add_argument('dim',type=int)
a=ap.parse_args()
check_upstream(a.base)
sys.path.insert(0,str(a.base))
solver=__import__(f'plateau{a.dim}d')
ref=solver.RS
data=object.__new__(Opened)
data.bytes=ref._read
data.status_sha=sha((a.base/'vendor/tree/experiments/jcp-references/store/STATUS.json').read_bytes())
store=data.refstore(ref)
co={2:'dev6',3:'dev4'}[a.dim]
records=[]
for case in (0,1):
    for k in ('S','ST'):
        f,m=ref.load(a.dim,co,case,k)
        actual,meta=store.load(a.dim,co,case,k)
        assert meta==m and np.array_equal(f,actual)
        delta=actual*1.00001
        historical=ref.score(delta,a.dim,co,case,k)
        current=store.score(delta,a.dim,co,case,k)
        assert store.reference_status(a.dim,co,case,k)==ref.reference_status(a.dim,co,case,k)
        assert current['e']==historical['e'] and np.array_equal(current['e_t'],historical['e_t'])
        records.append({'case':case,'kind':k,'e':current['e'],'eligible':current['eligible']})
print(json.dumps({'dim':a.dim,'cohort':co,'passed':True,'matched':records}))
