"""Separate sampling discrepancy from nonlinear stopping using saved native-tight rollouts."""
from pathlib import Path
import hashlib,json
import numpy as np
from analyse import sha

HERE=Path(__file__).resolve().parent
run=HERE/'runs/pilot/collected/output';j=json.loads((run/'result.json').read_text())
ref=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references/store')
assert sha(ref/'MANIFEST.sha256')==j['reference']['store_version']
manifest={line.split(maxsplit=1)[1]:line.split()[0] for line in (ref/'MANIFEST.sha256').read_text().splitlines()}
basis=HERE/'inputs/Phi.npz';assert sha(basis)==j['artifacts']['files']['Phi.npz']
Phi=np.load(basis)['Phi'][:,:512];gram=Phi.T@Phi
out=dict(source_result_sha256=sha(run/'result.json'),basis_sha256=sha(basis),scope='same four selected validation cases; native tight solves',
         cases={},arrays={},field_bar=j['config']['admission']['field_distance'])
for i in j['config']['cases']:
    m=ref/f'3d/val64/ST/c{i:03}.json';assert sha(m)==manifest[str(m.relative_to(ref))]
    n0=json.loads(m.read_text())['judge']['n0'];w={}
    for arm in ['k8','k16','none']:
        p=run/'arrays'/f'{arm}_c{i:03}_tight.npz';out['arrays'][str(p.resolve())]=sha(p);w[arm]=np.load(p)['W']
    row={}
    for arm in ['k8','k16']:
        delta=w[arm][1:]-w['none'][1:]
        row[f'{arm}_to_none']=float(np.sqrt(np.maximum(np.sum((delta@gram)*delta,axis=1),0)).max()/n0)
    out['cases'][str(i)]=row
out['k16_still_fails_field_bar']=max(x['k16_to_none'] for x in out['cases'].values())>out['field_bar']
(HERE/'checks/pilot/tight-vs-none.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out['cases'],indent=2))
