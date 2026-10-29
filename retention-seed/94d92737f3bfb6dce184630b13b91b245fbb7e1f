"""Regenerate the five applicable empirical field R rows from audited raw arrays."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B))
from immutable_rule import array_digest
P=B/'artifacts/g2-cache-primary01/results/cache_primary_control'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(ref):
    path=P/ref['path']
    if sha(path)!=ref['file_sha256']:raise ValueError('raw replay file changed')
    with np.load(path,allow_pickle=False) as z:value=z['array']
    if array_digest(value)!=ref['content_sha256']:raise ValueError('raw replay array changed')
    return value
def main():
    report=json.loads((P/'control.json').read_text());records=[x for x in report['records'] if x['tag'].endswith('/common')]
    fine=next(x for x in records if x['tag']=='p4_g4_h3/common')
    weights_path=P/'caches/p4_g4_h3/common/arrays.npz'
    with np.load(weights_path,allow_pickle=False) as z:w=z['weights']
    reference=load(fine['arrays']['velocity']['expected']);den={'velocity':max(float(np.sqrt(np.sum(w[:,None]*reference**2))),1e-12*.2*float(np.sqrt(w.sum()))),'pressure':.2**2*float(np.sqrt(w.sum()))};rows={}
    for record in records:
        values={}
        for key in ('velocity','pressure'):
            refs=record['arrays'][key];difference=load(refs['actual'])-load(refs['expected']);squared=np.sum(difference**2,axis=1) if difference.ndim==2 else difference**2;numerator=float(np.sqrt(np.sum(w*squared)))
            values[key]={'numerator':numerator,'denominator':den[key],'value':numerator/den[key],'raw_array_references':refs}
        rows[record['tag'].removesuffix('/common')]=values
    result={'schema':'audited-Re20-baseline-field-R-v1','Re':20,'control_sha256':sha(P/'control.json'),'archive_sha256':'1cc1cdd437e4a4c1fb0f9cf348e7817bd30e11abf6bd0cbad7b81ff109df3cab','weights_file_sha256':sha(weights_path),'weights_content_sha256':array_digest(w),'rows':rows,'scope':'Only these five baseline states; no other parameters or S/Q states; force R must be independently retained','resolution_floor':1e-12}
    (B/'g2/RE20-REPLAY-NOISE.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
