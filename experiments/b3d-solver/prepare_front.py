"""Freeze original all64 coefficient identities and their distinct Q1/Q2 evidence; CPU only."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE=Path(__file__).resolve().parent
DIMS=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-18-jcp-dims/experiments/jcp-dims')
SPEED=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-14-jcp-speed3d/experiments/jcp-speed3d')
PICKS=[('m3a','W256|lat8192|qn|BDF2|0.025|k4'),('k3b','W512k8|lat16384|fs|CN|0.01|K1'),
       ('m3b','W768|gl40|fs|CN|0.01|K1'),('m3b','W1024|gl56|fs|CN|0.01|K1'),
       ('q3r5','M2_512|gl32|qn|BDF2|0.016666667|k4')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pin(p):return dict(path=str(p),sha256=sha(p))

def main():
    ap=DIMS/'checks/analysis-3d.json';sp=SPEED/'checks/analysis-grid.json'
    analysis=json.loads(ap.read_text());speed=json.loads(sp.read_text())
    arrays={};out=dict(sources=dict(dims_analysis=pin(ap),speed_analysis=pin(sp)),entries={})
    for attempt,key in PICKS:
        d=DIMS/f'runs/{attempt}/slim/output/d3run_{attempt}'
        wp=d/f'W_{attempt}.npz';rp=d/'result.json';z=np.load(wp,allow_pickle=False)
        rawkey=key if attempt!='q3r5' else key+'#q|'+'|'.join(key.split('|')[2:])
        ids=np.asarray(z['cases|'+rawkey],int);assert np.array_equal(ids,np.arange(64))
        W=np.asarray(z[rawkey]);assert W.shape[:2]==(64,6) and np.isfinite(W).all()
        arrays[key]=W
        cert=analysis['configurations'].get(key)
        if attempt=='q3r5':
            q1=speed['configs'][key]
            assert all(q1[k] for k in ('Ea','Eb','Ec','adt_valid','adt_gate','evidence_ok','selectable'))
            # Q1 and Q2 come from different archived invocations; prove their coefficient identity.
            gp=SPEED/'runs/g1/archive/output/W_grid.npz';gz=np.load(gp,allow_pickle=False)
            gi=np.asarray(gz['cases|'+key],int);assert np.array_equal(gi,ids)
            relative=float(np.linalg.norm(W-gz[key])/np.linalg.norm(gz[key]));assert relative<=1e-9
            cert=dict(Q1_source_record=q1,Q1_pass=True,Q2=analysis['rounds'][key],qualified=True,
                      Q1_Q2_coefficient_identity_relative=relative,Q1_coefficients=pin(gp))
        elif key.startswith('W1024'):
            assert cert['Q1_pass']
            raw=json.loads(rp.read_text());block=next(b['result'][key] for b in raw['blocks'] if key in b.get('result',{}))
            finerpath=HERE/'results/qcheck801/raw/output/result.json';finer=json.loads(finerpath.read_text())
            assert finer['complete'] and finer['config']['cases']==list(range(8))
            assert finer['cohort_sha256']==raw['cohort']['table_sha256']
            assert block['cases']==list(range(64)) and all(block['valid']) and all(block['next_valid'])
            assert max(block['dist_next'])<=2.5e-4 and np.isfinite(block['rho']).all() and np.max(block['rho'])<=.116
            assert all(r['valid'] and r['dist_gl64_gl80']<=2.5e-4 for r in finer['runs'].values())
            cert=dict(cert,qualified=True,Q2=dict(label='pass',scope='historical full64 gl56→64 plus original first8 gl64→80',
                main_population=64,finer_population=list(range(8)),worst_dist_qp=max(block['dist_next']),rho_max=float(np.max(block['rho'])),
                check=[True,max(r['dist_gl64_gl80'] for r in finer['runs'].values())],original_main_record=block,
                finer_result=pin(finerpath),finer_fresh_bad=sum(sum(not(g<=1e-3) for g in r['fresh_gamma']) for r in finer['runs'].values()),
                finer_steps=sum(len(r['fresh_gamma']) for r in finer['runs'].values()),finer_rule_itself_qualified=False))
        else:assert cert['qualified'] and cert['Q1_pass'] and cert['Q2']['label']=='pass'
        out['entries'][key]=dict(ordered_cases=ids.tolist(),raw_key=rawkey,coefficients=pin(wp),run=pin(rp),
            cohort=json.loads(rp.read_text())['cohort'],qualification=cert)
    np.savez_compressed(HERE/'inputs/front_archive.npz',**arrays)
    out['archive_sha256']=sha(HERE/'inputs/front_archive.npz')
    (HERE/'inputs/front_archive.json').write_text(json.dumps(out,indent=1)+'\n')
    print(json.dumps(dict(entries=len(out['entries']),bytes=(HERE/'inputs/front_archive.npz').stat().st_size)))

if __name__=='__main__':main()
