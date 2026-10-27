"""Freeze the audited minimum-worst no-HR endpoint before the sole A3 replay."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sys
import numpy as np

sys.dont_write_bytecode=True
os.environ['JAX_PLATFORMS']='cpu'  # schedule metadata only; no PDE solve occurs in this verifier.
import hr_interface as I
from analyse import sha

HERE=Path(__file__).resolve().parent
def raw_hash(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--result',required=True);p.add_argument('--config',required=True)
    p.add_argument('--audit-status',required=True);p.add_argument('--attempt',required=True)
    p.add_argument('--references',default='/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-10-06-jcp-references/experiments/jcp-references')
    a=p.parse_args();result=Path(a.result);config=Path(a.config);audit=Path(a.audit_status)
    j=json.loads(result.read_text());ar=json.loads(audit.read_text())['jobs'][a.attempt]
    assert ar['results'].startswith('READY') and j['complete'] and j['config']['cases']==[0,5,30,43]
    assert sha(config)==j['config_sha256']
    choices=[dict(subject_id=name,worst_error=r['worst_error'],median_error=r['median_error'])
        for name,r in j['selection'].items() if r['qualified_subset'] and r['eligible_cases']==[0,5,30,43]]
    assert choices;winner=min(choices,key=lambda r:(r['worst_error'],r['median_error'],r['subject_id']))['subject_id']
    rows={i:j['runs'][f'{winner}|case{i}'] for i in [0,5,30,43]};row=rows[0]
    assert all((r['scheme'],r['dt'],r['steps'],r['production_stopping_gtol'])==
        (row['scheme'],row['dt'],row['steps'],row['production_stopping_gtol']) for r in rows.values())
    selected=dict(subject_id=winner,scheme=row['scheme'],dt=row['dt'],steps=row['steps'],
        gtol=row['production_stopping_gtol'],tolf=I.R.LEVELS['prod'][1])
    ss=I.H.sched(selected['scheme'],selected['dt'],nmax=selected['steps'])
    schedule=dict({k:np.asarray(v).tolist() for k,v in ss.items() if k!='nsave'},gtol=selected['gtol'],tolf=selected['tolf'])
    assert sha(HERE/'inputs/Phi.npz')==I.PHI_SHA
    Phi=np.load(HERE/'inputs/Phi.npz')['Phi'][:,:512]
    sys.path.insert(0,a.references);import refstore as RS
    data={};pins={};reconstruction={};slot=np.flatnonzero(np.asarray(ss['slot'])>=0)
    for i,r in rows.items():
        q=result.parent/f'{winner}_c{i:03d}.npz';z=np.load(q,allow_pickle=False)
        C,W=z['C'],z['W'];assert C.shape==(selected['steps']+1,512) and W.shape==(6,512)
        assert C.dtype==W.dtype==np.float64 and np.isfinite(C).all() and np.isfinite(W).all()
        assert raw_hash(C)==r['coefficient_sha256'] and raw_hash(W)==raw_hash(C[slot])
        F=W@Phi.T;ref,rm=RS.load(3,'val64',i,'ST',field='finest');e=np.linalg.norm(F-ref,axis=1)/rm['judge']['n0']
        assert e.shape==(6,) and np.max(np.abs(e-np.asarray(r['score']['e_t'])))<=1e-10
        assert r['score']['eligible']
        data.update({f'C_c{i:03}':C,f'W_c{i:03}':W,f'fields_c{i:03}':F})
        pins[str(i)]=dict(path=str(q.resolve()),sha256=sha(q),source_C_raw_sha256=r['coefficient_sha256'],
            source_fused_fields_raw_sha256=r['output_sha256'])
        reconstruction[str(i)]=dict(method='NumPy W @ frozen Phi.T; reconstructed bytes are not asserted equal to old fused bytes',
            raw_sha256=raw_hash(F),six_source_errors=r['score']['e_t'],six_recomputed_errors=e.tolist())
    fixture=HERE/'inputs/fine_endpoint.npz';dest=HERE/'inputs/FINE-SELECTION.json'
    assert not fixture.exists() and not dest.exists(),'endpoint already frozen; no replacement'
    np.savez_compressed(fixture,**data)
    copies={}
    for src,name in [(result,'fine_tuning_result.json'),(config,'fine_tuning_config.json'),(audit,'fine_tuning_audit.json')]:
        dst=HERE/'inputs'/name;assert not dst.exists();shutil.copyfile(src,dst);copies[name]=dict(path=str(src.resolve()),sha256=sha(src))
    sel=dict(independent_results_gate='READY',source_audit_attempt=a.attempt,source_pins=copies,
        source_commit=j['source_commit'],source_job_id=j['job_id'],cohort_sha256=j['cohort_sha256'],
        eligible_candidates=choices,selection_rule='minimum worst error, then median error, then subject ID',selected=selected,
        actual_schedule=schedule,input_sha256={str(i):r['input_sha256'] for i,r in rows.items()},
        source_input_hash_kind='raw-contiguous-bytes',basis_sha256=I.PHI_SHA,source_arrays=pins,
        reference_fixture=str(fixture.relative_to(HERE)),reference_fixture_sha256=sha(fixture),
        reference_field_reconstruction=reconstruction,reference_store_version=RS.store_version())
    dest.write_text(json.dumps(sel,indent=2)+'\n')
    cfg=dict(attempt='k16fine01',driver='fine_replay',cases=[0,5,30,43],tighter_cases=[0,5],timing_cases=[0,5],
        timing_reps=2,burn_seconds=.2,cohort_seed=923801,cohort_count=64,rank=512,L=64,
        selection_manifest=str(dest.relative_to(HERE)),selection_sha256=sha(dest),wall_hours=.4,budget_hours=.35,
        scope='one A3 archived k16 control at independently audited stronger-policy no-HR minimum-worst endpoint')
    cp=HERE/'configs/k16fine01.json';assert not cp.exists();cp.write_text(json.dumps(cfg,indent=2)+'\n')
    print(json.dumps(dict(selected=selected,selection_sha256=sha(dest),fixture_sha256=sha(fixture)),indent=2))


if __name__=='__main__':main()
