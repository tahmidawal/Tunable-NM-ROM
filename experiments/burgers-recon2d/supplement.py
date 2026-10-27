"""One pre-reconstruction extra time rung for only failed coarse-input admissions.

Runtime NST capacity increases only; the vendored PDE/time equations are unchanged.
Original records/levels are copied into a new output tree, never overwritten.
"""
from pathlib import Path
import argparse,json,shutil,time
import numpy as np
import adapter as A
import data_manifest as DM

EXTRA_DT=.00015625
CAPACITY=2000

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',required=True); ap.add_argument('--out',required=True); args=ap.parse_args()
    src=Path(args.data); out=Path(args.out)
    if out.exists() and any(out.iterdir()): raise RuntimeError('fresh output required')
    out.mkdir(parents=True,exist_ok=True)
    A.verify_inputs(); A.write_json(out/'runtime.json',A.runtime())
    source_manifest_sha256=DM.verify(src)
    A.RC.NST=CAPACITY
    source_summary=json.loads((src/'coarse/summary.json').read_text())
    records=[]; failures=[]
    # Bases are shared read-only. Their original hashes/manifest stay bound.
    (out/'bases').symlink_to((src/'bases').resolve(),target_is_directory=True)
    (out/'coarse').mkdir()
    before=sum(not r['accepted'] for r in source_summary)
    for L in (32,64,128):
        solve=A.RC.make_solver(2,L,adv='central',max_newton=30,lin_maxiter=400)
        for row in [r for r in source_summary if r['L']==L]:
            key=row['key']; old=src/'coarse'/key; new=out/'coarse'/key
            md=json.loads(old.with_suffix('.json').read_text())
            if md['accepted']:
                shutil.copyfile(old.with_suffix('.json'),new.with_suffix('.json')); shutil.copyfile(old.with_suffix('.npz'),new.with_suffix('.npz'))
                records.append(row); continue
            with np.load(old.with_suffix('.npz')) as z: fields=np.asarray(z['fields']); ref=np.asarray(z['reference_levels']); desc=np.asarray(z['desc'])
            if md['levels'][-1]['dt']!=.0003125: raise RuntimeError('unexpected failed ladder endpoint')
            u0=A.CH.ic_2d(L,desc)[1:-1,1:-1].reshape(-1)
            t=time.perf_counter(); ans=solve(u0,desc[-1],EXTRA_DT,'cnr',1e-11,1e-12); A.sync(ans)
            u=np.asarray(ans['out']); n=int(ans['stages']); residual=np.asarray(ans['res'])[:n]
            complete=bool(n==A.RC.n_stages(EXTRA_DT,'cnr') and int(ans['units'])==2*int(round(.25/EXTRA_DT)))
            solver=bool(complete and np.isfinite(u).all() and np.isfinite(residual).all() and np.all(residual<=2e-11))
            gap=float(np.max(np.linalg.norm(u[1:]-fields[-1,1:],axis=1))/(md['n0']*L/A.LF))
            level=dict(dt=EXTRA_DT,stages=n,units=int(ans['units']),complete=complete,minimum_field=float(u.min()),maximum_field=float(u.max()),max_residual=float(residual.max()),solver_pass=solver,relative_refinement_gap=gap,seconds=time.perf_counter()-t,newton=np.asarray(ans['nwt'])[:n].tolist(),residual=residual.tolist(),compile_seconds=solve.compile_seconds)
            md['levels'].append(level); md['accepted']=bool(solver and md['levels'][-2]['solver_pass'] and gap<=md['tolerance']); md['selected']+=1; md['input_sha256']=A.sha_array(u)
            md['amendment']=dict(extra_rung=EXTRA_DT,runtime_NST=CAPACITY,original_json_sha256=A.sha_file(old.with_suffix('.json')),original_npz_sha256=A.sha_file(old.with_suffix('.npz')),selection='only failed original coarse-input admissions; no reconstruction outcomes observed')
            np.savez(new.with_suffix('.npz'),fields=np.concatenate([fields,u[None]],axis=0),reference_levels=ref,desc=desc)
            A.write_json(new.with_suffix('.json'),md)
            rr={k:v for k,v in md.items() if k not in ['levels','reference_meta','amendment']}; records.append(rr)
            if not md['accepted']: failures.append(key)
            A.log('extra rung',key,'accepted',md['accepted'],'gap',gap)
    A.write_json(out/'coarse/summary.json',records)
    A.write_json(out/'COMPLETE.json',dict(before_failed=before,after_failed=len(failures),remaining_failures=failures,extra_dt=EXTRA_DT,runtime_NST=CAPACITY,source_summary_sha256=A.sha_file(src/'coarse/summary.json'),no_reconstruction_performed=True,source_consumed_manifest_sha256=source_manifest_sha256))
    DM.build(out)
if __name__=='__main__': main()
