"""Independent CPU reconstruction and accounting audit of the dynamics pilot.

No JAX or experiment solver imports. Reconstructs all first-repeat ROM full fields
from saved bank/coefficients, and common fields for every method. Full-mesh timed
FOM errors at the finer grid are not reconstructable for the nonreference CFL;
those are explicitly outside this audit. References are retained, not regenerated.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import scipy.linalg

NAMES=("displacement","velocity","energy_state")

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def restriction(a,n,target,bc):
    r=n//target;s=slice(r-1,None,r) if bc=='dirichlet' else slice(None,None,r)
    return a[...,s,s]
def mass(n,bc):
    w=np.ones(n-1 if bc=='dirichlet' else n+1)/n
    if bc!='dirichlet':w[[0,-1]]*=.5
    return w[:,None]*w[None,:]
def energy2(u,v,n,bc,c):
    m=mass(n,bc)
    if bc=='dirichlet':
        up=np.pad(u,[(0,0)]*(u.ndim-2)+[(1,1),(1,1)])
        potential=np.sum(np.diff(up,axis=-1)**2,axis=(-2,-1))+np.sum(np.diff(up,axis=-2)**2,axis=(-2,-1))
    else:
        w=np.ones(n+1);w[[0,-1]]*=.5
        potential=np.sum(np.diff(u,axis=-1)**2*w[:,None],axis=(-2,-1))+np.sum(np.diff(u,axis=-2)**2*w,axis=(-2,-1))
    return np.sum(m*v*v,axis=(-2,-1))+c*c*potential

def recompute(u,v,ut,vt,n,bc,c):
    l2=lambda x:np.sqrt(np.sum(mass(n,bc)*x*x,axis=(-2,-1)))
    phase=np.sqrt(energy2(ut[0],vt[0],n,bc,c));u0=l2(ut[0])
    result={}
    for name,err,scale,current in zip(NAMES,(l2(u-ut),l2(v-vt),np.sqrt(np.maximum(0,energy2(u-ut,v-vt,n,bc,c)))),(u0,phase,phase),(l2(ut),l2(vt),np.sqrt(np.maximum(0,energy2(ut,vt,n,bc,c))))):
        result[name]={'absolute':err,'initial_normalized':err/scale,'reference_norm':current,'initial_scale':scale}
    return result

def compare_metrics(actual,saved):
    differences=[]
    for name in NAMES:
        for key in actual[name]:
            differences.append(float(np.max(abs(np.asarray(actual[name][key])-np.asarray(saved[name][key])))))
        differences.append(abs(float(np.max(actual[name]['initial_normalized']))-saved[name]['max_initial_normalized']))
    return max(differences)

def geometry(head,transform,z,w=None):
    def silu_all(x):
        sig=1/(1+np.exp(-x));base=sig*(1-sig)
        return x*sig,sig+x*base,2*base+x*base*(1-2*sig)
    x1=z@head['p/l1/w']+head['p/l1/b'];h1,d1,s1=silu_all(x1)
    x2=h1@head['p/l2/w']+head['p/l2/b'];h2,d2,s2=silu_all(x2)
    scale=float(head['frozen/output_scale'])
    a=transform@(head['p/bias']+head['p/linear']@z+scale*(h2@head['p/out/w']+head['p/out/b']))
    jac=transform@(head['p/linear']+scale*head['p/out/w'].T@(d2[:,None]*head['p/l2/w'].T)@(d1[:,None]*head['p/l1/w'].T))
    if w is None:return a,jac
    t1=w@head['p/l1/w'];t2=(d1*t1)@head['p/l2/w']
    curvature=transform@(scale*((s2*t2*t2+d2*((s1*t1*t1)@head['p/l2/w']))@head['p/out/w']))
    return a,jac@w,jac,curvature

def audit(record):
    cluster=record/'cluster';native=cluster/'out/pilot';out=record/'analysis';out.mkdir(exist_ok=True)
    result=json.loads((native/'result.json').read_text());cfg=result['config'];meta=result['provenance']
    assert result['complete'] and not result['final_test_opened'] and meta['jax_backend']=='gpu' and meta['x64'] and meta['matmul_precision']=='highest'
    cleanup=json.loads((record/'cleanup.json').read_text());assert cleanup['remote_deleted_and_absence_checked'] and cleanup['all_three_manifests_verified']
    assert (cluster/'EXIT_CODE').read_text().strip()=='0'
    log=(cluster/'logs'/f"{meta['job_id']}.out").read_text();err=(cluster/'logs'/f"{meta['job_id']}.err").read_text()
    assert 'jax_backend=gpu' in log
    for bad in ('captured constant','Captured constant','out of memory','RESOURCE_EXHAUSTED','Traceback','No space left','cuInit'):
        assert bad not in log+err,bad
    submission=json.loads((record/'submission.json').read_text())
    for path,sha in submission['source_hashes'].items():
        assert digest(cluster/path)==sha
        relative='experiments/'+str(Path(path).relative_to('code'))
        payload=subprocess.check_output(['git','show',submission['source_commit']+':'+relative],cwd=Path(__file__).resolve().parents[2])
        assert hashlib.sha256(payload).hexdigest()==sha
    for name,sha in result['output_sha256'].items():assert digest(native/name)==sha
    groups=defaultdict(list)
    for row in result['invocations']:groups[row['boundary'],row['intervals'],row['case'],row['method'],row['setting']].append(row)
    reconstruction=[];diagnostics=[];linear_checks=[];training=[]
    for bc in cfg['boundaries']:
        with np.load(native/f'training_ladder_{bc}.npz') as tt:
            a=tt['a'].reshape(-1,64);center=a.mean(axis=0)
            np.testing.assert_allclose(center,tt['center'],atol=1e-14)
            np.testing.assert_allclose((a-center).T@(a-center)/len(a),tt['covariance'],atol=1e-14)
            np.testing.assert_allclose(tt['basis'].T@tt['basis'],np.eye(64),atol=2e-14)
            np.testing.assert_allclose((a-center).T@(a-center)@tt['basis'],tt['basis']*tt['singular_values']**2,atol=1e-10)
            tmeta=next(x for x in result['training'] if x['boundary']==bc)
            assert hashlib.sha256(tt['a'].tobytes()).hexdigest()==tmeta['training_coefficients_sha256']
            training.append(dict(boundary=bc,saved_projector_defect=tmeta['saved_projector_defect'],saved_center_defect=tmeta['saved_center_defect'],data_hashes_match=tmeta['data_hashes_match'],basis_covariance_verified=True))
        with np.load(cluster/'in'/bc/'head.npz') as hh:head={k:hh[k] for k in hh.files}
        for n in cfg['meshes']:
            with np.load(native/f'mesh_{bc}_{n}.npz') as mm:mesh={k:mm[k] for k in mm.files}
            g=mesh['g'];m=mesh['mass']
            np.testing.assert_allclose(g.T@(m[:,None]*g),np.eye(64),atol=2e-10)
            for ci in cfg['validation_indices']:
                with np.load(native/f'samegrid_{bc}_{n}_{ci}.npz') as ss:su,sv=ss['u'],ss['v']
                with np.load(native/f'reference_{bc}_{ci}.npz') as rr:ut,vt=rr['u'],rr['v']
                for key,rows in groups.items():
                    if key[:3]!=(bc,n,ci):continue
                    assert sorted(r['repetition'] for r in rows)==list(range(cfg['repetitions']))
                    assert len({json.dumps(r['output_sha256'],sort_keys=True) for r in rows})==1
                    first=next(r for r in rows if r['repetition']==0);c=first['parameters'][5]
                    with np.load(native/(first['invocation_id']+'.npz')) as ff:field={k:ff[k] for k in ff.files}
                    u,v=field['u'],field['v']
                    physical=recompute(u,v,ut,vt,cfg['comparison_intervals'],bc,c)
                    discrepancy=max(compare_metrics(physical,row['physical_reference_error']) for row in rows)
                    native_difference=None;field_difference=None
                    if 'coefficients' in field:
                        shape=(49,n-1,n-1) if bc=='dirichlet' else (49,n+1,n+1)
                        uf=(field['coefficients']@g.T).reshape(shape);vf=(field['velocity_coefficients']@g.T).reshape(shape)
                        field_difference=max(float(np.max(abs(restriction(a,n,cfg['comparison_intervals'],bc)-b))) for a,b in ((uf,u),(vf,v)))
                        native_difference=max(compare_metrics(recompute(uf,vf,su,sv,n,bc,c),row['same_grid_discrepancy']) for row in rows)
                    if key[3]=='rom':
                        actual=[geometry(head,mesh['transform'],z,w)[:2] for z,w in zip(field['rollout_z'],field['rollout_w'])]
                        np.testing.assert_allclose(np.stack([x[0] for x in actual]),field['coefficients'],atol=2e-12)
                        np.testing.assert_allclose(np.stack([x[1] for x in actual]),field['velocity_coefficients'],atol=2e-11)
                    elif key[3] in ('affine16','affine32','full64'):
                        arm=key[3];basis,offset=mesh[arm+'_basis'],mesh[arm+'_center'];dim=basis.shape[1]
                        gen=np.zeros((2*dim+1,2*dim+1));gen[:dim,dim:2*dim]=np.eye(dim)
                        gen[dim:2*dim]=np.linalg.solve(basis.T@basis,np.column_stack((-c*c*basis.T@mesh['stiffness']@basis,-c*basis.T@mesh['damping']@basis,-c*c*basis.T@mesh['stiffness']@offset)))
                        gd=float(np.max(abs(gen-field['generator'])))
                        truth=np.stack([scipy.linalg.expm(gen*i*cfg['observation_dt'])@field['states'][0] for i in range(49)])
                        sd=float(np.max(abs(truth-field['states'])))
                        assert gd<1e-7 and sd<1e-7
                        linear_checks.append(dict(invocation_id=first['invocation_id'],generator_difference=gd,state_difference=sd))
                    for row in rows:
                        components=sum(v for k,v in row['seconds'].items() if k!='complete_query')
                        assert abs(components-row['seconds']['complete_query'])<1e-12
                        assert row['output_bytes']==2*8*49*su.shape[-1]**2
                        assert row['completed']
                    assert discrepancy<2e-10 and (native_difference is None or native_difference<2e-9) and (field_difference is None or field_difference<2e-10)
                    reconstruction.append(dict(invocation_id=first['invocation_id'],repetitions_verified=len(rows),physical_metric_difference=discrepancy,native_grid_ROM_metric_difference=native_difference,common_field_difference=field_difference))
                drow=next(x for x in result['diagnostics'] if (x['boundary'],x['intervals'],x['case'])==(bc,n,ci))
                with np.load(native/f'diagnostic_{bc}_{n}_{ci}.npz') as ff:fit={k:ff[k] for k in ff.files}
                budget=cfg['diagnostic_fit_budgets'][-1];selected=fit[f'fit_{budget}_selected'];z=fit[f'fit_{budget}_z'][np.arange(len(selected)),selected]
                for i,zi in enumerate(z):
                    a,jac=geometry(head,mesh['transform'],zi)
                    w=np.linalg.lstsq(jac,fit['velocities'][i],rcond=None)[0]
                    aa,b,jj,curvature=geometry(head,mesh['transform'],zi,w)
                    q=np.linalg.qr(jj)[0];force=-c*c*mesh['stiffness']@aa-c*mesh['damping']@b-curvature
                    normal=force-q@(q.T@force)
                    np.testing.assert_allclose(normal,fit['mlp16_normal_force'][i],atol=2e-8,rtol=2e-10)
                    np.testing.assert_allclose(curvature,fit['mlp16_curvature'][i],atol=2e-8,rtol=2e-10)
                for arm in drow['arms']:
                    name=arm['arm'];a=fit[name+'_coefficients'];b=fit[name+'_velocity_coefficients']
                    ur=(a[:-1]@g.T).reshape(fit['truth_u'].shape);vr=(b[:-1]@g.T).reshape(fit['truth_v'].shape)
                    difference=compare_metrics(recompute(ur,vr,fit['truth_u'],fit['truth_v'],n,bc,c),arm['snapshot_metrics'])
                    assert difference<2e-9
                    assert abs(np.linalg.norm(a[-1])-arm['zero_field_mass_norm'])<1e-12
                    diagnostics.append(dict(boundary=bc,intervals=n,case=ci,arm=name,snapshot_metric_difference=difference,zero_mass_norm=float(np.linalg.norm(a[-1]))))
    report=dict(passed=True,scope=__doc__,result_sha256=digest(native/'result.json'),training=training,
        reconstructed_queries=reconstruction,linear_controls=linear_checks,diagnostics=diagnostics,
        audited_invocations=sum(x['repetitions_verified'] for x in reconstruction),file_hashes=len(result['output_sha256']),
        maximum_physical_metric_difference=max(x['physical_metric_difference'] for x in reconstruction),
        maximum_ROM_native_grid_metric_difference=max(x['native_grid_ROM_metric_difference'] or 0 for x in reconstruction),
        final_cohort_opened=False)
    write(out/'audit.json',report);print(json.dumps({k:report[k] for k in ('passed','audited_invocations','file_hashes','maximum_physical_metric_difference','maximum_ROM_native_grid_metric_difference')},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('record',type=Path);audit(ap.parse_args().record)
