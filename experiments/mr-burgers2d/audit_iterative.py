"""Independent NumPy/SciPy audit; no JAX import or new GPU solve."""
import argparse,collections,hashlib,itertools,json,pickle,subprocess
from pathlib import Path
import numpy as np
from scipy.special import expit


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(1024*1024):h.update(chunk)
    return h.hexdigest()

def errors(f,t):
    delta=np.linalg.norm((f-t).reshape(len(f),-1),axis=1);n0=np.linalg.norm(t[0]);norm=np.linalg.norm(t.reshape(len(t),-1),axis=1)
    return dict(fixed_initial_per_time=(delta/n0).tolist(),current_relative_per_time=(delta/np.maximum(norm,1e-300)).tolist(),
        fixed_initial_max=float(max(delta/n0)),current_relative_max=float(max(delta/np.maximum(norm,1e-300)))),delta

def mlp(layers,x):
    for i,(w,b) in enumerate(layers):
        x=x@w+b
        if i<len(layers)-1:x=x*expit(x)
    return x

def head(p,z):
    x=z
    if 'hB' in p:
        a=2*np.pi*(z@p['hB']);x=np.concatenate((z,np.sin(a),np.cos(a)),axis=-1)
    return mlp(p['h'],x)+z@p['h_lin']

def features(p,xy):
    a=2*np.pi*(xy@p['B']);x,y=xy.T
    return (p['out_scale']*16*x*(1-x)*y*(1-y))[:,None]*mlp(p['g'],np.concatenate((np.sin(a),np.cos(a)),axis=-1))

def spatial(f,L):
    c=f[1:-1,1:-1];xm=f[:-2,1:-1];xp=f[2:,1:-1];ym=f[1:-1,:-2];yp=f[1:-1,2:]
    adv=c*L*(np.where(c>0,c-xm,xp-c)+np.where(c>0,c-ym,yp-c));lap=L*L*(xm+xp+ym+yp-4*c)
    return adv,lap

def sample_field(f,xy,L):
    x=xy*L;lo=np.minimum(np.floor(x).astype(int),L-1);t=x-lo;i,j=lo.T;a,b=t.T
    return (1-a)*(1-b)*f[i,j]+a*(1-b)*f[i+1,j]+(1-a)*b*f[i,j+1]+a*b*f[i+1,j+1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('record',type=Path);a=ap.parse_args();record=a.record;run=record/'archive';out=run/'out'
    d=json.loads((out/'result.json').read_text());cfg=d['config'];root=Path(__file__).resolve().parents[2]
    assert d['complete'] and d['backend']=='gpu' and d['x64'] and d['matmul_precision']=='highest' and d['network_weights_frozen']
    assert d['checkpoint_sha256']==d['checkpoint_sha256_after']==sha(run/'in/checkpoint.pkl')
    assert d['commit']==(run/'COMMIT.txt').read_text().strip() and str(d['job_id'])==json.loads((record/'SUBMISSION.json').read_text())['job_id']
    for r in json.loads((run/'PROVENANCE.json').read_text()):
        saved=subprocess.check_output(['git','show',r['commit']+':'+r['source']],cwd=root)
        assert hashlib.sha256(saved).hexdigest()==sha(run/r['staged'])==r['sha256']
    check=json.loads((record/'COLLECTION-CHECK.json').read_text());assert check['archive_verified']
    stdout='\n'.join(x.read_text() for x in (run/'logs').glob('*.out'));stderr='\n'.join(x.read_text() for x in (run/'logs').glob('*.err'))
    assert 'jax_backend=gpu' in stdout and 'ITERATIVE BURGERS COMPLETE' in stdout and 'ALL-DONE' in stdout
    assert not stderr.strip(),stderr
    cases=range(cfg['cases']);reps=range(cfg['reps']);meshes=cfg['meshes'];largest=max(meshes)
    names=['nmrom']+[s['name'] for s in cfg['fom_settings']]
    actual=[(r['intervals'],r['name'],r['case'],r['rep']) for r in d['invocations']]
    expected=set(itertools.product(meshes,names,cases,reps));assert len(actual)==len(expected) and set(actual)==expected
    rng=np.random.default_rng(cfg['seed']);n=cfg['cases']
    physical=np.stack([rng.uniform(.15,.85,n),rng.uniform(.15,.85,n),rng.uniform(.05,.2,n),rng.uniform(.5,2.,n),np.exp(rng.uniform(np.log(.01),np.log(.1),n))],1)
    assert np.array_equal(physical,np.array(d['physical_cases']))
    order_rng=np.random.default_rng(cfg['order_seed'])
    for L,rep,case in itertools.product(meshes,reps,cases):
        assert [names[i] for i in order_rng.permutation(len(names))]==[r['name'] for r in d['invocations'] if (r['intervals'],r['rep'],r['case'])==(L,rep,case)]
    ck=pickle.load(open(run/'in/checkpoint.pkl','rb'));params=ck['params']
    rf=cfg['reference_mesh'];rt=cfg['reference_dt'];refs={};ref_delta=0.
    for r in d['reference']:
        z=np.load(out/r['artifact']);f=z['fields'];assert f.shape==(6,largest+1,largest+1) and f.dtype==np.float64 and np.isfinite(f).all()
        assert max(z['residuals'])<2e-11 and abs(max(z['residuals'])-r['max_relative_residual'])<1e-15
        refs[r['intervals'],r['dt'],r['case']]=f
    for L in meshes:
        for row in d['reference_metrics'][str(L)]:
            case=row['case']
            def diff(n1,t1,n2,t2):return errors(refs[n1,t1,case][:,::largest//L,::largest//L],refs[n2,t2,case][:,::largest//L,::largest//L])[0]['fixed_initial_max']
            sd=diff(rf//2,rt,rf,rt);td=diff(rf,2*rt,rf,rt);sc=diff(rf//4,2*rt,rf//2,2*rt);sf=diff(rf//2,2*rt,rf,2*rt);tc=diff(rf,4*rt,rf,2*rt)
            calculated=dict(space_difference=sd,time_difference=td,margin=sd+td,spatial_coarse=sc,spatial_fine=sf,temporal_coarse=tc,observed_space_order=np.log2(sc/sf),observed_time_order=np.log2(tc/td))
            ref_delta=max(ref_delta,max(abs(v-row[k]) for k,v in calculated.items()));assert row['decrease']==bool(sf<sc and td<tc)
    assert ref_delta<1e-10
    groups=collections.defaultdict(list)
    for r in d['invocations']:groups[r['artifact']].append(r)
    worst_metric=worst_decoder=worst_weak=worst_fom=worst_initial=0.;weak_pairs=fom_pairs=0;summary=[]
    for L in meshes:
        ops=np.load(out/f'operators_L{L}.npz');info=next(x for x in d['mesh_setup'] if x['intervals']==L)
        assert info['instrumented_rom_relative_parity']<1e-11 and info['instrumented_rom_counter_parity']
        xy=ops['cold_xy'];w=ops['cold_w'];Q=ops['cold_Q'];R=ops['cold_R']
        # Independent spatial-track identity at every fitted point and every advection stencil.
        gc=features(params,xy)*w[:,None]
        assert np.linalg.norm(gc-Q@R)/np.linalg.norm(gc)<2e-12
        assert np.linalg.norm(Q.T@Q-np.eye(Q.shape[1]))<1e-10
        ij=np.stack(np.unravel_index(np.array(info['eq_indices']),(L-1,L-1)),1)+1
        offsets=np.array([[0,0],[1,0],[-1,0],[0,1],[0,-1]])
        g5=features(params,((ij[:,None,:]+offsets)/L).reshape(-1,2)).reshape(ops['G5'].shape)
        assert np.linalg.norm(g5-ops['G5'])/np.linalg.norm(g5)<2e-12
        modeids=np.array(info['mode_ids']);phi=(2./L)*np.sin(np.pi*ij[:,0,None]/L*modeids[:,0])*np.sin(np.pi*ij[:,1,None]/L*modeids[:,1])
        assert np.linalg.norm(phi*np.array(info['eq_weights'])[:,None]-ops['Pq'])<1e-12
        mode_lam=4*L*L*(np.sin(np.pi*modeids[:,0]/(2*L))**2+np.sin(np.pi*modeids[:,1]/(2*L))**2)
        assert np.max(abs(mode_lam-ops['lam']))<1e-10
        ns=min(31,L-1);ids=np.unique(np.rint(np.linspace(1,L-1,ns)).astype(int));ix,iy=np.meshgrid(ids,ids,indexing='ij');g=features(params,np.stack((ix.ravel()/L,iy.ravel()/L),1))
        for artifact,rows in groups.items():
            row=rows[0]
            if row['intervals']!=L:continue
            case=row['case'];nu=physical[case,4];z=np.load(out/artifact);f=z['fields'];truth=refs[rf,rt,case][:,::largest//L,::largest//L]
            assert f.shape==(6,L+1,L+1) and f.dtype==np.float64 and np.isfinite(f).all()
            assert np.max(abs(f[:,[0,-1],:]))==0 and np.max(abs(f[:,:,[0,-1]]))==0
            h=hashlib.sha256(f.tobytes()).hexdigest();metric,_=errors(f,truth)
            xx,yy=np.meshgrid(np.arange(L+1)/L,np.arange(L+1)/L,indexing='ij');cx,cy,width,amp,_=physical[case]
            u0=amp*np.exp(-((xx-cx)**2+(yy-cy)**2)/(2*width*width));u0[[0,-1]]=0;u0[:,[0,-1]]=0
            worst_initial=max(worst_initial,float(np.max(abs(u0-truth[0]))));assert np.max(abs(u0-truth[0]))<1e-14
            for row in rows:
                assert row['field_sha256']==h and row['output_bytes']==f.nbytes and row['gpu_seconds']>0 and row['host_seconds']>=row['gpu_seconds'] and row['finite']
                worst_metric=max(worst_metric,max(float(np.max(abs(np.array(v)-row['error'][k]))) for k,v in metric.items()))
                assert len(row['residuals'])==len(row['iterations'])==round(.25/row['dt'])
            if row['method']=='rom':
                internal=z['internal_latents'];assert np.array_equal(internal,np.array(row['internal_latents']))
                states=internal[::int(round(.05/row['dt']))];assert np.array_equal(states,np.array(row['latent_states']))
                heads=head(params,internal);pred=(g@head(params,states).T).T.reshape(6,len(ids),len(ids));actual=f[:,ids[:,None],ids[None,:]]
                worst_decoder=max(worst_decoder,float(np.linalg.norm(pred-actual)/np.linalg.norm(actual)))
                weak=[]
                for i in range(len(internal)-1):
                    h0,h1=heads[i:i+2];us=np.einsum('msr,r->ms',ops['G5'],h1);c,xp,xm,yp,ym=us.T
                    adv=c*L*(np.where(c>0,c-xm,xp-c)+np.where(c>0,c-ym,yp-c));ah=ops['A']@h1
                    residual=(ah-ops['A']@h0+row['dt']*(ops['Pq'].T@adv+nu*ops['lam']*ah))/(1+row['dt']*nu*ops['lam'])
                    weak.append(np.linalg.norm(residual))
                for r in rows:
                    weak_pairs+=len(weak);worst_weak=max(worst_weak,float(np.max(abs(np.array(weak)-r['residuals']))))
                    assert r['ic_reason'] in [0,1,2,3] and 0<=r['ic_iterations']<=180 and all(0<=v<=30 for v in r['iterations'])
                    assert set(r['stop_reasons'])<=set([0,1,2,3])
                    tolerance=1e-9*np.linalg.norm(sample_field(u0,xy,L)*w)*np.sqrt(len(w))
                    assert all(v<=tolerance+1e-10 for v,s in zip(weak,r['stop_reasons']) if s==1)
            else:
                prev=z['previous_output_steps'];assert prev.shape==(5,L+1,L+1)
                output_r=[]
                for i in range(5):
                    adv,lap=spatial(f[i+1],L);res=f[i+1,1:-1,1:-1]-prev[i,1:-1,1:-1]+row['dt']*(adv-nu*lap)
                    output_r.append(np.linalg.norm(res)/np.linalg.norm(prev[i,1:-1,1:-1]))
                for r in rows:
                    fom_pairs+=5;worst_fom=max(worst_fom,float(np.max(abs(np.array(output_r)-np.array(r['residuals'])[int(round(.05/r['dt']))-1::int(round(.05/r['dt']))]))))
                    assert r['nonlinear_converged']==bool(max(r['residuals'])<=r['ntol']*(1+1e-9))
                    assert r['linear_converged']==bool(all(max(x[:int(n)],default=0.)<=r['ltol']*(1+1e-7) for x,n in zip(r['linear_relative_residuals'],r['iterations'])))
                    assert np.array_equal(f[0],u0)
        for name in names:
            rows=[r for r in d['invocations'] if r['intervals']==L and r['name']==name];first=rows[0];case_gpu=[];case_host=[];outliers=0
            for case in cases:
                cr=[r for r in rows if r['case']==case];g=np.array([r['gpu_seconds'] for r in cr]);h=np.array([r['host_seconds'] for r in cr]);case_gpu.append(float(np.median(g)));case_host.append(float(np.median(h)));outliers+=int(np.sum(g>2*np.median(g)))
            worst=max(r['error']['fixed_initial_max'] for r in rows);worst_cur=max(r['error']['current_relative_max'] for r in rows)
            reference_pass=all(x['margin']<=cfg['target_fixed_initial']*cfg['reference_margin_fraction'] and x['decrease'] for x in d['reference_metrics'][str(L)])
            error_pass=all(r['error']['fixed_initial_max']+d['reference_metrics'][str(L)][r['case']]['margin']<=cfg['target_fixed_initial'] for r in rows)
            eligible=reference_pass and error_pass and (all(r['nonlinear_converged'] and r['linear_converged'] for r in rows) if first['method']=='fom' else all(r['ic_reason'] not in [0,3] and all(s not in [0,3] for s in r['stop_reasons']) for r in rows))
            summary.append(dict(intervals=L,nodes_per_axis=L+1,name=name,method=first['method'],gpu_ms=1000*float(np.median(case_gpu)),host_ms=1000*float(np.median(case_host)),
                case_gpu_ms=(1000*np.array(case_gpu)).tolist(),case_host_ms=(1000*np.array(case_host)).tolist(),worst_fixed_initial=worst,worst_current_relative=worst_cur,
                median_case_fixed_initial=float(np.median([max(r['error']['fixed_initial_max'] for r in rows if r['case']==c) for c in cases])),
                worst_initial_error=max(r['error']['fixed_initial_per_time'][0] for r in rows),
                worst_final_fixed_initial=max(r['error']['fixed_initial_per_time'][-1] for r in rows),
                worst_final_current_relative=max(r['error']['current_relative_per_time'][-1] for r in rows),
                median_total_iterations=float(np.median([sum(r['iterations']) for r in rows])),timing_outliers_gt2x_case_median=outliers,
                reference_pass=reference_pass,error_plus_margin_pass=error_pass,eligible_development_5percent=eligible,
                initial_reasons=dict(collections.Counter(r['ic_reason'] for r in rows)) if first['method']=='rom' else None,
                evolution_reasons=dict(collections.Counter(s for r in rows for s in r['stop_reasons'])) if first['method']=='rom' else None,
                nonlinear_failed_invocations=sum(not r['nonlinear_converged'] for r in rows) if first['method']=='fom' else None,
                linear_failed_invocations=sum(not r['linear_converged'] for r in rows) if first['method']=='fom' else None,
                maximum_newton_iterations=max(max(r['iterations']) for r in rows),ntol=first.get('ntol'),ltol=first.get('ltol')))
    assert worst_metric<1e-11 and worst_decoder<2e-11 and worst_weak<2e-10 and worst_fom<2e-10
    comparisons=[]
    for L in meshes:
        rom=next(r for r in summary if r['intervals']==L and r['name']=='nmrom')
        for fom in [r for r in summary if r['intervals']==L and r['method']=='fom']:
            comparisons.append(dict(intervals=L,fom=fom['name'],ratio_of_median_gpu_ms=fom['gpu_ms']/rom['gpu_ms'],ratio_of_median_host_ms=fom['host_ms']/rom['host_ms'],
                median_paired_gpu_ratio=float(np.median(np.array(fom['case_gpu_ms'])/rom['case_gpu_ms'])),both_eligible=rom['eligible_development_5percent'] and fom['eligible_development_5percent']))
    audit=dict(passed=True,job_id=d['job_id'],source_commit=d['commit'],gpu=d['gpu'],invocations=len(actual),unique_fields=len(groups),full_metric_max_delta=worst_metric,
        reference_metric_max_delta=ref_delta,sampled_decoder_relative_max_delta=worst_decoder,weak_residual_pairs=weak_pairs,weak_residual_max_delta=worst_weak,
        output_fom_residual_pairs=fom_pairs,output_fom_residual_max_delta=worst_fom,initial_reference_max_delta=worst_initial,
        scope='Saved full-field and source audit; independent output-step FOM residuals and every weak-step residual; linear inner residuals checked from source/records, not rebuilt from unavailable Newton corrections.')
    (record/'AUDIT.json').write_text(json.dumps(audit,indent=2)+'\n')
    panel=dict(pde='burgers2d',status='audited development',source_commit=d['commit'],job_id=d['job_id'],gpu=d['gpu'],config=cfg,checkpoint_sha256=d['checkpoint_sha256'],
        model=dict(K=d['K'],R=d['R'],M=d['M'],m=d['m'],operator='preassembled linear weak operators and sampled full sign-upwind advection; not historical r64 polynomial tensor'),
        norm='fixed initial L2 primary; current-relative also reported',reference='regenerated refined implicit upwind FOM, empirical continuum refinement margins',reference_metrics=d['reference_metrics'],
        timing_contract=d['timing_contract'],aggregation='median of per-case repetition medians; ratio of cohort medians and median paired ratios separately',rows=summary,comparisons=comparisons,audit=audit)
    (record/'PANEL.json').write_text(json.dumps(panel,indent=2)+'\n');print(json.dumps(audit),flush=True)

if __name__=='__main__':main()
