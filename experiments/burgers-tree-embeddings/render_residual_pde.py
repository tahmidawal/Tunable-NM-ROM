"""Plot retained 4096² PDE outputs; no fitting, evolution or qualification."""
import argparse
from common import *
from neural.build_data import resolve
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True)
    ap.add_argument('--source-job',default='b8treepde');ap.add_argument('--allow-partial',action='store_true')
    ap.add_argument('--display-size',type=int,default=512);a=ap.parse_args()
    assert 64<=a.display_size<=1024
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);meta_=preflight()
    ns=Path(os.environ['STUDY_NAMESPACE']);rp=resolve(ns/a.source_job/'output/result.json',ns)
    r=json.loads(rp.read_text());complete=r['status'] in ['complete_fixed_tree_diagnostic','complete_fixed_tree_solve']
    assert complete or a.allow_partial, 'Incomplete source requires --allow-partial'
    assert not r['final_test_opened']
    outputs=r['outputs'];assert outputs and len(outputs)<=5, 'No actual output fields available to plot'
    assert len({float(v['time']) for v in outputs})==len(outputs)
    assert all(any(abs(float(v['time'])-t)<1e-10 for t in [.05,.10,.15,.20,.25]) for v in outputs)
    if complete:assert len(outputs)==5 and r['requested_outputs_complete']
    samples=[];verified=[];field_min=np.inf;field_max=-np.inf;error_max=0.
    # Only display arrays persist across iterations; one full-resolution NPZ at a time.
    idx=np.unique(np.rint(np.linspace(0,4095,a.display_size)).astype(int))
    for row in sorted(outputs,key=lambda v:v['time']):
        rec=row['fields'];assert rec is not None, 'Source omitted requested full fields'
        path=rp.parent/rec['file'];assert path.parent.resolve()==rp.parent.resolve()
        assert digest(path)==rec['sha256'] and row['shape']==[4096,4096]
        with np.load(path,allow_pickle=False) as f:
            pred=f['prediction'];truth=f['reference'];err=f['error']
            assert pred.shape==truth.shape==err.shape==(4096,4096)
            assert pred.dtype==truth.dtype==err.dtype==np.float64
            assert np.isclose(float(f['time']),row['time'],atol=1e-12,rtol=0)
            numerator=denominator=0.;max_consistency=0.
            for start in range(0,4096,128):
                sl=slice(start,start+128);p,t,e=pred[sl],truth[sl],err[sl]
                assert np.isfinite(p).all() and np.isfinite(t).all() and np.isfinite(e).all()
                max_consistency=max(max_consistency,float(np.max(np.abs((p-t)-e))))
                numerator+=float(np.sum(e*e));denominator+=float(np.sum(t*t))
                field_min=min(field_min,float(p.min()),float(t.min()));field_max=max(field_max,float(p.max()),float(t.max()))
                error_max=max(error_max,float(np.max(np.abs(e))))
            assert max_consistency<1e-12 and denominator>0
            actual=float(np.sqrt(numerator/denominator));assert np.isclose(actual,row['relative_error'],atol=1e-12,rtol=1e-10)
            samples.append((float(row['time']),truth[np.ix_(idx,idx)].copy(),pred[np.ix_(idx,idx)].copy(),np.abs(err[np.ix_(idx,idx)]).copy()))
            verified.append(dict(time=row['time'],file=str(path),sha256=rec['sha256'],shape=list(pred.shape),
                recomputed_relative_error=actual,reference_allowance=row['reference_allowance'],error_consistency_max=max_consistency))
            del pred,truth,err,p,t,e
    prefix='Completed diagnostic' if complete else 'PARTIAL diagnostic — missing outputs shown blank'
    title=f"{prefix}: {r['case']} · nonlinear neural tree · {r['nodes']} nodes"
    fig,axes=plt.subplots(3,5,figsize=(17,10),layout='constrained')
    fieldnorm=Normalize(field_min,field_max if field_max>field_min else field_min+1)
    errnorm=Normalize(0,max(error_max,1e-15));fieldimage=errorimage=None
    expected=[.05,.10,.15,.20,.25]
    for col,t in enumerate(expected):
        match=[s for s in samples if abs(s[0]-t)<1e-10]
        for ri in range(3):
            ax=axes[ri,col];ax.set_aspect('equal');ax.set_xlim(0,1);ax.set_ylim(0,1)
            if match:
                image=ax.imshow(match[0][ri+1].T,origin='lower',extent=(0,1,0,1),interpolation='nearest',
                    cmap='magma' if ri==2 else 'viridis',norm=errnorm if ri==2 else fieldnorm)
                if ri==2:errorimage=image
                else:fieldimage=image
            else:ax.text(.5,.5,'Not available',ha='center',va='center')
            if ri==0:ax.set_title(f't = {t:.2f}')
            if col==0:ax.set_ylabel(['Reference · y','Neural tree · y','Absolute error · y'][ri])
            if ri==2:ax.set_xlabel('x')
    fig.colorbar(fieldimage,ax=axes[:2,:],shrink=.8,label='Field value — common scale')
    fig.colorbar(errorimage,ax=axes[2,:],shrink=.8,label='Absolute error — common scale')
    fig.suptitle(title+'\nActual retained/scored arrays: 4096 × 4096; display: '+str(len(idx))+' × '+str(len(idx))+' samples\nFixed tree; no adaptation or method qualification',fontsize=14)
    fig.savefig(out/'pde_fields_actual4096.png',dpi=160);plt.close(fig)
    times=np.array([v['time'] for v in verified]);errors=np.array([v['recomputed_relative_error'] for v in verified]);allow=np.array([v['reference_allowance'] for v in verified])
    fig,(ax,initax)=plt.subplots(1,2,figsize=(12,4.8),gridspec_kw={'width_ratios':[3,1]},layout='constrained')
    ax.plot(times,100*errors,'o-',label='Actual4096 relative error');ax.plot(times,100*(errors+allow),'s--',label='Error + reference allowance')
    ax.plot(times,100*allow,':',label='Reference allowance');ax.axhline(1,color='crimson',ls='--',label='1% requirement')
    ax.set(xlabel='Time',ylabel='Relative L2 error (%)',title='Current-time reference normalization');ax.grid(alpha=.2);ax.legend(fontsize=9)
    initial=r['initialization']['initial_callable_relative_error']
    initax.bar(['Initialization\nQ'+str(r['Q'])+' fit grid'],[100*initial],color='#9970ab');initax.axhline(1,color='crimson',ls='--')
    initax.set(ylabel='Relative error (%)',title='Separate t = 0 fit\nNot actual4096 scoring')
    fig.suptitle(title);fig.savefig(out/'pde_error_over_time.png',dpi=160);plt.close(fig)
    timing=r['timing'];components=dict(timing['components'])
    components={'Source loading':r.get('source_load_seconds',0.),**components,
                'Explicit compilation':sum(timing['explicit_compilation'].values()),'PDE steps':timing['step_seconds_sum']}
    names=list(components);values=[float(components[k]) for k in names]
    fig,(ax,bx)=plt.subplots(1,2,figsize=(14,max(5,.35*len(names))),gridspec_kw={'width_ratios':[2,1]},layout='constrained')
    ax.barh([n.replace('_',' ') for n in names],values,color='#4c78a8');ax.invert_yaxis();ax.set(xlabel='Seconds',title='Instrumented components — incomplete decomposition')
    wall=float(timing['complete_cold_query_seconds']);elapsed=float(timing['elapsed_to_timing_snapshot_seconds'])
    bx.bar(['Cold online\nquery wall','Invocation to\ntiming snapshot'],[wall,elapsed],color=['#f58518','#999999'])
    bx.set(ylabel='Seconds',title='Enclosing measurements — not additive')
    for i,v in enumerate([wall,elapsed]):bx.text(i,v,f'{v:.2f}s',ha='center',va='bottom')
    fig.suptitle('Single diagnostic run; no paired CROM/SNF comparison\nCold online wall excludes conditioning, optional host transfer, reference scoring and archival I/O')
    fig.savefig(out/'pde_timing.png',dpi=160);plt.close(fig)
    metadata=dict(meta=meta_,source_job=a.source_job,source_result_sha256=digest(rp),source_status=r['status'],
        source_model_job=r['model_job'],source_model_sha256=r['model_sha256'],case=r['case'],layout=r['layout'],nodes=r['nodes'],
        complete_source=complete,display_shape=[len(idx),len(idx)],display_method='Uniform index subsampling including endpoints; no full-resolution scoring on display pixels',
        full_shape=[4096,4096],verified_fields=verified,normalization=r['normalization'],
        initialization=dict(time=0,relative_error=initial,Q=r['Q'],actual4096=False),
        common_field_range=[field_min,field_max],common_absolute_error_range=[0,error_max],
        timing=timing,timing_contract=r['timing_contract'],instrumented_components=components,
        method_admitted=False,qualification_claim=False,adaptation=r['adaptation'],final_test_opened=False,
        plot_hashes={p.name:digest(p) for p in sorted(out.glob('*.png'))})
    save(out/'report_metadata.json',metadata);print('rendered',len(verified),'verified full-resolution outputs',flush=True)


if __name__=='__main__':main()
