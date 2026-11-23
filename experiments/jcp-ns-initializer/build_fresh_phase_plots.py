"""All-case fresh physical trajectories; missing/failed cases are never dropped."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

HERE=Path(__file__).resolve().parent
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
    'svg.hashsalt':'jcp-fresh-phase-20261123','pdf.fonttype':42,'ps.fonttype':42})
COLORS={'head_common':'#1768ac','span_common':'#c14440','span_native':'#84735c'}
LABELS={'head_common':'Head · common initial field','span_common':'Span · common initial field','span_native':'Span · native initial field'}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def plain(x):
    if isinstance(x,dict):return {str(k):plain(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [plain(v) for v in x]
    if isinstance(x,np.ndarray):return plain(x.tolist())
    if isinstance(x,np.generic):return plain(x.item())
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def dump(path,value):Path(path).write_text(json.dumps(plain(value),indent=2,allow_nan=False)+'\n')
def save(fig,path):
    for ext in ('png','pdf','svg'):
        metadata={'Creator':'Measured JCP fresh physical diagnostics'}
        if ext=='pdf':metadata.update(CreationDate=None,ModDate=None)
        if ext=='svg':metadata['Date']=None
        fig.savefig(path.with_suffix('.'+ext),dpi=200,metadata=metadata,bbox_inches='tight')
    plt.close(fig)

def build(run):
    base=HERE/'runs'/run;root=base/'output/phase';report=json.loads((root/'summary.json').read_text())
    cfg=report['config'];n=cfg['eval_cases'];ids=list(range(n));arms=cfg['arms'];tol=min(cfg['solver_gtols'])
    times=np.arange(round(cfg['horizon']/cfg['dt'])+1)*cfg['dt'];nt=len(times)
    raw_ids=[c['index'] for c in report['cases']]
    assert len(raw_ids)==len(set(raw_ids)) and set(raw_ids)<=set(ids)
    by_id={c['index']:c for c in report['cases']};cases=[];actual_fits=[]
    truth_concentration=np.full((n,nt),np.nan)
    matrices={arm:{key:np.full((n,nt),np.nan) for key in ('ordinary','shape','phase','concentration')} for arm in arms}
    for i in ids:
        raw=by_id.get(i);case={'index':i,'present':raw is not None,'qualification':raw.get('qualification',{}) if raw else {},'arms':{}}
        if raw is not None:
            case.update(parameters=raw['parameters'],initial_qualification=raw['initial_qualification'],reference=raw['reference'],
                        actual_common_initial_state_checks=raw.get('actual_common_initial_state_checks',[]))
            case['actual_initial_fit_invocations']=[]
            for fit_tol in cfg['fit_gtols']:
                source=root/f'case_{i:03d}'/f'initial_g{fit_tol:g}.json'
                detail=json.loads(source.read_text())
                entry={'case':i,'gtol':fit_tol,'qualified':detail['qualified'],
                    'qualification':detail['qualification'],
                    'ordinary_direct_objective_guard':detail['ordinary_direct_objective_guard'],
                    'source':str(source.relative_to(root)),'source_sha256':sha(source)}
                case['actual_initial_fit_invocations'].append(entry);actual_fits.append(entry)
            for arm in arms:
                rr=[r for r in raw['runs'] if r['arm']==arm and r['gtol']==tol]
                if not rr:continue
                assert len(rr)==1;r=rr[0]
                assert r['rows']==cfg['rows'] and r['dt']==cfg['dt'] and np.allclose(r['times'],times,atol=1e-15,rtol=0)
                assert len(r['metrics'])==nt
                metrics=r['metrics'];a=matrices[arm]
                a['ordinary'][i]=np.asarray([m['relative_error'] for m in metrics],dtype=float)
                a['shape'][i]=np.asarray([m['shape_aligned_relative_error'] for m in metrics],dtype=float)
                a['phase'][i]=np.linalg.norm(np.asarray([m['phase_displacement'] for m in metrics],dtype=float),axis=1)
                a['concentration'][i]=np.min(np.asarray([m['pred_centroid_concentration'] for m in metrics],dtype=float),axis=1)
                if arm=='head_common':
                    truth_concentration[i]=np.min(np.asarray([m['truth_centroid_concentration'] for m in metrics],dtype=float),axis=1)
                case['arms'][arm]={'metrics':metrics,'gtol':tol,'rows':r['rows'],'dt':r['dt'],
                    'all_steps_stationary':r['all_steps_stationary'],'jacobian_action_checked':r.get('jacobian_action_checked',False),
                    'initial_error':float(a['ordinary'][i,0]),'evolved_max_error':float(np.max(a['ordinary'][i,1:])),
                    'evolved_max_shape_error':float(np.max(a['shape'][i,1:])),
                    'evolved_max_phase_displacement':float(np.max(a['phase'][i,1:]))}
        cases.append(case)
    assert [c['index'] for c in cases]==ids
    complete_metrics=all(np.isfinite(matrices[arm][key]).all() for arm in arms for key in ('ordinary','shape','phase','concentration'))
    paired={}
    for key in ('ordinary','shape','phase'):
        head=np.max(matrices['head_common'][key][:,1:],axis=1)
        span=np.max(matrices['span_common'][key][:,1:],axis=1)
        paired[key]=span/np.maximum(head,1e-300)
    stats={}
    for arm in arms:
        stats[arm]={}
        for key in ('ordinary','shape','phase'):
            data=matrices[arm][key]
            # Full-cohort statistics require every registered case at that time.
            coverage=np.isfinite(data).sum(axis=0);valid=coverage==n
            stats[arm][key]={'coverage':coverage,'median':np.where(valid,np.median(data,axis=0),np.nan),
                'minimum':np.where(valid,np.min(data,axis=0),np.nan),'maximum':np.where(valid,np.max(data,axis=0),np.nan)}
    out=HERE/'plots'/run/'phase';out.mkdir(parents=True,exist_ok=True)
    full_pass=sum(bool(c['qualification'].get('passed',False)) for c in cases)
    actual_direct_flags=sum(not f['ordinary_direct_objective_guard'] for f in actual_fits)
    exported={'cohort_role':'untouched confirmation','seed':cfg['eval_seed'],'registered_ids':ids,'observed_ids':raw_ids,
        'complete_metrics':complete_metrics,'source_commit':report['source_commit'],'frozen_protocol':report['frozen_protocol'],
        'registration':report['registration'],'selected_solver_gtol':tol,'times':times,'cases':cases,
        'truth_min_coordinate_centroid_concentration':truth_concentration,
        'full_cohort_time_statistics':stats,'paired_case_evolved_max_span_over_head':paired,
        'full_qualification_pass_count':full_pass,'expected_case_count':n,
        'actual_phase_initial_fit_count':len(actual_fits),'actual_phase_initial_fit_ordinary_direct_flags':actual_direct_flags,
        'parameter_names':['center_x','center_y','center_z','vortex_separation','relative_strength','viscosity']}
    dump(out/'plotted-data.json',exported)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,key,title,factor,ylabel in [(axes[0,0],'ordinary','Ordinary physical-field error',100,'Relative field error (%)'),
        (axes[0,1],'shape','After diagnostic centroid alignment',100,'Aligned field error (%)'),
        (axes[1,0],'phase','Wrapped physical-centroid displacement',1,'Domain-length fraction')]:
        for arm in arms:
            s=stats[arm][key];ax.fill_between(times[1:],np.asarray(s['minimum'])[1:]*factor,np.asarray(s['maximum'])[1:]*factor,color=COLORS[arm],alpha=.10)
            ax.plot(times[1:],np.asarray(s['median'])[1:]*factor,color=COLORS[arm],lw=2,label=LABELS[arm])
        ax.set_title(title);ax.set_xlabel('Time');ax.set_ylabel(ylabel);ax.grid(alpha=.18)
    ax=axes[1,1]
    ax.scatter(ids,paired['ordinary'],color='#1768ac',label='Ordinary error',s=25)
    ax.scatter(ids,paired['shape'],color='#139776',marker='+',label='Aligned error',s=35)
    ax.axhline(1,color='#b52128',ls='--',lw=1);ax.set_yscale('log')
    ax.set_xlabel('Registered case (zero based)');ax.set_ylabel('Span / head evolved-maximum error')
    ax.set_title('Paired common-initial-field comparison')
    handles,labels=axes[0,0].get_legend_handles_labels()
    ratio_handles,ratio_labels=ax.get_legend_handles_labels()
    fig.legend(handles+ratio_handles,labels+ratio_labels,loc='lower center',ncol=3,frameon=False)
    fig.suptitle(f'Untouched NS cohort · seed {cfg["eval_seed"]} · full qualification {full_pass}/{n}')
    fig.tight_layout(rect=(0,.10,1,.95));save(fig,out/'fresh-phase-shape')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for arm in arms:
        axes[0].plot(ids,np.max(matrices[arm]['phase'][:,1:],axis=1),'.',color=COLORS[arm],label=LABELS[arm])
    axes[0].set_xlabel('Registered case');axes[0].set_ylabel('Maximum evolved displacement (domain fraction)')
    axes[0].set_title('Physical centroid displacement: every case')
    p=paired['phase'];higher=np.isfinite(p)&(p<1)
    axes[1].scatter(np.asarray(ids)[~higher],p[~higher],color=COLORS['head_common'],s=25)
    axes[1].scatter(np.asarray(ids)[higher],p[higher],color='#c14440',s=40,marker='x')
    axes[1].axhline(1,color='#b52128',ls='--',lw=1);axes[1].set_yscale('log')
    axes[1].set_xlabel('Registered case');axes[1].set_ylabel('Common-span / head maximum displacement')
    axes[1].set_title(f'Phase benefit is not universal: {int(higher.sum())}/{n} head losses')
    for ax in axes:ax.grid(alpha=.18)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
    fig.tight_layout(rect=(0,.08,1,1));save(fig,out/'fresh-phase-displacement-pairs')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for arm in arms:axes[0].scatter(ids,matrices[arm]['ordinary'][:,0]*100,s=25,label=LABELS[arm],color=COLORS[arm])
    axes[0].set_xlabel('Registered case');axes[0].set_ylabel('Time-zero field error (%)');axes[0].set_title('Initial fit kept separate from evolved error')
    for arm in arms:axes[1].plot(ids,np.min(matrices[arm]['concentration'],axis=1),'.',label=LABELS[arm],color=COLORS[arm])
    axes[1].plot(ids,np.min(truth_concentration,axis=1),'x',label='Reference centroid',color='#222222',markersize=5)
    axes[1].axhline(cfg['centroid_concentration_threshold'],color='#b52128',ls='--')
    axes[1].set_xlabel('Registered case');axes[1].set_ylabel('Minimum coordinate concentration');axes[1].set_title('Centroid diagnostic reliability')
    fig.legend(*axes[1].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False)
    fig.tight_layout(rect=(0,.08,1,1));save(fig,out/'fresh-initial-error-centroid-concentration')
    checks=['initialization','operator','rollout','jacobian','refinement','reference','common_state','passed']
    q=np.asarray([[2 if not c['present'] or key not in c['qualification'] else int(c['qualification'][key]) for key in checks] for c in cases])
    fig,ax=plt.subplots(figsize=(9,11));ax.imshow(q,cmap=ListedColormap(['#c14440','#b9dfcf','#dddddd']),vmin=0,vmax=2,aspect='auto',interpolation='nearest')
    ax.set_xticks(np.arange(len(checks)),checks,rotation=40,ha='right');ax.set_yticks(ids,ids);ax.set_ylabel('Every registered case')
    ax.set_title('Fresh trajectory qualification · green pass · red fail · gray missing')
    fig.tight_layout();save(fig,out/'fresh-trajectory-qualification')
    for key,label in [('ordinary','Ordinary field error'),('shape','Centroid-aligned field error')]:
        columns=6;gridrows=math.ceil(n/columns);fig,axes=plt.subplots(gridrows,columns,figsize=(18,2.25*gridrows),squeeze=False)
        for i,ax in enumerate(axes.flat):
            if i>=n:ax.set_visible(False);continue
            for arm in arms:ax.plot(times,matrices[arm][key][i]*100,color=COLORS[arm],lw=1.2,label=LABELS[arm])
            status='' if cases[i]['qualification'].get('passed',False) else ' · unqualified/missing'
            ax.set_title(f'Case {i}{status}',fontsize=9,color='black' if not status else '#a32222')
            ax.grid(alpha=.15);ax.tick_params(labelsize=8)
            if i//columns==gridrows-1:ax.set_xlabel('Time',fontsize=9)
            if i%columns==0:ax.set_ylabel('Error (%)',fontsize=9)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False)
        fig.suptitle(f'{label}: all {n} registered untouched cases')
        fig.tight_layout(rect=(0,.03,1,.97));save(fig,out/f'fresh-all-cases-{key}')
    captions=f'''# Fresh physical phase/shape figures

These are measured outcomes from seed{cfg['eval_seed']}, with {n} registered
cases and {len(raw_ids)} observed case records. Every registered ID is present
in the export and qualification matrix; missing cases are explicit. Full
trajectory qualification passes {full_pass}/{n}. No failed case is filtered out.

The primary curves show pointwise medians and full case ranges, at evolved
times only, using the tighter solver tolerance{tol:g}. Ordinary and diagnostic
centroid-aligned field errors are both divided by each case's original physical
initial-field norm. A full-cohort statistic is absent where any registered case
has missing/nonfinite data. Initial errors are plotted separately. Physical phase
is the norm of wrapped energy-centroid displacement in unit-domain fractions.
The concentration panel includes each prediction and the reference; its line at
{cfg['centroid_concentration_threshold']:g} is a descriptive ambiguity threshold,
not an accuracy or trajectory gate.

The paired panel divides each common-span case's maximum evolved error by the
same head case's maximum evolved error, separately for ordinary and aligned
errors. Ratios above one favor the head. It does not divide timewise medians or
select a favorable time. Native-span initialization is labeled separately.
The separate centroid-displacement pairing applies the same per-case evolved
maximum rule to physical displacement. Red crosses identify a larger head
displacement; all such counterexamples are retained. This diagnostic does not
equate phase displacement with the ordinary or aligned field error.

The reference is the same-grid discrete CNAB2 solution at dt{cfg['reference']['dt']:g},
checked at dt{cfg['reference']['check_dt']:g}, under the unchanged absolute and
fractional gates. This is not a continuum-error claim. Actual initial fits,
operator checks, all rollout stops, tighter-trajectory Jacobian-action checks,
solver refinement, both reference initial fields and actual common time-zero
states are independently qualified. Loose trajectories are not separately
Jacobian-action checked. The {len(actual_fits)} actual phase initializer invocations
retain {actual_direct_flags} ordinary-direct objective flags in this export, alongside
their separate qualification checks. These are distinct from the standalone
initializer matrix's flags. Neither set is silently relabeled as a pass under
the original ordinary-arithmetic objective comparison.

The head, bank, four starting-code identities and solver method were frozen
before this cohort. The new initializer changes arithmetic and bounded fitting
work, not training. No timing comparison, speed claim, new classical comparator
qualification, tangent-spectrum claim or manifold-distance claim follows from
these figures. Historical failed confirmations remain preserved.
'''
    (out/'captions.md').write_text(captions)
    dump(out/'MANIFEST.json',{'builder_sha256':sha(__file__),'raw_phase_summary_sha256':sha(root/'summary.json'),
        'result_manifest_sha256':sha(base/'RESULTS.sha256'),'registered_case_count':n,'observed_case_count':len(raw_ids),
        'plotted_ids_equal_registered_ids':True,'all_metric_records_finite':complete_metrics,
        'files':{p.name:sha(p) for p in sorted(out.iterdir()) if p.name!='MANIFEST.json'}})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run');args=parser.parse_args();build(args.run)
