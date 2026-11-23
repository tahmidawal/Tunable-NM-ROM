"""Measured initializer evidence only; every registered case is retained."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

HERE=Path(__file__).resolve().parent
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
    'svg.hashsalt':'jcp-ns-initializer-20261123','pdf.fonttype':42,'ps.fonttype':42})

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def save(fig,path):
    for ext in ('png','pdf','svg'):
        metadata={'Creator':'Measured JCP initializer analysis'}
        if ext=='pdf':metadata.update(CreationDate=None,ModDate=None)
        if ext=='svg':metadata['Date']=None
        fig.savefig(path.with_suffix('.'+ext),dpi=200,metadata=metadata,bbox_inches='tight')
    plt.close(fig)

def load(run):
    parts=run.split('/')
    root=HERE/'runs'/parts[0]/'output'
    if len(parts)>1:
        assert parts[1:]==['initializers'];root=root/'initializers'
    rows=json.loads((root/'records.json').read_text())
    provenance=json.loads((root/'provenance.json').read_text())
    expected=[f"{c['seed']}/{i}" for c in provenance['config']['cohorts'] for i in range(c['count'])]
    assert len(rows)==len(expected)*12
    result=[]
    for row in rows:
        d=json.loads((root/row['detail']).read_text());f=d['repair'];idx=f['original_compiled_selected']
        core=(f['controls'][2*idx+1]['direct'] if 'controls' in f else f['baseline_starts'][idx])
        q=f['qr_only_starts'][f['qr_only_selected']]
        qdirect=q.get('direct_check',q)
        item={**row,'case_position':expected.index(row['case_key']),
            'parameter':d['parameter'],'cohort_role':provenance['config'].get('role','exposed development'),
            'old_phase_fresh_gradient':d['original_phase_fresh_common_check']['gradient'],
            'old_core_fresh_gradient':core['gradient'],
            'qr_fresh_gradient':qdirect['gradient'],
            'returned_direct_gradient':f['direct_returned_check']['gradient'],
            'returned_anchored_gradient':f.get('anchored_returned_check',f['direct_returned_check'])['gradient'],
            'ordinary_direct_phase_objective_quality':d.get('ordinary_direct_phase_objective_quality',row['phase_objective_quality']),
            'ordinary_direct_objective_guard':f.get('direct_original_objective_guard',f['global_original_objective_quality']),
            'anchored_objective_guard':f['global_original_objective_quality'],
            'arithmetic_audit_pass':row.get('arithmetic_audit_pass'),
            'old_phase_initial_field_error':d['original_phase_initial_field_error'],
            'method_version':f['policy']['version'],'source_detail_sha256':sha(root/row['detail']),
            'audit_array_sha256':d['audit_arrays_sha256']}
        if 'independent_arithmetic_audit' in d:
            audit=d['independent_arithmetic_audit']
            item['independent_objective_delta_bands']=float(audit['longdouble_delta'])/float(audit['longdouble_original_band']) if 'longdouble_delta' in audit else None
            item['decimal_executed']=audit.get('decimal_required',False)
            item['decimal_objective_pass']=audit.get('decimal_pass')
        result.append(item)
    for family in ('learned','pod'):
        for path in ('phase','query'):
            for tol in (1e-7,1e-9,1e-11):
                subset=[r for r in result if (r['family'],r['input_path'],r['gtol'])==(family,path,tol)]
                assert len(subset)==len(expected) and {r['case_key'] for r in subset}==set(expected)
    return result,provenance,expected

def build(run):
    rows,provenance,expected=load(run);out=HERE/'plots'/run;out.mkdir(parents=True,exist_ok=True)
    dump(out/'plotted-data.json',{'run':run,'source_commit':provenance['source_commit'],
        'config_sha256':provenance['config_sha256'],'cohorts':provenance['config']['cohorts'],
        'case_order':expected,'expected_records':len(expected)*12,
        'parameter_names':['center_x','center_y','center_z','vortex_separation','relative_strength','viscosity'],
        'records':rows})
    role=rows[0]['cohort_role'];role_label='untouched confirmation' if role=='untouched-confirmation' else 'exposed development'
    n=len(expected);x=np.arange(n)
    boundaries=np.cumsum([c['count'] for c in provenance['config']['cohorts']])[:-1]-.5
    series=[('old_phase_fresh_gradient','Old host phase','#666666','o'),
            ('old_core_fresh_gradient','Old compiled core','#b18a26','s'),
            ('returned_direct_gradient','Returned direct','#1768ac','D')]
    if rows[0]['method_version'].startswith('v2'):
        series.append(('returned_anchored_gradient','Returned anchored','#139776','+'))
    for tol in (1e-7,1e-9,1e-11):
        fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,sharey=True)
        for ax,(family,path) in zip(axes.flat,[(f,p) for f in ('learned','pod') for p in ('phase','query')]):
            rr=sorted([r for r in rows if (r['family'],r['input_path'],r['gtol'])==(family,path,tol)],key=lambda r:r['case_position'])
            for key,label,color,marker in series:
                y=np.array([r[key]/tol for r in rr]);ax.scatter(x,np.maximum(y,1e-16),s=22,label=label,color=color,marker=marker,alpha=.8)
            ax.axhline(1,color='#b52128',lw=1.2,ls='--')
            for b in boundaries:ax.axvline(b,color='#777',lw=.8,ls=':')
            ax.set_yscale('log')
            ax.set_xlim(-.7,n-.3);ax.grid(alpha=.18,axis='y')
            failed=sum(not r['initializer_qualified'] for r in rr)
            bank_label='POD' if family=='pod' else 'Learned'
            ax.set_title(f'{bank_label} · {path} input\nFull qualification failures: {failed}/{len(rr)}',fontsize=11)
        fig.supylabel('Normalized stationarity / unchanged tolerance',x=.005)
        for ax in axes[-1]:ax.set_xlabel(f'Registered {role_label} case position (zero based)')
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='lower center',ncol=len(series),bbox_to_anchor=(.5,-.01),frameon=False)
        method='Fixed-anchor initializer' if rows[0]['method_version'].startswith('v2') else 'QR/SVD attempt'
        fig.suptitle(f'{method} · {role_label} · tolerance {tol:g}')
        fig.tight_layout(rect=(.02,.05,1,.93));save(fig,out/f'initializer-stationarity-{tol:g}')
    checks=[('repaired_pass','Core initializer'),('phase_objective_quality','Common objective'),
            ('arithmetic_audit_pass','Independent arithmetic'),('initial_field_agreement_pass','Tolerance fields'),
            ('increment_checks_pass','Increment/J check'),('activation_ranges_pass','Activation range'),
            ('old_new_initial_agreement_pass','Old/new fields'),('initializer_qualified','Full qualification')]
    fig,axes=plt.subplots(2,2,figsize=(13,13),sharey=True)
    for ax,(family,path) in zip(axes.flat,[(f,p) for f in ('learned','pod') for p in ('phase','query')]):
        columns=[];labels=[]
        for tol in (1e-7,1e-9,1e-11):
            rr=sorted([r for r in rows if (r['family'],r['input_path'],r['gtol'])==(family,path,tol)],key=lambda r:r['case_position'])
            for key,label in checks:
                columns.append([2 if r.get(key) is None else int(bool(r[key])) for r in rr])
                labels.append(f'{tol:g}\n{label}')
        matrix=np.asarray(columns).T
        ax.imshow(matrix,aspect='auto',cmap=ListedColormap(['#c14440','#b9dfcf','#dddddd']),vmin=0,vmax=2,interpolation='nearest')
        ax.set_xticks(np.arange(len(labels)),labels,rotation=90,fontsize=7)
        ax.set_yticks(np.arange(n),expected,fontsize=7);ax.set_title(f'{family.capitalize()} · {path} input')
        for b in boundaries:ax.axhline(b,color='black',lw=.8)
        for b in (len(checks)-.5,2*len(checks)-.5):ax.axvline(b,color='white',lw=2)
    fig.suptitle(f'{role_label.capitalize()} initializer qualification: green pass · red fail · gray not part of this version')
    fig.tight_layout(rect=(0,0,1,.96));save(fig,out/'initializer-qualification-matrix')
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    colors={'learned':'#1768ac','pod':'#c28c20'}
    for family in ('learned','pod'):
        rr=[r for r in rows if r['family']==family]
        axes[0].scatter([r['old_phase_initial_field_error']*100 for r in rr],
                        [r['initial_field_error']*100 for r in rr],s=15,alpha=.5,color=colors[family],label=family)
        axes[1].scatter(np.arange(len(rr)),[max(r['old_new_field_gap'],1e-18) for r in rr],s=13,alpha=.5,color=colors[family],label=family)
    lim=max(axes[0].get_xlim()[1],axes[0].get_ylim()[1]);axes[0].plot([0,lim],[0,lim],color='#666',lw=1,ls='--')
    axes[0].set_xlabel('Original host initial-field error (%)');axes[0].set_ylabel('Returned initial-field error (%)')
    axes[0].set_title('Same physical input and original norm')
    axes[1].axhline(1e-6,color='#b52128',ls='--');axes[1].set_yscale('log')
    axes[1].set_xlabel('Registered fit order within each bank');axes[1].set_ylabel('Old/new physical-field difference')
    axes[1].set_title('Unchanged initial-field agreement bound')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False)
    fig.tight_layout(rect=(0,.08,1,1));save(fig,out/'initializer-field-agreement')
    diagnostic_keys=[('ordinary_direct_phase_objective_quality','Ordinary f64\nphase comparator'),
        ('ordinary_direct_objective_guard','Ordinary f64\nall-control guard'),
        ('anchored_objective_guard',"Version's common\nobjective guard"),
        ('arithmetic_audit_pass','Independent\narithmetic audit'),('initializer_qualified','Full amended\nqualification')]
    fig,ax=plt.subplots(figsize=(10,4.5))
    available=[any(r.get(k) is not None for r in rows) for k,_ in diagnostic_keys]
    values=[sum(r.get(k) is False for r in rows) for k,_ in diagnostic_keys]
    bars=ax.bar(np.arange(len(values)),values,color=['#bb665d','#bb665d','#467fa8','#467fa8','#467fa8'])
    for bar,value,present in zip(bars,values,available):
        if not present:bar.set_facecolor('#cccccc');bar.set_hatch('//')
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+max(values+[1])*.025,str(value) if present else 'not run',ha='center')
    ax.set_xticks(np.arange(len(values)),[label for _,label in diagnostic_keys]);ax.set_ylabel('Reported failed flags (all registered fits)')
    ax.set_ylim(0,max(values+[1])*1.22)
    decimal_count=sum(r.get('decimal_executed',False) for r in rows)
    ax.set_title(f'{role_label.capitalize()}: ordinary arithmetic flags remain visible\nDecimal audits executed: {decimal_count}/{len(rows)}')
    ax.text(.5,-.23,'First two bars are retained ordinary-arithmetic diagnostics, not relabeled passes of the amended contract.',
            transform=ax.transAxes,ha='center',fontsize=9)
    fig.tight_layout();save(fig,out/'initializer-objective-diagnostics')
    captions=f'''# Measured initializer {role_label} figures ({run})

The cohort role is {role_label}. Registered cohorts are
{', '.join('seed'+str(c['seed'])+' with '+str(c['count'])+' cases' for c in provenance['config']['cohorts'])}.
No case is omitted. Phase input is F.initial;
query input includes the original extra spectral projection. The frozen learned
and POD heads are unchanged. These figures do not establish a timing benefit.

Stationarity panels use the original normalized residual-gradient formula,
divided by the displayed unchanged tolerance. The dashed line is one; values
above it fail. Original host and compiled-core returned codes are freshly
evaluated by the common ordinary-direct evaluator. The compiled control is a
standalone core replay, not bitwise reproduction of an earlier whole query.
V2 also shows the fixed-anchor gradient; both direct and anchored tests must
pass. Vertical dotted lines separate registered cohorts where applicable. Full qualification
counts include the objective and actual physical-field checks, not only gradients.

The matrix retains every case and all three tolerances. Gray marks a check not
included in that historical version. Ordinary-direct objective-ranking failures
remain in the exported data even if a separate higher-precision original-network
audit establishes objective quality. Such an audit is cluster-host arithmetic
evidence, not a replacement for the f64 GPU stationarity requirement.
The objective-diagnostic bars retain failed ordinary-direct phase and all-control
flags beside amended objective/arithmetic/full-qualification checks. The first
two bars are diagnostic, not silently passing old rules.

The field panels use explicit physical reconstructions normalized by the original
physical initial-field norm. The diagonal shows equal old/new initial error;
the horizontal line is the original1e-6 agreement bound. All tolerance and input
paths are retained. This is initialization evidence, not evolution, reference,
or complete-query cost qualification. Version1 remains a failed attempt: optional
SVD polishing can change the selected raw minimum and does not guarantee a lower
failure count. Version2 combines fixed-anchor arithmetic and both legacy controls;
no isolated attribution to one change is claimed.
'''
    (out/'captions.md').write_text(captions)
    dump(out/'MANIFEST.json',{'run':run,'builder_sha256':sha(__file__),
        'raw_result_manifest_sha256':sha(HERE/'runs'/run.split('/')[0]/'RESULTS.sha256'),
        'files':{p.name:sha(p) for p in sorted(out.iterdir()) if p.name!='MANIFEST.json'},
        'coverage_verified':True,'record_count':len(rows),'case_count':len(expected)})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('runs',nargs='+');args=ap.parse_args()
    for run in args.runs:build(run)
