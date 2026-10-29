"""Rebuild the plot atlas from retained compact evidence, without JAX or cluster access.

Every joint cost/error point uses a timed output or its hash-identical scoring witness.
No runtime curve crosses allocation boundaries. Historical 3D work curves use iteration
counts, never seconds_first (which includes compilation).
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import csv
import gzip
import hashlib
import html
import json
import math
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
FIG=HERE/'figs'
ROWS=[]
PLOTS=[]
DEV=[f'dev6|{i}' for i in range(6)]


def load(name):return json.loads(gzip.decompress((HERE/'data'/f'{name}.json.gz').read_bytes()))
def median(v):return float(np.median(list(v)))
def finite(v):return v is not None and math.isfinite(v)
def family(rule):return re.sub(r'[0-9].*','',rule)
def true(v):return v is True or isinstance(v,dict) and (v.get('passed') is True or v.get('pass') is True)


def summarize(row):
    seed=re.search(r'\d+s(\d+)$',row.get('rule',''))
    row['quadrature_shift']=int(seed.group(1)) if seed else None
    cases=row['cases']; es=[v['e'] for v in cases.values()]
    assert es and all(finite(e) and e>=0 for e in es)
    row.update(n_cases=len(cases),error_median=median(es),error_worst=max(es),
      error_over_1pct=sum(e>.01 for e in es),eps_max=max(v.get('eps',0) for v in cases.values()),
      error_lower_median=median(max(0,v['e']-v.get('eps',0)) for v in cases.values()),
      error_upper_median=median(v['e']+v.get('eps',0) for v in cases.values()))
    for key in ['milliseconds','iterations','residual','stationarity','evolution_ms','decode_ms']:
        vals=[v.get(key) for v in cases.values()]
        row[key]=median(vals) if all(finite(v) for v in vals) else None
    reps=[v for c in cases.values() for v in c.get('time_repetitions_ms',[])]
    outliers=0
    for c in cases.values():
        ts=c.get('time_repetitions_ms',[])
        if len(ts)>=4:
            a,b=np.percentile(ts,[25,75]);outliers+=sum(t<a-1.5*(b-a) or t>b+1.5*(b-a) for t in ts)
    row.update(timing_repetitions=len(reps),timing_outliers=outliers)
    ROWS.append(row)


def two_d():
    heads=load('heads-export');dims=load('dims-export');qa=load('qualify-analysis');p1=load('p1-analysis')
    for name in ['dims-timed','p1-timed']+[f'{lane}-{mesh}-timed' for lane in ['jcp-heads','jcp-qualify'] for mesh in [256,1024,4096]]:
        doc=load(name);raw=doc['result']; panel=f"2D {name} job {raw['job_id']}"
        for lb,own in doc['subjects'].items():
            subject=raw['subjects'][lb];sp=subject['spec'];meta=own['meta'];mesh=int(meta.get('mesh',sp['mesh']))
            is_head=sp.get('kind')=='head'
            if name.startswith('jcp-heads') and not is_head:continue # user deferred full-bank/head comparison
            K=int(re.search(r'H(\d+)',sp['name']).group(1)) if is_head and re.match(r'H\d+',sp['name']) else None
            if is_head and K is None:continue # incumbent replay is not another K level
            r=int(meta.get('Rp',512 if is_head else meta.get('dim',sp.get('Rp',0))))
            M=int(meta.get('M',subject.get('M'))); rule=meta.get('rule',sp['rule']);scheme=meta['scheme'];form=meta.get('form','LSPG')
            qp='not retained';qualified=False;timing_usable=False
            if name=='dims-timed':
                q=dims['costs_2d'].get(lb,{})
                qualified=q.get('qualified',False);timing_usable=q.get('usable',False);qp=str(q.get('q2','unverified'))
            elif name=='p1-timed':
                q=p1['timing']['timing_qualification'].get(lb,{})
                qualified=q.get('timing_job',q.get('both',False));timing_usable=p1['timing']['controls_ok'] and not p1['timing']['veto'].get(lb,False)
                qp=str(q.get('Q2','unverified'))
            elif name.startswith('jcp-heads'):
                q=heads['cost'][str(mesh)].get(lb,{})
                qualified=q.get('timing_Q1',{}).get('pass',False) and q.get('timing_Q2',{}).get('label')=='pass'
                tm=heads['timing_meta'][str(mesh)]
                drift=q.get('subject_drift')
                timing_usable=(q.get('complete',False) and q.get('equivalence',False) and tm['doubled_ok']
                   and tm['split_panel']['valid'] and finite(drift) and 1/1.1<=drift<=1.1)
                qp=q.get('timing_Q2',{}).get('label','unverified')
            else:
                q=qa['timing'][str(mesh)]['subjects'].get(lb,{})
                qualified=q.get('timing_label',{}).get('numerically_qualified',False)
                timing_usable=q.get('usable',False);qp=q.get('timing_label',{}).get('Q2',{}).get('label','unverified')
            if sp.get('inject_identity_fault') or q.get('injected_control'):continue
            fus=[v for v in raw['fused'] if v['B']==lb and v['case'] in DEV]
            inv=[v for v in raw['invocations'] if v['B']==lb and v['case'] in DEV]
            hash_bad=any(not v.get('hash_ok',False) or not v.get('all_hashes_ok',True) for v in fus)
            hash_bad |= any(not v.get('all_hashes_ok',False) for v in inv)
            cases={}
            for cid,score in own['scores'].items():
                ff=[v for v in fus if v['case']==cid and v['valid'] and v.get('hash_ok')]
                ii=[v for v in inv if v['case']==cid and v['valid'] and v.get('all_hashes_ok')]
                metrics=own['metrics']
                def met(k,agg='median'):return metrics.get(k,{}).get(cid,{}).get(agg)
                times=[v['t_fused']*1000 for v in ff]
                cases[cid]=dict(e=score['e'],eps=score['eps'],e_t=score['e_t'],
                    milliseconds=median(times) if times and timing_usable and not hash_bad else None,
                    time_repetitions_ms=times if timing_usable and not hash_bad else [],
                    iterations=median(sum(v['it']) for v in ff) if ff and not hash_bad else None,
                    residual=met('raw'),stationarity=met('cert') if 'cert' in metrics else met('gamma'),
                    evolution_ms=median(v['t_evolve']*1000 for v in ii) if ii and timing_usable and not hash_bad else None,
                    decode_ms=median(v['t_decode']*1000 for v in ii) if ii and timing_usable and not hash_bad else None)
            assert sorted(cases)==DEV
            m=subject.get('m',subject.get('describe',{}).get('m',{})).get('main')
            assert m is not None, (name,lb,'quadrature count missing')
            summarize(dict(dimension=2,panel=panel,id=lb,r=r,K=K,M=M,m=m,mesh=mesh,
                dt=meta['dt'],scheme=scheme,form=form,solver='head' if is_head else 'linear',
                bank='head512' if is_head else 'bank512',quad_family=family(rule),rule=rule,
                seed=int(re.search(r's(\d+)',sp['name']).group(1)) if is_head else None,
                policy=meta.get('policy',sp.get('policy','frozen')),qualified=bool(qualified),timing_usable=bool(timing_usable and not hash_bad),
                Q2=qp,source=name,source_result_sha256=doc['result_sha256'],source_npz_sha256=own['npz_sha256'],
                cost_kind='milliseconds',cases=cases,provisional=not bool(qualified)))


def three_d():
    doc=load('frontier3d')
    for sid,sub in doc['subjects'].items():
        cfg=sub['config']
        if cfg['kind']!='bank':continue
        timed=[v for v in doc['timing_repetitions'] if v['subject']==sid and v['role']=='B']
        cases={}
        for cid,original in doc['case_results'][sid].items():
            vv=[v for v in timed if str(v['case'])==cid and v['timing_admitted'] is True]
            for v in vv:
                assert abs(v['score']['e']-original['score']['e'])<1e-10
                assert v['output_sha256']==original['output_sha256']
            times=[v['descriptor_seconds']*1000 for v in vv]
            cases[cid]=dict(e=original['score']['e'],eps=original['score']['epsilon'],
               e_t=original['score']['e_t'],milliseconds=median(times) if times else None,
               time_repetitions_ms=times,iterations=original['work'].get('nonlinear_iterations'))
        summarize(dict(dimension=3,panel=f"3D full64 job {doc['execution']['job_id']}",id=sid,
            r=cfg['r'],M=cfg['M'],m=sub['effective_parameters']['m'],mesh=65,dt=cfg['dt'],
            scheme=cfg['scheme'],solver=cfg['solver'],form='LSPG',bank=cfg['setting'],
            K=None,seed=None,rule=cfg['rule'],quad_family=family(cfg['rule']),policy=cfg['work_policy'],
            qualified=sub['qualification']['status']=='qualified',timing_usable=all(v.get('milliseconds') is not None for v in cases.values()),
            source='frontier3d',source_result_sha256=json.loads((HERE/'SOURCES.json').read_text())['frontier3d']['sha256'],
            source_config_sha256=doc['execution']['config_sha256'],
            cost_kind='milliseconds',cases=cases,provisional=False))
    dim=load('dims-export');qualified=set(dim['qualified_3d'])
    for own in load('historical3d-work'):
        identity=own['identity'];cid=own['key'];cases={k:dict(e=v['e'],eps=v['eps'],
            iterations=v['iterations'],residual=v['residual_median'],stationarity=v['gamma_max']) for k,v in own['cases'].items()}
        if identity.get('parse_status')!='parsed':continue
        summarize(dict(dimension=3,panel='3D historical work (counts only)',id=cid+' @'+own['attempt'],
           r=identity['r'],M=identity['M'],m=identity['m'],mesh=identity['mesh_nodes'],dt=float(identity['dt_label']),
           scheme=identity['scheme'],solver=identity['solver'],form='LSPG',
           bank=identity['bank'],K=None,seed=None,rule=identity['rule'],quad_family=family(identity['rule']),
           policy='|'.join(identity['policy']),qualified=cid in qualified,timing_usable=False,
           source='historical3d-work',source_result_sha256=own['result_sha256'],
           source_scores_sha256=own['scores_sha256'],cost_kind='iterations',cases=cases,provisional=True))
    doc=load('mesh3d')
    for sid,q in doc['queries'].items():
        if 'Q2' not in q or q.get('contract') not in ['mf','mesh']:continue
        vv=[v for v in doc['timing_repetitions'] if v['query']==sid and v['role']=='B' and v['timing_admitted'] is True]
        # Anchor queries only appear as A; retain their actual timings without multiplying their weight.
        if not vv:vv=[v for v in doc['timing_repetitions'] if v['query']==sid and v['timing_admitted'] is True]
        cases={}
        for cid,o in doc['case_results'][sid].items():
            ts=[v for v in vv if str(v['case'])==cid]
            assert all(abs(v['score']['e']-o['score']['e'])<1e-10 for v in ts)
            times=[v['descriptor_seconds']*1000 for v in ts]
            cases[cid]=dict(e=o['score']['e'],eps=o['score']['epsilon'],e_t=o['score']['e_t'],
              milliseconds=median(times) if times else None,time_repetitions_ms=times,
              iterations=o['work'].get('nonlinear_iterations'))
        summarize(dict(dimension=3,panel='3D mesh '+str(doc['execution']['job_id']),id=sid,
            r=q['r'],M=q['M'],m=q['m'],mesh=q['N'],dt=q['dt'],scheme=q['scheme'],solver='qn',form='LSPG',
            bank=q['bank'],K=None,seed=None,rule=q['rule'],quad_family=family(q['rule']),policy=q['contract'],
            qualified=true(q['Q1']) and true(q['Q2']),timing_usable=all(v.get('milliseconds') is not None for v in cases.values()),
            source='mesh3d',source_result_sha256=json.loads((HERE/'SOURCES.json').read_text())['mesh3d']['sha256'],
            cost_kind='milliseconds',cases=cases,provisional=True))
    if (HERE/'data/timed3d-screen.json.gz').exists():three_d_screen()


def three_d_screen():
    doc=load('timed3d-screen');raw=doc['raw_timing'];qualified=set(load('dims-export')['qualified_3d'])
    stats=defaultdict(dict)
    for v in raw['rows']:
        if 'tag' in v and 'spec' in v:stats[v['tag']+'|'+v['spec']][str(v['case'])]=v
    for key,sub in doc['subjects'].items():
        parts=key.split('|');solver=parts[2]
        form=parts[3] if solver=='ad' else 'LSPG'
        scheme=parts[4] if solver=='ad' else parts[3]
        dt=float(parts[5] if solver=='ad' else parts[4]);policy=parts[-1]
        rule=sub['rule'];order=int(re.search(r'\d+',rule).group())
        m=order**3 if rule.startswith('gl') else order
        cases={}
        for cid,score in sub['scores'].items():
            tt=[v['tB']*1000 for v in sub['timing'] if str(v['case'])==cid]
            step=stats.get(key,{}).get(cid,{}).get('steps',{})
            it=step.get('it',step.get('step_it',step.get('q')))
            gn=step.get('gn',step.get('step_gn',step.get('gam')))
            cases[cid]=dict(e=score['e'],eps=score['eps'],e_t=score['e_t'],milliseconds=median(tt),
              time_repetitions_ms=tt,iterations=float(sum(it)) if it else None,
              residual=median(step['rn']) if 'rn' in step else None,
              stationarity=max(gn) if gn else None)
        checkkey=key.removesuffix('|prod') if solver=='ad' else key
        summarize(dict(dimension=3,panel=f"3D rescored screening job {raw['job_id']} (cases 0–3)",id=key,
            r=sub['r'],M=sub['M'],m=m,mesh=65,dt=dt,scheme=scheme,solver=solver,form=form,
            bank=sub['bank'],K=None,seed=None,rule=rule,quad_family=family(rule),policy=policy,
            qualified=checkkey in qualified,timing_usable=True,source='screen3d',
            source_result_sha256=doc['source_result_sha256'],source_coefficients_sha256=doc['source_coefficients_sha256'],
            cost_kind='milliseconds',cases=cases,provisional=True))


LABELS={'r':'Bank width r (coefficients)','K':'Nonlinear latent dimension K',
 'M':'Smooth test functions M (count)','m':'Quadrature points m (count)',
 'dt':'Time step Δt (simulation units)','mesh':'Mesh intervals / nodes per axis (source convention)'}


def save(fig,slug,title,caption,dimension,knob,points):
    fig.savefig(FIG/f'{slug}.png',dpi=135,bbox_inches='tight',facecolor='white')
    fig.savefig(FIG/f'{slug}.svg',bbox_inches='tight',facecolor='white')
    plt.close(fig)
    PLOTS.append(dict(slug=slug,title=title,caption=caption,dimension=dimension,knob=knob,
       point_ids=[r['id'] for r in points],point_keys=[(r['dimension'],r['panel'],r['id']) for r in points],
       panels=sorted(set(r['panel'] for r in points))))


def groups(rows,knob,coupled=False):
    fields=['panel','r','K','M','m','mesh','dt','scheme','solver','form','bank','quad_family','quadrature_shift','seed','policy']
    exclude={knob}
    if coupled and knob=='r':exclude|={'M','m','bank','quad_family'}
    if coupled and knob=='K':exclude|={'M','m'}
    out=defaultdict(list)
    for r in rows:
        if r.get(knob) is not None:
            key=tuple((k,r.get(k)) for k in fields if k not in exclude)
            out[key].append(r)
    return [(k,sorted(v,key=lambda x:x[knob])) for k,v in out.items() if len({r[knob] for r in v})>=2]


def group_title(key,knob):
    d=dict(key)
    label=[d['panel'].replace('2D ','').replace('3D ','')]
    for k in ['bank','scheme','solver','r','M','m','dt','mesh','seed','policy']:
        if k in d and d[k] is not None:label.append(f'{k}={d[k]}')
    return ' · '.join(label)


def draw_group(axs,rows,knob,kind):
    x=np.array([r[knob] for r in rows]); order=np.argsort(x);xx=x[order]
    # Duplicate x comes from separate historical invocations. Keep every point and avoid
    # a misleading line through an arbitrarily selected replicate.
    unique=len(set(x))==len(x)
    colour=['#176b87','#aa5533']
    for metric,label,c in [('error_median','Median',colour[0]),('error_worst','Worst',colour[1])]:
        yy=np.array([r[metric]*100 for r in rows])[order]
        if unique:axs[0].plot(xx,yy,color=c,alpha=.55,lw=1)
        for status,marker in [(True,'o'),(False,'x')]:
            sel=[i for i,j in enumerate(order) if rows[j]['qualified']==status]
            if sel:axs[0].scatter(xx[sel],yy[sel],color=c,marker=marker,s=24,label=label if status else None,zorder=3)
    axs[0].set(ylabel='Conditional relative L2 error (%)',yscale='log')
    costkey=kind;valid=[r for r in rows if finite(r.get(costkey)) and r[costkey]>0]
    if valid:
        vx=[r[knob] for r in valid];vy=[r[costkey] for r in valid]
        if len(set(vx))==len(vx):axs[1].plot(vx,vy,color='#334155',alpha=.55,lw=1)
        for status,marker in [(True,'o'),(False,'x')]:
            a=[r for r in valid if r['qualified']==status]
            axs[1].scatter([r[knob] for r in a],[r[costkey] for r in a],color='#334155',marker=marker,s=24)
        axs[1].set(yscale='log',ylabel='Fused query cost (ms)' if kind=='milliseconds' else 'Nonlinear iterations (count)')
        if max(vy)/min(vy)<5:axs[1].set_yscale('linear')
        for r in valid:
            for metric,c in [('error_median',colour[0]),('error_worst',colour[1])]:
                axs[2].scatter(r[costkey],r[metric]*100,color=c,marker='o' if r['qualified'] else 'x',s=24)
            axs[2].annotate(f"{r[knob]:g}",(r[costkey],r['error_median']*100),xytext=(3,3),textcoords='offset points',fontsize=6)
        axs[2].set(xscale='log',yscale='log',xlabel='Fused query cost (ms)' if kind=='milliseconds' else 'Nonlinear iterations (count)',ylabel='Conditional relative L2 error (%)')
        if max(vy)/min(vy)<5:axs[2].set_xscale('linear')
    else:
        # No empty chart is emitted: show supported work in place of missing seconds.
        vv=[r for r in rows if finite(r.get('iterations'))]
        axs[1].scatter([r[knob] for r in vv],[r['iterations'] for r in vv],color='#334155',s=22)
        axs[1].set(ylabel='Nonlinear iterations (count)')
        axs[2].remove()
    for ax in axs[:2]:ax.set_xlabel(LABELS[knob])
    for ax in axs:
        ax.grid(True,alpha=.18);ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=8)
        from matplotlib.ticker import NullFormatter
        ax.xaxis.set_minor_formatter(NullFormatter());ax.yaxis.set_minor_formatter(NullFormatter())
    if knob in ['m','dt','mesh']:
        axs[0].set_xscale('log');axs[1].set_xscale('log')
    from matplotlib.lines import Line2D
    axs[0].legend(handles=[Line2D([],[],color=colour[0],label='Median'),Line2D([],[],color=colour[1],label='Worst')],fontsize=7,loc='best')


def sweep_pages(only_sources=None,tag=''):
    for dim in [2,3]:
        rows=[r for r in ROWS if r['dimension']==dim and (only_sources is None or r['source'] in only_sources)]
        for knob in LABELS:
            coupled=knob=='K'
            gg=groups(rows,knob,coupled=coupled)
            # Full64 width comparisons co-vary tests and quadrature, retained explicitly.
            if knob=='r':gg+=groups([r for r in rows if r['source']=='frontier3d'],knob,coupled=True)
            for start in range(0,len(gg),4):
                page=gg[start:start+4];fig,axs=plt.subplots(len(page),3,figsize=(15,3.45*len(page)),squeeze=False)
                points=[]
                for i,(key,rr) in enumerate(page):
                    kind=rr[0]['cost_kind'];draw_group(axs[i],rr,knob,kind)
                    axs[i,0].set_title(group_title(key,knob),fontsize=8,loc='left',pad=15,wrap=True)
                    points+=rr
                caveat='K policy sweep: M and quadrature may co-vary.' if coupled else 'Hold listed settings fixed; a width package is labelled separately in the source table.'
                title=f'{dim}D Burgers · {LABELS[knob]} · page {start//4+1}'
                fig.suptitle(title,fontsize=17,x=.06,ha='left',y=1.005)
                fig.text(.06,.005,'Circles: reported numerical qualification passed. Crosses: failed or unverified. '
                    'Error = max evolved-output relative L2, fixed initial-reference norm.\n'
                    'Each row is one timing allocation, or hardware-free historical iteration counts. '+caveat,fontsize=8)
                fig.tight_layout(h_pad=3,rect=(0,.045,1,.99))
                save(fig,f'{dim}d-{tag}{knob}-{start//4+1:02}',title,caveat+' Median over the common cases, then worst over those cases; timing is median within case then median across cases.',dim,knob,points)


def residual_pages(only_sources=None,tag=''):
    for dim in [2,3]:
        rows=[r for r in ROWS if r['dimension']==dim and finite(r.get('residual')) and r['residual']>0
              and (only_sources is None or r['source'] in only_sources)]
        gg=groups(rows,'m')
        for start in range(0,len(gg),6):
            page=gg[start:start+6];fig,axs=plt.subplots(len(page),2,figsize=(12,2.7*len(page)),squeeze=False);pts=[]
            for i,(key,rr) in enumerate(page):
                for ax,k,label in [(axs[i,0],'residual','Median per-step weak residual norm (source units)'),(axs[i,1],'stationarity','Step certificate: case-median '+('of maxima' if dim==3 else 'of medians'))]:
                    vv=[r for r in rr if finite(r.get(k)) and r[k]>0]
                    ax.scatter([r['m'] for r in vv],[r[k] for r in vv],color='#176b87',s=20)
                    ax.set(xscale='log',yscale='log',xlabel='Quadrature points m (count)',ylabel=label)
                    ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
                axs[i,0].set_title(group_title(key,'m'),fontsize=8,loc='left',wrap=True);pts+=rr
            title=f'{dim}D · residual and stationarity versus quadrature · page {start//6+1}'
            fig.suptitle(title,fontsize=16,y=1.01)
            fig.text(.06,.005,'Source residual norms are not physical error and are not pooled across changing test spaces, time steps or formulations.\n'
                     'Source certificates may be stationarity ratios or residual tests; exact fields and source arrays are retained in data.',fontsize=8)
            fig.tight_layout(h_pad=3,rect=(0,.045,1,.99))
            save(fig,f'{dim}d-{tag}residual-{start//6+1:02}',title,'Within each row bank, test space, time step and method are fixed. Raw residual is not squared training loss.',dim,'loss',pts)


def training():
    d=load('heads-export');v=[(name,r) for name,r in d['training'].items() if name.startswith('mid_k')]
    fig,axs=plt.subplots(1,3,figsize=(14,4))
    for seed in range(3):
        own=sorted([r for name,r in v if name.endswith(f'_s{seed}')],key=lambda r:r['k'])
        for ax,key,label in zip(axs,['recon','oracle','seconds'],['Training reconstruction error (relative, source metric)','Bounded fitted reconstruction error (relative, source metric)','Offline training duration (seconds)']):
            ax.plot([r['k'] for r in own],[r[key] for r in own],marker='o',label=f'seed {seed}')
            ax.set(xlabel=LABELS['K'],ylabel=label,yscale='log');ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    axs[0].legend();fig.suptitle('2D nonlinear head · training and fitted reconstruction · separate from online rollout',fontsize=14)
    fig.text(.07,.005,'Source: jcp-heads training job '+str(d['training_meta']['job'])+'. Matched training recipe; bounded fitting is not a certified manifold optimum. No 3D head sweep is present.',fontsize=8)
    fig.tight_layout(rect=(0,.07,1,.98));save(fig,'2d-K-training','2D head training versus K','Offline duration and reconstruction metrics from the same training record; not online solve cost or physical rollout error.',2,'training',[])


def overview():
    specs={2:[
      ('r','Bank width · fixed M and quadrature',lambda r:r['source']=='dims-timed' and r['M']==2048 and r['m']==65536 and r['scheme']=='CNR' and r['form']=='LSPG'),
      ('K','Latent dimension · selected-rule policy',lambda r:r['source']=='jcp-heads-1024-timed' and r['seed']==0 and r['scheme']=='BE'),
      ('M','Test-space size · fixed bank and quadrature',lambda r:r['source']=='dims-timed' and r['r']==384 and r['m']==65536 and r['scheme']=='CNR' and r['form']=='LSPG'),
      ('m','Quadrature · fixed bank and tests',lambda r:r['source']=='dims-timed' and r['r']==384 and r['M']==1536 and r['scheme']=='CNR' and r['form']=='LSPG'),
      ('dt','Time step · fixed bank and tests',lambda r:r['source']=='jcp-qualify-1024-timed' and r['r']==384 and r['scheme']=='CNR' and r['form']=='LSPG'),
      ('mesh','Mesh · fixed reduced model',lambda r:r['source']=='p1-timed' and r['r']==384 and r['scheme']=='CNR' and r['policy']=='frozen')],
      3:[
      ('r','Bank width · fixed M and quadrature',lambda r:r['source']=='screen3d' and r['M']==2052 and r['m']==65536 and r['scheme']=='CN' and r['solver']=='fs' and r['dt']==.01),
      ('M','Test-space size · fixed bank and quadrature',lambda r:r['source']=='screen3d' and r['r']==512 and r['rule']=='lat65536' and r['scheme']=='CN' and r['solver']=='ad' and r['form']=='LSPG' and r['dt']==.01),
      ('m','Quadrature · fixed bank and tests',lambda r:r['source']=='screen3d' and r['bank']=='M2' and r['r']==512 and r['M']==2052 and r['quad_family']=='gl' and r['scheme']=='CN' and r['solver']=='fs' and r['dt']==.01),
      ('dt','Time step · same-allocation production timings',lambda r:r['source']=='frontier3d' and r['r']==1024 and r['solver']=='qn'),
      ('mesh','Mesh · fixed output contract',lambda r:r['source']=='mesh3d' and r['policy']=='mf')]}
    for dim,ss in specs.items():
        good=[]
        for knob,label,selector in ss:
            rr=sorted([r for r in ROWS if r['dimension']==dim and selector(r)],key=lambda r:r[knob])
            if len({r[knob] for r in rr})>=2:good.append((knob,label,rr))
        fig,axs=plt.subplots(len(good),3,figsize=(15,3.1*len(good)),squeeze=False);points=[]
        for i,(knob,label,rr) in enumerate(good):
            draw_group(axs[i],rr,knob,rr[0]['cost_kind']);points+=rr
            axs[i,0].set_title(label,fontsize=11,loc='left',pad=14)
            small,sax=plt.subplots(1,3,figsize=(15,4))
            draw_group(sax,rr,knob,rr[0]['cost_kind'])
            controls=', '.join(f'{k}={rr[0][k]}' for k in ['r','M','m','dt','scheme','solver'] if k!=knob)
            if knob=='K':controls='Selected-rule policy: M and m may vary; seed '+str(rr[0]['seed'])
            small.suptitle(f'{dim}D · {label}',fontsize=15,x=.04,ha='left')
            small.text(.04,.005,controls+'\n'+rr[0]['panel']+f" · {rr[0]['n_cases']} common cases · evolved outputs t=0.05–0.25. Circles pass source checks; crosses fail/unverified.",fontsize=8)
            small.tight_layout(rect=(0,.13,1,.96))
            small.subplots_adjust(left=.075,right=.99,wspace=.30)
            save(small,f'{dim}d-{knob}-highlight',f'{dim}D {label}',controls,dim,'highlight',rr)
        fig.suptitle(f'{dim}D Burgers · accuracy and cost knobs',fontsize=20,x=.06,ha='left',y=1.005)
        fig.text(.06,.005,'Each row holds the stated controls fixed. The K row follows its selected test-space/quadrature policy.\n'
          'Circles: source numerical checks passed. Crosses: failed/unverified. 2D: six development cases; 3D screening/mesh: four exposed cases; 3D time endpoints: full exposed validation.\n'
          'Fused cost includes input/initial fit, evolution and output. Iteration counts are not seconds. Numbers use conditional reference estimates.',fontsize=9)
        fig.tight_layout(h_pad=3,rect=(0,.055,1,.99))
        save(fig,f'{dim}d-overview',f'{dim}D accuracy and cost overview','Representative controlled sweeps; consult the exhaustive pages and source table for all saved settings and admission details.',dim,'overview',points)


def clean(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    return value


def output():
    (HERE/'POINTS.json').write_text(json.dumps(clean(ROWS),indent=1,allow_nan=False)+'\n')
    columns=['dimension','panel','id','r','K','M','m','mesh','dt','scheme','solver','policy','qualified','timing_usable',
      'n_cases','error_median','error_worst','error_over_1pct','milliseconds','iterations','residual','stationarity',
      'timing_repetitions','timing_outliers','source','source_result_sha256']
    with (HERE/'POINTS.csv').open('w') as f:
        w=csv.DictWriter(f,columns,extrasaction='ignore');w.writeheader();w.writerows(clean(ROWS))
    (HERE/'PLOTS.json').write_text(json.dumps(PLOTS,indent=2)+'\n')
    coverage=[]
    for dim in [2,3]:
        for knob in LABELS:
            rr=[r for r in ROWS if r['dimension']==dim]
            gg=groups(rr,knob,coupled=knob=='K')
            coverage.append({'dimension':dim,'knob':knob,'matched_groups':len(gg),
               'timed_groups':sum(all(finite(r['milliseconds']) for r in pts) for _,pts in gg),
               'work_groups':sum(all(finite(r['iterations']) for r in pts) for _,pts in gg),
               'qualification':'source population only; failed/unverified points retained',
               'scope':'K policy co-varies M/m' if knob=='K' and gg else 'absent; no nonlinear K sweep' if knob=='K' else 'listed controls held fixed'})
    (HERE/'COVERAGE.json').write_text(json.dumps(coverage,indent=2)+'\n')
    intro='''# Accuracy and cost knobs for Burgers in two and three dimensions

This atlas regenerates plots from preserved experiments and newly rescored saved timing witnesses. Numbers are final for those saved invocations; historical qualification labels, conditional reference estimates and missing controlled sweeps limit interpretation. No new training or PDE solve was run.

Open `index.html` for the filterable plot gallery. Each figure is available as PNG and SVG. `POINTS.csv` contains every plotted configuration; `POINTS.json` retains per-case scores, timings and repetition arrays. `PLOTS.json` maps figures to configurations. `SOURCES.json` and the retained compact data pin provenance.

## Reading the plots

The primary error is the maximum over evolved saved outputs at simulation times 0.05 through 0.25 of the field L2 difference divided by the initial reference-field L2 norm. It is neither training loss nor a pointwise error. Reference uncertainties are conditional Richardson estimates, not rigorous bounds. The main linked 2D panels use the same six development cases. The newly rescored 3D screening panel uses four exposed validation cases; the later frontier uses the full exposed validation population; historical work uses that full population; mesh diagnostics use their fixed diagnostic subset. Populations are never mixed inside a curve.

Each runtime curve stays within one allocation. Cost and error refer to the same timed returned field, or a scoring trajectory that the timed invocation reproduced by its recorded identity check. Initialization, evolution and output decoding are included in fused query time. Compilation, training and postprocessing are excluded. Separate phase measurements are diagnostic and are never summed to manufacture a fused time. Historical 3D controls without admitted warmed timing use nonlinear iteration counts; iterations do not measure equal work across methods or dimensions.

The recovered 3D screening panel is rescored from its own saved coefficient trajectories, not from a separate accuracy run. Its original in-job output links and A–B–A timing controls passed. New f64 physical-lattice Gram rescoring is checked against direct field reconstruction, with an independent NumPy implementation included for replay. The small screening population supports descriptive knob curves, not a new full-population or sealed validation claim. Minimal models, references, coefficients and products are preserved; a large replay archive is divided into checksum-verified parts so every committed file can be backed up to GitHub.

Markers retain the original numerical qualification outcome: circles pass the reported solver/quadrature checks; crosses fail or remain unverified. A passing development check is not full validation or a claim of physical accuracy. No failed configuration is used to assert an accepted speedup. Every point retains its case count, worst error and count of errors above the explicitly descriptive one-percent threshold. Timing outliers use within-case Tukey fences when at least four repeats exist; small repetition counts cannot establish tail behavior.

## Scope and gaps

- Bank width, test-space size, quadrature count/family, time step/method and mesh resolution are covered wherever saved records supply multiple settings. Exhaustive gallery pages hold the other listed settings fixed.
- The nonlinear head K panels vary K under the historical selected-rule policy. Test-space size and quadrature can change with K; those curves do not isolate a pure K effect. Separate seeds remain separate. The recorded 2D qualified K trend is inconclusive.
- The saved 3D bank is a linear-coefficient model. Its `K1` and `k4` labels are solver work policies, not nonlinear latent dimensions. A matched nonlinear-head K sweep in 3D is absent and cannot be inferred from width or work-policy curves.
- Solver tolerance and fixed/refreshed nonlinear work policies are available as diagnostic controls rather than a complete warmed runtime ladder. Their evidence is retained in the source data; this atlas does not use first-call/compilation seconds as production cost.
- Raw weak-residual norms and stationarity certificates are plotted separately from physical error. Changing test functions, residual weights, integration scheme or time step changes the objective, so residual values are not compared as a common accuracy scale. A certificate is not a loss value, and residual norm is not silently squared into training loss.
- A systematic 3D head sweep, a controlled fixed-M/fixed-quadrature K study, and any missing matched tolerance timings require new experiments. This atlas completes the saved-evidence plotting, not those unrun experiments. Full-decoder comparisons, adaptive/local refinement and other PDE extensions remain deferred at the user's request.

## Reproduce

Run `/home/tahmid/Dev/.venv/bin/python build.py` from this directory, with `OPENBLAS_NUM_THREADS=1`. It needs only the compact `data/` files and plotting libraries. `collect.py` documents the source extraction and can refresh it read-only from the original local lanes and checksum-verified cluster archives. Do not delete shared archives to reproduce these figures.

## Plot inventory

'''
    lines=[intro]
    lines.append('| Dimension | Knob | Matched groups | Groups with complete warmed timing | Groups with iteration counts |\n|---|---|---:|---:|---:|')
    for v in coverage:lines.append(f"| {v['dimension']}D | {v['knob']} | {v['matched_groups']} | {v['timed_groups']} | {v['work_groups']} |")
    lines.append('\nCounts describe coverage, not independent replications. Multiple sweeps can reuse configurations.\n')
    for dim in [2,3]:
        lines.append(f'### {dim}D\n')
        lines.append(f'- [Overview](figs/{dim}d-overview.png)')
        for knob in [*LABELS,'loss','training']:
            pp=[p for p in PLOTS if p['dimension']==dim and p['knob']==knob]
            if pp:lines.append(f"- {LABELS.get(knob,knob)}: "+', '.join(f"[page {i+1}](figs/{p['slug']}.png)" for i,p in enumerate(pp)))
    lines.append('''
## Plain-language glossary

- **r / bank width:** number of coefficients in the ordered linear dictionary used to represent the field.
- **K / latent dimension:** number of nonlinear-head input coordinates. Solver labels `K1` and `k4` in historical 3D records mean something different: work or refresh policies.
- **M / test-space size:** number of smooth functions against which the PDE residual is projected.
- **m / quadrature count:** number of integration points used to approximate that projection. A tensor rule's per-axis order is not its total point count.
- **Mesh:** spatial discretization size. Source conventions differ: historical 2D uses intervals per axis; the newer 3D mesh panel uses nodes per axis, including boundary nodes.
- **Time step / dt:** time advanced in one numerical step. Changing it can also change residual weights and the directions used by a least-squares solver.
- **BE, CN, CNR, BDF2:** backward Euler, Crank–Nicolson, Crank–Nicolson with the recorded startup treatment, and second-order backward differentiation: alternative time integrators.
- **LSPG / GAL:** a least-squares solve of the projected time-discrete residual / a Galerkin reduced equation. Their objectives differ.
- **fs / qn / ad:** fixed-sweep, refreshed quasi-Newton, and adaptive nonlinear-solver policies in the original records.
- **Quadrature family / rule:** integration construction, such as Gauss, lattice or Fibonacci; the rule name specifies the saved point set.
- **Seed:** recorded random initialization or sampling identifier; separate seeds are separate repetitions of the model-building recipe.
- **Conditional L2 error:** field-distance estimate relative to the reference, normalized by the initial reference field. **Median** is the middle case value; **worst** is the largest case value.
- **eps / reference uncertainty:** conditional estimated error of the reference, used to judge whether differences are resolved; it is not an absolute guarantee.
- **Residual / loss:** mismatch with the discrete equations being minimized. The saved raw norm is reported in its source units; it is not interchangeable with field error or a squared loss.
- **Stationarity / certificate:** a check that a returned solver state satisfies a recorded stopping condition. It does not prove that the reconstructed field is accurate.
- **Qualified / Q1 / Q2:** the source solver-convergence checks / source quadrature-consistency checks passed for their recorded population. A failed or missing check remains visible.
- **Fused query cost:** measured input preparation, initial fit, evolution and output synthesis under the stated source contract, in milliseconds. A **timing panel** is one allocation on one GPU.
- **Iteration / correction:** one recorded nonlinear update. Different solvers and dimensions can perform different work per update.
- **Invocation / witness / hash:** one solver execution / its saved output / a content fingerprint linking the output to a repeated timed execution.
- **Development / exposed validation:** cases already available for research and selection; neither is sealed confirmation. No sealed or out-of-distribution cases are read here.
- **POINTS table:** dimension, panel and id identify the run; r, K, M, m, mesh, dt, scheme, solver and policy specify its controls; qualified and timing_usable retain admission; n_cases records population; error columns record median, worst and count above one percent; milliseconds, iterations, residual and stationarity give the named metrics; timing_repetitions/outliers describe retained timings; source and SHA256 columns identify evidence.
''')
    (HERE/'2026-10-29-burgers-accuracy-and-cost-knobs.md').write_text('\n'.join(lines))
    cards=[]
    for p in sorted(PLOTS,key=lambda p:p['knob']!='overview'):
        slug=p['slug'];cards.append(f'''<section data-dim="{p['dimension']}" data-knob="{p['knob']}"><h2>{html.escape(p['title'])}</h2>
<p>{html.escape(p['caption'])}</p><a href="figs/{slug}.png"><img loading="lazy" src="figs/{slug}.png" alt="{html.escape(p['title'])}"></a>
<p><a href="figs/{slug}.svg">Vector SVG</a> · <a href="figs/{slug}.png">PNG</a></p></section>''')
    opts=''.join(f'<option value="{k}">{html.escape(LABELS.get(k,k))}</option>' for k in ['overview','highlight',*LABELS,'loss','training'])
    (HERE/'index.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Burgers accuracy and cost knobs</title><style>body{max-width:1500px;margin:36px auto;padding:0 24px;font:16px/1.5 system-ui;color:#17212b;background:#fafafa}h1{font-size:30px}h2{font-size:21px}header{max-width:1000px}nav{position:sticky;top:0;background:#fafafa;padding:16px 0;border-bottom:1px solid #ddd;z-index:5}select{padding:8px;margin-right:16px}section{padding:28px 0;border-bottom:1px solid #ddd}img{width:100%;height:auto}a{color:#176b87}.note{padding-left:16px;border-left:3px solid #176b87}section[hidden]{display:none}</style>
<header><h1>Accuracy and cost knobs</h1><p>Burgers in 2D and 3D · preserved experiment evidence · compiled October 29, 2026</p>
<p>Read each figure from left to right: physical error, query cost or nonlinear work, then their tradeoff. Circles pass the source numerical checks; crosses fail or remain unverified. Runtime curves stay within one GPU allocation.</p>
<p class="note">The 2D nonlinear latent K curves follow a selected test-space/quadrature policy. There is no matched 3D nonlinear-head K sweep. Training, raw weak residual and physical rollout error are separate metrics.</p>
<p><a href="2026-10-29-burgers-accuracy-and-cost-knobs.md">Methods, limitations and glossary</a> · <a href="POINTS.csv">Plot data CSV</a> · <a href="POINTS.json">Per-case data JSON</a> · <a href="SOURCES.json">Source provenance</a></p></header>
<nav><label>Dimension <select id="dim"><option value="all">Both</option><option value="2">2D</option><option value="3">3D</option></select></label><label>Knob <select id="knob"><option value="all">All knobs</option>'''+opts+'''</select></label><span id="count"></span></nav>'''+''.join(cards)+'''
<script>const dim=document.querySelector('#dim'),knob=document.querySelector('#knob');function update(){let n=0;document.querySelectorAll('section').forEach(s=>{s.hidden=!((dim.value==='all'||s.dataset.dim===dim.value)&&(knob.value==='all'||s.dataset.knob===knob.value));if(!s.hidden)n++});document.querySelector('#count').textContent=n+' figure pages'}dim.onchange=knob.onchange=update;update();</script></html>''')
    expected={p['slug'] for p in PLOTS}
    for p in FIG.iterdir():
        if p.suffix in ['.png','.svg'] and p.stem not in expected:p.unlink()
    manifest={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.rglob('*'))
       if p.is_file() and p.name not in ['MANIFEST.json','VALIDATION.json'] and '__pycache__' not in str(p)}
    (HERE/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'configurations':len(ROWS),'plots':len(PLOTS),'by_dimension':{d:sum(r['dimension']==d for r in ROWS) for d in [2,3]}}))


def main():
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
      'svg.fonttype':'none','savefig.pad_inches':.15})
    import sys
    if '--extend-timed3d' in sys.argv:
        ROWS.extend(r for r in json.loads((HERE/'POINTS.json').read_text()) if r['source']!='screen3d')
        PLOTS.extend(p for p in json.loads((HERE/'PLOTS.json').read_text()) if p['knob'] not in ['overview','highlight'] and '-screen-' not in p['slug'])
        three_d_screen();overview();sweep_pages(only_sources={'screen3d'},tag='screen-');residual_pages(only_sources={'screen3d'},tag='screen-');output()
    elif '--refresh-overview' in sys.argv:
        ROWS.extend(json.loads((HERE/'POINTS.json').read_text()))
        PLOTS.extend(p for p in json.loads((HERE/'PLOTS.json').read_text()) if p['knob'] not in ['overview','highlight'])
        overview();output()
    else:
        two_d();three_d();overview()
        old={r['source'] for r in ROWS}-{'screen3d'}
        sweep_pages(only_sources=old);residual_pages(only_sources=old);training()
        if 'screen3d' in {r['source'] for r in ROWS}:
            sweep_pages(only_sources={'screen3d'},tag='screen-');residual_pages(only_sources={'screen3d'},tag='screen-')
        output()


if __name__=='__main__':main()
