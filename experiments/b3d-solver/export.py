"""Generate the coordinator's small numerical export directly from preserved job outputs."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ledger=json.loads((HERE/'JOB-LEDGER.json').read_text())
    audits=json.loads((HERE/'AUDIT-STATUS.json').read_text())
    out=dict(audits=audits,status='diagnostic subset; consult per-job independent audit verdicts',scope='validation diagnostics; no new frontier or comparator win claim',
             gpu_hours=sum(e.get('gpu_hours',0.) for e in ledger['entries']),budget_gpu_hours=ledger['budget_gpu_hours'],
             jobs=ledger['entries'],observations=[],same_history=[],archive_identities=[],quadrature=[],quadrature_witnesses=[],time_diagnostics=[],time_quadrature=[],POD_tuning=[],POD_tuning_jobs=[],full64_frontier=[],mesh_results=[],retention_manifests=[],
             time_intervention_scope='Time-step/method package: dt changes integration, inherited diffusion-scaled residual row weights and time-discrete LSPG directions. Q1/Q2 control nonlinear/quadrature effects; this does not isolate pure temporal truncation.',
             equation_notes=dict(POD_none='Trial states are restricted to the frozen POD span, but the common LM objective minimizes the full central-FD residual norm. It does not first Galerkin-project the residual. This explicitly corrects ambiguous inherited raw metadata wording without modifying raw results.',
                 bank='Smooth-test weak residual with analytic continuum advection quadrature and discrete sine diffusion; inherited diffusion-dependent row weighting. No upwind substitution.',
                 spectral='Frozen3/2 oversampled sine ETDRK4 implementation; no exact de-aliasing claim'),
             next_open=[])
    for entry in ledger['entries']:
        rp=HERE/'results'/entry['attempt']/'raw/output/result.json'
        if not rp.exists():continue
        z=json.loads(rp.read_text());source=dict(path=str(rp.relative_to(HERE)),sha256=sha(rp),job_id=entry['job_id'],source_commit=z['source_commit'],gpu=z['gpu'],config_sha256=z['config_sha256'])
        original_manifest=rp.parents[1]/'MANIFEST.sha256'
        if original_manifest.exists():source['original_job_manifest']=dict(path=str(original_manifest.relative_to(HERE)),sha256=sha(original_manifest))
        for mode,filename,collection in [('front','front_result.json','full64_frontier'),('mesh','mesh_result.json','mesh_results')]:
            actual=rp.parent/filename
            if z['config']['mode']==mode and actual.exists():
                out[collection].append(dict(source=source,result=dict(path=str(actual.relative_to(HERE)),sha256=sha(actual)),
                    independent_audit=audits['jobs'].get(entry['attempt'],{}),complete=json.loads(actual.read_text())['execution']['complete']))
                summary=rp.parents[2]/'SUMMARY.json'
                if summary.exists():out[collection][-1]['summary']=dict(path=str(summary.relative_to(HERE)),sha256=sha(summary))
        if z['config']['mode']=='pod_tune':
            out['POD_tuning_jobs'].append(dict(source=source,scope=z['scope'],execution_complete=z['complete'],failure=z.get('failure'),
                selection=z.get('selection',{}),frozen_picks=z.get('frozen_picks',[]),identity_probes=z.get('identity_probes',{})))
        for key,v in z['runs'].items():
            if z['config']['mode']=='pod_tune':
                out['POD_tuning'].append(dict(key=key,source=source,scope=z['scope'],**v));continue
            if z['config']['mode']=='quad_witness':
                out['quadrature_witnesses'].append(dict(key=key,source=source,scope=z['scope'],
                    tight_tighter_distance=z.get('tight_tighter_distance'),anchor_usable=z.get('anchor_usable'),**v));continue
            if z['config']['mode']=='time':
                payload=dict(v)
                if 'source' in payload:payload['trajectory_source']=payload.pop('source')
                out['time_diagnostics'].append(dict(payload,key=key,source=source,scope=z['scope']));continue
            if 'nonlinear_iterations' not in v:
                out['quadrature'].append(dict(key=key,source=source,**v));continue
            obs=dict(key=key,source=source,scope=z['scope'],score=v['score'],fresh_nonstationary_steps=v['nonstationary'],
                     nonlinear_iterations=v['nonlinear_iterations'],jacobian_contractions=v['jacobian_contractions'],
                     residual_evaluations=v['residual_evaluations'],historical_reproduction_relative=v.get('historical_reproduction_relative'),
                     distance_historical_tight=v.get('distance_historical_tight'),distance_gate_diagnostic=v.get('distance_gate_diagnostic'),
                     valid=v.get('valid',v['finite']),Q1_full_population='not established',Q2='historical gl40 evidence; not recertified by this single-case pilot')
            obs['timing_seconds']=[]
            obs['timing_admission']='pilot descriptive; drift gate not applied' if entry['attempt']=='spilot02' else 'bracket drift ratio in [1/1.1,1.1]'
            obs['timing_all_admitted']=True
            for tk,trips in z['timing'].items():
                sn,policy,case=key.split('|')
                if not tk.startswith(sn+'|') or not tk.endswith('|'+case):continue
                for trip in trips:
                    for r in trip:
                        if r['arm']==policy:
                            obs['timing_seconds'].append(r['seconds'])
                            obs['timing_all_admitted'] &= r.get('timing_admitted',False)
            obs['query_median_seconds']=float(np.median(obs['timing_seconds'])) if obs['timing_seconds'] else None
            obs['max_evolved_error_time_index']=int(np.argmax(v['score']['e_t'][1:])+1)
            obs['max_evolved_error_time']=.05*obs['max_evolved_error_time_index']
            out['observations'].append(obs)
        for k,v in z['restarts'].items():
            sn,case,_=k.split('|');baseline=z['runs'][sn+'|baseline|'+case]
            mt=.05*(int(np.argmax(baseline['score']['e_t'][1:]))+1);st=.025*(v['step']+1)
            out['same_history'].append(dict(key=k,source=source,step_output_time=st,physical_max_error_time=mt,
                physical_max_precedes_replayed_step=mt<st,step_fails_stationarity=v['diagnostics']['baseline_gamma']>1e-3,**v))
        out['archive_identities'] += [dict(key=k,source=source,**v) for k,v in z['archive_identity'].items()]
        out['time_quadrature'] += [dict(key=k,source=source,**v) for k,v in z.get('quadrature',{}).items()]
    qr=HERE/'results/historical-tight-quadrature.json'
    if qr.exists():out['historical_tight_quadrature']=dict(path=str(qr.relative_to(HERE)),sha256=sha(qr),data=json.loads(qr.read_text()))
    ts=HERE/'results/time-selection.json'
    if ts.exists():out['time_selection']=dict(path=str(ts.relative_to(HERE)),sha256=sha(ts),data=json.loads(ts.read_text()))
    for p in sorted((HERE/'results').glob('*/RETENTION.json')):
        retained=json.loads(p.read_text())
        out['retention_manifests'].append(dict(path=str(p.relative_to(HERE)),sha256=sha(p),attempt=retained['attempt'],job_id=retained['job_id'],
            archive=dict(path=retained['independent_local_collection_archive'],sha256=retained['collection_archive_sha256']),
            bulk_arrays=[r for r in retained['arrays'] if r['retention'].startswith('local bulk')],archival_precondition=retained['archival_precondition']))
    accepted=lambda rows:any(r['complete'] and r['independent_audit'].get('results','').startswith('READY') for r in rows)
    front_ready=accepted(out['full64_frontier']);mesh_ready=accepted(out['mesh_results'])
    active=[e for e in ledger['entries'] if e.get('state',e.get('status')) in ('RUNNING','PENDING','SUBMITTED','COMPLETING','CONFIGURING')]
    out['campaign_status']=dict(full64_complete_results_gate=front_ready,mesh_complete_results_gate=mesh_ready,active_jobs=[dict(attempt=e['attempt'],job_id=e['job_id'],state=e.get('state',e.get('status'))) for e in active])
    if not front_ready:out['next_open'].append('Complete immutable full64 collection and independent results gate')
    if not mesh_ready:out['next_open'].append('Complete immutable fixed-mode mesh collection and independent results gate')
    out['status']='complete independently audited frozen campaign' if front_ready and mesh_ready else 'frozen campaign in progress; consult actual collection and independent-gate state'
    out['scope']='Same-allocation full64 paired frontier and separate within-job fixed-mode mesh diagnostics; finite exposed validation sets and per-query masks apply.' if front_ready and mesh_ready else 'Accepted diagnostic subsets plus any actual collected full64/mesh data; pending or incomplete data remain unreleased.'
    out['metadata_generator_sha256']=sha(Path(__file__))
    out['source_job_pins']=[dict(attempt=e['attempt'],job_id=e['job_id'],source_commit=e.get('source_commit'),manifest_sha256=e.get('manifest_sha256')) for e in ledger['entries']]
    (HERE/'EXPORT.json').write_text(json.dumps(out,indent=1)+'\n')
    print(json.dumps(dict(observations=len(out['observations']),same_history=len(out['same_history']),gpu_hours=out['gpu_hours'])))

if __name__=='__main__':main()
