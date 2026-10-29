"""Development-only actual FOM calibration; no validation or eligibility writer.

A separately reviewed finite candidate ladder is timed within one allocation. Every
actual returned state/output is retained and scored; failures stay in the ladder.
"""
from pathlib import Path
import gc,hashlib,json,os,re,resource,shutil,sys,time
import numpy as np
from fe_adapter import sha_file
from qualification import require_qualification
from pilot2d import DEV_RE,OUTPUTS,load_reference_population,save_evidence
from fom_comparator2d import (FrozenFOMComparator,checked_file,read_json,native_digest,
                              source_digest as comparator_digest)
from fom_bridge2d import score_fom_invocation
from timing import calibration_cpu


def source_digest():
    return hashlib.sha256(json.dumps({'entry':sha_file(__file__),'comparator':comparator_digest(),
        'readout':sha_file(Path(__file__).with_name('fom_readout2d.py'))},sort_keys=True).encode()).hexdigest()


def verify(plan_path,release_path):
    plan_path=Path(plan_path);p=read_json(plan_path);r=read_json(release_path)
    if (r.get('schema')!='ns-obstacle-fom-development-calibration-release-v1' or
        r.get('authorized_stage')!='development_fom_calibration' or r.get('source_sha256')!=source_digest() or
        r.get('plan_sha256')!=sha_file(plan_path) or r.get('single_execution') is not True):
        raise ValueError('exact separately reviewed calibration plan/release required')
    if (p.get('schema')!='ns-obstacle-fom-development-calibration-plan-v1' or p.get('development_Re')!=list(DEV_RE) or
        p.get('validation_allowed') is not False or p.get('output_contracts')!=list(OUTPUTS) or
        p.get('native_source_sha256')!=native_digest() or p.get('selection_policy')!='minimum_pooled_development_median_all_invocations_eligible_ties_ladder_order'):
        raise ValueError('frozen development-only ladder/selection policy required')
    if type(r.get('allocation_seconds')) is not int or not 0<r['allocation_seconds']<=2400:
        raise ValueError('fixed calibration allocation cap required')
    if type(r.get('finalization_reserve_seconds')) is not int or not 60<=r['finalization_reserve_seconds']<=300 or r['finalization_reserve_seconds']>=r['allocation_seconds']:
        raise ValueError('explicit finalization reserve within original allocation required')
    if r.get('launcher_hard_wall_seconds')!=r['allocation_seconds']:raise ValueError('reviewed launcher hard wall must enforce original cap')
    if type(p.get('cores')) is not int or not 1<=p['cores']<=16:raise ValueError('fixed CPU policy required')
    if type(p.get('max_common_points')) is not int or not 0<p['max_common_points']<=1300000:raise ValueError('full common point cap required')
    if p.get('repetitions')!=7:raise ValueError('seven actual repetitions required')
    versions=p.get('runtime_versions')
    if not isinstance(versions,dict) or set(versions)!={'python','ngsolve','numpy','scipy','jax'} or any(not isinstance(v,str) or not v for v in versions.values()):
        raise ValueError('exact native runtime versions required')
    if type(p.get('common_rule_points')) is not int or not 0<p['common_rule_points']<=p['max_common_points']:
        raise ValueError('complete common rule count required')
    if not re.fullmatch('[0-9a-f]{64}',p.get('volume_rule_sha256','')):raise ValueError('common rule digest required')
    base=plan_path.parent;truth=checked_file(base,p['truth_admission'])
    qualification=require_qualification(read_json(truth))
    if require_qualification(r.get('reference_qualification'))!=qualification or require_qualification(p.get('reference_qualification'))!=qualification:
        raise ValueError('actual FD1 scope differs')
    population=checked_file(base,p['development_population']);meta=read_json(population)
    if meta.get('split')!='development' or meta.get('reference_admission_sha256')!=sha_file(truth):raise ValueError('development truth identity differs')
    geometry=checked_file(base,p['common_geometry'])
    candidates=p.get('candidates',[])
    if not 1<=len(candidates)<=6 or len({c['id'] for c in candidates})!=len(candidates):raise ValueError('finite unique candidate ladder required')
    resolved=[]
    for c in candidates:
        if not re.fullmatch('[a-zA-Z0-9_-]+',c['id']) or c.get('field_rule_mode') not in ('native_exact','certified_requested_points'):
            raise ValueError('safe unique identity and frozen native field mode required')
        if type(c.get('runtime_workspace_reserve_bytes')) is not int or not 0<c['runtime_workspace_reserve_bytes']<=16*2**30:
            raise ValueError('reviewed prospective native workspace reserve required')
        files={key:checked_file(base,c[key]) for key in ('config','mesh','layout','boundary','eligibility')}
        files['common_geometry']=geometry;cfg=read_json(files['config']);eligibility=read_json(files['eligibility'])
        fixed={'kind':'dfg','outlet':'gradient','xmin':0.,'xmax':2.2,'H':.41,'center':[.2,.2],'radius':.05,'U':.2,'D':.1,'precision':'float64'}
        if any(cfg.get(k)!=v for k,v in fixed.items()) or cfg.get('threads')!=p['cores']:raise ValueError('candidate fixed channel/CPU policy differs')
        if eligibility.get('role')!='development_calibration_only' or eligibility.get('truth_reference_admission_sha256')!=sha_file(truth):
            raise ValueError('distinct development candidate eligibility required; no coarse truth or frozen selection')
        resolved.append((c,files,cfg,read_json(files['layout'])))
    return p,r,qualification,truth,population,resolved


class DevelopmentCandidate(FrozenFOMComparator):
    def check_bindings(self):
        lift=self.native.lift;r=self.release
        if self.configuration_digest({'release':r,'qualification':self.qualification})!=self.binding_digest:
            raise ValueError('development release/scope mutated')
        if (lift.usage_role!='development_calibration_only' or lift.selection_frozen or
            lift.eligibility_sha256!=r['eligibility_sha256'] or lift.truth_reference_admission_sha256!=r['reference_admission_sha256'] or
            lift.configuration_sha256!=self.expected_configuration or self.configuration_digest(self.native.cfg)!=self.expected_configuration):
            raise ValueError('actual development candidate binding changed')

    def decorate(self,invocation):
        return dict(invocation,development_calibration={'candidate':self.release['candidate_id'],
            'release_sha256':self.release_file_sha256,'reference_admission_sha256':self.release['reference_admission_sha256'],
            'reference_qualification':self.qualification,'volume_rule_sha256':self.rule_hash,'validation_used':False})

    def query(self,reynolds,contract):
        self.check_bindings()
        if reynolds not in DEV_RE or contract not in OUTPUTS:raise ValueError('only frozen development parameters/contracts')
        try:return self.decorate(self.native.query(reynolds,contract))
        except Exception as exc:
            if isinstance(getattr(exc,'invocation',None),dict):exc.invocation=self.decorate(exc.invocation)
            raise


def choose(candidates,blocks,contract):
    """No successful-only median: every required invocation/control must qualify."""
    rankings=[]
    for index,c in enumerate(candidates):
        rows=[b for b in blocks if b['candidate']==c['id'] and b['contract']==contract]
        eligible=(len(rows)==len(DEV_RE) and {b['Re'] for b in rows}==set(DEV_RE) and
                  all(b['eligible'] is True and len(b['elapsed_seconds'])==7 and
                      np.isfinite(b['elapsed_seconds']).all() and min(b['elapsed_seconds'])>0 for b in rows))
        seconds=[v for b in rows for v in b.get('elapsed_seconds',[])]
        rankings.append({'candidate':c['id'],'eligible':eligible,'median_seconds':float(np.median(seconds)) if eligible else None,'ladder_index':index})
    admitted=[r for r in rankings if r['eligible']]
    winner=min(admitted,key=lambda r:(r['median_seconds'],r['ladder_index']))['candidate'] if admitted else None
    return winner,rankings


def run(plan_path,release_path,output):
    started=time.perf_counter();plan,release,qualification,truth,population,resolved=verify(plan_path,release_path)
    if Path(release['job_directory']).resolve()!=Path.cwd().resolve() or Path(release['output_directory']).resolve()!=Path(output).resolve():
        raise ValueError('released allocation/output differs')
    # Native runtime/GPU/f64 policy is checked again by shared construction.
    if sys.executable!='/cluster/tufts/paralab/tawal01/ae-research/venv/bin/python' or not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('one released cluster allocation required')
    allocation_start=float(os.environ['ROM_ALLOCATION_START_UNIX_SECONDS'])
    elapsed_before_entry=time.time()-allocation_start
    if not np.isfinite(elapsed_before_entry) or elapsed_before_entry<0 or elapsed_before_entry>=release['allocation_seconds']:
        raise TimeoutError('original allocation setup already exhausted or invalid')
    deadline=time.perf_counter()+release['allocation_seconds']-elapsed_before_entry-release['finalization_reserve_seconds']
    with (Path.cwd()/'FOM-CALIBRATION-CLAIM.json').open('x') as claim:json.dump({'release_sha256':sha_file(release_path),'job_id':os.environ['SLURM_JOB_ID']},claim)
    out=Path(output);out.mkdir(exist_ok=False);blocks=[];failures=[];source=source_digest()
    ledger={'status':'running','blocks':blocks,'construction_failures':failures,'physical_admission':False,'validation_used':False,
            'plan_sha256':sha_file(plan_path),'source_sha256':source,'reference_qualification':qualification,
            'allocation_start_unix_seconds':allocation_start,'elapsed_setup_before_entry_seconds':elapsed_before_entry,
            'finalization_reserve_seconds':release['finalization_reserve_seconds'],
            'cooperative_checks_do_not_bound_inflight_native_work':True,'launcher_hard_wall_seconds':release['launcher_hard_wall_seconds']}
    try:
        for candidate,files,cfg,layout in resolved:
            identifier=candidate['id'];directory=out/identifier;directory.mkdir()
            runtime={**plan,'candidate_id':identifier,'field_rule_mode':candidate['field_rule_mode'],'eligibility_sha256':sha_file(files['eligibility']),
                'reference_admission_sha256':sha_file(truth),'reynolds':list(DEV_RE)}
            declared={**plan,**candidate}
            query=None
            try:
                if time.perf_counter()>=deadline:raise TimeoutError('original calibration allocation exhausted')
                query=DevelopmentCandidate._construct(runtime,declared,qualification,files,cfg,layout,files['eligibility'],release_path,directory/'construction')
                references=load_reference_population(population,plan['development_population']['sha256'],'development',query.rule_hash)
                for reference in references:
                    for contract in OUTPUTS:
                        if time.perf_counter()>=deadline:raise TimeoutError('science budget exhausted; preserve finalization reserve')
                        blockdir=directory/f"Re{reference['Re']}_{contract}";blockdir.mkdir()
                        values={'full_velocity_pressure_fields':len(query.points)*3,'fixed_probes':6,'integrated_forces':2}[contract]
                        payload=7*8*(values+query.native.X.ndof)+len(query.points)*9*8+2**20
                        if shutil.disk_usage(out).free<10*2**30+2*(payload+payload//50+16*2**20):raise OSError('next actual block/durable copy exceeds disk reserve')
                        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024+3*payload+candidate['runtime_workspace_reserve_bytes']>16*2**30:
                            raise MemoryError('native workspace plus actual output retention exceeds16GiB')
                        def call():
                            if time.perf_counter()>=deadline:raise TimeoutError('original calibration allocation exhausted')
                            return query.query(reference['Re'],contract)
                        timing=calibration_cpu(call,contract,plan['cores']);receipt=save_evidence(blockdir,'actual_invocations',timing)
                        scores=[];control=None
                        for row in timing['actual_invocation_outputs']:
                            if time.perf_counter()>=deadline:
                                scores.append({'eligible_accuracy':False,'diagnostics_unavailable_budget':True});continue
                            try:score=score_fom_invocation(query.native,row['invocation'],reference,query.weights,query.points)
                            except Exception as exc:score={'eligible_accuracy':False,'diagnostic_failed':True,'exception':repr(exc)}
                            scores.append(score)
                        successful=[row for row in timing['actual_invocation_outputs'] if not row['invocation'].get('query_failed')]
                        if successful and time.perf_counter()<deadline:
                            from fom_readout2d import readout_control
                            try:control=readout_control(query,successful[-1]['invocation'],directory/'construction',blockdir,deadline)
                            except Exception as exc:control={'passed':False,'exception':repr(exc)}
                        eligible=all(s.get('eligible_accuracy') is True for s in scores) and bool(control and control.get('passed'))
                        scored=save_evidence(blockdir,'accuracy',{'same_invocation_scores':scores,'readout_control':control,'timing_record':receipt})
                        blocks.append({'candidate':identifier,'Re':reference['Re'],'contract':contract,'eligible':eligible,'readout_passed':bool(control and control.get('passed')),
                            'elapsed_seconds':[r['elapsed_seconds'] for r in timing['repetitions']],
                            'evidence':{'path':str(blockdir/scored['record_file']),'sha256':scored['record_sha256']},
                            'raw_evidence':receipt,'measured_block_bytes':sum(p.stat().st_size for p in blockdir.rglob('*') if p.is_file())})
                        ledger['elapsed_seconds']=time.perf_counter()-started
                        (out/'CALIBRATION.json').write_text(json.dumps(ledger,indent=2)+'\n')
                        del timing,scores,successful
            except Exception as exc:failures.append({'candidate':identifier,'exception':repr(exc)})
            finally:del query;gc.collect()
        ledger['status']='completed_pending_independent_review'
        evidence=out/'CALIBRATION.json';evidence.write_text(json.dumps(ledger,indent=2)+'\n')
        for contract in OUTPUTS:
            winner,ranking=choose(plan['candidates'],blocks,contract)
            selection={'schema':'ns-obstacle-fom-development-common-target-screen-v1','selection_frozen':False,
                'development_only':True,'validation_used':False,'matched_accuracy':False,
                'common_targets_passed':winner is not None,'bank_matching_performed':False,
                'development_Re':list(DEV_RE),'output_contract':contract,'diagnostic_best_common_target_candidate':winner,'rankings':ranking,
                'reference_qualification':qualification,'reference_admission_sha256':sha_file(truth),
                'native_source_sha256':native_digest(),'calibration_source_sha256':source,'runtime_versions':plan['runtime_versions'],
                'common_rule_points':plan['common_rule_points'],'volume_rule_sha256':plan['volume_rule_sha256'],
                'development_evidence':{'path':str(evidence.resolve()),'sha256':sha_file(evidence)}}
            if winner is not None:
                c,files,_,_=next(row for row in resolved if row[0]['id']==winner)
                selection.update(field_rule_mode=c['field_rule_mode'],**{k:{'path':str(files[k]),'sha256':sha_file(files[k])} for k in ('config','mesh','layout','boundary','common_geometry')})
            (out/(contract+'-common-target-screen.json')).write_text(json.dumps(selection,indent=2)+'\n')
    except Exception as exc:ledger.update(status='failed',exception=repr(exc));raise
    finally:
        # Final accounting stays separate: frozen selections bind immutable CALIBRATION.json.
        final={'status':ledger['status'],'elapsed_seconds':time.perf_counter()-started,
            'peak_host_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'allocation_elapsed_seconds':time.time()-allocation_start,
            'actual_output_bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),'physical_admission':False}
        (out/'FINAL.json').write_text(json.dumps(final,indent=2)+'\n')
    return ledger


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('plan','release','out'):parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();run(args.plan,args.release,args.out)
