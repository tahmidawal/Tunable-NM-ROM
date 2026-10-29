"""Assemble raw common-rule contrasts, empirical noise and conditional uncertainty.

This produces evidence, never reference admission. Physical state gates remain a
separate reviewed requirement. No pressure shift or numerical threshold is fitted.
"""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from immutable_rule import array_digest

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def load_fields(folder,record,point_count):
    path=folder/record['field_file']
    if sha(path)!=record['field_sha256']:raise ValueError('actual readout archive changed')
    with np.load(path,allow_pickle=False)as z:fields={k:z[k]for k in ('velocity','pressure')}
    if any(array_digest(v)!=record['field_content_sha256'][k]for k,v in fields.items()):raise ValueError('actual readout field changed')
    if fields["velocity"].shape!=(point_count,2)or fields["pressure"].shape!=(point_count,):raise ValueError("field shape differs from complete rule")
    if any(v.dtype!=np.dtype("float64")or not np.isfinite(v).all()for v in fields.values()):raise ValueError("fields must be finite f64")
    return fields

def main():
    p=argparse.ArgumentParser();p.add_argument('--common',type=Path,required=True);p.add_argument('--fresh-replay',type=Path);p.add_argument('--re20-noise',type=Path);p.add_argument('--out',type=Path,required=True);p.add_argument('--original-rule',type=Path);p.add_argument('--plan',type=Path,default=ROOT/'g2/population01/PLAN.json');a=p.parse_args()
    report=read(a.common/'readout.json')
    if report.get('complete')is not True or len(report['states'])!=15:raise ValueError('all15 actual states required')
    plan=read(a.plan)
    if report['complete_common_rule']!=plan['common_rule']:raise ValueError('declared original rule differs from frozen population plan')
    original=a.original_rule or ROOT/plan['common_rule']['source']
    if sha(original)!=plan['common_rule']['sha256']:raise ValueError('original physical rule container changed')
    with np.load(a.common/'common-rule.npz',allow_pickle=False)as z:w=z['weights'];points=z['points']
    with np.load(original,allow_pickle=False)as z:
        if not np.array_equal(w,z['weights'])or not np.array_equal(points,z['points']):raise ValueError('readout points/weights differ from original immutable rule')
    if len(w)!=plan['common_rule']['point_count']or w.dtype!=np.dtype('float64')or points.dtype!=np.dtype('float64')or not np.isfinite(points).all():raise ValueError('rule count/precision/finite identity invalid')
    if w.shape!=(len(points),)or points.shape!=(len(w),2)or np.any(w<=0)or not np.isfinite(w).all():raise ValueError('invalid complete physical rule')
    states=report['states'];fields={k:load_fields(a.common,r,len(w))for k,r in states.items()};fine=states['p4_g4_h3_baseline'];truth=fields['p4_g4_h3_baseline']
    den={'velocity':float(np.sqrt(np.sum(w[:,None]*truth['velocity']**2))),'pressure':.2**2*float(np.sqrt(w.sum())),'drag':abs(fine['force_order16'][0]),'lift':1.}
    if any(not np.isfinite(v)or v<=0 for v in den.values()):raise ValueError('nonpositive adopted-P4 normalization')
    def difference(left,right,left_force,right_force):
        for force in (left_force,right_force):
            if np.asarray(force).shape!=(2,)or not np.isfinite(force).all():raise ValueError('invalid actual force coefficients')
        raw={'velocity':float(np.sqrt(np.sum(w[:,None]*(left['velocity']-right['velocity'])**2))),'pressure':float(np.sqrt(np.sum(w*(left['pressure']-right['pressure'])**2))),'drag':abs(float(left_force[0])-float(right_force[0])),'lift':abs(float(left_force[1])-float(right_force[1]))}
        return {k:{'numerator':v,'denominator':den[k],'value':v/den[k]}for k,v in raw.items()}
    def pair(left,right):return difference(fields[left],fields[right],states[left]['force_order16'],states[right]['force_order16'])
    if report['Re']==20:
        if a.re20_noise is None:raise ValueError('reviewed retained Re20 R required, no silent replacement')
        retained=read(a.re20_noise)
        if retained['Re']!=20 or retained['weights_content_sha256']!=array_digest(w):raise ValueError('retained R belongs to another rule')
        fresh=None
    else:
        if a.fresh_replay is None:raise ValueError('actual fresh baseline replay required')
        fresh=read(a.fresh_replay/'readout.json')
        if fresh.get('complete')is not True or fresh['Re']!=report['Re']or len(fresh['states'])!=5:raise ValueError('incomplete or wrong-parameter replay')
        with np.load(a.fresh_replay/'common-rule.npz',allow_pickle=False)as z:
            if not np.array_equal(w,z['weights'])or not np.array_equal(points,z['points']):raise ValueError('fresh replay physical rule differs')
    noise={};extra_quadrature={}
    for space in ('p4_g4_h1','p4_g4_h2','p4_g4_h3','p3_g3_h3','p3_g4_h3'):
        base=space+'_baseline';s=pair(base,space+'_solver_100x');q=pair(base,space+'_quadrature_plus4')
        if fresh is None:
            r={k:retained['rows'][space][k]['numerator']/den[k]for k in ('velocity','pressure')}
        else:
            replay_entry=fresh['states'][base]
            if replay_entry['solution_sha256']!=states[base]['solution_sha256']:raise ValueError('replay changed actual coefficients')
            replay_fields=load_fields(a.fresh_replay,replay_entry,len(w));replay_diff=difference(fields[base],replay_fields,states[base]['force_order16'],replay_entry['force_replay_order16']);r={k:replay_diff[k]['value']for k in ('velocity','pressure')}
        force=states[base]['force_order16'];force_replay=states[base]['force_replay_order16']
        r.update(drag=abs(force[0]-force_replay[0])/den['drag'],lift=abs(force[1]-force_replay[1]))
        noise[space]={k:{'S':s[k],'Q':q[k],'R':r[k],'resolution_floor':1e-12,'eta':s[k]['value']+q[k]['value']+max(r[k],1e-12)}for k in den}
        qentry=states[space+'_quadrature_plus4'];q20=qentry['other_force_study']['physical_symmetric_order20']['coefficient']
        extra_quadrature[space]={'baseline16_to_Qstate20':{'drag':abs(force[0]-q20[0])/den['drag'],'lift':abs(force[1]-q20[1])},'Qstate16_to_Qstate20':{'drag':abs(qentry['force_order16'][0]-q20[0])/den['drag'],'lift':abs(qentry['force_order16'][1]-q20[1])},'not_substituted_for_canonical_Q':True}
    contrasts={'h_earlier':pair('p4_g4_h1_baseline','p4_g4_h2_baseline'),'h_finest':pair('p4_g4_h2_baseline','p4_g4_h3_baseline'),'p_same_geometry':pair('p4_g4_h3_baseline','p3_g4_h3_baseline'),'p_combined':pair('p4_g4_h3_baseline','p3_g3_h3_baseline'),'geometry_at_P3':pair('p3_g3_h3_baseline','p3_g4_h3_baseline')}
    intervals={}
    for key,left,right in [('h_earlier','p4_g4_h1','p4_g4_h2'),('h_finest','p4_g4_h2','p4_g4_h3'),('geometry_at_P3','p3_g3_h3','p3_g4_h3')]:
        intervals[key]={}
        for metric in den:
            d=contrasts[key][metric]['value'];n=noise[left][metric]['eta']+noise[right][metric]['eta'];intervals[key][metric]={'difference':d,'pair_noise':n,'interval':[max(0.,d-n),d+n],'resolved':d>n}
    uncertainty={}
    for metric in den:
        dp=max(contrasts['p_same_geometry'][metric]['value'],contrasts['p_combined'][metric]['value']);f=noise['p4_g4_h3'][metric]
        uncertainty[metric]={'p_envelope':dp,'last_h':contrasts['h_finest'][metric]['value'],'delta':2*max(contrasts['h_finest'][metric]['value'],dp)+f['S']['value']+f['Q']['value'],'resolved_decreasing_h':intervals['h_finest'][metric]['resolved']and intervals['h_earlier'][metric]['interval'][0]>intervals['h_finest'][metric]['interval'][1],'rigorous_bound_claim':False}
    named={'points_dimensionless':points/.1,'weights_dimensionless':w/.1**2};rule_hash=hashlib.sha256()
    for name,value in sorted(named.items()):
        value=np.ascontiguousarray(value);rule_hash.update(name.encode());rule_hash.update(str((value.shape,str(value.dtype))).encode());rule_hash.update(value.tobytes())
    mapping={'velocity':'velocity_relative','pressure':'pressure_scaled','drag':'drag_relative','lift':'transverse_absolute'}
    component_evidence={'readout_json':sha(a.common/'readout.json'),'rule_npz':sha(a.common/'common-rule.npz'),'assembler_source':sha(Path(__file__)),'population_plan':sha(a.plan),'original_rule':sha(original),'R_record':sha(a.re20_noise if report['Re']==20 else a.fresh_replay/'readout.json')}
    for key,state in [('truth','p4_g4_h3_baseline'),('h_fine','p4_g4_h2_baseline'),('p_same_geometry','p3_g4_h3_baseline'),('p_combined','p3_g3_h3_baseline'),('solver','p4_g4_h3_solver_100x'),('quadrature','p4_g4_h3_quadrature_plus4')]:component_evidence[key+'_field_npz']=states[state]['field_sha256']
    consumer_metrics={}
    for key,name in mapping.items():
        value=uncertainty[key];f=noise['p4_g4_h3'][key]
        consumer_metrics[name]={'h_fine':value['last_h'],'p':value['p_envelope'],'solver':f['S']['value'],'quadrature':f['Q']['value'],'delta':value['delta'],'raw_norms':{'h_fine':contrasts['h_finest'][key],'p_same_geometry':contrasts['p_same_geometry'][key],'p_combined':contrasts['p_combined'][key],'solver':f['S'],'quadrature':f['Q']},'p_definition':'max(p_same_geometry,p_combined)','resolved_h':{'earlier':intervals['h_earlier'][key],'finest':intervals['h_finest'][key],'decreasing':value['resolved_decreasing_h']},'evidence_sha256':component_evidence}
    out={'schema':'FD1-direct-parameter-noise-ledger-v1','Re':report['Re'],'split':report['split'],'admission':False,'common_readout_sha256':sha(a.common/'readout.json'),'rule_content_sha256':{'weights':array_digest(w),'points':array_digest(points)},'adopted_P4_denominators':den,'noise_by_baseline_state':noise,'contrasts':contrasts,'h_intervals':intervals,'conditional_uncertainty':uncertainty,'additional_force_quadrature':extra_quadrature,'geometry_checked_at_P3':True,'P4_g5_interaction_measured':False,'force_convention':'physical symmetric stress order16; no mixed-rule substitution','original_G2_domain_gate':'failed','metrics':consumer_metrics,'volume_rule_sha256':rule_hash.hexdigest(),'volume_rule_hash_algorithm':'sha256_sorted_named_c_arrays_shape_dtype_v1','physical_rule_identity':{'points_sha256':array_digest(points),'weights_sha256':array_digest(w),'hash_algorithm':'sha256_json_dtype_shape_and_contiguous_bytes_v1','original_container_sha256':report['complete_common_rule']['sha256']}}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':main()
