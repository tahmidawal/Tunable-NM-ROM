"""Read-only FD1 qualification candidate assembly. Never writes an admission.

The fixed reviewed check contract owns requiredness; raw data own values. Missing
controls remain missing. Shared controls are separately audited evidence records.
"""
from pathlib import Path
import argparse,hashlib,json,math,sys
import numpy as np
B=Path(__file__).resolve().parents[1];sys.path.insert(0,str(B))
from reference_primary import verify_runtime_inventory
from reference_scope import FD1_GATE,FD1_SCOPE,FD1_CONTRACT_SHA256,validate_geometry
SPACES=('p4_g4_h1','p4_g4_h2','p4_g4_h3','p3_g3_h3','p3_g4_h3')
VARIANTS=('baseline','solver_100x','quadrature_plus4')
def sha(p):
 with p.open('rb')as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
def confined(name):
 p=(B/name).resolve()
 if not p.is_relative_to(B):raise ValueError('evidence path outside reference lane')
 return p
def member(folder,export,name):
 p=folder/name
 if sha(p)!=export['files'][name]:raise ValueError('export member changed:'+str(p))
 return read(p)if p.suffix=='.json'else p

def checked(contract,value,evidence):
 out={k:contract[k]for k in ('name','role','threshold')};out.update(value=value,evidence_sha256=evidence)
 if value is None:out['status']='missing';return out
 op=contract['operator'];limit=contract['threshold']
 if op=='true':valid=type(value)is bool and value is True
 elif op=='eq':valid=type(value)is type(limit)and value==limit
 else:
  if type(value)not in (int,float)or not math.isfinite(value):raise ValueError('nonfinite/nonnumeric gate value')
  valid={'le':lambda:value<=limit,'lt':lambda:value<limit,'ge':lambda:value>=limit,'gt':lambda:value>limit}[op]()
 out['status']='pass'if valid else'fail';return out

def assemble_row(group,artifact,noise_path,contract,retention_index,raw_index):
 noise=read(noise_path);readout=read(artifact/'results/common/readout.json')
 if noise['Re']!=group['Re']or noise['split']!=group['split']or noise['common_readout_sha256']!=sha(artifact/'results/common/readout.json'):raise ValueError('noise/readout parameter provenance differs')
 cases={};states=[];evidence={'noise_ledger':sha(noise_path),'common_readout':sha(artifact/'results/common/readout.json')}
 for entry in group['states']:
  key=entry['id'];folder=confined(entry['source_directory'])if entry['retained']else artifact/'results/states'/key
  export_path=folder/'EXPORT-MANIFEST.json';export=read(export_path)
  if sha(export_path)!=readout['states'][key]['source_export_sha256']:raise ValueError('readout used different state export')
  data={name:member(folder,export,name)for name in ('config.json','result.json','reload_checks.json','mesh_identity.json','fe_layout.json','jacobian_checks.json','newton_history.json','normal_orientation.json')}
  for name in ('solution.sol','mesh.vol.bin'):member(folder,export,name)
  if 'PRIMARY-REFERENCE.json'in export['files']:verify_runtime_inventory(folder.resolve(),export)
  else:
   for name,digest in export['files'].items():
    if sha(folder/name)!=digest:raise ValueError('legacy export closure changed')
  if data['result.json']['configuration']!=data['config.json']:raise ValueError('result/config mismatch')
  case={'folder':folder,'export':export,'data':data,'retained':entry['retained']};cases[key]=case
  role='adopted_truth'if key=='p4_g4_h3_baseline'else('p_geometry_diagnostic'if entry['space'].startswith('p3')else'refinement_or_noise_diagnostic')
  states.append({'space':entry['space'],'variant':entry['variant'],'role':role,'directory':str(folder.relative_to(B)),'export_manifest_sha256':sha(export_path),'result_sha256':export['files']['result.json'],'config_sha256':export['files']['config.json']})
  evidence[key+'_result']=export['files']['result.json'];raw_index[str(export_path.relative_to(B))]=sha(export_path)
  owner=next((p for p in (folder,*folder.parents)if p.is_relative_to(B)and(p/'RETENTION.json').is_file()),None)
  if owner is None:raise ValueError('missing retained archive owner')
  retention=read(owner/'RETENTION.json')
  if not retention['status'].startswith('complete_all_regular_files_verified')or not retention.get('archive_sha256'):raise ValueError('retention incomplete')
  retention_index[str(owner.relative_to(B))]={'retention_sha256':sha(owner/'RETENTION.json'),'archive_sha256':retention['archive_sha256'],'verified_regular_files':retention['verified_regular_files']}
 fine=cases['p4_g4_h3_baseline'];f=fine['data'];result=f['result.json'];cfg=f['config.json'];m=result['metrics'];geo=result['geometry'];forces=m['forces'];physical=forces['physical_symmetric_order16']['coefficient'];gradient=forces['gradient_order16']['coefficient'];reaction=forces['reaction_coefficient']
 if not all(math.isfinite(x)for v in (physical,gradient,reaction)for x in v)or gradient[0]==0:raise ValueError('force values invalid')
 values={'exact_population_state_coverage':{(s['space'],s['variant'])for s in states}=={(s,v)for s in SPACES for v in VARIANTS},'source_and_retention_closure':True}
 # The complete rule and actual fields are independently validated by population_noise.
 values['complete_common_rule_no_dropped_points']=noise['physical_rule_identity']['original_container_sha256']==readout['complete_common_rule']['sha256']and len(readout['states'])==15
 fixed=True;algebra=True;diagnostics={}
 for key,case in cases.items():
  d=case['data'];c=d['config.json'];r=d['result.json'];m2=r['metrics'];scope_cfg={**c,'scope_contract_sha256':FD1_CONTRACT_SHA256}if case['retained']else c
  try:validate_geometry(scope_cfg)
  except ValueError:fixed=False
  fixed &= c['precision']=='float64'and math.isclose(c['U']*c['D']/c['nu'],group['Re'],rel_tol=1e-14,abs_tol=0.)
  checks=d['jacobian_checks.json'];useful=[x['agreement']['value']for x in checks if x['signal_norm']>x['signal_floor']];jac=bool(useful)and min(useful)<=1e-6 and len(useful)>=2 and(max(useful)<=1e-8 or any(b<=a for a,b in zip(useful,useful[1:])))
  algebra &= r['status']=='completed_control_not_reference_admission'and d['reload_checks.json']['passed']is True and d['normal_orientation.json']['passed']is True and m2['independent_residual']['value']<=1e-9 and jac
  diagnostics[key]={'divergence_scaled':m2['divergence']['value'],'mass_relative':m2['mass']['value'],'physical_gradient_transverse_absolute':abs(m2['forces']['physical_symmetric_order16']['coefficient'][1]-m2['forces']['gradient_order16']['coefficient'][1]),'independent_free_residual':m2['independent_residual']['value'],'native_jacobian_gate_pass':jac}
 values.update(fixed_FD1_geometry_and_units=bool(fixed),all_states_native_algebra_and_reload_valid=bool(algebra),direct_five_state_S_Q_R_coverage=set(noise['noise_by_baseline_state'])==set(SPACES),geometry_volume_relative=geo['volume_error']['value'],geometry_perimeter_relative=geo['obstacle_perimeter_error']['value'],independent_free_residual=m['independent_residual']['value'],mass_relative=m['mass']['value'],divergence_scaled=m['divergence']['value'],essential_trace_over_U=m['prescribed_FE_essential_trace']['upper_bound_over_U'],physical_energy_balance=m['energy']['corrected']['value'],physical_momentum_balance_max=max(x['corrected']['value']for x in m['momentum']),physical_gradient_drag_relative=abs(physical[0]-gradient[0])/abs(gradient[0]),physical_gradient_transverse_absolute=abs(physical[1]-gradient[1]),reaction_gradient_drag_relative=abs(reaction[0]-gradient[0])/abs(gradient[0]),reaction_gradient_transverse_absolute=abs(reaction[1]-gradient[1]),positive_mapped_jacobian=geo['minimum_sampled_jacobian_determinant'],no_obstacle_quadrature=geo['analytic_obstacle_quadrature']['no_obstacle_quadrature']and geo['analytic_obstacle_quadrature']['actually_removed_count']==0,correct_boundary_labels_and_normal_orientation=set(geo['boundary_measures'])=={'inlet','outlet','obstacle','wall_bottom','wall_top'}and result['normal_orientation']['passed'])
 divs=[cases[s+'_baseline']['data']['result.json']['metrics']['divergence']['value']for s in SPACES[:3]];values['divergence_decreases_under_h_refinement']=divs[0]>divs[1]>divs[2]
 metric_names={'velocity':'velocity_relative','pressure':'pressure_scaled','drag':'drag_relative','lift':'transverse_absolute'}
 for raw,name in metric_names.items():
  v=noise['metrics'][name];values['h_finest_'+name]=v['h_fine'];values['resolved_decreasing_h_'+name]=v['resolved_h']['decreasing'];values['conditional_delta_'+name]=v['delta']
  budget=next(x['threshold']for x in contract['per_parameter']if x['name']=='conditional_delta_'+name);assessments=[]
  for contrast,left,right in [('p_same_geometry','p4_g4_h3','p3_g4_h3'),('p_combined','p4_g4_h3','p3_g3_h3'),('geometry_at_P3','p3_g3_h3','p3_g4_h3')]:
   difference=noise['contrasts'][contrast][raw]['value'];floor=noise['noise_by_baseline_state'][left][raw]['eta']+noise['noise_by_baseline_state'][right][raw]['eta'];assessments.append(difference>floor or difference+floor<=budget)
  values['p_geometry_evidence_'+name]=bool(all(assessments)and v['p']==max(v['raw_norms']['p_same_geometry']['value'],v['raw_norms']['p_combined']['value']))
  if raw in ('velocity','pressure'):values['solver_change_'+name]=v['solver']
 values['solver_change_drag_absolute']=noise['metrics']['drag_relative']['raw_norms']['solver']['numerator'];values['solver_change_transverse_absolute']=noise['metrics']['transverse_absolute']['raw_norms']['solver']['numerator']
 values['higher_surface_quadrature_retained']=all(k in forces for k in ('physical_symmetric_order12','physical_symmetric_order16'))and all('Qstate16_to_Qstate20'in v for v in noise['additional_force_quadrature'].values())
 boundary=member(fine['folder'],fine['export'],'boundary_rule.json');outlet=[r for r in boundary if r['label']=='outlet'];flux=np.array([np.dot(r['velocity'],r['normal'])for r in outlet]);weights=np.array([r['weight']for r in outlet]);backflow={'signed_flux':float(weights@flux),'negative_flux_magnitude':float(weights@np.maximum(-flux,0.)),'minimum_normal_velocity':float(flux.min()),'sample_count':len(outlet),'stress_convention':cfg['outlet'],'no_added_cutoff':True}
 values['outlet_flux_backflow_and_stress_audited']=bool(len(outlet)>0 and np.isfinite(flux).all()and np.isfinite(weights).all()and np.all(weights>0)and cfg['outlet']=='gradient')
 benchmark=read(B/'benchmark.json');probe=forces['pressure_probes']
 if group['Re']==20:
  if not probe['valid']or not probe['G2_pressure_gate_eligible']:raise ValueError('ineligible exact pressure probes')
  values.update(dfg_drag_relative=abs(physical[0]-benchmark['drag_coefficient'])/abs(benchmark['drag_coefficient']),dfg_lift_absolute=abs(physical[1]-benchmark['lift_coefficient']),dfg_pressure_drop_relative=abs(probe['difference']-benchmark['pressure_drop'])/abs(benchmark['pressure_drop']))
 values.update(refinement_and_P3_physical_diagnostics_retained=len(diagnostics)==15,P3_fixed_order_geometry_diagnostic_retained='geometry_at_P3'in noise['contrasts'],P4_g5_interaction_unmeasured_disclosed=noise['P4_g5_interaction_measured']is False,original_domain_sensitivity_pass=False)
 evidence_path=noise_path.with_name(group['id']+'-CHECK-EVIDENCE.json');evidence_path.write_text(json.dumps({'constituent_sha256':evidence,'source_paths':{str(case['folder'].relative_to(B)):case['export']['files']for case in cases.values()},'noise_ledger':str(noise_path.relative_to(B))},indent=2)+'\n');raw_index[str(evidence_path.relative_to(B))]=sha(evidence_path)
 checks=[]
 for rule in contract['per_parameter']:
  item=checked(rule,values.get(rule['name']),sha(evidence_path))
  if rule['applies_to_re']!='all'and group['Re']not in rule['applies_to_re']:item.update(status='not_applicable',value=None)
  checks.append(item)
 export=fine['export'];folder=fine['folder'];adopted={'directory':str(folder.relative_to(B)),'export_manifest_sha256':sha(folder/'EXPORT-MANIFEST.json'),'config_sha256':export['files']['config.json'],'mesh_sha256':export['files']['mesh.vol.bin'],'layout_sha256':f['fe_layout.json']['sha256'],'layout_file_sha256':export['files']['fe_layout.json'],'solution_sha256':export['files']['solution.sol'],'storage_mode':'coefficient_primary'if'PRIMARY-REFERENCE.json'in export['files']else'expanded_full_fields'}
 return {'Re':group['Re'],'split':group['split'],'adopted_reference':adopted,'states':states,'checks':checks,'noise_ledger':{'path':str(noise_path.relative_to(B)),'sha256':sha(noise_path)},'conditional_uncertainty':noise['metrics'],'missing_or_unresolved':[x['name']for x in checks if x['status']in('missing','unresolved')],'diagnostics':{'all_states':diagnostics,'divergence_h_sequence':divs,'outlet_backflow':backflow,'force_orders':forces,'conditional_noise':noise}}

def main():
 p=argparse.ArgumentParser();p.add_argument('--index',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out=a.out.resolve();index=read(a.index);plan=read(B/'g2/population01/PLAN.json');contract=read(B/'g2/FD1-CHECK-CONTRACT.json');retention_index={};raw_index={};rows=[]
 for group in plan['groups']:
  entry=index['groups'].get(group['id'])
  if entry is None:continue
  rows.append(assemble_row(group,confined(entry['artifact_directory']),confined(entry['noise_ledger']),contract,retention_index,raw_index))
 a.out.parent.mkdir(parents=True,exist_ok=True)
 shared=[]
 for rule in contract['shared']:
  item=index.get('shared_controls',{}).get(rule['name']);evidence={}
  if item:
   for path,digest in item['evidence_sha256'].items():
    if sha(confined(path))!=digest:raise ValueError('shared control evidence changed')
    evidence[path]=digest;raw_index[path]=digest
  evidence_path=a.out.with_name('SHARED-'+rule['name']+'-EVIDENCE.json');evidence_path.write_text(json.dumps(evidence,indent=2)+'\n');raw_index[str(evidence_path.relative_to(B))]=sha(evidence_path)
  shared.append(checked(rule,None if item is None else item['value'],sha(evidence_path)))
 a.out.parent.mkdir(parents=True,exist_ok=True);retention_path=a.out.with_name('RETENTION-INDEX.json');raw_path=a.out.with_name('RAW-EVIDENCE-INDEX.json')
 for path,data in ((retention_path,retention_index),(raw_path,raw_index)):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
 out={'schema':'ns-obstacle-FD1-reference-qualification-candidate-v1','qualification_gate':FD1_GATE,'qualification_scope':FD1_SCOPE,'admission':False,'training_released':False,'original_domain_gate':'failed','population':{split:[g['Re']for g in plan['groups']if g['split']==split]for split in ('train','development','validation')},'rows':rows,'missing_parameter_groups':[g['id']for g in plan['groups']if g['id']not in index['groups']],'shared_controls':{'checks':shared},'check_contract_sha256':sha(B/'g2/FD1-CHECK-CONTRACT.json'),'provenance':{'assembler_sha256':sha(Path(__file__)),'retention_index_sha256':sha(retention_path),'raw_evidence_index_sha256':sha(raw_path),'population_plan_sha256':sha(B/'g2/population01/PLAN.json'),'input_index_sha256':sha(a.index)}}
 a.out.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
