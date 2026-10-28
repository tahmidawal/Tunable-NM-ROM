"""Always preserve G1 outcome, including missing/failed cases. Never grants G2."""
import argparse,json
from pathlib import Path

ap=argparse.ArgumentParser();ap.add_argument('root');ap.add_argument('--exit-code',type=int,default=0)
ap.add_argument('--failed-stage',default='completed');args=ap.parse_args()
root=Path(args.root);root.mkdir(parents=True,exist_ok=True)
campaign=json.loads(Path(__file__).with_name('campaign-A2.json').read_text());g=campaign['reference_gates']
checks=[];case_states={};nongating={}
for kind in ['channel','manufactured','dfg']:
 path=root/kind/'result.json'
 if not path.exists():
  fail=root/kind/'FAILED.json'
  case_states[kind]={'state':'failed' if fail.exists() else 'not_run',
                     'failure':json.loads(fail.read_text()) if fail.exists() else None}
  checks.append(dict(case=kind,name='completed',passed=False))
  continue
 result=json.loads(path.read_text());m=result['metrics'];geo=result['geometry']
 case_states[kind]={'state':'completed_control_not_reference'}
 for name,actual,limit in [('independent_residual',m['independent_residual']['value'],g['free_dof_residual']),
                          ('geometry_volume',geo['volume_error']['value'],g['geometry_relative'])]:
  checks.append(dict(case=kind,name=name,value=actual,threshold=limit,passed=actual<=limit))
 if kind=='channel':
  for name,limit in [('velocity_error',g['analytic_velocity_relative']),('pressure_error',g['analytic_pressure_scaled']),('boundary_trace',g['essential_trace_over_U'])]:
   checks.append(dict(case=kind,name=name,value=m[name]['value'],threshold=limit,passed=m[name]['value']<=limit))
  shear=m['wall_shear_sign_control']['axial_error']['value']
  checks.append(dict(case=kind,name='wall_force_sign_and_magnitude',value=shear,threshold=1e-6,passed=shear<=1e-6))
 if kind=='manufactured':
  for name,limit in [('velocity_error',.02),('pressure_error',.02)]:
   checks.append(dict(case=kind,name='coarse_'+name,value=m[name]['value'],threshold=limit,passed=m[name]['value']<=limit))
  force=m['forces']['reaction_analytic_error']['value']
  checks.append(dict(case=kind,name='coarse_known_curved_force',value=force,threshold=.005,passed=force<=.005))
 if kind=='dfg':
  b=json.loads(Path(__file__).with_name('benchmark.json').read_text());f=m['forces']
  quantities={'reaction':f['reaction_coefficient']}
  for key,val in f.items():
   if key.startswith('gradient_order') or key.startswith('physical_symmetric_order'):
    quantities[key]=val['coefficient']
  rows={}
  for key,coeff in quantities.items():
   rows[key]={'coefficient':coeff,'drag_reference_abs_gap':abs(coeff[0]-b['drag_coefficient']),
              'lift_reference_abs_gap':abs(coeff[1]-b['lift_coefficient']),
              'reaction_drag_abs_gap':abs(coeff[0]-f['reaction_coefficient'][0]),
              'reaction_lift_abs_gap':abs(coeff[1]-f['reaction_coefficient'][1])}
  pd=f['pressure_probes'];dp=pd['difference']
  nongating['dfg']={'status':'coarse_smoke_diagnostic_no_reference_admission',
    'literature_source':b['source'],'force_rows':rows,'pressure_probes':pd,
    'pressure_drop_reference_abs_gap':abs(dp-b['pressure_drop']) if dp is not None else None,
    'quadrature_orders':[result['configuration']['diagnostic_order'],result['configuration']['diagnostic_order']+4],
    'G2_pressure_probe_blocked':not pd['valid']}
preflight_path=root/'preflight.json'
preflight=json.loads(preflight_path.read_text()) if preflight_path.exists() else {'passed':False,'state':'not_completed'}
stages_path=root/'stage_exit_codes.jsonl'
stages=[json.loads(line) for line in stages_path.read_text().splitlines()] if stages_path.exists() else []
summary={'gate':'G1_smoke','checks':checks,'cases':case_states,'preflight':preflight,
         'run_exit_code':args.exit_code,'last_stage':args.failed_stage,'stage_exit_codes':stages,
         'passed':args.exit_code==0 and preflight.get('passed',False) and all(c['passed'] for c in checks),
         'nongating':nongating,'G2_steady_reference_admitted':False,'training_released':False,
         'reason':'mesh/p/tolerance/force/domain/outlet and result reviews remain required'}
(root/'smoke_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'gate':summary['gate'],'passed':summary['passed'],'run_exit_code':args.exit_code,'case_states':case_states},indent=2))
raise SystemExit(0 if summary['passed'] else 2)
