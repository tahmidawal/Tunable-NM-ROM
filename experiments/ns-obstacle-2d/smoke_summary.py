"""G1 controls only. Never grants G2 reference/data admission."""
import argparse,json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('root');args=ap.parse_args()
root=Path(args.root)
campaign=json.loads(Path(__file__).with_name('campaign-A2.json').read_text());g=campaign['reference_gates']
checks=[]
for kind in ['channel','manufactured','dfg']:
 result=json.loads((root/kind/'result.json').read_text());m=result['metrics'];geo=result['geometry']
 for name,actual,limit in [('independent_residual',m['independent_residual']['value'],g['free_dof_residual']),
                          ('geometry_volume',geo['volume_error']['value'],g['geometry_relative'])]:
  checks.append(dict(case=kind,name=name,value=actual,threshold=limit,passed=actual<=limit))
 if kind=='channel':
  for name,limit in [('velocity_error',g['analytic_velocity_relative']),('pressure_error',g['analytic_pressure_scaled']),('boundary_trace',g['essential_trace_over_U'])]:
   checks.append(dict(case=kind,name=name,value=m[name]['value'],threshold=limit,passed=m[name]['value']<=limit))
  shear=m['wall_shear_sign_control']['axial_error']['value']
  checks.append(dict(case=kind,name='wall_force_sign_and_magnitude',value=shear,threshold=1e-6,passed=shear<=1e-6))
 if kind=='manufactured':
  # Source derivatives checked inside solver. Coarse MMS solve is a sign/assembly
  # smoke, not a final asymptotic accuracy or force certificate.
  for name,limit in [('velocity_error',.02),('pressure_error',.02)]:
   checks.append(dict(case=kind,name='coarse_'+name,value=m[name]['value'],threshold=limit,passed=m[name]['value']<=limit))
  force=m['forces']['reaction_analytic_error']['value']
  checks.append(dict(case=kind,name='coarse_known_curved_force',value=force,threshold=.005,passed=force<=.005))
summary={'gate':'G1_smoke','checks':checks,'passed':all(c['passed'] for c in checks),
         'G2_steady_reference_admitted':False,'training_released':False,
         'reason':'mesh/p/tolerance/force/domain/outlet and result reviews remain required'}
(root/'smoke_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
raise SystemExit(0 if summary['passed'] else 2)
