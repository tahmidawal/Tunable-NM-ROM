"""Read-only saved-result screen. Never writes a training admission certificate.

A plan binds every input by SHA256. Missing cross-run evidence remains missing;
per-case success alone never certifies a reference or the parameter population.
"""
from pathlib import Path
import argparse, hashlib, json, math

REQUIRED_CONTROLS = ('two_decreasing_h_differences', 'p4_g4_crosscheck', 'p3_g4_geometry',
    'solver_100x', 'quadrature_plus4', 'upstream_extension', 'downstream_extension',
    'symmetric_outlet_long_domain', 'conditional_uncertainty', 'shared_mesh_lift_reuse',
    'parameter_population_coverage', 'independent_raw_evidence_review')

UNASSESSED_RAW_CHECKS = ('positive_mapped_Jacobians_and_fluid_points', 'normal_and_binary_reload_certificates',
    'surface_rule_12_vs_16', 'outlet_backflow', 'manufactured_force_and_derivative_controls',
    'wall_shear_sign_control', 'manufactured_global_flux')

def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def assess(plan_path, output):
    plan_path=Path(plan_path).resolve();plan=json.loads(plan_path.read_text());sources=[]
    def read(entry):
        path=Path(entry['path']);path=path if path.is_absolute() else plan_path.parent/path
        actual=digest(path)
        if actual != entry['sha256']:raise ValueError('input hash mismatch: '+str(path))
        sources.append({'path':str(path.resolve()),'sha256':actual})
        return json.loads(path.read_text())
    contract=read(plan['campaign']);gates=contract['reference_gates'];benchmark=read(plan['benchmark'])
    cases=[]
    for entry in plan['cases']:
        result=read(entry);cfg=result['configuration'];m=result['metrics'];geo=result['geometry'];rows=[]
        if cfg['kind'] not in ('dfg','manufactured','channel') or cfg['precision']!='float64':
            raise ValueError('unsupported physical case/precision')
        def check(name,value,threshold,scope='per_case'):
            if not isinstance(value,(int,float)) or not math.isfinite(value) or threshold<0:
                raise ValueError('invalid metric '+name)
            rows.append(dict(check=name,value=value,threshold=threshold,within_threshold=value<=threshold,scope=scope))
        check('fluid_area_relative',geo['volume_error']['value'],gates['geometry_relative'])
        if cfg['kind']!='channel':
            check('obstacle_perimeter_relative',geo['obstacle_perimeter_error']['value'],gates['geometry_relative'])
        check('independent_free_residual',m['independent_residual']['value'],gates['free_dof_residual'])
        check('prescribed_FE_trace_bound_over_U',m['prescribed_FE_essential_trace']['upper_bound_over_U'],gates['essential_trace_over_U'])
        check('scaled_divergence',m['divergence']['value'],gates['divergence_scaled'])
        if cfg['kind']=='dfg':
            check('inlet_outlet_mass',m['mass']['value'],gates['mass_2d'])
        else:
            check('analytic_velocity',m['velocity_error']['value'],gates['analytic_velocity_relative'])
            check('analytic_pressure',m['pressure_error']['value'],gates['analytic_pressure_scaled'])
        check('physical_energy_balance',m['energy']['corrected']['value'],gates['balance_scaled'])
        for component,row in enumerate(m['momentum']):
            check('physical_momentum_'+str(component),row['corrected']['value'],gates['balance_scaled'])
        if cfg['kind']=='dfg':
            anchor = dict(nu=benchmark['nu'], D=benchmark['diameter'], U=benchmark['inlet_umean'],
                          xmin=0., xmax=benchmark['length'], H=benchmark['height'],
                          center=benchmark['center'], radius=benchmark['radius'], outlet='gradient')
            if any(cfg[key] != value for key,value in anchor.items()):
                raise ValueError('this DFG screen requires the canonical Re20 geometry and outlet; other references need separate coverage')
            f=m['forces'];physical=f['physical_symmetric_order16']['coefficient'];gradient=f['gradient_order16']['coefficient'];reaction=f['reaction_coefficient']
            if gradient[0]==0 or benchmark['drag_coefficient']==0 or benchmark['pressure_drop']==0:raise ValueError('zero force/probe denominator')
            check('physical_gradient_drag_relative',abs(physical[0]-gradient[0])/abs(gradient[0]),gates['physical_gradient_drag_gap_relative'])
            check('physical_gradient_lift_absolute',abs(physical[1]-gradient[1]),gates['physical_gradient_transverse_gap_absolute'])
            check('reaction_gradient_drag_relative',abs(reaction[0]-gradient[0])/abs(gradient[0]),gates['h_drag_relative'])
            check('reaction_gradient_lift_absolute',abs(reaction[1]-gradient[1]),gates['h_transverse_absolute_2d'])
            check('physical_DFG_drag_relative',abs(physical[0]-benchmark['drag_coefficient'])/abs(benchmark['drag_coefficient']),gates['dfg_drag_relative'])
            check('physical_DFG_lift_absolute',abs(physical[1]-benchmark['lift_coefficient']),gates['dfg_lift_absolute'])
            probe=f['pressure_probes']
            if not probe['valid'] or not probe['G2_pressure_gate_eligible']:raise ValueError('exact eligible pressure probes required')
            check('DFG_pressure_drop_relative',abs(probe['difference']-benchmark['pressure_drop'])/abs(benchmark['pressure_drop']),gates['dfg_pressure_drop_relative'])
        cases.append({'name':entry['name'],'kind':cfg['kind'],'mesh_maxh':cfg['maxh'],
                      'rows':rows,'all_listed_per_case_checks_pass':all(r['within_threshold'] for r in rows)})
    # This first stage deliberately does not consume caller-supplied pass booleans.
    # Cross-run checks require their raw arrays and derived uncertainty constituents.
    record={'scope':'provisional per-case screen only; cross-run and raw evidence audit required',
            'physical_admission':False,'training_released':False,'cases':cases,
            'required_unassessed_controls':list(REQUIRED_CONTROLS),
            'unassessed_raw_per_case_checks':list(UNASSESSED_RAW_CHECKS),'sources':sources,
            'plan_sha256':digest(plan_path),'generator_sha256':digest(__file__)}
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/'assessment.json').write_text(json.dumps(record,indent=2)+'\n')
    lines=['# Provisional reference qualification screen','',record['scope']+'. No training admission is issued.','',
           '| Case | Check | Saved value | Threshold | Result |','|---|---|---:|---:|---|']
    for case in cases:
        for row in case['rows']:
            lines.append(f"| {case['name']} | {row['check']} | {row['value']:.9g} | {row['threshold']:.9g} | {'within threshold' if row['within_threshold'] else 'fails'} |")
    lines+=['','Cross-run controls not assessed by this screen: '+', '.join(REQUIRED_CONTROLS)+'.','',
            'Raw per-case controls not assessed here: '+', '.join(UNASSESSED_RAW_CHECKS)+'.','',
            '## Glossary','', 'Case: one saved solver invocation. Saved value: the recorded metric or the explicit force/probe difference derived from it. Threshold: the frozen campaign limit. Within threshold: this one check passes; it does not qualify the reference. FE trace bound: bound on error against prescribed finite-element boundary velocity, including a fixed roundoff allowance; it is not a raw observed error. Raw polynomial coefficients and independent check residuals are retained in the producer certificate. Divergence: local incompressibility error. DFG: cylinder-in-channel benchmark. Reaction force: force recovered from the discrete equations; gradient and physical forces use different stress conventions. Cross-run: evidence requiring more than one independently retained solve. Conditional uncertainty: a refinement-based estimate, not a rigorous error bound.']
    (out/'assessment.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();assess(a.plan,a.out)
