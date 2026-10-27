"""Generate a NONEXECUTABLE G2 matrix/projection, optionally from collected G1 counts."""
from pathlib import Path
import argparse
import hashlib
import json
import math

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--baseline-result');args=ap.parse_args()
    campaign=json.loads((ROOT/'g2/campaign-A6.json').read_text());base=json.loads((ROOT/'configs/dfg-smoke.json').read_text())
    campaign_sha256=hashlib.sha256((ROOT/'g2/campaign-A6.json').read_bytes()).hexdigest()
    case=campaign['case2d'];rows=[]
    def add(name,kind='dfg',h_factor=1.,p=3,g=3,**extra):
        rows.append({'case':name,'kind':kind,'Re':20.,'relative_h_factor':h_factor,'velocity_order':p,'pressure_order':p-1,
                     'geometry_order':g,'assembly_bonus_order':base['assembly_bonus_order'],'diagnostic_order':base['diagnostic_order'],
                     'nonlinear_relative_tolerance':base['nonlinear_relative_tolerance'],'linear_relative_tolerance':base['linear_relative_tolerance'],
                     'xmin':case['xmin'],'xmax':case['xmax'],'outlet':'gradient','exact_maxh':None,'exact_obstacle_h':None,
                     'reference_export_controls':True,'release':'G2_steady_control_candidate','export_common_stokes_lift':False,
                     'campaign_file':'g2/campaign-A6.json','campaign_sha256':campaign_sha256,**extra})
    add('empty_channel_control',kind='channel')
    for level,factor in enumerate((1.,.5,.25)):
        add(f'manufactured_h{level}',kind='manufactured',h_factor=factor)
        add(f'dfg_h{level}',h_factor=factor,export_common_stokes_lift=level==2)
    add('dfg_p4_g4',h_factor=.25,p=4,g=4,role='independent p cross-check; geometry difference resolved by separate p3/g4 case')
    add('dfg_p3_g4',h_factor=.25,g=4,role='geometry order only, fixed solution order')
    add('dfg_solver_100x',h_factor=.25,nonlinear_relative_tolerance=base['nonlinear_relative_tolerance']/100,
        linear_relative_tolerance=base['linear_relative_tolerance']/100)
    add('dfg_quadrature_plus4',h_factor=.25,assembly_bonus_order=base['assembly_bonus_order']+4,diagnostic_order=base['diagnostic_order']+4)
    add('upstream_extension',h_factor=.25,xmin=case['upstream_extension_xmin'],comparison='common-domain velocity and force; common-pole pressure difference, no shifted pressure L2')
    add('downstream_extension',h_factor=.25,xmax=case['downstream_extension_xmax'],comparison='common-domain velocity and force; common-pole pressure difference, no shifted pressure L2')
    add('symmetric_outlet_long_domain',h_factor=.25,xmax=case['downstream_extension_xmax'],outlet='symmetric',problem='separate symmetric-traction-zero boundary model; never DFG')
    baseline=None
    if args.baseline_result:
        path=Path(args.baseline_result).resolve();data=json.loads(path.read_text())
        if data['configuration']['kind']!='dfg' or data['configuration']['release']!='G1_smoke_only_not_reference_admission':
            raise ValueError('explicit collected G1 DFG result required for projections')
        import zipfile
        with zipfile.ZipFile(path.parent/'full_rule.npz') as z:
            # Read only NPY array headers to determine shape; never load field values.
            import numpy.lib.format as nf
            counts={}
            for name in ('points','weights'):
                with z.open(name+'.npy') as f:
                    version=nf.read_magic(f)
                    shape,fortran,dtype=(nf.read_array_header_1_0(f) if version==(1,0) else nf.read_array_header_2_0(f))
                    counts[name]=shape[0]
        memory=json.loads((path.parent/'memory_checks.json').read_text())
        baseline={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'elements':data['geometry']['element_count'],
                  'dofs':data['geometry']['ndof'],'volume_points':counts['points'],'nnz':max(m['nnz'] for m in memory),
                  'peak_rss_bytes':data['peak_rss_bytes'],'wall_seconds_reference_pipeline':data['elapsed_seconds']}
    limits={'elements':base['max_elements'],'dofs':base['max_dofs'],'nnz':base['max_nnz'],'volume_points':100000,
            'boundary_points':30000,'factor_estimate_plus_coarse_peak_bytes':base['memory_stop_gib']*2**30}
    for row in rows:
        domain_factor=(row['xmax']-row['xmin'])/(base['xmax']-base['xmin'])
        h_scale=domain_factor/row['relative_h_factor']**2
        p=row['velocity_order'];q=p-1
        dof_p=(p*p+.5*q*q)/(3*3+.5*2*2)
        local=lambda r:(r+1)*(r+2)/2
        nnz_p=((2*local(p)+local(q))/(2*local(3)+local(2)))**2
        qo=row['diagnostic_order'];rule_factor=(math.ceil((qo+2)/2)/math.ceil((base['diagnostic_order']+2)/2))**2
        row['projection_multipliers']={'elements':h_scale,'dofs_asymptotic_triangles':h_scale*dof_p,
                                       'nnz_local_stencil_estimate':h_scale*nnz_p,'volume_points_rule_estimate':h_scale*rule_factor}
        row['projection']=None
        if baseline:
            ne=math.ceil(baseline['elements']*h_scale);nd=math.ceil(baseline['dofs']*h_scale*dof_p)
            nnz=math.ceil(baseline['nnz']*h_scale*nnz_p);nq=math.ceil(baseline['volume_points']*h_scale*rule_factor)
            factor=nnz*16*100+nd*8*32
            predicted={'elements':ne,'dofs':nd,'nnz':nnz,'volume_points':nq,
                       'factor_estimate_plus_coarse_peak_bytes':factor+baseline['peak_rss_bytes'],
                       'full_volume_arrays_bytes_only':nq*120+math.ceil(nd*(2+9/11))*8,
                       'work_model':'2D h^-2 only; no runtime promise; curved local refinement and native fill can differ'}
            predicted['cap_violations']=[key for key,cap in limits.items() if key in predicted and predicted[key]>cap]
            row['projection']=predicted
    matrix={'status':'DRAFT_UNFROZEN_NOT_EXECUTABLE_NO_G2_RELEASE','campaign_sha256':campaign_sha256,
        'adapter_contract_commit':'97847a479','baseline_G1_counts':baseline,'case_count':len(rows),'cases':rows,
        'existing_caps_unchanged':limits,'resource_grouping_proposal':{'maximum_initial_allocations':4,'seconds_each':1200,'gpus_each':1,'gpu_hours_max':4*1200/3600,
            'status':'proposal only; fit grouping to measured G1/G2 counts and times before release; no automatic submissions'},
        'reference_gates_from_campaign':campaign['reference_gates'],
        'additional_design_gates':{'tightening_u_relative':1e-5,'tightening_p_scaled':2e-5,'tightening_force_coefficient_absolute':1e-5,
            'domain_u_relative':2e-3,'domain_drag_relative':2e-3,'domain_transverse_absolute':2e-4},
        'immutable_training_family':'not released; each future training/development/validation reference must pass common gates; no inheritance of Re20 certification',
        'before_freeze':['G1 completes and saved artifacts reviewed','producer/native reload and polynomial trace code review','measured counts versus existing point/DOF/memory caps',
            'review exact common physical-volume comparison rule and validity policy','freeze exact h values, source/config hashes, allocation grouping and budget','explicit coordinator release'],
        'comparison_policy':'same-domain velocity reference-relative and pressure U²sqrtV; domain pressure uses common-pole differences; no nearest nodes or nudges',
        'timing_policy':'reference Python per-point export, diagnostics, reload and hashing are offline setup; no speed claim from these timers; future tuned FOM/POD/bank queries need same-job repetitions and identical outputs'}
    (ROOT/'g2/CASE-MATRIX.DRAFT.json').write_text(json.dumps(matrix,indent=2)+'\n')
    table=['# G2 case matrix and resource projection (draft)','','Generated from `plan.py`; exact sizes and execution are not released. No G2 numerical result exists.','',
           '| Case | h/h0 | P velocity / pressure | Geometry | Element multiplier | DOF multiplier | Full-rule multiplier |',
           '|---|---:|---|---:|---:|---:|---:|']
    for r in rows:
        m=r['projection_multipliers'];table.append(f"| {r['case']} | {r['relative_h_factor']:g} | {r['velocity_order']}/{r['pressure_order']} | {r['geometry_order']} | {m['elements']:.4g} | {m['dofs_asymptotic_triangles']:.4g} | {m['volume_points_rule_estimate']:.4g} |")
    table+=['','Absolute DOFs, points and memory are unresolved until explicit collected G1 counts are provided. Multipliers are mesh-scaling estimates, not measured counts or memory guarantees. The existing adapter caps remain unchanged. A cap violation requires a reviewed resource-only amendment or separately qualified rule, never silent coarsening or quadrature weakening.','',
            '## Glossary','','h/h0: proposed mesh-size ratio to the eventual frozen coarse level. P: finite-element polynomial degree. Geometry: curved-mapping polynomial order. DOF: one solver coefficient. Full rule: all positive mapped element quadrature points. Multiplier: estimated work/storage growth relative to the coarse case. Draft: no numerical execution is authorized.']
    (ROOT/'g2/CASE-MATRIX.DRAFT.md').write_text('\n'.join(table)+'\n')

if __name__=='__main__':main()
