"""Refresh only this owner's canonical record from retained completed evidence."""
from pathlib import Path
import datetime,hashlib,json
H=Path(__file__).resolve().parent
sources={}
def read(name):
    p=H/'results'/name/'output/result.json';sources[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())
r=read('b1directaudit');g=read('b1globalfull');n=read('b1child8refit')
stable=read('b1stableglobal');child=read('b1child16');refit=read('b1child16refit');head=read('b1headspan2')
growth=read('b1childgrowth64');wide=read('b1globalwide128')
pilots=[]
for name in ['b1weakpilot1','b1weakL2048','b1weakL4096','b1weakdt125','b1weakdt625','b1cn2048','b1cn2048half','b1cnM8','b1cnQ512','b1cnQ1024']:
    if (H/'results'/name/'output/result.json').exists():
        p=read(name)
        if p['status']=='complete_fixed_tree_diagnostic_pilot':pilots.append((name,p))
ledger=json.loads((H/'cluster/ledger.json').read_text());terminal={'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}
active=[v for v in ledger if v.get('state') not in terminal]
seconds=sum(v.get('elapsed_seconds',0) for v in ledger);stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
status=f'''**New Burgers tree-embedding study: B1 ACTIVE, qualified nonlinear representation advancing to coupled physics.** The same child16 head with64 retained nodes reaches {100*growth['actual4096_witness']['error']:.4f}% on actual4096 output at1152 variables; every development snapshot passes the representation screen. The direct24 tree gives {100*r['dense_witness']['relative_l2']:.4f}% with1376 variables. A prescribed global POD-plus-Fourier space reaches {100*stable['controls']['960']['worst']:.4f}% with960 variables, so a tree advantage is not established. CN improves common PDE accuracy; time halving leaves the direct tree's early-time failure, and independent test enrichment is next. The wider global128 neural repair remains at{100*wide['worst_projection']:.4f}% projection error. No complete-runtime win or final-test result. [Generated evidence and plots](reports/2026-11-18-tree-embedding-representation.md). Old campaign remains PAUSED; no merge.'''
title='### 2026-11-18 — Tree-embedding B1 frozen/global audits and physics: '+', '.join(name for name,_ in pilots)
lines=[]
for name,p in pilots:
    for c in p['cases']:
        for method,v in c['methods'].items():
            line=f"- {name}, {c['case']}, {method}: L{p['L']}, {p.get('time_integrator','be').upper()}, Q{p.get('quadrature_intervals',p['L'])}, dt{p['dt']:g}, worst evolved error {100*max(v['physical_error'][1:]):.6f}%, actual4096 final {100*v['actual4096_final_error']:.6f}%; all steps stationary {v['all_steps_stationary']}, accuracy pass {v['physical_accuracy_pass']}."
            if 'previous_state_relative_change' in v:line+=f" Maximum physical state change from prior setting {100*max(v['previous_state_relative_change'][1:]):.6f}%."
            lines.append(line)
entry=f'''{title}

Published {stamp} from checksum-verified results in the owned lane. These are complete truth-free fixed-tree trajectories on a small exposed pilot, not all-development qualification or timing confirmation. Exact fine-grid upwind remains inside the common scaled weak residual. Physical projection and numerical/dynamics error are reported separately where measured. The first pilot lacks explicit immediate timer burn-in; its times remain unqualified diagnostics. Subsequent runs add burn-in without altering the original record.

{chr(10).join(lines)}

The extended saved global POD1280 space gives worst {100*g['POD']['1280']['projection']['worst']:.6f}% versus direct-tree1376 {100*r['dense_witness']['relative_l2']:.6f}%. The trailing relative training singular value is {g['POD']['1280']['relative_training_singular_value']:.4g}: this is a numerical-null completion, not reliably learned variance. Keep its real saved-space error visible and require stability qualification. The data-independent complete sine48 space gives {100*g['complete_sine']['48']['projection']['worst']:.6f}% with {g['complete_sine']['48']['variables']} variables. No error-per-variable result establishes cost.

Frozen child8 dense250 fitting gives {100*n['worst_projection']:.6f}%, maximum stationarity {n['max_stationarity']:.4g}; weights unchanged. This does not prove a global fitting minimum. A frozen global output-span relaxation is prospectively registered to separate final-layer shape restrictions from nonlinear coding. No additional trained recipe is claimed by that audit.

The completed global-head audit gives frozen affine-space bounds {', '.join(f"latent{v['latent']}: {100*v['panels']['dev']['worst']:.6f}% at rank{v['affine_span_rank']}" for v in head['models'])}. Those bounds cannot be reduced by code optimization on the same frozen head. The gap above the bound does not distinguish nonlinear image restriction from imperfect fitting. Its first job failed on an absent label in legacy metadata; logs and allocated time are retained, and the retry reads the matching span's labels. No scientific result is taken from the failed attempt.

Child16 training ends at its registered stop: selected update{child['selected_update']}, worst{100*child['worst_projection']:.6f}%; all{child['zero_jacobian']['independent_directions']}/{child['zero_jacobian']['child_variables']} representative zero-child directions are independent after parent removal. Frozen250-step Cholesky refitting gives{100*refit['worst_projection']:.6f}% with maximum stationarity{refit['max_stationarity']:.4g}. Only alignment cases miss0.5%. Completed same-head64-node growth now passes all44x11 development snapshots: worst{100*growth['projection']['worst']:.6f}%, actual4096{100*growth['actual4096_witness']['error']:.6f}%. All geometries preserve all1024 independent zero-child directions; maximum zero-extension change{growth['max_zero_child_transfer']:.3g}. This admits the nonlinear representation to coupled PDE diagnostics, not a speed claim. The larger architecture uses1152 variables versus the original128-variable failure. It is an offline projection diagnostic with initial-callable-only node selection, not online truth assistance.

The combined wider global repair removes the narrow frozen feature/output restrictions using the stable960-feature space and hidden widths256,256,960. It stops at the registered limit with selected update{wide['selected_update']} and worst{100*wide['worst_projection']:.6f}%. The selected code fits are stationary, while the training/development gap remains. This unsuccessful seventh B1 recipe is preserved; it does not justify endlessly increasing training or claiming the original global architecture reproduced faithfully.

The prescribed global completion of training POD512 gives960-variable error{100*stable['controls']['960']['worst']:.6f}%, orthogonality defect{stable['orthogonality_defect']:.3g}. This defeats child16's projection error at the same count; all complete-runtime conclusions remain open. The two fixed training-row permutations give pure saved POD1280 errors {', '.join(f"{100*v['controls']['1280']['projection']['worst']:.6f}%" for v in stable['permutations'])}: modest score changes in this limited check do not turn null directions into learned variance. The retained integer-scatter future warning in the completion job affects index dtype, not float64 arithmetic; the source now explicitly uses int32 indices.

Registered common CN time-integration and structured weak-integration diagnostics preserve the fine sign-upwind spacing and discrete diffusion. The local sub-minute fixtures verify independent old/new CN stencils, all five sparse fine-grid samples and dense-equivalent returned states. Composite integration weights are deterministic, not trained on residuals; its real convergence test is still required. Every method in a comparison shares the same operator/integrator/rule.

Ledger snapshot: {seconds} elapsed allocated GPU-seconds ({seconds/3600:.6f}h), including active elapsed time. Outstanding full reservations remain charged by the submission guard. Active owned names: {', '.join(v['name'] for v in active) or 'none'}. Old paused namespace and final-test solution fields remain untouched. Source, selected models and generated report are mirrored; raw evidence has local and designated cluster copies. No merge.
'''
lab=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/LAB-LOG.md');text=lab.read_text()
a=text.index('**New Burgers tree-embedding study:');b=text.index('\n\n',a);text=text[:a]+status+text[b:]
if title not in text:text=text.rstrip()+'\n\n'+entry+'\n'
lab.write_text(text)
record=dict(utc=stamp,own_status=status,own_entry=entry,source_hashes=sources)
tag=hashlib.sha256(title.encode()).hexdigest()[:10]
(H/f'manifests/B1-physics-progress-{tag}.json').write_text(json.dumps(record,indent=2)+'\n')
owner=json.loads((H/'OWNER.json').read_text());owner.update(updated_utc=stamp,status='B1_ACTIVE_FIXED_TREE_NUMERICAL_REFINEMENT',
    allocated_gpu_seconds=seconds,allocated_gpu_hours=seconds/3600,active_jobs=len(active),terminal_jobs=len(ledger)-len(active),
    decision='Direct and nonlinear64-node representations qualified; coupled PDE and independent numerical checks continue',final_test_solution_fields_opened=False)
(H/'OWNER.json').write_text(json.dumps(owner,indent=2)+'\n')
claim=Path('/home/tahmid/Dev/QuadtreeNMROM/execution/tree-embeddings-owner.lock/owner.json')
assert json.loads(claim.read_text())['owner']==owner['owner'];claim.write_text(json.dumps(owner,indent=2)+'\n')
print(status)
