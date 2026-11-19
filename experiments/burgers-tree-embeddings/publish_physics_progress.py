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
m8=read('b1cnM8');q512=read('b1cnQ512')
allocation_text='Matched same-head uniform/prescribed controls continue.'
if (H/'results/b1childallocation/output/result.json').exists():
    ac=read('b1childallocation')
    if ac['status']=='complete_frozen_child_allocation_controls':
        allocation_text=f"At1472 variables the same nonlinear checkpoint gives uniform error{100*ac['controls']['uniform84']['projection']['worst']:.4f}% versus prescribed{100*ac['controls']['prescribed84']['projection']['worst']:.4f}%, with actual4096 worst witnesses. These are best-found, partly underconverged fits; the84-node allocation is not independently admitted to physics."
pilot_worst=max(max(v['physical_error'][1:]) for c in m8['cases'] for v in c['methods'].values())
pilots=[]
for name in ['b1weakpilot1','b1weakL2048','b1weakL4096','b1weakdt125','b1weakdt625','b1cn2048','b1cn2048half','b1cnM8','b1cnM16','b1cnM32','b1cnQ512','b1cnQ1024','b1childpde','b1childpdeR1','b1childpdeR2']:
    if (H/'results'/name/'output/result.json').exists():
        p=read(name)
        if p['status'] in ['complete_fixed_tree_diagnostic_pilot','complete_coupled_child_diagnostic_pilot']:pilots.append((name,p))
ledger=json.loads((H/'cluster/ledger.json').read_text());terminal={'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}
active=[v for v in ledger if v.get('state') not in terminal]
seconds=sum(v.get('elapsed_seconds',0) for v in ledger);stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
status=f'''**New Burgers tree-embedding study: B1 ACTIVE, representation passed; fixed-tree numerical qualification continues.** The same child16 head with64 retained nodes reaches {100*growth['actual4096_witness']['error']:.4f}% on actual4096 output at1152 variables; every development snapshot passes the representation screen. Direct24 gives {100*r['dense_witness']['relative_l2']:.4f}% with1376 variables; prescribed global POD-plus-Fourier gives {100*stable['controls']['960']['worst']:.4f}% with960. The direct PDE pilot passes physical accuracy, but the latest test sensitivity is0.10684%, just above0.1%. The first nonlinear PDE pilot also passes physical accuracy but fails stationarity on160/300 steps; a finite solver repair is running. Matched same-head uniform/prescribed controls continue. The wider global128 neural repair remains at{100*wide['worst_projection']:.4f}% projection error. No all-development PDE admission, complete-runtime win or final-test result. [Generated evidence and plots](reports/2026-11-18-tree-embedding-representation.md). Old campaign remains PAUSED; no merge.'''
title='### 2026-11-18 — Tree-embedding B1 frozen/global audits and physics: '+', '.join(name for name,_ in pilots)
status=status.replace('Matched same-head uniform/prescribed controls continue.',allocation_text)
coupled=[(name,p) for name,p in pilots if name.startswith('b1childpde')]
coupled_text=''
if coupled:
    latest_name,latest=coupled[-1];tree_methods=[c['methods']['tree_child16_n64'] for c in latest['cases']]
    failures=sum(sum(s[1] is None or s[1]>latest['stationarity_tolerance'] for s in v['step_status']) for v in tree_methods)
    count=sum(len(v['step_status']) for v in tree_methods)
    changes=[max(v['previous_state_relative_change']) for v in tree_methods if 'previous_state_relative_change' in v]
    coupled_text=f"Latest nonlinear PDE diagnostic {latest_name}: {failures}/{count} steps miss stationarity; evolved physical accuracy passes {all(v['physical_accuracy_pass'] for v in tree_methods)}."
    if changes:coupled_text+=f" Largest returned-field change from its preceding setting is{100*max(changes):.6f}%."
    if latest_name=='b1childpdeR2' and not latest['pilot_gate']:
        coupled_text+=' Two distinct online solver repairs failed; stop optimizer tuning for this checkpoint. Its representation remains admitted, while the direct-coefficient branch continues numerical qualification. No nonlinear adaptation or timing confirmation is admitted.'
    status=status.replace('The first nonlinear PDE pilot also passes physical accuracy but fails stationarity on160/300 steps; a finite solver repair is running.',coupled_text)
if 'ac' in globals() and ac['status']=='complete_frozen_child_allocation_controls':title+='; frozen child allocation'
test_admission_text='Further test-enrichment qualification is pending.'
for name,p in pilots:
    if name=='b1cnM32':
        change=max(max(v['previous_state_relative_change']) for c in p['cases'] for v in c['methods'].values())
        passed=all(v['physical_accuracy_pass'] and v['all_steps_stationary'] and max(v['previous_state_relative_change'])<=.001 for c in p['cases'] for v in c['methods'].values())
        test_admission_text=f'The completed44032-test check changes returned fields by at most{100*change:.6f}%; its stationary/physical/sensitivity gate passes{passed}. This qualifies22016 tests for the three-case direct24/global768 pilot only. Independent integration and remaining numerical checks continue.'
        status=status.replace('The direct PDE pilot passes physical accuracy, but the latest test sensitivity is0.10684%, just above0.1%.',test_admission_text)
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

Completed11008-test CN comparison: all three cases and both methods pass physical accuracy and stationarity, worst{100*pilot_worst:.6f}%. The direct tree's largest state change from5504 tests is{100*max(max(c['methods']['tree_direct24']['previous_state_relative_change']) for c in m8['cases']):.6f}%, so the original smaller test set is not numerically admitted. The completed22016-test check changes the tree field by up to0.106838%, narrowly failing the unchanged0.1% sensitivity gate. A finite44032-test review is registered before further integration changes. The independent Q512 integration diagnostic changes the tree states by at most{100*max(max(c['methods']['tree_direct24']['previous_state_relative_change']) for c in q512['cases']):.6f}% versus dense integration at5504 tests, preserving the underlying fine stencil. It inherits that smaller test set's physical failure and is not admission of a combined new rule. Pilot times charge joint setup to both methods and cannot serve as method-specific speed ratios.

The first coupled child16/64-node pilot has physical error below1% on all three exposed cases, but74 dev000 and86 align002 steps fail stationarity1e-8. Smooth dev003 is stationary. All requested-state tangent visibility minima exceed0.35, so the useful child directions remain observed by the tests in this diagnostic. Initial fits are separately underconverged, especially alignment stationarity3.44e-6. The first bounded online-solver repair increases only the common PDE LM cap30 to120; its unchanged cold fits and all costs remain visible. An optional coordinate-scaling implementation passes a sub-minute independent CPU fixture but is not enabled in that repair. Workstation GPU initialization failed before fixture arithmetic; no unrelated process was changed. All scientific experiments use preflight-verified float64 cluster GPUs.

{allocation_text} Uniform20 and prescribed20 also retain identical448-variable budgets. Every allocation uses the same selected child head and resolved physical metric, with encoder, zero and temporal starts; no weights were retrained. Fit stationarity is retained in the generated table, so the comparison is not presented as a certified manifold optimum.

{coupled_text} The completed iteration-limit repair is retained even though it fails. Its successor, safeguarded inverse BFGS with Armijo line search, is a separately registered second repair of the same online solver bottleneck; it uses the identical physical objective and tolerance. All rejected trials count. If it also fails, no further optimizer search occurs without a written scoped decision or specific amendment. The unchanged initial fits remain a separate qualification issue.

{test_admission_text} Earlier small-rule passes and failures remain in the ledger; a quadrature result at5504 tests is not automatically a result at22016 tests.

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
