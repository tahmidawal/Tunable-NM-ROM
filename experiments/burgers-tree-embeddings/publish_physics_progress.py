"""Refresh only this owner's canonical record from retained completed evidence."""
from pathlib import Path
import datetime,hashlib,json
H=Path(__file__).resolve().parent
sources={}
def read(name):
    p=H/'results'/name/'output/result.json';sources[name]=hashlib.sha256(p.read_bytes()).hexdigest()
    return json.loads(p.read_text())
r=read('b1directaudit');g=read('b1globalfull');n=read('b1child8refit')
pilots=[]
for name in ['b1weakpilot1','b1weakL2048','b1weakL4096','b1weakdt125','b1weakdt625']:
    if (H/'results'/name/'output/result.json').exists():
        p=read(name)
        if p['status']=='complete_fixed_tree_diagnostic_pilot':pilots.append((name,p))
ledger=json.loads((H/'cluster/ledger.json').read_text());terminal={'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}
active=[v for v in ledger if v.get('state') not in terminal]
seconds=sum(v.get('elapsed_seconds',0) for v in ledger);stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
status=f'''**New Burgers tree-embedding study: B1 ACTIVE, fixed-tree physics under numerical refinement.** The direct local representation passes its screen: worst {100*r['dense_witness']['relative_l2']:.4f}% on actual4096 output. Initial complete PDE pilots converge but pulse errors remain above1%; spatial/time/test refinement is required. Frozen child8 refitting reaches {100*n['worst_projection']:.4f}%, with unresolved stationarity. The saved global1280 span reaches {100*g['POD']['1280']['projection']['worst']:.4f}%, but includes numerically null training directions and needs stability qualification. No complete-runtime win or final-test result. [Generated evidence and plots](reports/2026-11-18-tree-embedding-representation.md). Old campaign remains PAUSED; no merge.'''
title='### 2026-11-18 — Tree-embedding B1 physics evidence: '+', '.join(name for name,_ in pilots)
lines=[]
for name,p in pilots:
    for c in p['cases']:
        for method,v in c['methods'].items():
            line=f"- {name}, {c['case']}, {method}: L{p['L']}, dt{p['dt']:g}, worst evolved error {100*max(v['physical_error'][1:]):.6f}%, actual4096 final {100*v['actual4096_final_error']:.6f}%; all steps stationary {v['all_steps_stationary']}, accuracy pass {v['physical_accuracy_pass']}."
            if 'previous_state_relative_change' in v:line+=f" Maximum physical state change from prior setting {100*max(v['previous_state_relative_change'][1:]):.6f}%."
            lines.append(line)
entry=f'''{title}

Published {stamp} from checksum-verified results in the owned lane. These are complete truth-free fixed-tree trajectories on a small exposed pilot, not all-development qualification or timing confirmation. Exact fine-grid upwind remains inside the common scaled weak residual. Physical projection and numerical/dynamics error are reported separately where measured. The first pilot lacks explicit immediate timer burn-in; its times remain unqualified diagnostics. Subsequent runs add burn-in without altering the original record.

{chr(10).join(lines)}

The extended saved global POD1280 space gives worst {100*g['POD']['1280']['projection']['worst']:.6f}% versus direct-tree1376 {100*r['dense_witness']['relative_l2']:.6f}%. The trailing relative training singular value is {g['POD']['1280']['relative_training_singular_value']:.4g}: this is a numerical-null completion, not reliably learned variance. Keep its real saved-space error visible and require stability qualification. The data-independent complete sine48 space gives {100*g['complete_sine']['48']['projection']['worst']:.6f}% with {g['complete_sine']['48']['variables']} variables. No error-per-variable result establishes cost.

Frozen child8 dense250 fitting gives {100*n['worst_projection']:.6f}%, maximum stationarity {n['max_stationarity']:.4g}; weights unchanged. This does not prove a global fitting minimum. A frozen global output-span relaxation is prospectively registered to separate final-layer shape restrictions from nonlinear coding. No additional trained recipe is claimed by that audit.

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
    decision='Direct representation qualified; common fixed-tree PDE pilot requires numerical refinement',final_test_solution_fields_opened=False)
(H/'OWNER.json').write_text(json.dumps(owner,indent=2)+'\n')
claim=Path('/home/tahmid/Dev/QuadtreeNMROM/execution/tree-embeddings-owner.lock/owner.json')
assert json.loads(claim.read_text())['owner']==owner['owner'];claim.write_text(json.dumps(owner,indent=2)+'\n')
print(status)
