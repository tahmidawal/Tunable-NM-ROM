"""Publish this owner's data-generated progress without changing other campaigns."""
from pathlib import Path
import json,datetime,hashlib
H=Path(__file__).resolve().parent
def read(n):return json.loads((H/'results'/n/'output/result.json').read_text())
r=read('b1directaudit');n=read('b1child8');c=read('b1cache');g=read('b1global128')
rows=json.loads((H/'cluster/ledger.json').read_text());terminal={'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}
active=[v for v in rows if v.get('state') not in terminal];seconds=sum(v.get('elapsed_seconds',0) for v in rows)
stamp=datetime.datetime.now(datetime.timezone.utc).isoformat();owner=json.loads((H/'OWNER.json').read_text())
owner.update(updated_utc=stamp,status='B1_ACTIVE_QUALIFIED_DIRECT_REPRESENTATION',allocated_gpu_seconds=seconds,
    allocated_gpu_hours=seconds/3600,active_jobs=len(active),terminal_jobs=len(rows)-len(active),
    decision='Direct local POD24 passes expanded representation and dense-output gates; nonlinear compression and fixed-tree physics continue',
    final_test_solution_fields_opened=False)
(H/'OWNER.json').write_text(json.dumps(owner,indent=2)+'\n')
claim=Path('/home/tahmid/Dev/QuadtreeNMROM/execution/tree-embeddings-owner.lock/owner.json');assert json.loads(claim.read_text())['owner']==owner['owner']
claim.write_text(json.dumps(owner,indent=2)+'\n')
status=f'''**New Burgers tree-embedding study: B1 ACTIVE with qualified direct representation.** Shared local POD24 on the expanded training128/development32 plus alignment panel reaches {100*r['dense_scoring']['2048']['worst']:.4f}% worst current-time relative L2 error at1376 total variables; actual4096 confirms {100*r['dense_witness']['relative_l2']:.4f}%. The richer uncompressed span is more accurate but larger. The first nonlinear child8 reaches {100*n['worst_projection']:.4f}% with incomplete fitting convergence; a frozen-checkpoint numerical audit and child16 training are underway. The verified weak-upwind fixed-tree solver is implemented and its algebra fixture passes; complete PDE accuracy/runtime is not established. Final tests remain unopened. [Generated diagnosis and plots](reports/2026-11-18-tree-embedding-representation.md). The November17 campaign remains PAUSED; no merge.'''
title='### 2026-11-18 — Tree-embedding B1 qualified direct representation and nonlinear compression'
entry=f'''{title}

Published {stamp} from retained run JSONs. Same owned branch/worktree/namespace; no old-campaign writes. Shared direct local prefixes4/8/16/24/36 give worst errors {', '.join(f"{100*v['projection']['worst']:.6f}%" for v in c['direct_controls'].values())} at total counts {', '.join(str(v['active_variables']) for v in c['direct_controls'].values())}. Independent fixed-code dense scoring admits prefix24: actual4096 error {100*r['dense_witness']['relative_l2']:.8f}%, largest physical-metric-to2048 change {r['max_metric_2048_change']:.3g}. Child Jacobian directions after removing the parent: {r['zero_child_jacobian']['independent_directions']}/{r['zero_child_jacobian']['child_variables']}, squared singular-value ratio {r['zero_child_jacobian']['smallest_to_largest']:.3g}. Same-checkpoint allocation controls are retained with exact variable counts; uniform and prescribed are compared at matched20/84-node budgets.

The expanded global nonlinear128 control has worst projection {100*g['worst_projection']:.6f}%. These spectral-feature controls remain architecture-inspired, not faithful coordinate-MLP Tunable reproductions. Child8 training stopped at its registered early-stop rule; selected update {n['selected_update']}, worst error {100*n['worst_projection']:.6f}%, with nonzero stationarity. Before interpreting a capacity floor, registered a frozen dense-LM audit and the numerical linear-solver correction for the already-registered child16 recipe. The pilot's focused prescribed-tree/multiple-depth training is an explicit amendment; the original balanced random/uniform/oracle mixture is not claimed for B1. Full confirmations and all strong NM-ROM competitors remain outstanding.

Implemented exact dense fine-grid tested weak-upwind physics with the original discrete sine eigen-diffusion and scaling; a local sub-minute fixture agrees with explicit dense tests and checks its field/Jacobian/step. The finite fixed-tree pilot is separately registered for dev000/dev003/align002 at L1024 and dt0.0025 with common tests, actual4096 final output, truth-free initialization and tree selection. Dense integration is a physics diagnostic, not a runtime win or hyper-reduced deployment. Further spatial/time/test/solver refinement and all-development coverage are mandatory before advancement.

Current reconciled allocated usage is {seconds} GPU-seconds ({seconds/3600:.6f} hours), including failed/setup time, under the finite B1 cumulative ceiling. Active names at publication: {', '.join(v['name'] for v in active) or 'none'}. All completed results have verified local/cluster copies; selected checkpoints and source are mirrored. The aliased preliminary rich-span result and the interval/mode implementation failure remain explicitly retracted/retained. No final-test fields, automatic merge, or unrelated three-dimensional work.
'''
lab=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/LAB-LOG.md');s=lab.read_text();a=s.index('**New Burgers tree-embedding study:');b=s.index('\n\n',a);s=s[:a]+status+s[b:]
if title not in s:s=s.rstrip()+'\n\n'+entry+'\n'
lab.write_text(s)
publication=dict(utc=stamp,own_status=status,own_entry=entry,source_hashes={name:hashlib.sha256((H/'results'/name/'output/result.json').read_bytes()).hexdigest() for name in ['b1directaudit','b1child8','b1cache','b1global128']})
(H/'manifests/B1-canonical-progress.json').write_text(json.dumps(publication,indent=2)+'\n')
print(status)
