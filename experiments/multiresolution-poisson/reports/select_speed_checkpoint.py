"""Apply the predeclared all-development checkpoint rule for later review."""
from pathlib import Path
import argparse,json,hashlib,statistics as st

p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();run=a.run
d=json.loads((run/'summary.json').read_text());cfg=d['config'];assert d['audit']['passed']
proposal=json.loads((Path(__file__).resolve().parents[1]/'speed_factorial_proposal.json').read_text())
assert proposal['status'].startswith('proposal_only')
scores=[]
for ck in d['checkpoints']:
    if ck['model']=='original_frozen':continue
    rows=[s for s in d['summaries'] if s['cohort']=='all_development' and s['model']==ck['model'] and s['arm']=='rom_modular' and s['tau']==0]
    assert {s['intervals'] for s in rows}==set(cfg['intervals'])
    assert all(len(s['cases'])==cfg['cohort_count'] for s in rows)
    errors=[c['adjusted_error'] for s in rows for c in s['cases']]
    valid=ck['valid_endpoint'] and all(c['solver_valid'] for s in rows for c in s['cases'])
    scores.append(dict(model=ck['model'],checkpoint_sha256=ck['sha256'],eligible=valid,
        worst_adjusted_error=max(errors),median_adjusted_error=st.median(errors),
        errors_by_mesh_and_case=[dict(intervals=s['intervals'],case=c['case'],adjusted_error=c['adjusted_error'],solver_valid=c['solver_valid']) for s in rows for c in s['cases']]))
eligible=[r for r in scores if r['eligible']]
selected=min(eligible,key=lambda x:(x['worst_adjusted_error'],x['median_adjusted_error'],x['model'])) if eligible else None
original=next(x for x in d['checkpoints'] if x['model']=='original_frozen')
result=dict(status='ready_for_coordinator_review_no_submission',source_summary_sha256=hashlib.sha256((run/'summary.json').read_bytes()).hexdigest(),
    selection_rule=proposal['checkpoint_selection'],case_count=cfg['cohort_count'],intervals=cfg['intervals'],
    original_checkpoint=original,candidates=scores,selected_continuation=selected,proposed_budget=proposal)
if selected:
    result['proposed_budget']['checkpoints']=['original_frozen',selected['model']]
    result['selected_checkpoint_path']=str(run/'checkpoints'/(selected['model']+'.pkl'))
    payload=Path(result['selected_checkpoint_path']).read_bytes()
    assert hashlib.sha256(payload).hexdigest()==selected['checkpoint_sha256']
(run/'SPEED-SELECTION.json').write_text(json.dumps(result,indent=2)+'\n')
lines=['# Proposed next Poisson speed experiment: checkpoint selection','',
    'This generated selection uses every development case and both completed query meshes. It prepares a concrete experiment for coordinator review; no new job is submitted.','',
    '| Scheduled endpoint | Eligible | Worst adjusted physical error | Median adjusted physical error | Checkpoint SHA256 |',
    '|---|---|---:|---:|---|']
for r in scores:lines.append(f"| {r['model']} | {r['eligible']} | {r['worst_adjusted_error']:.9g} | {r['median_adjusted_error']:.9g} | `{r['checkpoint_sha256']}` |")
if selected:
    lines+=['',f"Proposed frozen controls are `original_frozen` and `{selected['model']}`. The continuation minimizes worst adjusted error over {cfg['cohort_count']} cases on meshes {cfg['intervals']}, among complete endpoints with solver-valid tighter generic outputs throughout. "
        'Ties use median error and then model name. No inspected-case-only selection is used.','',
        f"The proposed job retains both development cohorts and uses {proposal['repetitions']} repetitions, {proposal['expected_timed_invocations']} total measured invocations, and a `{proposal['slurm_walltime']}` limit on generic A100. "
        f"Requested test count is {proposal['requested_modes']}; primary residual-reduction targets are {proposal['taus']}. "
        f"Separate stationary accuracy controls add {proposal['stationary_accuracy_controls']['expected_invocations']} fully recorded single calls excluded from speed selection. "
        'The projection-by-initialization factorial, direct parity checks and DST controls are specified in the linked proposal.']
else:lines+=['','No continuation meets the predeclared selection gate. Coordinator review is required to revise the proposed scope.']
lines+=['','See [algorithm and interpretation](../../AFTER-PILOT04-SPEED-PROPOSAL.md) and [complete budget](../../speed_factorial_proposal.json). All errors remain development measurements with empirical reference adjustment, without a rigorous continuum bound.','',
    '## Plain-language glossary','',
    '- **Endpoint / eligible:** fixed final training checkpoint / complete training and solver-valid tighter outputs on every declared case and mesh.',
    '- **Adjusted physical error:** common-observation relative error enlarged using the recorded empirical reference-refinement difference.',
    '- **Worst / median:** largest value over both meshes and all sources / middle value of that same collection.',
    '- **SHA256 / frozen control / factorial:** content identity / unchanged comparison checkpoint / independently crossed implementation choices.','']
(run/'SPEED-SELECTION.md').write_text('\n'.join(lines));print(json.dumps(result,indent=2))
