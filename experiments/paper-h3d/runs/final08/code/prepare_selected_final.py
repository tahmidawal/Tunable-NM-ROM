"""Freeze development-only head, CG frontier and operator transfer selections.

This script reads audited development files only and never generates parameters.
"""
import argparse,copy,hashlib,json
from pathlib import Path
import final_freeze as F


def write(path,value):
    assert not path.exists(),path
    path.write_text(json.dumps(value,indent=2)+'\n')


def passing(row):
    return row['nonfinite_cases']==0 and row['cases_with_nonstationary_solves']==0


def error(row):return row['same_grid_current_evolved_worst']
def time(row):return row['device_ms_median']


def frontier(rows):
    return [a for a in rows if not any(error(b)<=error(a) and time(b)<=time(a)
        and (error(b)<error(a) or time(b)<time(a)) for b in rows)]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--attempt',default='development06');args=parser.parse_args()
    assert args.attempt.isalnum();lane=Path(__file__).resolve().parent;run=lane/'runs'/args.attempt;source=run/'archive/out'
    for name in ['audit-local.json','audit-states-primary.json','audit-panel-primary.json',
                 'audit-field-seedB.json','audit-states-seedB.json','audit-panel-seedB.json',
                 'audit-frozen-replay-primary.json','audit-frozen-replay-seedB.json']:
        assert json.loads((run/name).read_text())['passed'],name
    record=json.loads((source/'result.json').read_text());other=json.loads((source/'seedB/result.json').read_text())
    assert record['complete'] and other['complete'] and not record['final_cohort_opened'] and not other['final_cohort_opened']
    cfg=record['config'];native=cfg['train_intervals'];rows=json.loads((source/'summary.json').read_text())['rows']
    lookup={(r['intervals'],r['method']):r for r in rows};head_candidates=[]
    for k in cfg['latent_dimensions']:
        rungs=[lookup[(native,f'nmrom_K{k}_q{q}_dense')] for q in cfg['q_ladder'] if k+q<=cfg['bank_rank']]
        head_candidates.append(dict(k=k,passing_all_native_rungs=all(passing(r) for r in rungs),
            q0_error=error(rungs[0]),q0_device_ms=time(rungs[0]),rungs=rungs))
    eligible=[r for r in head_candidates if r['passing_all_native_rungs']]
    assert eligible,'Repair numerical convergence on development before freezing a head.'
    chosen=min(eligible,key=lambda r:(r['q0_error'],r['q0_device_ms'],r['k']))['k']
    operators={};cg={};meshes=[]
    for n in cfg['evaluation_intervals']:
        metadata=next(m['methods'] for m in record['meshes'] if m['intervals']==n)
        selected=[];operator_candidates={}
        for base in cfg['frozen_operators']:
            candidates=[lookup[(n,name)] for name,meta in metadata.items()
                if meta['kind']=='neural_operator' and (name==base or name.startswith(base+'_'))
                and (n!=native or name==base)]
            acceptable=[r for r in candidates if passing(r)]
            assert acceptable,(n,base,'no finite passing operator')
            winner=min(acceptable,key=lambda r:(error(r),time(r),r['method']))
            selected.append(winner['method']);operator_candidates[base]=candidates
        operators[str(n)]=selected
        candidates=[lookup[(n,name)] for name in metadata if name.startswith('fom_cn_cg_')]
        available=frontier([r for r in candidates if passing(r)]);assert available
        # Original descriptive targets stay fixed. Add the actual selected-method
        # errors solely to choose a no-less-accurate development comparator.
        desired=[dict(label=f'declared_{t:g}',error=t) for t in [.05,.02,.01]]
        ranks={cfg['bank_rank']}|{chosen+q for q in cfg['q_ladder'] if chosen+q<=cfg['bank_rank']}
        named=selected+[f'nmrom_K{chosen}_q{q}_dense' for q in cfg['q_ladder'] if chosen+q<=cfg['bank_rank']]
        named += [name for name in metadata if name.startswith('linear_bank_') or name in {f'pod{rank}_exact' for rank in ranks}]
        desired += [dict(label=name,error=error(lookup[(n,name)])) for name in named if passing(lookup[(n,name)])]
        picks={min(available,key=lambda r:(error(r),time(r),r['method']))['method']};matches=[]
        for target in desired:
            feasible=[r for r in available if error(r)<=target['error']]
            winner=min(feasible,key=lambda r:(time(r),error(r),r['method'])) if feasible else None
            if winner:picks.add(winner['method'])
            matches.append(dict(**target,selected_cg=winner['method'] if winner else None))
        cg[str(n)]=sorted(picks)
        meshes.append(dict(intervals=n,operator_candidates=operator_candidates,selected_operators=selected,
            cg_candidates=candidates,cg_passing_frontier=[r['method'] for r in available],selected_cg=sorted(picks),targets=matches,
            unmatched_target_policy='retain direct DST comparison and report no eligible measured CG comparator; do not invent a matched speedup'))
    selection=dict(schema='heat3d-prospective-final-selection-v1',final_data_accessed=False,
        source_result_sha256=dict(primary=F.digest(source/'result.json'),seedB=F.digest(source/'seedB/result.json')),
        source_commit=record['source_commit'],source_job_id=record['job_id'],head_candidates=head_candidates,
        selected_head=chosen,operator_methods_by_mesh=operators,cg_methods_by_mesh=cg,meshes=meshes,
        policy='Primary seed fixed prospectively. Among numerically passing native ladders choose lowest q0 worst error, then q0 median time; never maximize span by selecting a weak q0. Each fine operator uses minimum worst development error, then median time. Keep only passing non-dominated CG settings needed for fixed targets and selected-method comparisons, plus the most accurate passing CG. All rejected development variants remain archived. No final-based pruning.')
    report=lane/'final-selection.json';write(report,selection)
    common=dict(evaluation_cohort='final',representation_oracles=False,fit_quadrature=False,retain_solver_states=True,
        latent_dimensions=[chosen],final_selection_report_file='code/final-selection.json',final_selection_report_sha256=F.digest(report))
    primary=copy.deepcopy(cfg);primary.update(common);primary.pop('frozen_source_seed',None)
    primary.update(frozen_source_attempt=args.attempt,frozen_input_directory=f'inputs/{args.attempt}_primary',
        operator_methods_by_mesh=operators,cg_methods_by_mesh=cg,final_freeze_path='code/final-primary-freeze.json',
        companion_config_file='code/finalB.json')
    write(lane/'finalA.json',primary);write(lane/'final-primary-freeze.json',F.prepare(primary,source))
    robust=copy.deepcopy(other['config']);robust.update(common)
    robust.update(frozen_source_attempt=args.attempt,frozen_source_seed='seedB',frozen_input_directory=f'inputs/{args.attempt}_seedB',
        operator_methods_by_mesh={str(native):list(cfg['frozen_operators'])},final_freeze_path='code/final-robustness-freeze.json',
        primary_freeze_sha256=F.digest(lane/'final-primary-freeze.json'))
    write(lane/'finalB.json',robust);write(lane/'final-robustness-freeze.json',F.prepare(robust,source/'seedB'))
    print('FINAL_SELECTION_PREPARED_FROM_DEVELOPMENT_ONLY',chosen,cg,operators)


if __name__=='__main__':main()
