"""Generate the failed frozen-constructor summary; never select or validate weights."""
from pathlib import Path
import csv
import json
import numpy as np
from analyse import sha


def main():
    here=Path(__file__).resolve().parent;d=here/'runs/fit32/collected'
    p=d/'output/fit.json';j=json.loads(p.read_text());q=j['path'];cfg=j['config']
    assert j['complete'] is True and j['attempts']==1 and j['accepted'] is False
    assert q['reason']=='incomplete (KKT violated)' and not q['kkt']['ok']
    assert not j['target_reached'] and not j['earlier_certified_optimum']
    z=np.load(d/'output/NNLS_terminal.npz');cand=np.load(d/'output/candidate_nodes.npy')
    support=z['support'];weights=z['weights'];drops=np.asarray(q['drops'])
    assert np.array_equal(z['nodes'],cand[support])
    assert len(set(support.tolist()))==len(support) and np.isfinite(weights).all() and (weights>0).all()
    assert int((drops>=cfg['max_drops']).sum())==q['banned']
    assert len(support)<cfg['target_points'] and q['banned']>.01*len(cand)
    rows=list(csv.DictReader((d/'sacct.txt').read_text().splitlines(),delimiter='|'))
    acct=next(r for r in rows if r['JobID']==str(j['job_id']))
    assert acct['State']=='COMPLETED' and acct['ExitCode']=='0:0'
    fact=dict(result_path=str(p),result_sha256=sha(p),terminal_path=str(d/'output/NNLS_terminal.npz'),
        terminal_sha256=sha(d/'output/NNLS_terminal.npz'),source_commit=j['source_commit'],job_id=j['job_id'],
        complete=True,accepted=False,attempts=j['attempts'],construction_failure_reason=q['reason'],
        training_count=cfg['training_count'],training_trajectory_count=len(j['training']['descriptors']),
        training_state_count=len(j['training']['states']),candidate_count=len(cand),
        support_target=cfg['target_points'],terminal_support=len(support),
        terminal_weights_positive=True,relative_training_residual=q['relres'],
        kkt=q['kkt'],banned=q['banned'],banned_fraction=q['banned']/len(cand),banned_fraction_limit=.01,
        qr_drift_ok=q['drift_ok'],peak_device_allocated_bytes=j['peak_device_bytes'],
        gpu=j['gpu'],precision=j['precision'],contributions=j['contributions'],
        slurm_state=acct['State'],exit_code=acct['ExitCode'],elapsed_seconds=int(acct['ElapsedRaw']),
        spent_gpu_hours=int(acct['ElapsedRaw'])/3600,
        rollout_validation_run=False,HR_admission_evaluated=False,A2_release_not_reached=True,
        independent_result_gate='READY_FAILED_FROZEN_FIT',
        interpretation='No accepted support was constructed. This is not a k32 rollout failure or evidence POD loses; qualified HR comparison remains open.')
    out=here/'checks/fit32';out.mkdir(exist_ok=True)
    (out/'SUMMARY.json').write_text(json.dumps(fact,indent=2)+'\n')
    lines=['# Fixed larger-support POD construction','',
        'Final results for the single predeclared training-only support extension. The constructor failed its frozen acceptance predicates; no new-support trajectory validation was released.','',
        '| construction quantity | saved result |','|---|---:|',
        f"| target support | {fact['support_target']} |",f"| terminal support | {fact['terminal_support']} |",
        f"| candidate nodes | {fact['candidate_count']} |",f"| relative training residual | {fact['relative_training_residual']:.12g} |",
        f"| banned nodes | {fact['banned']} |",f"| banned fraction | {fact['banned_fraction']:.6%} |",
        f"| maximum permitted banned fraction | {fact['banned_fraction_limit']:.6%} |",
        f"| maximum off-support gradient | {fact['kkt']['off_support_max_gradient']:.12g} |",
        f"| KKT gradient tolerance | {fact['kkt']['tol']:.12g} |",
        f"| QR drift quality passed | {fact['qr_drift_ok']} |",f"| GPU-hours | {fact['spent_gpu_hours']:.9g} |",'',
        'The global KKT certificate failed and the banned-node fraction exceeded its quality limit. The fixed path stopped without reaching its target or a certified earlier optimum. No retry, alternate support, new-support accuracy score or trajectory admission claim follows from this run.','',
        'The independent reviewer recomputed the residual and complete candidate-gradient vector in NumPy from the retained contribution matrix and terminal weights, corroborating the recorded failure. The exact matrix, target, candidate nodes, descriptors, terminal arrays and source capsule are retained with checksums. The large matrix is preserved locally outside Git and must survive any later worktree archival.','',
        'The separately qualified full-residual POD fallback remains available. A qualified hyper-reduced POD comparison remains unresolved. This failed constructor does not establish that POD loses to the coordinate bank.','',
        '## Glossary','',
        '- support: the grid nodes assigned positive fitting weights.',
        '- candidate nodes: the fixed pool from which the greedy path could select support.',
        '- relative training residual: the remaining fitting discrepancy divided by the target norm.',
        '- banned nodes: candidates excluded after the frozen number of active-set drops.',
        '- NNLS: nonnegative least squares, the offline fitting problem for positive weights.',
        '- KKT certificate: a check that active and excluded candidate gradients satisfy first-order optimality conditions.',
        '- off-support gradient: a candidate-gradient component at a node outside the terminal support.',
        '- QR drift: disagreement in the updated matrix factorization used by the fit.',
        '- GPU-hours: allocated GPU time, including setup and the stopped fit.',
        '- full-residual POD: a reduced linear model whose online objective uses every grid residual row.',
        '- hyper-reduced POD: a reduced linear model using weighted selected residual rows.','']
    (out/'report.md').write_text('\n'.join(lines))
    print(json.dumps(fact,indent=2))


if __name__=='__main__':main()
