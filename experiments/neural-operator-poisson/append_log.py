"""Append only this lane's source-generated closing entry under the shared lock."""
import fcntl
import json
from pathlib import Path
import subprocess
from dataset import HERE,ROOT
LOG=Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/LAB-LOG.md')


def entry():
    matched=json.loads((HERE/'runs/matched01/matched-summary.json').read_text())
    pilot=json.loads((HERE/'runs/pilot01/summary.json').read_text())
    submissions={name:json.loads((HERE/'runs'/name/'submission.json').read_text()) for name in ['pilot01','matched01']}
    for name in submissions:
        collection=json.loads((HERE/'runs'/name/'collection.json').read_text())
        assert collection['checksums_verified'] and collection['numpy_audit_pass'] and collection['remote_exact_directory_removed']
    assert matched['complete']
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    lines=['','## 2026-09-14','### Poisson lane — independent diagnosis and matched capacity screen; broader operator study remains open','',
        'Completed the authorized first Poisson diagnosis and its conditional matched-data capacity screen in `worktrees/2026-09-14-no-poisson`. '
        f'The retained branch head is `{commit}`; all source, trained checkpoints, references, output fields, repetitions and independent audits are under '
        '`experiments/neural-operator-poisson/runs/`. No merge or final-cohort access occurred. The table values below are generated from the retained JSONs by '
        '`experiments/neural-operator-poisson/append_log.py`, not transcribed from conversation.','',
        'Execution and evidence:']
    for name,s in submissions.items():
        lines.append(f"- `{name}`: source `{s['source_commit']}`, job `{s['job_id']}`, {s['gpu_type']} on `{s['allocated_node']}`, elapsed {s['elapsed_seconds']} seconds. "
                     'GPU preflight, float64/highest precision, source/output checksums and independent numerical audits passed. The exact remote job directory is removed.')
    lines += ['',f"The shared dataset contains {pilot['case_count']} training/validation cases, split into the recorded independent training and validation cohorts, plus the separate calibration cohort. "
        f"The largest calibration relative change between the two recorded fine meshes is {pilot['calibration_maximum_change']:.12e}; all validation empirical refinement gates pass. "
        'Training targets are exact discrete FD/DST solutions. Error tables use the recorded fine-grid candidate restricted to the requested mesh; continuum interpretation remains provisional under empirical refinement, not a rigorous bound. '
        'Original copied indices are unchanged, with explicit calibration-path relocation; future generated indices embed portable calibration evidence.','',
        'The historical checkpoint is an unmatched-training diagnostic. Fresh arms share hidden widths, Fourier features, latent dimension, training cases, phase update counts, source/point sampling streams and correction policy; only the bank feature count varies. '
        'The smaller fresh arm is a common-width control, not the historical architecture. Fresh weights and training-field PCA codes initialize both arms. '
        'Training never opens validation. Each correction basis is refitted from normalized training residuals only. All phase checkpoints, optimizer timing blocks, losses, exposures and parameter counts are retained. '
        'This is one initialization seed with fixed phase budgets; it does not establish a matched tuning-budget comparison against FNO.','',
        '| Model | Bank median / worst error (%) | Best-found fit median / worst (%) | Online median / worst (%) | Online cases above 5% | Invalid online invocations |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    bundles=[('historical r128',pilot)]+[(f'fresh r{rank}',v['summary']) for rank,v in matched['results'].items()]
    for label,s in bundles:
        o=s['oracles']['physical_candidate'];bank=o['bank_projection'];fit=o['best_found_nonlinear_linear_fit'];online=s['methods']['nmrom']['physical_candidate_relative_error']
        fmt=lambda d:f"{100*d['median']:.9f} / {100*d['maximum']:.9f}"
        lines.append(f"| {label} | {fmt(bank)} | {fmt(fit)} | {fmt(online)} | {online['count_above_005']} | {s['methods']['nmrom']['invalid_invocations']} |")
    larger=matched['results']['256']['summary'];smaller=matched['results']['128']['summary']
    assert all(v['summary']['methods']['nmrom']['physical_candidate_relative_error']['count_above_005'] for v in matched['results'].values())
    lines += ['', 'Neither fresh model passes the diagnostic accuracy target on every validation case. '
        'The larger bank lowers the projection floor in this single-seed screen, while its worst best-found fit and online errors increase. '
        'Both fresh models generalize poorly compared with their training reconstruction errors; they are retained as negative accuracy-screen outcomes, not selected replacements. '
        'The original diagnosis suggested that its converged online solve nearly reaches the best-found fit. After fresh training, some cases also show sizeable online-versus-fit gaps, '
        'so that earlier observation must not be generalized to these new heads. Stationarity alone does not establish physical accuracy.']
    parity=matched['initial_pca_code_parity']
    lines += ['',f"The initial audit's byte-equality check for separately computed PCA starting codes failed; its log is retained. "
        f"Maximum absolute code difference is {parity['max_abs']:.12e} and relative difference is {parity['relative']:.12e}. "
        f"The explicit numerical audit uses absolute/relative tolerances {parity['atol']:.1e}/{parity['rtol']:.1e}; shared initial hidden weights and dataset hashes remain byte-identical. "
        'This is a documented audit-assumption correction after the initial failure, not a change to training, data, outputs or a claim of byte-identical PCA codes.']
    lines += ['','Full learned-bank and augmented-bank projections agree: the physical correction fields lie inside the learned bank. '
        'Projection, multistart best-found fit and online errors are compared on the same cases and reference; their cohort maxima are not an additive decomposition. '
        'The fits are not certified global optima. Independent NumPy reconstruction and analytic head derivatives check full/reduced weak stationarity; physical accuracy is a separate gate.','',
        '| Fresh-model panel | Method | Median GPU time (ms) | Median complete host query (ms) | Worst candidate-relative error (%) |',
        '| --- | --- | ---: | ---: | ---: |']
    for rank,v in matched['results'].items():
        for name in ['nmrom','dst','cg_0.1']:
            m=v['summary']['methods'][name]
            lines.append(f"| r{rank} | {name} | {1000*m['gpu_seconds_median']:.9f} | {1000*m['total_seconds_median']:.9f} | {100*m['physical_candidate_relative_error']['maximum']:.9f} |")
    lines += ['','These cost/accuracy pairs come from the same saved invocations on the same allocation, with warmup, GPU burn-in and complete repetition arrays. '
        'All tested CG tolerances remain in the JSON, including failures or slower settings. Direct DST is included as the efficient linear FOM control. '
        'No ratio is assembled across the historical and fresh jobs, and no FNO speedup or broader-study completion is claimed.','',
        'Nothing from the earlier accepted campaign is retracted: the new independent validation cases differ from its cohort. '
        'The open work is the tuned-operator comparison and a joint paired cost/accuracy panel, followed by any selected accuracy or initialization change and the later resolution/data/seed controls. '
        'The final paper cohort remains sealed; the worktree remains separate.']
    return '\n'.join(lines)+'\n'


if __name__=='__main__':
    text=entry(); marker='### Poisson lane — independent diagnosis and matched capacity screen; broader operator study remains open'
    with LOG.open('a+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        stream.seek(0)
        if marker in stream.read():raise RuntimeError('This closing entry already exists')
        stream.seek(0,2);stream.write(text);stream.flush()
        fcntl.flock(stream,fcntl.LOCK_UN)
