"""Bounded CPU-only collection of owned job 3702709; no new GPU submissions.

The full archive and numerical audits are Git committed before exact job-directory
cleanup. A checksummed cluster-generated dataset cache remains for the FNO owner.
Any failure preserves the remote evidence and records an explicit failure status.
"""
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[3]
HERE=ROOT/'experiments/neural-operator-burgers'
LAB=Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/LAB-LOG.md')
JOB='3702709'
ATTEMPT='refinement02'
REMOTE='/cluster/tufts/paralab/tawal01/no_burgers_20260914/'+ATTEMPT
CACHE='/cluster/tufts/paralab/tawal01/no_burgers_20260914/pilot-data01'
STATUS=HERE/'checks/refinement02-collection-status.json'
ARCHIVE=HERE/'runs'/ATTEMPT/'archive'
ARTIFACT=HERE/'artifacts'/ATTEMPT


def save(value):
    temporary=STATUS.with_suffix('.partial')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(STATUS)


def ssh(command,timeout=600):
    return subprocess.check_output(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
                                    'tufts-login',command],text=True,timeout=timeout)


def run(*args):
    subprocess.run([str(x) for x in args],cwd=ROOT,check=True)


def append(text,current=None):
    with LAB.open('r+') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        existing=stream.read()
        if current is not None:
            start='<!-- burgers-20260914-current-start -->'
            end='<!-- burgers-20260914-current-end -->'
            block=start+'\n'+current+'\n'+end
            if start in existing:
                left=existing.index(start);right=existing.index(end,left)+len(end)
                existing=existing[:left]+block+existing[right:]
            else:
                anchor='## Read this first\n\n'
                assert existing.count(anchor)==1
                existing=existing.replace(anchor,anchor+block+'\n\n',1)
        stream.seek(0);stream.write(existing+text);stream.truncate();stream.flush();os.fsync(stream.fileno())


def closing_text(accounting,commit,cache):
    ref=json.loads((ARCHIVE/'output/refinement/index.json').read_text())
    diagnosis=json.loads((ARCHIVE/'output/diagnosis/index.json').read_text())
    audit=json.loads((HERE/'checks/refinement02-diagnosis-audit.json').read_text())
    dataset=json.loads((HERE/'checks/refinement02-dataset-audit.json').read_text())
    archive=json.loads((ARTIFACT/'archive.json').read_text())
    old=json.loads((HERE/'artifacts/calibration01/calibration-index.json').read_text())
    selected=ref['gate']['by_output']['256']['selected']
    old_candidate=next(c for c in old['gate']['by_output']['256']['candidates'] if c['intervals']==4096)
    date=time.strftime('%Y-%m-%d',time.gmtime())
    text=f'\n\n## {date}\n### Burgers lane — refined reference, unchanged-head diagnosis and shared data collection\n\n'
    text+=f'Owned job `{JOB}` finished on `{diagnosis["provenance"]["gpu"]}` with GPU backend, float64 and highest matrix precision. Accounting: `{accounting.strip()}`. '
    text+=f'Immutable execution source is `{diagnosis["provenance"]["source_commit"]}`; collection/audit/archive commit is `{commit}` in `worktrees/2026-09-14-no-burgers`. '
    text+=f'Full raw arrays, sources, manifests, logs, stopping records and all timing repetitions are retained in `experiments/neural-operator-burgers/artifacts/{ATTEMPT}` as bounded compressed archive chunks, whole SHA256 `{archive["sha256"]}`. Source and output checksums, the full reference audit, dataset audit and independent NumPy diagnosis audit passed before exact remote job-directory cleanup.\n\n'
    text+=f'The original job 3701221 calibration failure is preserved unchanged: its selected-mesh candidate worst empirical margin was {old_candidate["worst_empirical_margin"]:.16g}, above the {old["gate"]["empirical_budget"]:.16g} budget. '
    text+=f'The focused refinement retained {selected["intervals"]} spatial intervals and used time step {selected["dt"]:.16g}; its worst margin is {selected["worst_empirical_margin"]:.16g}. '
    text+=f'All {ref["count"]} independent calibration cases pass the unchanged budget with decreasing recorded refinements. The complete effective protocol and new source hashes are frozen, and reused original fields keep their old source/configuration/index/hash links. This is empirical refinement evidence, not a rigorous continuum certificate or a per-generated-case error bound. No cheaper candidate passed this reference gate.\n\n'
    text+='The same-case diagnosis uses the inherited checkpoint with unmatched training history. Its frozen bank projection, multistart best-found nonlinear fit and native online rollout are compared against the same evolved reference fields. The fit is not a certified global optimum and these maxima are not an additive error decomposition.\n\n'
    text+='| Diagnostic | Worst all-times error (%) | Worst evolved-times error (%) |\n| --- | ---: | ---: |\n'
    for key,label in [('bank','Free bank projection'),('nonlinear_best_found','Best-found nonlinear fit'),('online','Native online ROM')]:
        allmax=max(r['errors'][key]['maximum'] for r in diagnosis['cases'])
        evolved=max(max(r['errors'][key]['per_time'][1:]) for r in diagnosis['cases'])
        text+=f'| {label} | {100*allmax:.9f} | {100*evolved:.9f} |\n'
    text+='\n| Paired method | Median GPU (ms) | Median complete host query (ms) | Worst candidate-relative error (%) | Failed stopping checks | Upper-Tukey latency outliers |\n| --- | ---: | ---: | ---: | ---: | ---: |\n'
    for key,row in audit['summary'].items():
        text+=f'| {key} | {row["gpu_median_ms"]:.9f} | {row["host_to_host_median_ms"]:.9f} | {100*row["worst_fixed_initial_error"]:.9f} | {row["failed_stopping_invocations"]} | {row["latency_upper_tukey_outliers"]} |\n'
    text+=f'\nAll {audit["invocations_checked"]} invocations have paired cost/accuracy fields and retained repetition arrays, with GPU burn-in before each timed block. The efficient loose-tolerance FOM is faster and more accurate than this ROM: no ROM speed/accuracy advantage is established. These are within-job comparisons; no cross-job timing ratio is used. '
    text+='The native ROM retains fitted initial output for the archived compression diagnostic. A future deployable panel must return supplied initial output exactly for every method while still charging internal initial fitting and evolution, with compression error separate; that wrapper change must not silently replace these archived metrics.\n\n'
    eq=audit['eq_fidelity']['case_results']
    # Keep numerical evidence in its saved component-level scope.
    eqmax=max(max(row['relative_projected_advection_difference']) for row in eq)
    normalized=max(max(row['preconditioned_step_component_over_initial_norm']) for row in eq)
    text+=f'The independent audit also records sampled-state empirical-quadrature advection discrepancies for {len(eq)} cases in `checks/refinement02-diagnosis-audit.json`: worst relative projected-advection difference {eqmax:.16g}, and worst preconditioned step-component difference over initial norm {normalized:.16g}. These component comparisons are not trajectory bounds or causal attribution. '
    text+='Frozen-head quadrature-ablation source is prepared, but this collection monitor submits no GPU work. Trajectory training, correction and operator comparison remain open.\n\n'
    text+=f'Dataset collection checked {dataset["total_records_checked"]} model-facing cases; complete planned dataset: `{dataset["complete_dataset"]}`. '
    for split in dataset['splits']:text+=f'{split["split"]}: {split["records_checked"]}/{split["expected_count"]}. '
    text+='Exact four-key NPZ schema, finite float64 fields, zero boundaries, supplied initial output, independently regenerated seeds/initial fields, cross-split uniqueness, source hashes and every saved solver-step residual passed. The input is the sampled initial field plus viscosity; generation descriptors remain offline. Fine references evaluate the analytic Gaussian initial family, so this is a Gaussian continuum-family pilot, not evidence for an arbitrary sampled-field operator. '
    if cache:text+=f'A verified cluster-generated handoff cache remains at `{cache}` for the FNO owner; its `RELOCATION.json` preserves original index hashes and explicitly relocates calibration paths. The exact completed job directory is removed; the shared cache needs cleanup after downstream cluster copying. '
    text+='The original failure is not retracted, final cohorts remain sealed, and no merge occurred. The broader experiment campaign and eventual user merge decision remain open.\n'
    return text


def main():
    assert sys.executable=='/home/tahmid/Dev/.venv/bin/python'
    started=time.monotonic();last=None
    save(dict(phase='monitoring',job_id=JOB,pid=os.getpid(),remote=REMOTE,maximum_monitor_hours=12))
    while time.monotonic()-started<12*3600:
        try:
            # Slurm returns exit 1 for an expired explicit job selector. Query
            # the account queue successfully, then select only our exact ID.
            queue=ssh('squeue -u tawal01 -h -o "%i %j %T"',timeout=45)
            matches=[line.strip() for line in queue.splitlines()
                     if line.split() and line.split()[0]==JOB]
            assert len(matches)<=1
            queued=matches[0] if matches else ''
            if queued:
                parts=queued.split();assert parts[0]==JOB and parts[1]=='ctol_nob_refinement02'
                if queued!=last:print(queued,flush=True);last=queued
                time.sleep(30);continue
            accounting=ssh(f'sacct -X -j {JOB} -n -P -o JobID,JobName%40,State,ElapsedRaw,NodeList',timeout=45).strip()
            parts=accounting.split('|');assert parts[0]==JOB and parts[1]=='ctol_nob_refinement02'
            assert parts[2] in ('COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL') or parts[2].startswith('CANCELLED')
            break
        except subprocess.SubprocessError as error:
            print(f'Transient scheduler read error: {error}',flush=True);time.sleep(30)
    else:raise RuntimeError('Bounded monitor deadline exceeded; cluster directory preserved')
    assert int(parts[3])+1575<=8*3600
    save(dict(phase='collecting',job_id=JOB,accounting=accounting))
    run(sys.executable,HERE/'cluster/collect.py',ATTEMPT,'--kind','followup')
    (ARCHIVE/'scheduler.txt').write_text(accounting+'\n')
    logs=[p for p in ARCHIVE.rglob('*') if p.is_file() and (p.suffix in ('.out','.err','.log'))]
    combined='\n'.join(p.read_text(errors='replace') for p in logs)
    assert 'jax_backend=gpu' in combined
    bad=re.findall(r'Traceback|RESOURCE_EXHAUSTED|CUDA_ERROR|out of memory|No space left|jax_backend=cpu|captured.{0,30}constant',combined,re.I)
    assert not bad,bad
    run(sys.executable,HERE/'audit_calibration.py',ARCHIVE/'output/refinement/index.json',
        '--refined','--out',HERE/'checks/refinement02-reference-audit.json')
    run(sys.executable,HERE/'audit_diagnosis.py','--diagnosis',ARCHIVE/'output/diagnosis/index.json',
        '--reference',ARCHIVE/'output/refinement/index.json','--out',HERE/'checks/refinement02-diagnosis-audit.json')
    run(sys.executable,HERE/'audit_dataset.py',ARCHIVE/'output','--allow-partial',
        '--out',HERE/'checks/refinement02-dataset-audit.json')
    result=json.loads((HERE/'checks/refinement02-dataset-audit.json').read_text())
    cache=None
    if result['complete_dataset']:
        # All source arrays were generated on the cluster. Copy there, never sync
        # regenerated local data; leave original indices byte-identical.
        q=shlex.quote
        command=f'set -euo pipefail; mkdir {q(CACHE)}; cp -a {q(REMOTE+"/output/refinement")} {q(REMOTE+"/output/train")} {q(REMOTE+"/output/validation")} {q(CACHE+"/")}; cd {q(CACHE)}; find refinement train validation -type f -print0 | sort -z | xargs -0 sha256sum > DATA.sha256; sha256sum -c DATA.sha256 --quiet'
        ssh(command)
        relocation=dict(source_job_directory=REMOTE,cache_directory=CACHE,
            calibration_path_map={REMOTE+'/output/refinement/index.json':CACHE+'/refinement/index.json'},
            original_indices_unchanged=True,job_id=JOB,generated_on_cluster=True,
            downstream_cleanup_owner='Burgers lane after FNO owner copies verified cache',
            index_sha256={r['split']:r['index_sha256'] for r in result['splits']},
            reference_index_sha256=result['splits'][0]['calibration_index_sha256'])
        relocation_path=HERE/'checks/refinement02-data-handoff.json'
        relocation_path.write_text(json.dumps(relocation,indent=2)+'\n')
        run('scp',relocation_path,'tufts-login:'+CACHE+'/RELOCATION.json')
        run('scp','tufts-login:'+CACHE+'/DATA.sha256',HERE/'checks/refinement02-data-handoff.sha256')
        # Confirm copied cluster hashes equal the independently verified archive.
        handoff_lines=(HERE/'checks/refinement02-data-handoff.sha256').read_text().splitlines()
        expected_names={str(p.relative_to(ARCHIVE/'output')) for split in ['refinement','train','validation']
                        for p in (ARCHIVE/'output'/split).rglob('*') if p.is_file()}
        assert {line.split('  ',1)[1] for line in handoff_lines}==expected_names
        for line in handoff_lines:
            digest,name=line.split('  ',1)
            import hashlib
            h=hashlib.sha256()
            with (ARCHIVE/'output'/name).open('rb') as stream:
                for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
            assert h.hexdigest()==digest
        cache=CACHE
    run(sys.executable,HERE/'cluster/preserve_archive.py',ATTEMPT)
    save(dict(phase='audited_archived',job_id=JOB,accounting=accounting,dataset_complete=result['complete_dataset'],handoff_cache=cache))
    checks=list((HERE/'checks').glob('refinement02-*.json'))+list((HERE/'checks').glob('refinement02-*.sha256'))
    paths=[ARTIFACT,*checks]
    run('git','add','-f',*paths)
    run('git','commit','--only','-m','Archive and audit Burgers refined references, diagnosis and data',*paths)
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    closing=closing_text(accounting,commit,cache)
    # Explicit exact completed directory; no namespace or other-job deletion.
    ssh(f'rm -rf -- {shlex.quote(REMOTE)} && test ! -e {shlex.quote(REMOTE)}')
    save(dict(phase='complete',job_id=JOB,accounting=accounting,dataset_complete=result['complete_dataset'],
              handoff_cache=cache,archive_commit=commit,exact_job_directory_removed=True))
    run('git','add',STATUS)
    run('git','commit','--only','-m','Record completed Burgers collection and exact cleanup',STATUS)
    append(closing,current=f'**Burgers first-stage collection — finished.** Job `{JOB}` is checksum-collected, audited and Git-archived in the Burgers worktree; its exact completed job directory is removed. Complete shared dataset: `{result["complete_dataset"]}`. The refined independent reference gate passed; the inherited ROM is slower and less accurate than the efficient FOM in the same-job diagnostic. No advantage over FNO or FOM is established. A cluster-generated dataset handoff cache remains at `{cache}` for downstream copying and later cleanup. No additional GPU job or merge was performed by the collector; the broader campaign remains open. See the dated entry for retained evidence and limitations.')
    print(STATUS.read_text(),flush=True)


if __name__=='__main__':
    lock=(HERE/'runs/refinement02/collection-monitor.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:main()
    except BaseException as error:
        traceback.print_exc()
        save(dict(phase='failed_collection_requires_review',job_id=JOB,error=repr(error),remote=REMOTE,
                  caution='No numerical acceptance or remote cleanup may be assumed; inspect monitor log and exact paths'))
        append(f'\n\n## {time.strftime("%Y-%m-%d",time.gmtime())}\n### Burgers collection monitor stopped for review\n\nOwned job `{JOB}` collection encountered `{type(error).__name__}: {error}`. The failure is recorded in `worktrees/2026-09-14-no-burgers/experiments/neural-operator-burgers/checks/refinement02-collection-status.json` and its monitor log. Do not assume collection, numerical acceptance or exact cleanup completed; inspect those records. No additional GPU job was submitted by the monitor.\n')
        raise
