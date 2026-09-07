"""Copy a verified fresh job's scientific outputs into this owned worktree.

The coordinator retains the complete once-only split archive. This importer
verifies every copied scientific file against its pulled SHA-256 manifest.
"""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

COORDINATOR=Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude/worktrees/2026-09-06-burgers3d-repair/experiments/separable-decoder/runs/fresh_wave_campaign')
CELL=Path(__file__).resolve().parent


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--label',required=True)
    ap.add_argument('--allow-incomplete',action='store_true')
    ap.add_argument('--type',choices=('campaign','refinement'),default='campaign')
    args=ap.parse_args()
    if not re.fullmatch(r'[a-z][a-z0-9_]*',args.label):
        raise ValueError('Unsafe job label')
    source=(COORDINATOR/args.label).resolve()
    if source.parent!=COORDINATOR.resolve():
        raise ValueError('Unexpected source namespace')
    submission=json.loads((source/'submission.json').read_text())
    expected_remote='/cluster/tufts/paralab/tawal01/wave_head_transfer_20260906/'+args.label
    entry='fresh_campaign.py' if args.type=='campaign' else 'fresh_checkpoint_refine.py'
    if submission['remote']!=expected_remote or submission['entry']!=entry:
        raise ValueError('Not a fresh scientific wave campaign job')
    cluster=source/'cluster'
    output_prefix=Path('out')/args.type
    outputs=cluster/output_prefix
    result=json.loads((outputs/'result.json').read_text())
    if result.get('final_test_opened') or result['provenance']['jax_backend']!='gpu' or not result['provenance']['x64'] or result['provenance']['matmul_precision']!='highest':
        raise ValueError('Cohort/backend/precision provenance failed')
    if result['provenance']['source_commit']!=submission['source_commit'] or str(result['provenance']['job_id'])!=str(submission['job_id']):
        raise ValueError('Scientific result and submission lineage differ')
    run_completed=result.get('completed',False) if args.type=='campaign' else result.get('old_fine_parity_passed',False)
    exit_ok=(cluster/'EXIT_CODE').read_text().strip()=='0'
    if not args.allow_incomplete and not (run_completed and exit_ok):
        raise ValueError('Job is not a completed campaign; preserve failed attempts explicitly')
    expected={}
    for line in (cluster/'PULL.sha256').read_text().splitlines():
        if not line.strip():
            continue
        checksum,name=line.split(maxsplit=1)
        expected[name.lstrip('*').removeprefix('./')]=checksum
    files=[]
    for path in sorted(outputs.rglob('*')):
        if path.is_symlink():
            raise ValueError('Unexpected symbolic link in job output')
        if not path.is_file():
            continue
        relative=path.relative_to(cluster).as_posix()
        if relative not in expected or digest(path)!=expected[relative]:
            raise ValueError('Scientific output checksum mismatch: '+relative)
        if path.stat().st_size>=100*1024*1024:
            raise ValueError('Split oversized artifact before importing: '+relative)
        files.append(path)
    destination=CELL/'runs'/args.label
    temporary=CELL/'runs'/('.'+args.label+'.importing')
    if destination.exists() or temporary.exists():
        raise FileExistsError('Import directory already exists')
    temporary.mkdir(parents=True)
    copied={}
    for path in files:
        relative=output_prefix/path.relative_to(outputs)
        target=temporary/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
        checksum=digest(target)
        if checksum!=expected[path.relative_to(cluster).as_posix()]:
            raise RuntimeError('Post-copy checksum mismatch')
        copied[relative.as_posix()]=checksum
    for path in cluster.iterdir():
        if path.is_file():
            shutil.copyfile(path,temporary/path.name)
    for name in ('submission.json','ARCHIVE.json','ARCHIVE.sha256'):
        if (source/name).exists():
            shutil.copyfile(source/name,temporary/name)
    metadata={'source_coordinator_directory':str(source),'complete_raw_archive_branch':'exp/2026-09-06-burgers3d-repair','job_id':submission['job_id'],'source_commit':submission['source_commit'],'copied_scientific_files':copied,'post_copy_checksums_passed':True,'type':args.type,'run_completed':run_completed,'exit_ok':exit_ok}
    (temporary/'IMPORT.json').write_text(json.dumps(metadata,indent=2)+'\n')
    temporary.rename(destination)
    print(json.dumps({'destination':str(destination),'files':len(copied),'checksums_passed':True}))


if __name__=='__main__':
    main()
