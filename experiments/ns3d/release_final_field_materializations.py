"""Release coordinator-authorized duplicate final fields after acceptance.

All checkpoints, frozen final inputs, bases, operators, latent histories and
small numerical/provenance records remain materialized.
"""
import datetime,json
from pathlib import Path
from release_archived_materializations import ROOT,sha,verify_available


def main():
    assert Path.cwd().resolve()==ROOT.resolve()
    run=Path('experiments/ns3d/runs/final07');archive=Path('experiments/ns3d/artifacts/final07')
    destination=run/'materialization_release.json';assert not destination.exists()
    assert json.loads((run/'cleanup.json').read_text())['deleted']
    for name in ('audit','history_audit','pod_cold_audit','protocol_audit','parameter_runtime_audit','checksum_audit','input_checksum_audit','source_audit','git_archive_audit'):
        value=json.loads((run/(name+'.json')).read_text());assert value['passed'] and value.get('complete',True),name
    summary=json.loads((run/'paper_summary.json').read_text());assert summary['evaluation_cohort']=='final' and summary['cohort_count']==32
    proof=json.loads((run/'git_archive_audit.json').read_text());commit=proof['archive_commit']
    manifest=json.loads((archive/'archive.json').read_text());assert manifest['passed']
    names=['output/dev_data.npz','output/timed_fields.npz','output/timing_references.npz',*[f'output/refinement/case{i}.npz' for i in range(32)]]
    record=dict(complete=False,authorization='Root explicitly requested verified archive-backed duplicate relief through Heat coordinator at18:45UTC, after final acceptance; no scientific result changes',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=[],logical_bytes_released=0,allocated_bytes_released=0,
        exclusions='All frozen final inputs/checkpoints/POD/operator assets, latent histories, small JSON/source/audit files, archive parts and Git blobs remain materialized',
        accepted_summary_sha256=sha(run/'paper_summary.json'))
    for name in names:
        path=run/'collected'/name
        assert path.resolve().is_relative_to((ROOT/run/'collected/output').resolve())
        assert path.suffix=='.npz' and path.stat().st_size>=64*1024**2
        verify_available(commit,archive,manifest)
        assert sha(path)==manifest['files'][name]
        stat=path.stat();row=dict(path=str(path),archive_member='./'+name,sha256=manifest['files'][name],bytes=stat.st_size,
            allocated_bytes=stat.st_blocks*512 if stat.st_nlink==1 else 0,archive_commit=commit,
            archive_directory=str(archive),actual_git_byte_proof=str(run/'git_archive_audit.json'),removed=False,
            restore_command=f'cat {archive}/collection.tar.part* | tar -xf - -C {run}/collected ./{name}')
        record['files'].append(row);destination.write_text(json.dumps(record,indent=2)+'\n')
        path.unlink();row['removed']=True;record['logical_bytes_released']+=row['bytes'];record['allocated_bytes_released']+=row['allocated_bytes']
        destination.write_text(json.dumps(record,indent=2)+'\n');print('RELEASED',path,flush=True)
    record['complete']=True;record['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    destination.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({k:v for k,v in record.items() if k!='files'}))


if __name__=='__main__':main()
