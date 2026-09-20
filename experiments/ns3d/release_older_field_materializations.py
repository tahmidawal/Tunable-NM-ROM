"""Release only explicitly selected old, fully archived field/data duplicates.

No checkpoint, POD basis, solver matrix, active final input or small metadata is
eligible. Complete actual-Git archive byte proofs precede this operation.
"""
import datetime, hashlib, io, json, subprocess, tarfile
from pathlib import Path
from release_archived_materializations import ROOT, sha, verify_available
from retain_archive import Joined


def main():
    assert Path.cwd().resolve() == ROOT.resolve()
    destination = Path('experiments/ns3d/runs/older_field_materialization_release.json')
    assert not destination.exists(), 'Never overwrite a release record'
    selected = {
        'comparison02': ['output/dev_data.npz', 'output/reference_N32/reference_fields.npz',
            'output/reference_N64/reference_fields.npz', 'output/representation_fields.npz',
            'output/timed_fields.npz', 'output/timing_references.npz', 'output/train_data.npz'],
        'extra03': ['output/dev_data.npz', 'output/timed_fields.npz',
            'output/timing_references.npz', 'output/train_data.npz'],
        'confirmation06b': ['output/dev_training_selection.npz', 'output/timing_references.npz',
            'output/train_data.npz', *[f'output/refinement/case{i}.npz' for i in range(8)],
            *['output/deeponet_candidate/pretraining/'+name+'.npz' for name in
              ('branch_selected_coefficients', 'branch_teacher', 'training_teacher')]],
    }
    record = dict(complete=False, authorization='Parent approved own-tree archived duplicate materialization relief; Heat coordinator reaffirmed old NS field/data scope',
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), files=[],
        logical_bytes_released=0, allocated_bytes_released=0,
        exclusions='All checkpoints, POD bases, solver operators, current/final inputs, all small JSON/source/audit files and all archives remain materialized')
    for attempt, names in selected.items():
        run=Path('experiments/ns3d/runs')/attempt; archive=Path('experiments/ns3d/artifacts')/attempt
        proof=json.loads((run/'git_archive_audit.json').read_text())
        assert proof['passed'] and proof['complete']; commit=proof['archive_commit']
        manifest=json.loads((archive/'archive.json').read_text()); assert manifest['passed']
        verify_available(commit,archive,manifest)
        if 'files' in manifest:
            index=manifest['files']
        else:
            # Legacy archive manifests did not retain a per-file index. Recover
            # every member from the archive whose actual Git bytes just passed.
            index={}
            with io.BufferedReader(Joined([archive/p['path'] for p in manifest['parts']])) as stream, tarfile.open(fileobj=stream,mode='r|') as bundle:
                for member in bundle:
                    if not member.isfile(): continue
                    h=hashlib.sha256(); f=bundle.extractfile(member)
                    while block:=f.read(8*1024**2): h.update(block)
                    index[member.name.removeprefix('./')]=h.hexdigest()
            (run/'archive_file_index.json').write_text(json.dumps(dict(complete=True,archive_commit=commit,files=index),indent=2)+'\n')
        candidates=[(run/'collected'/name,name) for name in names]
        if attempt=='confirmation06b':
            aliases=['output/train_data.npz', *['output/deeponet_candidate/pretraining/'+name+'.npz'
                for name in ('branch_selected_coefficients','branch_teacher','training_teacher')]]
            candidates.extend((run/'pretraining_audit_input'/name,name) for name in aliases)
        for path,name in candidates:
            assert path.suffix=='.npz' and path.stat().st_size>=64*1024**2
            assert path.resolve().is_relative_to((ROOT/run).resolve())
            assert not path.resolve().is_relative_to((ROOT/'experiments/ns3d/runs/final07').resolve())
            verify_available(commit,archive,manifest)
            assert name in index and sha(path)==index[name]
            stat=path.stat()
            restore=f'cat {archive}/collection.tar.part* | tar -xf - -C {run}/collected ./{name}'
            if 'pretraining_audit_input' in path.parts:
                restore=f'mkdir -p {path.parent} && cat {archive}/collection.tar.part* | tar -xOf - ./{name} > {path}'
            row=dict(path=str(path),archive_member='./'+name,sha256=index[name],bytes=stat.st_size,
                allocated_bytes=stat.st_blocks*512 if stat.st_nlink==1 else 0,
                archive_commit=commit,archive_directory=str(archive),
                actual_git_byte_proof=str(run/'git_archive_audit.json'),restore_command=restore,removed=False)
            record['files'].append(row); destination.write_text(json.dumps(record,indent=2)+'\n')
            path.unlink(); row['removed']=True
            record['logical_bytes_released']+=row['bytes'];record['allocated_bytes_released']+=row['allocated_bytes']
            destination.write_text(json.dumps(record,indent=2)+'\n'); print('RELEASED',path,flush=True)
    record['complete']=True;record['completed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    destination.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k!='files'}))


if __name__=='__main__': main()
