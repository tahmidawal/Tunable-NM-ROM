"""Retain compact records, trained models and final replay assets in Git.

The full independently collected directory stays on local disk under its two
checksum manifests. This command never deletes local or remote experiment data.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    parser=argparse.ArgumentParser();parser.add_argument('attempt');args=parser.parse_args()
    exp=Path(__file__).resolve().parent;root=exp.parents[1];run=exp/'runs'/args.attempt;archive=run/'collected'
    assert (archive/'COLLECTION.sha256').exists() and (archive/'OUTPUTS.sha256').exists()
    kept=[]
    for path in run.rglob('*'):
        if not path.is_file() or path.name=='RETENTION.json':continue
        relative=path.relative_to(run)
        if relative.parts[0]=='stage' or '__pycache__' in relative.parts:continue
        if len(relative.parts)>1 and relative.parts[:2]==('collected','code'):continue
        if path.suffix in ('.json','.sha256','.log','.out','.err','.sbatch') or path.name in (
                'STARTED_UTC','FINISHED_UTC','EXIT_CODE','best.pkl','checkpoint.pkl','bank.pkl','directions.npz','bases.npz'):
            kept.append(path)
    for panel in (archive/'out').glob('seed*'):
        if not panel.is_dir() or not (panel/'result.json').exists():continue
        result=json.loads((panel/'result.json').read_text())
        for row in result['invocations']+result.get('operator_invocations',[]):
            if row['row'] in (512,519) and row['repetition']==0:kept.append(panel/row['artifact'])
        for row in (512,519):
            if row in result['config']['validation_rows']:
                kept.append(panel/f"reference_case{result['config']['validation_rows'].index(row)}.npz")
    kept=sorted(set(kept));records=[]
    for path in kept:
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        records.append(dict(path=str(path.relative_to(root)),sha256=digest,bytes=path.stat().st_size))
    retention=dict(full_local_archive=str(archive),checksums='collected/COLLECTION.sha256',
        scientific_checksums='collected/OUTPUTS.sha256',git_retained=records,
        dense_arrays_and_optimizer_states='retained in the complete local checksum-covered archive; no original file is removed',
        final_replay_assets='exact basis/direction arrays and repetition-zero examples at solver slots 512 and 519 retained in Git; each panel metadata identifies development versus final membership. Complete replay/prediction/reference fields are retained by the separate scientific split archive')
    target=run/'RETENTION.json';target.write_text(json.dumps(retention,indent=2)+'\n')
    subprocess.run(['git','-C',str(root),'add','-f','--',*[str(p.relative_to(root)) for p in kept],str(target.relative_to(root))],check=True)
    print(json.dumps(dict(files=len(records),bytes=sum(v['bytes'] for v in records),retention=str(target)),indent=2))


if __name__=='__main__':main()
