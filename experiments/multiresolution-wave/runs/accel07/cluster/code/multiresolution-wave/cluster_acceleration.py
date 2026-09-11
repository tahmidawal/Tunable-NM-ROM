"""Stage frozen model lineage plus immutable reflective acceleration protocol."""
import argparse
import json
import re
import subprocess
import cluster as transport
import cluster_iterative


def stage(label, configuration):
    cluster_iterative.stage(label)
    dest=transport.CELL/'stages'/label;record=transport.CELL/'runs'/label
    metadata=json.loads((record/'submission.json').read_text())
    cfgpath=transport.CELL/configuration
    payload=subprocess.check_output(['git','show',metadata['source_commit']+':'+str(cfgpath.relative_to(transport.TREE))],cwd=transport.TREE)
    assert payload==cfgpath.read_bytes()
    target=dest/'code/multiresolution-wave'/configuration;target.write_bytes(payload)
    metadata['source_hashes'][str(target.relative_to(dest))]=transport.digest(target)
    text=(dest/'job.sbatch').read_text().replace('iterative_replay.py --config code/multiresolution-wave/iterative-config.json',
        'acceleration_replay.py --config code/multiresolution-wave/'+configuration)
    (dest/'job.sbatch').write_text(text)
    metadata.update(protocol='reflective_wave_geometry_and_timestep_screen',configuration=configuration)
    for p in (dest/'CONFIG.json',record/'submission.json'):transport.write_json(p,metadata)
    (dest/'MANIFEST.sha256').write_text(''.join(f'{transport.digest(p)}  {p.relative_to(dest)}\n' for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='MANIFEST.sha256'))
    print(json.dumps(metadata,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['stage','submit','collect']);ap.add_argument('label')
    ap.add_argument('--config',default='acceleration-screen-config.json');args=ap.parse_args()
    assert re.fullmatch(r'accel[a-z0-9_]{0,19}',args.label)
    if args.action=='stage':stage(args.label,args.config)
    else:getattr(transport,args.action)(args.label)
