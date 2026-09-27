"""Stage the fresh-wave device contract; reuse checked submit/collect transport."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import numpy as np
import cluster as transport

CELL = Path(__file__).resolve().parent
SOURCE_COMMIT = '77af541a5995e3870583e6e957f9579d116a2760'


def stage(label):
    transport.stage(label)
    dest = CELL/'stages'/label
    record = CELL/'runs'/label
    metadata = json.loads((record/'submission.json').read_text())
    cfgpath = CELL/'device-replay-config.json'
    payload = subprocess.check_output(['git', 'show', metadata['source_commit']+':'+str(cfgpath.relative_to(transport.TREE))], cwd=transport.TREE)
    assert payload==cfgpath.read_bytes()
    target = dest/'code/multiresolution-wave'/cfgpath.name
    target.write_bytes(payload)
    metadata['source_hashes'][str(target.relative_to(dest))] = transport.digest(target)
    origin = json.loads((dest/'in/ORIGIN.json').read_text())
    prior = CELL/'runs/k32heads03/cluster/out/pilot'
    resultpath = 'experiments/multiresolution-wave/runs/k32heads03/cluster/out/pilot/result.json'
    priorbytes = subprocess.check_output(['git', 'show', SOURCE_COMMIT+':'+resultpath], cwd=transport.TREE)
    priorresult = json.loads(priorbytes)
    origin['head32_origin'] = dict(commit=SOURCE_COMMIT, result_path=resultpath,
        result_sha256=hashlib.sha256(priorbytes).hexdigest(), sources={})
    for bc in ('dirichlet', 'absorbing'):
        path = f'head_{bc}_new_mlp32_seed691200/head.npz'
        rel = str((prior/path).relative_to(transport.TREE))
        payload = subprocess.check_output(['git', 'show', SOURCE_COMMIT+':'+rel], cwd=transport.TREE)
        assert hashlib.sha256(payload).hexdigest()==priorresult['output_sha256'][path]
        (dest/'in'/bc/'head32.npz').write_bytes(payload)
        origin['head32_origin']['sources'][rel] = hashlib.sha256(payload).hexdigest()
        ladder = prior/f'training_ladder_{bc}.npz'
        expected = priorresult['output_sha256'][ladder.name]
        assert transport.digest(ladder)==expected
        with np.load(ladder) as f:
            np.savez_compressed(dest/'in'/bc/'initializer32.npz',
                                 linear=f['standardized_linear'][:, :32], center=f['center'])
        origin['head32_origin']['sources'][str(ladder.relative_to(transport.TREE))] = expected
    origin['initializer32_note'] = 'Only the frozen fitted PCA map and center are extracted from checksum-audited training lineage. No training trajectories or coefficient snapshots are uploaded; there is no retraining. Query fields and references regenerate from seed on the cluster.'
    transport.write_json(dest/'in/ORIGIN.json', origin)
    batch = (dest/'job.sbatch').read_text().replace('#SBATCH --time=02:00:00', '#SBATCH --time=01:00:00')
    batch = batch.replace('code/multiresolution-wave/pilot.py --config code/multiresolution-wave/config.json',
                          'code/multiresolution-wave/device_replay.py --config code/multiresolution-wave/device-replay-config.json')
    (dest/'job.sbatch').write_text(batch)
    metadata['protocol'] = 'fresh_verified_wave_device_resident_full_fields'
    metadata['requires_coordinator_review_before_submit'] = True
    transport.write_json(dest/'CONFIG.json', metadata)
    transport.write_json(record/'submission.json', metadata)
    (dest/'MANIFEST.sha256').write_text(''.join(f'{transport.digest(p)}  {p.relative_to(dest)}\n'
        for p in sorted(dest.rglob('*')) if p.is_file() and p.name!='MANIFEST.sha256'))
    print(json.dumps(metadata, indent=2))


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=('stage', 'submit', 'collect'))
    ap.add_argument('label')
    args = ap.parse_args()
    if not re.fullmatch(r'device[a-z0-9_]{0,18}', args.label):
        ap.error('Use a private device-prefixed attempt label')
    (stage if args.action=='stage' else getattr(transport, args.action))(args.label)
