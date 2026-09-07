"""Read closed-arm checkpoints while their parent campaign continues.

No remote file is edited or removed. Hash before/after pulls, then recheck against
the parent job's final verified archive before accepting the follow-up result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

NAMESPACE = '/cluster/tufts/paralab/tawal01/wave_head_transfer_20260906'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('label')
    parser.add_argument('arm')
    parser.add_argument('--bc', choices=('dirichlet', 'absorbing'), required=True)
    args = parser.parse_args()
    if not re.fullmatch(r'[a-z][a-z0-9_]{0,30}', args.label) or not re.fullmatch(r'(mlp|quadratic)(_velocity)?_69120[01]', args.arm):
        parser.error('unexpected immutable campaign/arm label')
    cell = Path(__file__).resolve().parents[1]
    parent = cell/'runs/fresh_wave_campaign'/args.label
    cfg = json.loads((parent/'submission.json').read_text())
    remote = f'{NAMESPACE}/{args.label}'
    if cfg['remote'] != remote:
        raise RuntimeError('Unexpected namespace')
    boundary = f'out/campaign/{args.bc}'
    arm = f'{boundary}/{args.arm}'
    files = {'bank_tables.npz':f'{boundary}/bank_tables.npz',
             'data_manifest.json':f'{boundary}/data_manifest.json',
             'head.npz':f'{arm}/head.npz', 'reconstruction.npz':f'{arm}/reconstruction.npz',
             'rollouts.npz':f'{arm}/rollouts.npz', 'arm_result.json':f'{arm}/result.json'}
    # result.json is written last, after every file in this completed arm closes.
    command = f'cd {shlex.quote(remote)} && sha256sum '+shlex.join(files.values())
    before = subprocess.check_output(['ssh','tufts-login',command], text=True)
    hashes = {line.split(maxsplit=1)[1].strip():line.split()[0] for line in before.splitlines()}
    destination = parent/'closed_checkpoints'/args.arm
    destination.mkdir(parents=True, exist_ok=False)
    for name, path in files.items():
        subprocess.run(['scp','-q',f'tufts-login:{remote}/{path}',str(destination/name)], check=True)
        if hashlib.sha256((destination/name).read_bytes()).hexdigest() != hashes[path]:
            raise RuntimeError('Checkpoint changed during transfer')
    after = subprocess.check_output(['ssh','tufts-login',command], text=True)
    if before != after:
        raise RuntimeError('Closed-arm files changed during transfer')
    result = json.loads((destination/'arm_result.json').read_text())
    if result['name']+'_'+str(result['optimizer_seed']) != args.arm:
        raise RuntimeError('Wrong arm identity')
    origin = dict(parent_label=args.label, parent_job=cfg['job_id'], source_commit=cfg['source_commit'],
                  remote=remote, arm=args.arm, bc=args.bc, hashes_before_after_equal=True,
                  files={name:dict(remote_relative=path,sha256=hashes[path]) for name,path in files.items()},
                  parent_campaign_still_running=True,
                  acceptance='Checkpoint transfer only; reconcile every hash with the final parent archive before scientific acceptance.')
    (destination/'ORIGIN.json').write_text(json.dumps(origin, indent=2)+'\n')
    print(str(destination))


if __name__ == '__main__':
    main()
