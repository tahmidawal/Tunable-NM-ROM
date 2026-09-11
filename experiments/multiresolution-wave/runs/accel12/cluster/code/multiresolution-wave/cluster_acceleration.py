"""Stage frozen model lineage plus immutable reflective acceleration protocol."""
import argparse
import json
import re
import subprocess
import hashlib
import io
import numpy as np
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
    cfg=json.loads(payload)
    assert cfg['primary_method']=='frozen_mlp32_seed691200','primary_method selects the frozen baseline lookup; trained alternatives belong in arms'
    assert not cfg.get('pending_capacity_screen',False),'Freeze the capacity selection before staging fresh development confirmation'
    if cfg.get('frozen_head_inputs'):
        origin=json.loads((dest/'in/ORIGIN.json').read_text())
        origin['new_head_origin']=dict(commit=metadata['source_commit'],sources={},training=[],parent_results={})
        for name,filename in cfg['frozen_head_inputs'].items():
            parent=cfg.get('head_source_runs',{}).get(name,cfg.get('head_source_run'))
            prefix='experiments/multiresolution-wave/runs/'+parent+'/cluster/out/pilot/'
            parent_bytes=subprocess.check_output(['git','show',metadata['source_commit']+':'+prefix+'result.json'],cwd=transport.TREE)
            parent_result=json.loads(parent_bytes)
            origin['new_head_origin']['parent_results'][parent]=dict(result_path=prefix+'result.json',result_sha256=hashlib.sha256(parent_bytes).hexdigest())
            origin['new_head_origin']['training'].extend(parent_result.get('head_training',[]))
            source=prefix+f'head_{name}.npz';head=subprocess.check_output(['git','show',metadata['source_commit']+':'+source],cwd=transport.TREE)
            assert hashlib.sha256(head).hexdigest()==parent_result['output_sha256'][f'head_{name}.npz']
            (dest/'in/dirichlet'/filename).write_bytes(head)
            origin['new_head_origin']['sources'][source]=hashlib.sha256(head).hexdigest()
            initializer=f'initializer_{name}.npz' if f'initializer_{name}.npz' in parent_result['output_sha256'] else 'fixed_encoder_training.npz'
            source=prefix+initializer
            coefficients=subprocess.check_output(['git','show',metadata['source_commit']+':'+source],cwd=transport.TREE)
            assert hashlib.sha256(coefficients).hexdigest()==parent_result['output_sha256'][initializer]
            with np.load(io.BytesIO(coefficients)) as f:
                np.savez_compressed(dest/'in/dirichlet'/f'initializer_{name}.npz',linear=f['linear'],center=f['center'])
            origin['new_head_origin']['sources'][source]=hashlib.sha256(coefficients).hexdigest()
        transport.write_json(dest/'in/ORIGIN.json',origin)
    text=(dest/'job.sbatch').read_text().replace('iterative_replay.py --config code/multiresolution-wave/iterative-config.json',
        'acceleration_replay.py --config code/multiresolution-wave/'+configuration)
    text=text.replace('#SBATCH --time=01:00:00','#SBATCH --time='+cfg.get('walltime','01:00:00'))
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
    elif args.action=='collect':
        config=json.loads((transport.CELL/'stages'/args.label/'code/multiresolution-wave'/json.loads((transport.CELL/'runs'/args.label/'submission.json').read_text())['configuration']).read_text())
        if 1024 in config['meshes']:
            import collect_iterative
            collect_iterative.collect(args.label)
        else:transport.collect(args.label)
    else:getattr(transport,args.action)(args.label)
