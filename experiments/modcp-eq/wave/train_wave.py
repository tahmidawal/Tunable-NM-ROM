"""All three separately trained wave decoder arms from one CP initialization."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import pickle
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import numpy as np
from common.decoders import DecoderConfig, init_decoder, initial_codes, add_modulation
from common.training import train, save_checkpoint
from data import training_data, save_json
from physics import provenance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--boundary', choices=('dirichlet', 'absorbing'), required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    meta = provenance()
    print(json.dumps(meta), flush=True)
    if meta['jax_backend'] != 'gpu' or not meta['x64'] or meta['matmul_precision'] != 'highest':
        raise RuntimeError('GPU/f64/highest required')
    hashes = {str(p.relative_to(Path(__file__).parents[1])): hashlib.sha256(p.read_bytes()).hexdigest()
              for folder in ('common', 'wave') for p in (Path(__file__).parents[1]/folder).glob('*.py')}
    result = {'config': cfg, 'boundary': args.boundary, 'provenance': meta,
              'source_sha256': hashes, 'complete': False, 'evaluation_opened': False}
    save_json(args.out/'training_result.json', result)
    grid, states, scales, manifest = training_data(cfg, args.boundary, args.out/'data')
    dc = DecoderConfig(architecture='cp', k=cfg['latent'], rank=cfg['rank'], outputs=2,
                       intervals=cfg['train_intervals'], boundary='dirichlet' if args.boundary == 'dirichlet' else 'free',
                       width=cfg['modulation_width'], inr_width=cfg['inr_width'])
    coords = jnp.asarray(grid.coordinates().reshape(-1, 2))
    weight = jnp.asarray(grid.mass().ravel())
    params = init_decoder(jax.random.PRNGKey(cfg['model_seed']), dc)
    codes = initial_codes(len(states), dc.k, cfg['model_seed']+1)
    kwargs = dict(states=states, coords=coords, scales=scales, point_weights=weight,
                  batch_size=cfg['batch_size'], point_batch=cfg['point_batch'], lr=cfg['learning_rate'],
                  code_lr_factor=cfg['code_lr_factor'], out_dir=args.out/'checkpoints')
    cp0, z0, hist0 = train(params, codes, config=dc, steps=cfg['pretrain_steps'], seed=cfg['model_seed']+2,
                           stage_name='cp_pretrain', **kwargs)
    cp, zcp, hcp = train(cp0, z0, config=dc, steps=cfg['comparison_steps'], seed=cfg['model_seed']+3,
                         stage_name='cp', **kwargs)
    dm = replace(dc, architecture='modcp')
    mod0 = add_modulation(cp0, dm, jax.random.PRNGKey(cfg['model_seed']+4))
    mod, zmod, hmod = train(mod0, z0, config=dm, steps=cfg['comparison_steps'], seed=cfg['model_seed']+3,
                            stage_name='modcp', **kwargs)
    df = replace(dc, architecture='film')
    film0 = init_decoder(jax.random.PRNGKey(cfg['model_seed']+5), df)
    film, zfilm, hfilm = train(film0, codes, config=df, steps=cfg['pretrain_steps']+cfg['comparison_steps'],
                               seed=cfg['model_seed']+6, stage_name='film', **kwargs)
    for name, p, z, conf in (('cp', cp, zcp, dc), ('modcp', mod, zmod, dm), ('film', film, zfilm, df)):
        save_checkpoint(args.out/'checkpoints'/f'{name}.pkl', p, z, conf,
                        {'fixed_component_scales': scales.tolist(), 'training_seed': cfg['train_seed'],
                         'model_seed': cfg['model_seed'], 'training_data_sha256': manifest['array_sha256'],
                         'nt': int(round(cfg['end_time']/cfg['observation_dt']))+1})
    result.update(complete=True, fixed_component_scales=scales.tolist(), data_manifest='data/training_data.json',
                  checkpoints={a: f'checkpoints/{a}.pkl' for a in ('cp', 'modcp', 'film')})
    save_json(args.out/'training_result.json', result)
    print('WAVE_TRAINING_COMPLETE', args.boundary, flush=True)


if __name__ == '__main__':
    main()
