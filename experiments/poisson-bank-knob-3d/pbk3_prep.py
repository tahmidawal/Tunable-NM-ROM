"""Offline rotation of the two frozen banks, from TRAINING data only (no development or final case).

    python pbk3_prep.py lshape  -> runs/prep_lshape.npz
    python pbk3_prep.py cube    -> runs/prep_cube.npz

L-shape: head_sdf_R512_K16 at its training mesh (256 intervals), the fit split of the 3072 training
sources (the split that carries stored codes). Cube: the accepted Poisson3D K16 checkpoint at its
training mesh (32^3), all 512 training sources (every one carries a stored code).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import pbk3_core as P3

HERE = Path(__file__).resolve().parent
EXP = HERE.parent


def lshape():
    import sep_common as sc
    import lsh_core as K_
    ld = EXP / 'hires-poisson' / 'lshape'
    models = {m['id']: m for m in json.loads((ld / 'models.json').read_text())}
    m = models['head_sdf_R512_K16']
    ck, bs = ld / m['checkpoint'], ld / m['basis']
    assert P3.sha_file(ck) == m['checkpoint_sha256'] and P3.sha_file(bs) == m['basis_sha256']
    params, codes, ckcfg = sc.load_pkl(ck)
    basis = np.load(bs)
    codes = np.asarray(codes)
    np.testing.assert_array_equal(codes, basis['training_latents'])
    train, tinfo = K_.cohort(0, 4608, 3072)
    fit, _ = K_.fit_validation_split(len(train), 20260916, 0.15)
    assert len(fit) == len(codes), (len(fit), codes.shape)
    ntr = 256
    geom = K_.Geometry(ntr, 'lshape')
    fom = K_.FOM(geom, build_ic0=False)
    U = K_.fields(fom, train[fit])
    G = K_.bank_of(K_.make_features(ckcfg['factor'], int(ckcfg['n_enrich'])), params, geom)
    H = np.asarray(jax.jit(jax.vmap(lambda z: sc.head(params, z)))(jnp.asarray(codes)))
    # the stored codes reproduce their training fields (sanity: alignment of U and H rows)
    Rg = np.asarray(P3.qr_r(G))
    recon = np.asarray(jnp.asarray(H) @ G.T)
    align = np.linalg.norm(recon - U, axis=1) / np.linalg.norm(U, axis=1)
    shuffled = np.linalg.norm(np.roll(recon, 1, axis=0) - U, axis=1) / np.linalg.norm(U, axis=1)
    T, L, info = P3.make_rotation(G, U, H)
    info.update(model=m['id'], checkpoint_sha256=m['checkpoint_sha256'], basis_sha256=m['basis_sha256'],
                training_intervals=ntr, training_cohort=tinfo, fit_split=dict(seed=20260916, validation_fraction=0.15,
                                                                              fit_count=int(len(fit))),
                stored_code_reconstruction_error=P3.summarise(align),
                control_shuffled_codes_error=P3.summarise(shuffled))
    return T, L, info, dict(Cfull=np.asarray(basis['coefficient_directions']))


def cube():
    import common as C
    import poisson as PP
    pd = EXP / 'paper-p3d' / 'runs' / 'final08' / 'checkpoints'
    shas = dict(bank='6c2298776709973b614b0ab3eed9fcd68d9d39e8d9639d40f1a8da5d0b973c63',
                head='258d28457800b3cceb7eab2e1f1075166f0287a246713034bb0bffb024409e01')
    assert P3.sha_file(pd / 'bank.pkl') == shas['bank'] and P3.sha_file(pd / 'head_K16.pkl') == shas['head']
    bankck = pickle.loads((pd / 'bank.pkl').read_bytes())
    model = pickle.loads((pd / 'head_K16.pkl').read_bytes())
    cfg = bankck['cfg']
    train_p = C.family(cfg['train_seed'], cfg['train_count'])
    coh = json.loads((pd / 'cohorts.json').read_text())
    np.testing.assert_allclose(np.asarray(coh['training_parameters']), train_p, rtol=0, atol=0)
    ntr = int(cfg['train_intervals'])
    U = PP.dataset(ntr, train_p).reshape(len(train_p), -1)
    G = C.bank_at(bankck['params'], ntr, cfg['field_chunk']) @ np.asarray(bankck['rotation'])
    codes = np.asarray(model['codes'])
    assert len(codes) == len(train_p)
    H = np.asarray(C.head(jax.device_put(model['params']), jnp.asarray(codes)))
    recon = H @ G.T
    align = np.linalg.norm(recon - U, axis=1) / np.linalg.norm(U, axis=1)
    shuffled = np.linalg.norm(np.roll(recon, 1, axis=0) - U, axis=1) / np.linalg.norm(U, axis=1)
    T, L, info = P3.make_rotation(G, U, H)
    info.update(model='paper-p3d final08 K16 (bank.pkl + head_K16.pkl)', checkpoint_sha256=shas,
                training_intervals=ntr, training_seed=int(cfg['train_seed']), training_count=int(cfg['train_count']),
                stored_code_reconstruction_error=P3.summarise(align),
                control_shuffled_codes_error=P3.summarise(shuffled))
    return T, L, info, dict(directions=np.asarray(model['directions']))


if __name__ == '__main__':
    which = sys.argv[1]
    T, L, info, extra = {'lshape': lshape, 'cube': cube}[which]()
    out = HERE / 'runs' / f'prep_{which}.npz'
    np.savez(out, T=T, L=L, rotation_info=json.dumps(info), **extra)
    print(json.dumps({k: v for k, v in info.items() if k != 'singular_values'}, indent=1))
    print('wrote', out, P3.sha_file(out))
