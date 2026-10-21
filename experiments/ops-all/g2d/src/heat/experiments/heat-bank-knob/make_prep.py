"""Offline rotation of a frozen heat bank from TRAINING data only (hbk_core.make_rotation) -> prep_<name>.npz + JSON.
Usage: make_prep.py <name> <bank.pkl rel. to inputs> <head.pkl rel. to inputs>"""
import json, sys
from pathlib import Path
import numpy as np
import jax
import core as C
import hbk_core as K

here = Path(__file__).resolve().parent
name, bank, head = sys.argv[1:4]
model = C.load_model(dict(kind='mlp', bank=bank, head=head), here / 'inputs')
tcfg = json.loads((here / 'inputs' / Path(bank).parent / 'training.json').read_text())
T, L, info = K.make_rotation(model, tcfg['config'])
assert info['training_draws_sha'] == tcfg['train_draws_sha'], (info['training_draws_sha'], tcfg['train_draws_sha'])
info.update(model_sha256=model['sha256'], backend=jax.default_backend(), device=str(jax.devices()[0]))
np.savez(here / f'prep_{name}.npz', T=T, L=L, rotation_info=json.dumps(info))
C.dump(here / f'prep_{name}.json', info)
print(json.dumps({k: v for k, v in info.items() if k != 'singular_values'}, indent=1))
