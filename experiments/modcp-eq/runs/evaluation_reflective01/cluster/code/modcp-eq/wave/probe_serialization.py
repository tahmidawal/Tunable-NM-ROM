"""Bounded local I/O probe using frozen TRAINING codes, never evaluation cases."""
import argparse
import json
from pathlib import Path
import pickle
import sys
import tempfile
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jnp
import numpy as np
from common.decoders import DecoderConfig, decode_grid
from physics import provenance
from fields import file_sha
from data import save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    with args.checkpoint.open('rb') as stream:
        ck = pickle.load(stream)
    dc = DecoderConfig(**ck['config'])
    p = jax.tree.map(jnp.asarray, ck['params'])
    decode = jax.jit(lambda p, z: jax.lax.map(lambda a: decode_grid(p, a, 512, dc)[1:-1, 1:-1], z))
    fields = np.asarray(decode(p, jnp.asarray(ck['Z'][:49])))
    result = {'provenance': provenance(), 'training_codes_only': True, 'evaluation_generated': False,
              'shape': list(fields.shape), 'checkpoint_sha256': file_sha(args.checkpoint),
              'scope': 'Local serialization implementation probe; not cluster or query timing evidence.', 'methods': {}}
    with tempfile.TemporaryDirectory(dir=args.out) as tmp:
        for name, write in [('stored', np.savez), ('deflate', np.savez_compressed)]:
            path = Path(tmp)/(name+'.npz')
            start = time.perf_counter()
            write(path, u=fields[..., 0], v=fields[..., 1])
            seconds = time.perf_counter()-start
            with np.load(path) as saved:
                exact = np.array_equal(saved['u'], fields[..., 0]) and np.array_equal(saved['v'], fields[..., 1])
            result['methods'][name] = {'seconds': seconds, 'bytes': path.stat().st_size, 'exact': exact}
            if not exact:
                raise RuntimeError('Lossless serialization changed a field')
    save_json(args.out/'result.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
