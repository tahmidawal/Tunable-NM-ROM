"""jcp-robust E2 3D: write the per-seed train2w.py configs (DESIGN.md 3.1, 3.3, A1.7).

    python make_config.py --seed 1 [--seed 2 ...]    -> configs/train_r1024_s<s>.json
    python make_config.py --pilot                    -> configs/train_r1024_pilot.json   (memory pilot, seed 1)
    python make_config.py --seed 1 --base configs/train_smoke_local.json --out <path>   (CPU chaining test only)

A recipe seed s changes exactly two keys of configs/train_r1024.json (the byte-identical incumbent config):
    model_seed  = 923715 + 1000 s,     subset_seed = 923002 + 1000 s.
Seed 0 reproduces the incumbent's values. Every other key is asserted equal to the base config.

The pilot (smoke, DESIGN 6) additionally sets bank_steps = 200 and warmup = 100. The warmup change is forced: the
bank optimiser is optax.warmup_cosine_decay_schedule(0, lr, warmup, bank_steps), which raises for bank_steps <=
warmup (500 in the recipe). Neither key changes any array shape, so the pilot's memory is the production memory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE / 'configs' / 'train_r1024.json'
MODEL_SEED0, SUBSET_SEED0 = 923715, 923002
PILOT = dict(bank_steps=200, warmup=100)


def seeded(base: dict, s: int, extra: dict | None = None) -> dict:
    cfg = dict(base)
    if 'local_smoke' not in base:        # the incumbent values are the s = 0 member of the family
        assert base['model_seed'] == MODEL_SEED0 and base['subset_seed'] == SUBSET_SEED0, 'unexpected base seeds'
    cfg['model_seed'] = MODEL_SEED0 + 1000 * s
    cfg['subset_seed'] = SUBSET_SEED0 + 1000 * s
    cfg.update(extra or {})
    allowed = {'model_seed', 'subset_seed'} | set(extra or {})
    assert set(cfg) == set(base), 'key set changed'
    diff = {k for k in base if cfg[k] != base[k]}
    assert diff <= allowed, f'unexpected changed keys {diff - allowed}'
    return cfg


def write(cfg: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=1) + '\n')
    print(f'wrote {path}  model_seed={cfg["model_seed"]} subset_seed={cfg["subset_seed"]} bank_steps={cfg["bank_steps"]}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--seed', type=int, action='append', default=[])
    ap.add_argument('--pilot', action='store_true')
    ap.add_argument('--base', default=str(BASE))
    ap.add_argument('--out', default=None, help='output path (only with exactly one --seed)')
    a = ap.parse_args()
    base = json.loads(Path(a.base).read_text())
    if a.out:
        assert len(a.seed) == 1 and not a.pilot
        write(seeded(base, a.seed[0]), Path(a.out))
        return
    assert Path(a.base) == BASE or a.out, 'named outputs are written from the recipe config only'
    for s in a.seed:
        assert s >= 1, 'seed 0 is the incumbent configuration itself (configs/train_r1024.json)'
        write(seeded(base, s), HERE / 'configs' / f'train_r1024_s{s}.json')
    if a.pilot:
        write(seeded(base, 1, PILOT), HERE / 'configs' / 'train_r1024_pilot.json')


if __name__ == '__main__':
    main()
