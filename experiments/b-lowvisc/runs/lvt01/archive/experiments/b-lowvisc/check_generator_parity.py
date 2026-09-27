"""Prove the patched generator is bit-identical to the incumbent's under the defaults.

`experiments/b-lowvisc/deps/burgers2d-coord-rom/burgers2d_film.py` differs from the copy the
incumbent's own training job staged (`experiments/b-seeds/deps/.../burgers2d_film.py`, whose
SHA256 equals the incumbent manifest's entry) only in reading the viscosity bounds from the
environment.  This script loads BOTH modules under the default bounds and asserts, for every
seed and count the training chain uses, that `sample_params` returns the same bytes for
`(cx, cy, w, a, nu)`; and then loads the patched one at the low-viscosity bounds and asserts
the family relation `nu_low == nu_incumbent / 10`.

It also records the ONE place the patch is not bit-identical under the defaults: the `z`
descriptor's log-nu column, whose centre and scale are now derived from the bounds.  `z` is
returned but discarded (`_z`) by `sep_burgers_r3.py:207` and `sep_coeff_extract.py:198,203`,
which are the only consumers in this lane's chain.

    python experiments/b-lowvisc/check_generator_parity.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
PATCHED = HERE / 'deps/burgers2d-coord-rom/burgers2d_film.py'
ORIGINAL = ROOT / 'experiments/b-seeds/deps/burgers2d-coord-rom/burgers2d_film.py'
# (seed, count): every draw the incumbent training recipe makes (b-seeds DESIGN section 4.1)
DRAWS = [(0, 576), (1000, 4032), (1, 8), (0, 128), (7090702, 4), (911702, 2), (17092026, 6)]


def load(path, name, env):
    keep = {k: os.environ.get(k) for k in ('BURGERS_NU_LO', 'BURGERS_NU_HI')}
    os.environ.update({k: str(v) for k, v in env.items()})
    for k in ('BURGERS_NU_LO', 'BURGERS_NU_HI'):
        if k not in env:
            os.environ.pop(k, None)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return mod


def sha(x):
    return hashlib.sha256(np.ascontiguousarray(np.asarray(x)).tobytes()).hexdigest()


def main():
    orig = load(ORIGINAL, 'bf_orig', {})
    same = load(PATCHED, 'bf_same', {})
    low = load(PATCHED, 'bf_low', {'BURGERS_NU_LO': 0.001, 'BURGERS_NU_HI': 0.01})
    rec = dict(
        patched=dict(path=str(PATCHED.relative_to(ROOT)),
                     sha256=hashlib.sha256(PATCHED.read_bytes()).hexdigest()),
        original=dict(path=str(ORIGINAL.relative_to(ROOT)),
                      sha256=hashlib.sha256(ORIGINAL.read_bytes()).hexdigest()),
        bounds=dict(default=[same.NU_LO, same.NU_HI], lowvisc=[low.NU_LO, low.NU_HI]),
        draws=[])
    ok = True
    for seed, m in DRAWS:
        a = orig.sample_params(seed=seed, m=m)
        b = same.sample_params(seed=seed, m=m)
        c = low.sample_params(seed=seed, m=m)
        phys_same = all(np.array_equal(x, y) for x, y in zip(a[:5], b[:5]))
        z_same = bool(np.array_equal(a[5], b[5]))
        desc_same = all(np.array_equal(x, y) for x, y in zip(a[:4], c[:4]))
        ratio = a[4] / c[4]
        dev = float(np.max(np.abs(ratio - 10.) / 10.))
        ok &= phys_same and desc_same and dev < 1e-13
        rec['draws'].append(dict(
            seed=seed, count=m,
            default_bounds_physical_bitwise_identical=phys_same,
            default_bounds_z_bitwise_identical=z_same,
            lowvisc_descriptors_bitwise_identical=desc_same,
            lowvisc_viscosity_ratio_max_relative_deviation_from_ten=dev,
            nu_sha256_original=sha(a[4]), nu_sha256_patched_default=sha(b[4]),
            nu_sha256_patched_lowvisc=sha(c[4])))
    rec['passed'] = bool(ok)
    rec['note'] = ('z is returned but discarded as `_z` by sep_burgers_r3.py and '
                   'sep_coeff_extract.py, the only consumers in this chain, so a last-bit '
                   'change in its log-nu column under the defaults touches no number.')
    out = HERE / 'checks/generator-parity.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2) + '\n')
    print(json.dumps({k: v for k, v in rec.items() if k != 'draws'}, indent=1))
    for d in rec['draws']:
        print(d['seed'], d['count'], 'phys_identical', d['default_bounds_physical_bitwise_identical'],
              'z_identical', d['default_bounds_z_bitwise_identical'],
              'nu_ratio_dev', f"{d['lowvisc_viscosity_ratio_max_relative_deviation_from_ten']:.2e}")
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
