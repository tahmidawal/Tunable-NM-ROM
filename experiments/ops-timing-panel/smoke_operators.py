"""Local smoke for the operator phase: every checkpoint in `operators.json` loads through the
no-second adapter and answers one query with the right shape, dtype and boundary mask.

    jaxrun /home/tahmid/Dev/.venv/bin/python smoke_operators.py

This is the only genuinely new path in the lane -- `fno_panel.py`, the panel and the audit are
copied unchanged apart from the multi-arm list -- so it is the one thing worth proving before a
cluster job. It does NOT check accuracy or speed: the GB10 is a different GPU and the numbers
from it are meaningless. Sub-minute, local, per CLAUDE.md.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WORKTREES = HERE.parents[2]
sys.path.insert(0, str(HERE / 'lib'))
import dataset  # noqa: E402
import model as adapter  # noqa: E402

L = 256                                  # the panel's mesh: 257 x 257 nodes
TIMES = 6                                # u(t0) .. u(t5)


def main():
    specs = json.loads((HERE / 'operators.json').read_text())['checkpoints']
    env = adapter.configure()
    print('environment', env)
    rng = np.random.default_rng(0)
    field = rng.standard_normal((1, 1, L + 1, L + 1))
    field[..., 0, :] = field[..., -1, :] = field[..., :, 0] = field[..., :, -1] = 0.
    parameters = np.array([[0.01]])          # burgers: the viscosity only (dataset.validate_case)
    ok = True
    for spec in specs:
        path = WORKTREES / spec['path'].removeprefix('worktrees/')
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        assert got == spec['sha256'], (spec['name'], got, spec['sha256'])
        ckpt = torch.load(path, map_location='cuda', weights_only=False)
        net = adapter.make_model(ckpt['pde'], ckpt['config'])
        net.load_state_dict(ckpt['model'])
        adapter.check_dtypes(net)
        net.eval()
        norm = tuple(v.cuda() for v in ckpt['normalization'])
        u0 = torch.from_numpy(field).cuda()
        pp = torch.from_numpy(parameters).cuda()
        with torch.no_grad():
            traj = adapter.predict(net, u0, pp, *norm, ckpt['pde'])
        f = traj.cpu().numpy()[0, :, 0]
        shape_ok = f.shape == (TIMES, L + 1, L + 1)
        t0_ok = np.array_equal(f[0], field[0, 0])
        mask_ok = not (f[:, 0, :].any() or f[:, -1, :].any() or f[:, :, 0].any() or f[:, :, -1].any())
        f64_ok = f.dtype == np.float64
        params = sum(p.numel() * (2 if p.is_complex() else 1) for p in net.parameters())
        good = shape_ok and t0_ok and mask_ok and f64_ok and np.isfinite(f).all()
        ok &= good
        print(f"{spec['name']:12s} family={spec['family']:10s} pde={ckpt['pde']} epoch={ckpt['epoch']:5d} "
              f"params={params:9d} shape={f.shape} t0_exact={t0_ok} masked={mask_ok} f64={f64_ok} "
              f"sha256_verified=True -> {'OK' if good else 'FAIL'}", flush=True)
        del net, ckpt
        torch.cuda.empty_cache()
    print('SMOKE', 'PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
