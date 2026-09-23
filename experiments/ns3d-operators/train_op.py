"""Train ONE neural-operator arm on the NS 3D family at ONE mesh (one cluster job per arm).

Protocol (the Table-2 cells' operator protocol, DESIGN.md section 4):
* data: the NM-ROM's own training seed 202609201, the 512 trajectories its head was
  trained on (prefix-preserving draw), truth = CNAB2 dt 0.001 at the target mesh, six
  frames t = 0, 0.04, ..., 0.2 (the bank/dev solver, `diag_floor.generate`), regenerated
  here from the seed. Split = the head's split: case index = 7 (mod 8) is validation
  (64 cases), the rest (448) is training. The development cohort (seed 202609202) is never
  generated here; its parameter rows are asserted disjoint.
* AdamW(lr 1e-3, weight decay 1e-4), batch 8 (micro-batches + gradient accumulation if 8
  does not fit), gradient clip 1.0, cosine learning rate in ELAPSED WALL TIME to 1e-5,
  Transolver warm-up 160 optimisation steps, 3000 s wall budget, epoch cap 4000,
  patience 250 epochs; checkpoint = best validation mean over cases of the per-case max
  evolved initial-relative error (the 2D cells' criterion).
* loss: mean over batch and the five evolved outputs of the squared initial-relative
  error ||pred_t - u_t||^2 / ||u0||^2 (the metric's normalisation), float64.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'experiments' / 'ns3d'))
sys.path.insert(0, str(HERE))

TRAIN_SEED, TRAIN_CASES, VAL_EVERY = 202609201, 512, 8
DEV_SEED, DEV_CASES = 202609202, 16
HELDOUT_SEED, HELDOUT_CASES = 202609221, 32
DT_TRUTH, HORIZON = 0.001, 0.2
STOP = False


def on_signal(signum, frame):
    global STOP
    STOP = True


def dump(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=1) + '\n')
    tmp.replace(path)


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(n, smoke=False):
    """JAX CNAB2 truth for the training seed (run before PyTorch touches the GPU)."""
    import jax
    jax.config.update('jax_enable_x64', True)
    jax.config.update('jax_default_matmul_precision', 'highest')
    import jax.numpy as jnp
    import ns3d_fom as F
    count = 16 if smoke else TRAIN_CASES
    dt = 0.005 if smoke else DT_TRUTH
    horizon = 0.05 if smoke else HORIZON
    params = F.parameters(TRAIN_SEED, count)
    dev = np.round(F.parameters(DEV_SEED, DEV_CASES), 9)
    held = np.round(F.parameters(HELDOUT_SEED, HELDOUT_CASES), 9)
    rows = {tuple(r) for r in np.round(params, 9)}
    overlap = dict(dev=int(sum(tuple(r) in rows for r in dev)), heldout=int(sum(tuple(r) in rows for r in held)))
    if any(overlap.values()):
        raise RuntimeError(f'training rows overlap an evaluation cohort: {overlap}')
    steps = int(round(horizon / dt))
    solver = F.make_solver(dt, steps, steps // 5)
    geom = F.geometry(n)
    frames = np.empty((count, 6, 3, n, n, n), dtype=np.float32)
    t0 = time.time()
    for i, p in enumerate(params):
        frames[i] = np.asarray(solver(jnp.asarray(F.initial(n, p)), float(p[-1]), geom), dtype=np.float32)
        if (i + 1) % 64 == 0:
            print(f'generated {i + 1}/{count} ({time.time() - t0:.0f}s)', flush=True)
    if not np.isfinite(frames).all():
        raise RuntimeError('nonfinite training truth')
    del solver, geom
    jax.clear_caches()
    return params, frames, overlap, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mesh', type=int, required=True)
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--wall-seconds', type=float, default=3000.)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    if not args.smoke and not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('production training runs in a cluster allocation')
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGUSR1, on_signal)
    n = args.mesh
    config = json.loads(args.config.read_text())
    name = args.config.stem
    if config['family'] == 'transolver':
        config['patch'] = max(1, n // 16)  # token grid 16^3 at every mesh
    if args.smoke:
        config = dict(config, **config.get('smoke_overrides', {}))
    args.out.mkdir(parents=True, exist_ok=True)

    params, frames, overlap, gen_seconds = generate(n, args.smoke)
    import torch
    import ops3d as O
    env = O.configure()
    count = len(params)
    is_val = (np.arange(count) % VAL_EVERY) == VAL_EVERY - 1
    tr_idx, va_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    # training data resident on the GPU in float32 (quantisation ~1e-7 relative, orders
    # below any operator error); every loss / error is computed in float64
    on_device = frames.nbytes < 0.35 * torch.cuda.get_device_properties(0).total_memory
    U = torch.from_numpy(frames)
    U = U.cuda() if on_device else U.pin_memory()
    del frames
    NU = torch.as_tensor(params[:, -1], dtype=torch.float64, device='cuda')
    s0 = s1 = 0.0
    c0 = c1 = 0
    for i in tr_idx:
        f = U[i].to('cuda').double()
        s0 += float(f[0].square().sum()); c0 += f[0].numel()
        s1 += float(f[1:].square().sum()); c1 += f[1:].numel()
    u_scale = torch.tensor(math.sqrt(s0 / c0), dtype=torch.float64, device='cuda')
    out_scale = torch.tensor(math.sqrt(s1 / c1), dtype=torch.float64, device='cuda')
    lognu = torch.log(NU[tr_idx])
    mu, sd = lognu.mean(), lognu.std().clamp_min(1e-12)
    norm = (u_scale, mu, sd, out_scale)
    projector = O.Projector(n)

    def get(ids):
        idc = torch.as_tensor(ids, device='cuda')
        batch = (U[idc] if on_device else U[torch.as_tensor(ids)].to('cuda', non_blocking=True)).double()
        return batch[:, 0], NU[idc], batch

    def evaluate(net):
        net.eval()
        errs = []
        with torch.no_grad():
            for i in va_idx:
                u0, nu, y = get([i])
                errs.append(O.initial_relative(O.predict(net, u0, nu, norm, projector), y, u0)[0].cpu().numpy())
        e = np.asarray(errs)
        if not np.isfinite(e).all():
            raise RuntimeError('nonfinite validation error')
        per = e[:, 1:].max(1)
        return dict(errors=e.tolist(), mean_case_max=float(per.mean()), median_case_max=float(np.median(per)),
                    worst_case_max=float(per.max()))

    def loss_of(net, ids):
        u0, nu, y = get(ids)
        e = O.initial_relative(O.predict(net, u0, nu, norm, projector), y, u0)[:, 1:]
        return e.square().mean()

    seed = int(config['seed'])
    batch_size = int(config['batch_size'])
    micro = None
    probe = {}
    for m in ((1,) if args.smoke else (8, 4, 2, 1)):
        m = min(m, batch_size)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(seed)
        net = O.make_model(config)
        try:
            loss_of(net, tr_idx[:m]).backward()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            probe[str(m)] = peak
            del net
            torch.cuda.empty_cache()
            if peak < 0.80 * torch.cuda.get_device_properties(0).total_memory or m == 1:
                micro = m
                break
        except torch.OutOfMemoryError:
            probe[str(m)] = 'oom'
            del net
            torch.cuda.empty_cache()
    if micro is None:
        raise RuntimeError('no micro-batch fits')

    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    net = O.make_model(config)
    O.check_dtypes(net)
    opt = torch.optim.AdamW(net.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    lr0, floor = float(config['learning_rate']), 1e-5
    warm = int(config.get('warmup_steps', 0))
    gen = torch.Generator().manual_seed(seed)
    wall = float(args.wall_seconds)
    best, best_epoch, history, step, stop_reason = float('inf'), -1, [], 0, 'epoch_cap'
    patience = int(config.get('patience', 250))
    provenance = dict(environment=env, mesh=n, arm=name, config=config, config_sha256=sha_file(args.config),
                      sources={p.name: sha_file(p) for p in sorted(HERE.glob('*.py'))},
                      ns3d_fom_sha256=sha_file(ROOT / 'experiments' / 'ns3d' / 'ns3d_fom.py'),
                      train_seed=TRAIN_SEED, train_cases=count, split='index % 8 == 7 -> validation',
                      train_count=int(len(tr_idx)), validation_count=int(len(va_idx)),
                      overlap_with_evaluation_cohorts=overlap, generation_seconds=gen_seconds,
                      parameter_sha256=hashlib.sha256(np.ascontiguousarray(params).tobytes()).hexdigest(),
                      normalization=[float(v) for v in norm], data_on_device=bool(on_device), micro_batch=micro, probe_peak_bytes=probe,
                      wall_budget_seconds=wall, job_id=os.environ.get('SLURM_JOB_ID'),
                      node=os.environ.get('SLURMD_NODENAME'), smoke=bool(args.smoke),
                      real_parameter_count=O.parameter_count(net))
    dump(args.out / 'provenance.json', provenance)
    print(json.dumps({k: provenance[k] for k in ('arm', 'mesh', 'micro_batch', 'real_parameter_count')}), flush=True)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(int(config['epochs'])):
        net.train()
        order = tr_idx[torch.randperm(len(tr_idx), generator=gen).numpy()]
        losses = []
        for s in range(0, len(order), batch_size):
            ids = order[s:s + batch_size]
            frac = min(1.0, (time.monotonic() - started) / wall)
            lr = floor + (lr0 - floor) * 0.5 * (1 + math.cos(math.pi * frac))
            if step < warm:
                lr = lr * (step + 1) / warm
            for g in opt.param_groups:
                g['lr'] = lr
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for m0 in range(0, len(ids), micro):
                sub = ids[m0:m0 + micro]
                loss = loss_of(net, sub) * len(sub) / len(ids)
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite training loss')
                loss.backward()
                total += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            step += 1
            losses.append(total)
            if STOP or time.monotonic() - started >= wall:
                break
        metrics = evaluate(net)
        if metrics['mean_case_max'] < best:
            best, best_epoch = metrics['mean_case_max'], epoch
            O.save_checkpoint(args.out / 'best.tmp', net, config, norm,
                              dict(epoch=epoch, step=step, best=best, mesh=n, arm=name))
            (args.out / 'best.tmp').replace(args.out / 'best.pt')
        history.append(dict(epoch=epoch, steps=step, train_loss=float(np.mean(losses)),
                            validation={k: v for k, v in metrics.items() if k != 'errors'},
                            learning_rate=lr, wall_seconds=time.monotonic() - started, best_so_far=best,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        dump(args.out / 'history.json', history)
        print(json.dumps(dict(arm=name, epoch=epoch, steps=step, train=history[-1]['train_loss'],
                              val_mean=metrics['mean_case_max'], val_worst=metrics['worst_case_max'],
                              lr=lr, wall=round(history[-1]['wall_seconds'], 1))), flush=True)
        if STOP:
            stop_reason = 'signal'; break
        if time.monotonic() - started >= wall:
            stop_reason = 'wall_budget'; break
        if epoch - best_epoch >= patience:
            stop_reason = 'patience'; break
    seconds = time.monotonic() - started
    net2, norm2, ck = O.load_checkpoint(args.out / 'best.pt')
    final = evaluate(net2)
    result = dict(complete=True, arm=name, family=config['family'], config=config, mesh=n, micro_batch=micro,
                  best_epoch=ck['epoch'], best_step=ck['step'], epochs_completed=len(history),
                  optimisation_steps=step, stop_reason=stop_reason, training_seconds=seconds,
                  wall_budget_seconds=wall, validation_best_checkpoint=final,
                  best_checkpoint_sha256=sha_file(args.out / 'best.pt'),
                  parameter_dtype=str(getattr(net2, 'parameter_dtype', torch.float64)),
                  real_parameter_count=O.parameter_count(net2), normalization=[float(v) for v in norm])
    dump(args.out / 'result.json', result)
    print('TRAIN DONE', json.dumps({k: result[k] for k in ('arm', 'mesh', 'stop_reason', 'epochs_completed',
                                                              'optimisation_steps', 'best_epoch')}),
          'val_mean_case_max', final['mean_case_max'], 'val_worst', final['worst_case_max'], flush=True)


if __name__ == '__main__':
    main()
