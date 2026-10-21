"""Train the operator arms of ONE Burgers 3D cell (one mesh) sequentially in one process
(data generated once). DESIGN.md §§2-4.

* data: the NM-ROM's training seed 923701 (prefix-preserving first `ntrain` draws) and its
  bank-validation seed 923751 (first `nval` draws) for checkpoint selection, regenerated here
  at the target mesh with the NM-ROM's data solver (Newton-BiCGStab dt 0.005, ntol 1e-8,
  ltol 1e-9; six frames t = 0, 0.05, ..., 0.25). Parameter rows asserted disjoint from the
  sealed held-out cohort 923901 x 32 and the ROM validation cohort 923801 x 64.
* optimiser (ns3d-operators protocol, unchanged): AdamW lr 1e-3 wd 1e-4, batch 8
  (micro-batches + accumulation if 8 does not fit), clip 1.0, cosine lr in elapsed wall time
  to 1e-5, Transolver warm-up 160 steps, 3000 s wall budget per arm, epoch cap 4000,
  patience 250; checkpoint = best validation mean over cases of the per-case max evolved
  initial-relative error. Loss = mean squared initial-relative error over the 5 evolved times.
* an arm that cannot train (no micro-batch fits, crash) is recorded as not trained with the
  error text; the next arm still runs.
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
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'lanes' / 'burgers3d-span'))
sys.path.insert(0, str(HERE))

TRAIN_SEED, VAL_SEED = 923701, 923751
HELDOUT = (923901, 32)
ROMVAL = (923801, 64)
DATA_NTOL, DATA_LTOL = 1e-8, 1e-9
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


def rowkeys(tab):
    return {(round(float(a), 12), round(float(b), 12)) for a, b in zip(tab['nu'], tab['A'])}


def generate(n, ntrain, nval, log):
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    import common as C
    tr, va = C.table(TRAIN_SEED, ntrain), C.table(VAL_SEED, nval)
    ev = {'heldout': C.table(*HELDOUT), 'romval': C.table(*ROMVAL)}
    overlap = {k: len(rowkeys(v) & (rowkeys(tr) | rowkeys(va))) for k, v in ev.items()}
    if any(overlap.values()):
        raise RuntimeError(f'training/validation rows overlap an evaluation cohort: {overlap}')
    fom = C.make_fom(n, C.DT, DATA_NTOL, DATA_LTOL)
    G, ni = n - 1, n - 2
    out = {}
    t0 = time.time()
    stats = dict(max_rel_residual=0., max_newton=0, cap_hits=0)
    for name, tab, cnt in (('train', tr, ntrain), ('val', va, nval)):
        fr = np.zeros((cnt, 6, G, G, G), dtype=np.float32)
        for j in range(cnt):
            f, it, rn = fom(jnp.asarray(C.initial_interior(n, tab, j)), float(tab['nu'][j]))
            f = np.asarray(f)
            if not np.isfinite(f).all():
                raise RuntimeError(f'nonfinite truth {name} row {j}')
            stats['max_rel_residual'] = max(stats['max_rel_residual'], float(np.max(rn)))
            stats['max_newton'] = max(stats['max_newton'], int(np.max(it)))
            stats['cap_hits'] += int((np.asarray(it) >= C.MAX_NEWTON).sum())
            fr[j, :, 1:, 1:, 1:] = f.reshape(6, ni, ni, ni)
            if (j + 1) % 128 == 0:
                log(f'generated {name} {j + 1}/{cnt} ({time.time() - t0:.0f}s)')
        out[name] = (fr, np.asarray(tab['nu'], dtype=np.float64)[:cnt], tab['sha256'])
    del fom
    jax.clear_caches()
    return out, overlap, stats, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mesh', type=int, required=True)
    ap.add_argument('--arms', required=True)
    ap.add_argument('--ntrain', type=int, required=True)
    ap.add_argument('--nval', type=int, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--wall-seconds', type=float, default=3000.)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGUSR1, on_signal)
    n = args.mesh
    begin = time.time()
    log = lambda s: print(f'[{time.time() - begin:.0f}s] {s}', flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    data, overlap, gstats, gen_seconds = generate(n, args.ntrain, args.nval, log)
    log(f'data generated in {gen_seconds:.0f}s: {gstats}')

    import torch
    import ops3d as O
    import opsb3d as P
    env = O.configure()
    emb = P.Embed(n)
    gpu_total = torch.cuda.get_device_properties(0).total_memory
    trf, trnu, trsha = data['train']
    vaf, vanu, vasha = data['val']
    del data
    on_device = trf.nbytes < 0.30 * gpu_total
    U = torch.from_numpy(trf)
    U = U.cuda() if on_device else U
    V = torch.from_numpy(vaf)
    NU = torch.as_tensor(trnu, device='cuda')
    NUV = torch.as_tensor(vanu, device='cuda')
    ni = n - 2
    s0 = s1 = 0.0
    c0 = c1 = 0
    for i in range(len(trnu)):
        f = U[i].to('cuda').double()
        s0 += float(f[0].square().sum()); c0 += ni ** 3
        s1 += float(f[1:].square().sum()); c1 += 5 * ni ** 3
    u_scale = torch.tensor(math.sqrt(s0 / c0), dtype=torch.float64, device='cuda')
    out_scale = torch.tensor(math.sqrt(s1 / c1), dtype=torch.float64, device='cuda')
    lognu = torch.log(NU)
    norm = (u_scale, lognu.mean(), lognu.std().clamp_min(1e-12), out_scale)
    pinned = None if on_device else torch.empty((8,) + tuple(U.shape[1:]), dtype=U.dtype).pin_memory()

    def flat(batch):                                  # (B, 6, G, G, G) -> (B, 6, ni^3) float64
        return emb.from_grid(batch.double())

    def get(ids):
        idc = torch.as_tensor(ids, device='cuda')
        if on_device:
            batch = U[idc]
        else:
            buf = pinned[:len(ids)]
            torch.index_select(U, 0, torch.as_tensor(ids), out=buf)
            batch = buf.to('cuda', non_blocking=True)
        y = flat(batch)
        return y[:, 0], NU[idc], y

    def evaluate(net, nrm):
        net.eval()
        errs = []
        with torch.no_grad():
            for i in range(len(vanu)):
                y = flat(V[i:i + 1].to('cuda'))
                u0 = y[:, 0]
                errs.append(P.initial_relative(P.predict(net, u0, NUV[i:i + 1], nrm, emb), y, u0)[0].cpu().numpy())
        e = np.asarray(errs)
        if not np.isfinite(e).all():
            raise RuntimeError('nonfinite validation error')
        per = e[:, 1:].max(1)
        return dict(errors=e.tolist(), mean_case_max=float(per.mean()), median_case_max=float(np.median(per)),
                    worst_case_max=float(per.max()))

    def loss_of(net, ids):
        u0, nu, y = get(ids)
        e = P.initial_relative(P.predict(net, u0, nu, norm, emb), y, u0)[:, 1:]
        return e.square().mean()

    common = dict(environment=env, mesh=n, grid=n - 1, train_seed=TRAIN_SEED, train_count=int(len(trnu)),
                  train_table_sha256=trsha, validation_seed=VAL_SEED, validation_count=int(len(vanu)),
                  validation_table_sha256=vasha, data_solver=dict(dt=0.005, ntol=DATA_NTOL, ltol=DATA_LTOL),
                  generation=gstats, generation_seconds=gen_seconds, overlap_with_evaluation_cohorts=overlap,
                  normalization=[float(v) for v in norm], data_on_device=bool(on_device),
                  job_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURMD_NODENAME'),
                  sources={p.name: sha_file(p) for p in sorted(HERE.glob('*.py'))})
    dump(args.out / 'data.json', common)

    ntr = len(trnu)
    tr_idx = np.arange(ntr)
    for arm in args.arms.split(','):
        adir = args.out / arm
        adir.mkdir(parents=True, exist_ok=True)
        cfg_path = HERE / 'configs' / 'ops' / f'{arm}.json'
        config = json.loads(cfg_path.read_text())
        if config['family'] == 'transolver':
            config['patch'] = max(1, (n - 1) // 16)
        if args.smoke:
            config = dict(config, **config.get('smoke_overrides', {}))
        log(f'=== arm {arm} {config}')
        try:
            train_arm(arm, config, cfg_path, adir, args, tr_idx, loss_of, evaluate, norm, gpu_total, log, common)
        except Exception as exc:  # noqa: BLE001  -- recorded as "not trained", next arm continues
            tb = traceback.format_exc()
            print(tb, flush=True)
            dump(adir / 'result.json', dict(complete=False, trained=False, arm=arm, family=config['family'],
                                            config=config, mesh=n, failure=f'{type(exc).__name__}: {exc}'[:2000]))
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if STOP:
            break
    log('ALL ARMS DONE')


def train_arm(arm, config, cfg_path, adir, args, tr_idx, loss_of, evaluate, norm, gpu_total, log, common):
    import torch
    import ops3d as O
    import opsb3d as P
    n = args.mesh
    seed = int(config['seed'])
    batch_size = int(config['batch_size'])
    micro, probe = None, {}
    for m in ((1,) if args.smoke else (8, 4, 2, 1)):
        m = min(m, batch_size)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(seed)
        net = popt = None
        try:
            net = P.make_model(config)
            popt = torch.optim.AdamW(net.parameters(), lr=1e-12, weight_decay=0.0)
            for _ in range(2):
                popt.zero_grad(set_to_none=True)
                loss_of(net, tr_idx[:m]).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                popt.step()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            probe[str(m)] = peak
            ok = peak < 0.85 * gpu_total or m == 1
        except torch.OutOfMemoryError:
            probe[str(m)] = 'oom'
            ok = False
        del net, popt
        torch.cuda.empty_cache()
        if ok:
            micro = m
            break
    if micro is None:
        raise RuntimeError(f'no micro-batch fits (probe {probe})')
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    net = P.make_model(config)
    O.check_dtypes(net)
    opt = torch.optim.AdamW(net.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    lr0, floor = float(config['learning_rate']), 1e-5
    warm = int(config.get('warmup_steps', 0))
    gen = torch.Generator().manual_seed(seed)
    wall = float(args.wall_seconds)
    best, best_epoch, history, step, stop_reason = float('inf'), -1, [], 0, 'epoch_cap'
    patience = int(config.get('patience', 250))
    dump(adir / 'provenance.json', dict(common, arm=arm, config=config, config_sha256=sha_file(cfg_path),
                                        micro_batch=micro, probe_peak_bytes=probe, wall_budget_seconds=wall,
                                        real_parameter_count=O.parameter_count(net)))
    log(f'{arm}: micro {micro} params {O.parameter_count(net)} probe {probe}')
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
        metrics = evaluate(net, norm)
        if metrics['mean_case_max'] < best:
            best, best_epoch = metrics['mean_case_max'], epoch
            torch.save(dict(model=net.state_dict(), config=config, normalization=[float(v) for v in norm],
                            epoch=epoch, step=step, best=best, mesh=n, arm=arm), adir / 'best.tmp')
            (adir / 'best.tmp').replace(adir / 'best.pt')
        history.append(dict(epoch=epoch, steps=step, train_loss=float(np.mean(losses)),
                            validation={k: v for k, v in metrics.items() if k != 'errors'},
                            learning_rate=lr, wall_seconds=time.monotonic() - started, best_so_far=best,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated()))
        dump(adir / 'history.json', history)
        print(json.dumps(dict(arm=arm, epoch=epoch, steps=step, train=history[-1]['train_loss'],
                              val_mean=metrics['mean_case_max'], val_worst=metrics['worst_case_max'],
                              lr=lr, wall=round(history[-1]['wall_seconds'], 1))), flush=True)
        if STOP:
            stop_reason = 'signal'; break
        if time.monotonic() - started >= wall:
            stop_reason = 'wall_budget'; break
        if epoch - best_epoch >= patience:
            stop_reason = 'patience'; break
    seconds = time.monotonic() - started
    del net, opt
    torch.cuda.empty_cache()
    net2, norm2, ck = P.load_checkpoint(adir / 'best.pt')
    final = evaluate(net2, norm2)
    reproduced = abs(final['mean_case_max'] - best) <= 1e-6 * max(best, 1e-12)
    result = dict(complete=bool(stop_reason != 'signal' and reproduced), trained=True, arm=arm,
                  validation_reproduced_from_checkpoint=bool(reproduced), family=config['family'], config=config,
                  mesh=n, micro_batch=micro, best_epoch=ck['epoch'], best_step=ck['step'],
                  epochs_completed=len(history), optimisation_steps=step, stop_reason=stop_reason,
                  training_seconds=seconds, wall_budget_seconds=wall, validation_best_checkpoint=final,
                  best_checkpoint_sha256=sha_file(adir / 'best.pt'),
                  parameter_dtype=str(getattr(net2, 'parameter_dtype', torch.float64)),
                  real_parameter_count=O.parameter_count(net2), normalization=[float(v) for v in norm])
    dump(adir / 'result.json', result)
    del net2
    log(f'TRAIN DONE {arm} stop {stop_reason} epochs {len(history)} steps {step} best_epoch {ck["epoch"]} '
        f'val_mean {final["mean_case_max"]:.4g} val_worst {final["worst_case_max"]:.4g}')


if __name__ == '__main__':
    main()
