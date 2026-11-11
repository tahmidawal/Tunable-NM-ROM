"""Train one operator arm on one rung of the data ladder (PROTOCOL.md section 4). Never sees the test cohort.

  train.py --pool DIR --rung {128,576,4608} --arm F0 --seed 0 --out DIR [--budget-hours H] [--max-steps N]
"""
import argparse, glob, json, os, signal, time
from pathlib import Path
import numpy as np
import torch
import opnet

STOP = False
EVAL_EVERY, PLATEAU_PATIENCE, STOP_PATIENCE, THRESHOLD, MIN_LR, BATCH = 500, 6, 20, 1e-3, 1e-5, 8   # PROTOCOL amendment A2


def on_signal(signum, frame):
    global STOP
    STOP = True


def write_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=1) + '\n')
    tmp.replace(path)


def load_pool(pool, count):
    parts, params, n = [], [], 0
    for f in sorted(glob.glob(os.path.join(pool, 'pool-*.fields.npy'))):
        if n >= count:
            break
        a = np.load(f, mmap_mode='r')
        take = min(len(a), count - n)
        parts.append(torch.from_numpy(np.array(a[:take])))
        params.append(np.load(f.replace('.fields.npy', '.meta.npz'))['params'][:take])
        n += take
    assert n == count, (n, count)
    return torch.cat(parts).cuda(), torch.from_numpy(np.concatenate(params)[:, 4].astype(np.float32)).cuda()


def evaluate(net, y, nu, ids, norm):
    net.eval()
    errs = []
    with torch.no_grad():
        for s in range(0, len(ids), 32):
            i = ids[s:s + 32]
            pred = opnet.predict(net, y[i, 0], nu[i], norm)
            errs.append(opnet.relative_errors(pred, y[i, 1:], y[i, 0]))
    per = torch.cat(errs).max(dim=1).values.double().cpu().numpy()
    assert np.isfinite(per).all()
    return dict(mean=float(per.mean()), median=float(np.median(per)), worst=float(per.max()))


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--pool', required=True); a.add_argument('--rung', type=int, required=True)
    a.add_argument('--arm', required=True); a.add_argument('--seed', type=int, default=0)
    a.add_argument('--out', required=True); a.add_argument('--budget-hours', type=float, default=1e9)
    a.add_argument('--max-steps', type=int, default=400000)
    a = a.parse_args()
    signal.signal(signal.SIGUSR1, on_signal); signal.signal(signal.SIGTERM, on_signal)
    opnet.configure()
    cfg = dict(opnet.ARMS[a.arm])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); np.random.seed(a.seed)
    y, nu = load_pool(a.pool, a.rung)
    nval = opnet.VALIDATION[a.rung]
    ntrain = a.rung - nval
    val_ids = torch.arange(ntrain, a.rung, device='cuda')
    u0 = y[:ntrain, 0]
    norm = dict(u_mean=float(u0.mean()), u_std=float(u0.std()), nu_mean=float(nu[:ntrain].mean()),
                nu_std=float(nu[:ntrain].std()),
                scale=float((sum(float(c.double().square().sum()) for c in y[:ntrain].split(64)) / y[:ntrain].numel()) ** .5))
    net = opnet.make(cfg)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=.5, patience=PLATEAU_PATIENCE, threshold=THRESHOLD,
                                                       threshold_mode='rel', min_lr=MIN_LR)
    warm = int(cfg.get('warmup_steps', 0))
    gen = torch.Generator(device='cuda').manual_seed(a.seed)
    meta = dict(arm=a.arm, config=cfg, rung=a.rung, seed=a.seed, train_cases=ntrain, validation_cases=nval,
                train_slice=[0, ntrain], validation_slice=[ntrain, a.rung], normalization=norm,
                real_parameters=opnet.real_parameters(net), gpu=torch.cuda.get_device_name(),
                job_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURMD_NODENAME'), torch=torch.__version__,
                eval_every=EVAL_EVERY, plateau_patience=PLATEAU_PATIENCE, stop_patience=STOP_PATIENCE,
                threshold=THRESHOLD, min_lr=MIN_LR, batch=BATCH, test_cohort_seen=False)
    write_json(out / 'meta.json', meta)
    best, best_thr, stale, history, step, losses = float('inf'), float('inf'), 0, [], 0, []
    started = time.monotonic()
    stop_reason = None
    net.train()
    while stop_reason is None:
        order = torch.randperm(ntrain, device='cuda', generator=gen)
        for s in range(0, ntrain - BATCH + 1, BATCH):          # drop the ragged last batch
            i = order[s:s + BATCH]
            if step < warm:
                for g in opt.param_groups:
                    g['lr'] = cfg['lr'] * (step + 1) / warm
            pred = opnet.predict(net, y[i, 0], nu[i], norm)
            loss = opnet.relative_errors(pred, y[i, 1:], y[i, 0]).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append(loss.detach())
            step += 1
            if step % EVAL_EVERY == 0:
                m = evaluate(net, y, nu, val_ids, norm)
                net.train()
                if step >= warm:
                    sched.step(m['mean'])
                if m['mean'] < best:
                    best = m['mean']
                    torch.save(dict(model=net.state_dict(), config=cfg, arm=a.arm, rung=a.rung, seed=a.seed, step=step,
                                    normalization=norm, validation=m), out / 'best.tmp')
                    os.replace(out / 'best.tmp', out / 'best.pt')
                if m['mean'] < best_thr * (1 - THRESHOLD):
                    best_thr, stale = m['mean'], 0
                else:
                    stale += 1
                history.append(dict(step=step, train_loss=float(torch.stack(losses).mean()), val_mean=m['mean'],
                                    val_median=m['median'], val_worst=m['worst'], lr=opt.param_groups[0]['lr'],
                                    wall_seconds=time.monotonic() - started, best=best))
                losses = []
                write_json(out / 'history.json', history)
                if step % 5000 == 0:
                    print(json.dumps(dict(run=out.name, **history[-1])), flush=True)
                if stale >= STOP_PATIENCE:
                    stop_reason = 'early_stopping'
                elif step >= a.max_steps:
                    stop_reason = 'step_cap'
                elif STOP or time.monotonic() - started > a.budget_hours * 3600:
                    stop_reason = 'time_limit'
                if stop_reason:
                    break
    ck = torch.load(out / 'best.pt', weights_only=False)
    write_json(out / 'result.json', dict(meta, complete=True, stop_reason=stop_reason, converged=stop_reason == 'early_stopping',
        steps=step, epochs=step * BATCH / ntrain, best_step=ck['step'], best_validation=ck['validation'],
        final_lr=opt.param_groups[0]['lr'], training_seconds=time.monotonic() - started,
        ms_per_step=1e3 * (time.monotonic() - started) / step, peak_gpu_bytes=torch.cuda.max_memory_allocated()))
    print('RUN DONE', out.name, stop_reason, step, json.dumps(ck['validation']), flush=True)


if __name__ == '__main__':
    main()
