"""Train one operator arm on one rung of the data ladder (PROTOCOL.md section 4). Never sees the test cohort.

  train.py --pool DIR --rung {128,576,4608} --arm F0 --seed 0 --out DIR [--budget-hours H] [--max-steps N]
           [--resume-from RUN_DIR]   (PROTOCOL amendment A5: continue a run cut by its budget from its best checkpoint)
"""
import argparse, glob, json, os, signal, time
from pathlib import Path
import numpy as np
import torch
import opnet

STOP = False
VALIDATION_CASES = 256   # P[4352:4608] at every rung (the template's 4,608-rung validation cases)
EVAL_EVERY, PLATEAU_PATIENCE, STOP_PATIENCE, THRESHOLD, MIN_LR, BATCH = 500, 6, 20, 1e-3, 1e-5, 8   # PROTOCOL amendment A2


def on_signal(signum, frame):
    global STOP
    STOP = True


def write_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=1) + '\n')
    tmp.replace(path)


def _chunks(directory, prefix):
    out = []
    for f in sorted(glob.glob(os.path.join(directory, f'{prefix}-*.fields.npy'))):
        a, b = map(int, os.path.basename(f).split('.')[0].split('-')[1:3])
        out.append((a, b, f))
    return out


def _take(chunks, start, stop, dst, at, nus):
    """Copy cases [start, stop) of a chunked set into dst[at:...] without a second full-size copy."""
    for a, b, f in chunks:
        lo, hi = max(a, start), min(b, stop)
        if lo >= hi:
            continue
        arr = np.load(f, mmap_mode='r')
        dst[at:at + hi - lo] = torch.from_numpy(np.ascontiguousarray(arr[lo - a:hi - a]))
        nus.append(np.load(f.replace('.fields.npy', '.meta.npz'))['params'][lo - a:hi - a, 4])
        at += hi - lo
    return at


def load_pool(pool, count, extra=None, device='cuda'):
    """Rung layout (operator-scaling PROTOCOL.md section 2). Returns y (count,6,257,257) float32, nu (count,), and
    the number of training cases; the last VALIDATION_CASES rows are always the template's validation cases P[4352:4608].
      count <= 4608: train P[0:count-256]                       (count = 4608 reproduces the template's P[0:4608] exactly)
      count >  4608: train P[0:4352] ++ X[0:count-4608]          (X = gen_data.py extra draw)"""
    nval = VALIDATION_CASES
    npool = min(count - nval, 4352)
    nextra = max(0, count - 4608)
    y = torch.empty((count, 6, 257, 257), dtype=torch.float32, pin_memory=(device == 'cpu'))
    nus = []
    at = _take(_chunks(pool, 'pool'), 0, npool, y, 0, nus)
    if nextra:
        at = _take(_chunks(extra, 'extra'), 0, nextra, y, at, nus)
    at = _take(_chunks(pool, 'pool'), 4352, 4608, y, at, nus)
    assert at == count, (at, count)
    nu = torch.from_numpy(np.concatenate(nus).astype(np.float32))
    return (y.cuda() if device == 'cuda' else y), nu.cuda(), count - nval


def evaluate(net, y, nu, ids, norm):
    net.eval()
    errs = []
    with torch.no_grad():
        for s in range(0, len(ids), 32):
            i = ids[s:s + 32]
            yi = y[i.to(y.device)].cuda(non_blocking=True)
            pred = opnet.predict(net, yi[:, 0], nu[i], norm)
            errs.append(opnet.relative_errors(pred, yi[:, 1:], yi[:, 0]))
    per = torch.cat(errs).max(dim=1).values.double().cpu().numpy()
    assert np.isfinite(per).all()
    return dict(mean=float(per.mean()), median=float(np.median(per)), worst=float(per.max()))


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--pool', required=True); a.add_argument('--rung', type=int, required=True)
    a.add_argument('--arm', required=True); a.add_argument('--seed', type=int, default=0)
    a.add_argument('--out', required=True); a.add_argument('--budget-hours', type=float, default=1e9)
    a.add_argument('--max-steps', type=int, default=400000)
    a.add_argument('--resume-from', default=None)
    a.add_argument('--extra', default=None, help='directory of gen_data.py extra chunks (rungs above 4,608)')
    a.add_argument('--deadline', type=float, default=1e18, help='unix time: stop (reason "deadline") at the next validation after it')
    a.add_argument('--snapshot-step', type=int, default=0, help='also keep the best checkpoint up to this step as best_upto_<step>.pt')
    a.add_argument('--schedule', choices=('plateau', 'cosine'), default='plateau',
                   help='plateau = protocol section 2; cosine = amendment A1 (cosine decay to 1e-6 over --max-steps, no early stopping)')
    a.add_argument('--continue-last', action='store_true', help='exact continuation from <out>/last.pt (same out dir)')
    a.add_argument('--data-device', choices=('cuda', 'cpu'), default='cuda')
    a = a.parse_args()
    signal.signal(signal.SIGUSR1, on_signal); signal.signal(signal.SIGTERM, on_signal)
    opnet.configure()
    cfg = dict(opnet.ARMS[a.arm])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=a.continue_last)
    torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed); np.random.seed(a.seed)
    t_load = time.monotonic()
    y, nu, ntrain = load_pool(a.pool, a.rung, a.extra, a.data_device)
    nval = a.rung - ntrain
    print('LOADED', a.rung, ntrain, nval, a.data_device, round(time.monotonic() - t_load, 1), 's', flush=True)
    val_ids = torch.arange(ntrain, a.rung, device='cuda')
    # same statistics as the template, accumulated in float64 over chunks (identical definition; works for host data)
    s1 = s2 = sq = 0.0
    for c in y[:ntrain].split(256):
        c = c.cuda().double()
        s1 += float(c[:, 0].sum()); s2 += float(c[:, 0].square().sum()); sq += float(c.square().sum())
    m0 = ntrain * y.shape[-1] * y.shape[-2]
    u_mean = s1 / m0
    norm = dict(u_mean=u_mean, u_std=((s2 - m0 * u_mean ** 2) / (m0 - 1)) ** .5, nu_mean=float(nu[:ntrain].mean()),
                nu_std=float(nu[:ntrain].std()), scale=(sq / y[:ntrain].numel()) ** .5)
    net = opnet.make(cfg)
    prior = None
    if a.resume_from:
        src = Path(a.resume_from)
        prior = json.loads((src / 'result.json').read_text())
        assert prior['stop_reason'] == 'time_limit' and (prior['arm'], prior['rung'], prior['seed']) == (a.arm, a.rung, a.seed)
        pck = torch.load(src / 'best.pt', map_location='cuda', weights_only=False)
        assert pck['config'] == cfg and pck['step'] == prior['best_step']
        assert all(abs(pck['normalization'][k] - norm[k]) <= 1e-6 * abs(norm[k]) for k in norm), (pck['normalization'], norm)
        norm = pck['normalization']
        net.load_state_dict({k: v for k, v in pck['model'].items() if k != '_metadata'})
        torch.save(pck, out / 'best.tmp'); os.replace(out / 'best.tmp', out / 'best.pt')
        prior_history = [h for h in json.loads((src / 'history.json').read_text()) if h['step'] <= pck['step']]
        cfg_lr0 = prior['final_lr']
    opt = torch.optim.AdamW(net.parameters(), lr=cfg_lr0 if prior else cfg['lr'], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=.5, patience=PLATEAU_PATIENCE, threshold=THRESHOLD,
                                                       threshold_mode='rel', min_lr=MIN_LR)
    warm = 0 if prior else int(cfg.get('warmup_steps', 0))
    gen = torch.Generator(device='cuda').manual_seed(a.seed + (1000 if prior else 0))
    meta = dict(arm=a.arm, config=cfg, rung=a.rung, schedule=a.schedule, data_device=a.data_device, extra_dir=a.extra, seed=a.seed, train_cases=ntrain, validation_cases=nval,
                train_layout='P[0:%d] ++ X[0:%d]' % (min(ntrain, 4352), max(0, a.rung - 4608)), validation_layout='P[4352:4608]', normalization=norm,
                real_parameters=opnet.real_parameters(net), gpu=torch.cuda.get_device_name(),
                job_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURMD_NODENAME'), torch=torch.__version__,
                eval_every=EVAL_EVERY, plateau_patience=PLATEAU_PATIENCE, stop_patience=STOP_PATIENCE,
                threshold=THRESHOLD, min_lr=MIN_LR, batch=BATCH, test_cohort_seen=False,
                continuation=None if not prior else dict(source=str(a.resume_from), source_job=prior['job_id'],
                    start_step=prior['best_step'], start_validation=prior['best_validation'], start_lr=prior['final_lr'],
                    first_round_steps=prior['steps'], first_round_seconds=prior['training_seconds'],
                    optimizer_state='restarted (not saved by the first round)'))
    write_json(out / 'meta.json', meta)
    best, best_thr, stale, history, step, losses = float('inf'), float('inf'), 0, [], 0, []
    wall0, step0 = 0.0, 0
    if prior:
        best = best_thr = prior['best_validation']['mean']
        history, step, step0, wall0 = prior_history, prior['best_step'], prior['best_step'], prior['training_seconds']
    if a.continue_last:
        st = torch.load(out / 'last.pt', map_location='cuda', weights_only=False)
        assert st['config'] == cfg and (st['arm'], st['rung'], st['seed']) == (a.arm, a.rung, a.seed)
        norm = st['normalization']
        net.load_state_dict(st['model']); opt.load_state_dict(st['opt']); sched.load_state_dict(st['sched'])
        gen.set_state(st['gen'].cpu() if hasattr(st['gen'], 'cpu') else st['gen'])
        best, best_thr, stale, history, step = st['best'], st['best_thr'], st['stale'], st['history'], st['step']
        step0, wall0 = step, st['wall_seconds']
        meta = dict(meta, **{k: v for k, v in json.loads((out / 'meta.json').read_text()).items() if k in ('job_id', 'node', 'gpu', 'continued_from_last')})
        meta.setdefault('continued_from_last', []).append(dict(step=step, job_id=os.environ.get('SLURM_JOB_ID'),
                                                                 gpu=torch.cuda.get_device_name()))
        write_json(out / 'meta.json', meta)
        print('CONTINUE', out.name, step, flush=True)
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
            elif a.schedule == 'cosine':
                import math
                for g in opt.param_groups:
                    g['lr'] = 1e-6 + .5 * (cfg['lr'] - 1e-6) * (1 + math.cos(math.pi * min(step, a.max_steps) / a.max_steps))
            yi = y[i.to(y.device)].cuda(non_blocking=True)
            pred = opnet.predict(net, yi[:, 0], nu[i], norm)
            loss = opnet.relative_errors(pred, yi[:, 1:], yi[:, 0]).square().mean()
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
                if step >= warm and a.schedule == 'plateau':
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
                                    wall_seconds=wall0 + time.monotonic() - started, best=best))
                losses = []
                write_json(out / 'history.json', history)
                if a.snapshot_step and step == a.snapshot_step:
                    import shutil
                    shutil.copyfile(out / 'best.pt', out / f'best_upto_{step}.tmp'); os.replace(out / f'best_upto_{step}.tmp', out / f'best_upto_{step}.pt')
                if step % 5000 == 0:
                    print(json.dumps(dict(run=out.name, **history[-1])), flush=True)
                    torch.save(dict(model=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), gen=gen.get_state(),
                                    config=cfg, arm=a.arm, rung=a.rung, seed=a.seed, normalization=norm, step=step,
                                    best=best, best_thr=best_thr, stale=stale, history=history,
                                    wall_seconds=wall0 + time.monotonic() - started), out / 'last.tmp')
                    os.replace(out / 'last.tmp', out / 'last.pt')
                if stale >= STOP_PATIENCE and a.schedule == 'plateau':
                    stop_reason = 'early_stopping'
                elif step >= a.max_steps:
                    stop_reason = 'step_cap'
                elif STOP or time.monotonic() - started > a.budget_hours * 3600:
                    stop_reason = 'time_limit'
                elif time.time() > a.deadline:
                    stop_reason = 'deadline'
                if stop_reason:
                    break
    ck = torch.load(out / 'best.pt', weights_only=False)
    write_json(out / 'result.json', dict(meta, complete=True, stop_reason=stop_reason, converged=stop_reason == 'early_stopping',
        steps=step, epochs=step * BATCH / ntrain, best_step=ck['step'], best_validation=ck['validation'],
        final_lr=opt.param_groups[0]['lr'], training_seconds=wall0 + time.monotonic() - started,
        round_seconds=time.monotonic() - started, round_steps=step - step0,
        ms_per_step=1e3 * (time.monotonic() - started) / max(step - step0, 1), peak_gpu_bytes=torch.cuda.max_memory_allocated()))
    print('RUN DONE', out.name, stop_reason, step, json.dumps(ck['validation']), flush=True)


if __name__ == '__main__':
    main()
