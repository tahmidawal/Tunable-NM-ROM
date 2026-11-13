"""Tune, train and development-score one arm (PROTOCOL.md "Tuning and training").

Usage: ops_train.py --arm node|grid|don --data <prep data dir> --out <dir> [--tune-min 20 --final-min 60]
Never reads the sealed held-out directory.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

import ops_common as oc
import ops_models as om

# Candidate configurations, fixed before training (PROTOCOL.md).
CONFIGS = {
    'node': [
        {'name': 'tr_K8', 'C': 128, 'L': 6, 'H': 8, 'M': 32, 'K': 8, 'noise': 0.0, 'lr': 1e-3, 'batch': 8, 'n_sub': 16384},
        {'name': 'tr_K8_noise', 'C': 128, 'L': 6, 'H': 8, 'M': 32, 'K': 8, 'noise': 0.01, 'lr': 1e-3, 'batch': 8, 'n_sub': 16384},
        {'name': 'tr_K32', 'C': 128, 'L': 6, 'H': 8, 'M': 32, 'K': 32, 'noise': 0.0, 'lr': 1e-3, 'batch': 8, 'n_sub': 16384},
    ],
    'grid': [
        {'name': 'fno_K8', 'net': 'fno', 'W': 40, 'L': 4, 'm1': 16, 'm2': 64, 'K': 8, 'noise': 0.0, 'lr': 1e-3, 'batch': 4},
        {'name': 'unet_K8', 'net': 'unet', 'base': 32, 'levels': 4, 'K': 8, 'noise': 0.0, 'lr': 1e-3, 'batch': 4},
        {'name': 'unet_K8_noise', 'net': 'unet', 'base': 32, 'levels': 4, 'K': 8, 'noise': 0.01, 'lr': 1e-3, 'batch': 4},
        {'name': 'unet_K32', 'net': 'unet', 'base': 32, 'levels': 4, 'K': 32, 'noise': 0.0, 'lr': 1e-3, 'batch': 4},
    ],
    'don': [
        {'name': 'don_p128', 'p': 128, 'width': 512, 'depth': 4, 'lr': 1e-3, 'batch': 64, 'n_pts': 4096},
        {'name': 'don_p256', 'p': 256, 'width': 768, 'depth': 5, 'lr': 5e-4, 'batch': 64, 'n_pts': 4096},
    ],
}
N_SENS = 1024


class Data:
    """Training/validation arrays of one representation, stacked over cases: X (ncase*T, ...) float32."""

    def __init__(self, data_dir, rep, cases, Re, ddev):
        self.cases = cases; self.rep = rep
        arrs = [np.load(Path(data_dir) / f'{c}_{rep}.npy') for c in cases]
        self.T = arrs[0].shape[0]
        self.X = torch.cat([torch.from_numpy(a) for a in arrs]).to(ddev)
        self.Re = torch.tensor([float(Re[c]) for c in cases])
        self.ddev = ddev

    def stats(self, K):
        ax = (0, 1) if self.rep != 'grid' else (0, 2, 3)
        sub = self.X[::7]
        mean = sub.mean(dim=ax).cpu().numpy(); std = sub.std(dim=ax).cpu().numpy()
        dstd = []
        for k in range(1, K + 1):
            acc = []
            for c in range(len(self.cases)):
                i = torch.arange(0, self.T - k, 23) + c * self.T
                acc.append((self.X[i + k] - self.X[i]))
            dstd.append((torch.cat(acc).std(dim=ax).cpu().numpy() / std))
        return {'mean': mean.tolist(), 'std': std.tolist(), 'dstd': np.array(dstd).tolist()}


def make_static(arm, geom, k, meta):
    if arm == 'grid':
        return {'feat': om.grid_static(geom.fluid), 'fluid': torch.as_tensor(geom.fluid, dtype=torch.float32)}
    xy = geom.node_xy(k)
    if arm == 'node':
        return {'feat': om.node_static(xy)}
    return {'sens': torch.as_tensor(np.asarray(meta['sens_idx'])), 'xy': torch.as_tensor(xy, dtype=torch.float32)}


def train(sur, data, cfg, budget_s, log, seed=0):
    """Optimise for a wall-time budget with warm-up + cosine decay; returns the loss history."""
    dev = sur.dev; arm = sur.arm; K = sur.K
    g = torch.Generator(device='cpu').manual_seed(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(sur.model.parameters(), lr=cfg['lr'], weight_decay=1e-5)
    T, nc = data.T, len(data.cases)
    X = data.X
    if arm == 'grid':
        wmask = sur.static['fluid']
    hist = []; it = 0; t0 = time.time(); acc = 0.0; nacc = 0
    sur.model.train()
    while True:
        el = time.time() - t0
        if el >= budget_s:
            break
        lr = cfg['lr'] * min(1.0, (it + 1) / 300) * 0.5 * (1 + math.cos(math.pi * el / budget_s))
        for pg in opt.param_groups:
            pg['lr'] = lr
        B = cfg['batch']
        c = torch.randint(0, nc, (B,), generator=g)
        re = sur.re_feat(data.Re[c])
        if arm == 'don':
            d = torch.randint(0, T, (B,), generator=g)
            i = (torch.rand(B, generator=g) * (T - d)).long()
            j = i + d
            P = cfg['n_pts']
            idx = torch.randint(0, X.shape[1], (B, P), generator=g).to(X.device)
            gi = (c * T + i).to(X.device); gj = (c * T + j).to(X.device)
            sens = sur.norm(X[gi][:, sur.static['sens'].to(X.device)].to(dev))
            tgt = sur.norm(X[gj[:, None], idx].to(dev))
            xy = sur.static['xy'][idx.to(dev)]
            pred = sur.model(sens, re, xy, (d.float() * sur.meta['dt']).to(dev))
            loss = ((pred - tgt) ** 2).mean()
        else:
            i = torch.randint(0, T - K, (B,), generator=g)
            gi = (c * T + i).to(X.device)
            lead = gi[:, None] + torch.arange(1, K + 1, device=X.device)[None]
            if arm == 'node':
                idx = torch.randperm(X.shape[1], generator=g)[:cfg['n_sub']].to(X.device)
                x = sur.norm(X[gi[:, None], idx[None]].to(dev))                                  # (B, n, 3)
                y = sur.norm(X[lead[:, :, None], idx[None, None]].to(dev))                            # (B, K, n, 3)
                if cfg['noise'] > 0:
                    x = x + cfg['noise'] * torch.randn_like(x)
                tgt = (y - x[:, None]).permute(0, 2, 1, 3) / sur.dstd[None, None]     # (B, n, K, 3)
                pred = sur.step_delta(x, re, idx.to(dev))
                loss = ((pred - tgt) ** 2).mean()
            else:
                x = sur.norm(X[gi].to(dev))                                           # (B, 3, NY, NX)
                y = (X[lead].to(dev) - sur.mean[:, None, None]) / sur.std[:, None, None]   # (B, K, 3, NY, NX)
                if cfg['noise'] > 0:
                    x = x + cfg['noise'] * torch.randn_like(x)
                tgt = (y - x[:, None]) / sur.dstd[None, :, :, None, None]
                pred = sur.step_delta(x, re)
                loss = (((pred - tgt) ** 2) * wmask).sum() / (wmask.sum() * B * K * 3)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(sur.model.parameters(), 1.0)
        opt.step()
        it += 1
        acc += float(loss.detach()) if it % 10 == 0 else 0.0; nacc += it % 10 == 0
        if it % 200 == 0:
            hist.append((time.time() - t0, it, acc / max(nacc, 1), lr)); acc = 0.0; nacc = 0
            if len(hist) % 10 == 1:
                log(f'  it {it} t {hist[-1][0]:.0f}s loss {hist[-1][2]:.4e} lr {lr:.2e}')
    h = np.array(hist) if hist else np.zeros((0, 4))
    out = {'iterations': it, 'seconds': time.time() - t0, 'history': h.tolist()}
    if len(h) >= 10:
        fr = h[:, 0] / budget_s
        a = h[(fr >= 0.76) & (fr <= 0.84), 2]; b = h[fr >= 0.96, 2]
        if len(a) and len(b):
            out['loss_at_80pct'] = float(a.mean()); out['loss_final'] = float(b.mean())
            out['relative_drop_last_20pct'] = float((a.mean() - b.mean()) / a.mean())
            out['plateau'] = bool(out['relative_drop_last_20pct'] <= 0.05)
    return out


def native_error(pred, ref, w, arm):
    """Relative velocity L2 error per save in the native representation (validation criterion)."""
    ref = ref.to(pred.device)
    if arm == 'grid':
        num = (((pred[:, :2] - ref[:, :2]) ** 2).sum(1) * w).sum((1, 2)); den = ((ref[:, :2] ** 2).sum(1) * w).sum((1, 2))
    else:
        num = (((pred[..., :2] - ref[..., :2]) ** 2).sum(-1) * w).sum(1); den = ((ref[..., :2] ** 2).sum(-1) * w).sum(1)
    return torch.sqrt(num / den).cpu().numpy()


def time_window(sur, x0, Re, n_saves, repeats=5):
    def run():
        if sur.dev != 'cpu':
            torch.cuda.synchronize()
        t = time.perf_counter(); sur.window(x0, Re, n_saves)
        if sur.dev != 'cpu':
            torch.cuda.synchronize()
        return time.perf_counter() - t
    warm = run(); ts = [run() for _ in range(repeats)]
    return {'warmup_seconds': warm, 'repeats': ts, 'median_seconds': float(np.median(ts)), 'min_seconds': min(ts), 'max_seconds': max(ts),
            'device': torch.cuda.get_device_name(0) if sur.dev != 'cpu' else 'cpu'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=list(CONFIGS)); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--tune-min', type=float, default=20.0); ap.add_argument('--final-min', type=float, default=60.0)
    ap.add_argument('--only', help='restrict to one named candidate (amendment runs)')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    logf = open(out / 'log.txt', 'a')

    def log(s):
        print(s, flush=True); logf.write(s + '\n'); logf.flush()
    data_dir = Path(a.data)
    cases = json.loads((data_dir / 'cases.json').read_text()); Re = {c: v['Re'] for c, v in cases.items()}
    floors = json.loads((data_dir / 'floors.json').read_text())
    k = floors['node_rule']['selected']
    rep = 'grid' if a.arm == 'grid' else f'n{k}'
    geom = oc.Geometry(data_dir, dev)
    big = dev != 'cpu' and torch.cuda.get_device_properties(0).total_memory > 20e9
    ddev = dev if big else 'cpu'
    gpu = torch.cuda.get_device_name(0) if dev != 'cpu' else 'cpu'
    log(f'arm {a.arm} rep {rep} device {gpu} data_device {ddev}')
    t_arr = np.load(data_dir / 're100_t.npy'); n_saves = len(t_arr); dt = float(np.mean(np.diff(t_arr)))
    meta = {'rep': rep, 'k': k, 'dt': dt, 'n_saves': n_saves}
    if a.arm == 'node':
        meta['n_static'] = int(om.node_static(geom.node_xy(k)[:4]).shape[1])
    if a.arm == 'don':
        rs = np.random.RandomState(0)
        ns = min(N_SENS, len(geom.node_xy(k)) // 2)
        meta['n_sens'] = ns; meta['sens_idx'] = sorted(rs.choice(len(geom.node_xy(k)), ns, replace=False).tolist())
    static = make_static(a.arm, geom, k, meta)
    w_native = torch.as_tensor(geom.fluid if a.arm == 'grid' else geom.node_weights(k), dtype=torch.float32, device=dev)
    summary = {'arm': a.arm, 'rep': rep, 'gpu': gpu, 'tune_minutes': a.tune_min, 'final_minutes': a.final_min, 'candidates': {}}
    cands = [c for c in CONFIGS[a.arm] if not a.only or c['name'] == a.only]

    # ---- selection: train on Re 90 + 110, validate the full-window rollout on Re 100
    if len(cands) > 1:
        dtr = Data(data_dir, rep, ['re90', 're110'], Re, ddev)
        val = torch.from_numpy(np.load(data_dir / f're100_{rep}.npy'))
        for cfg in cands:
            sur = om.Surrogate(a.arm, cfg, meta, dtr.stats(cfg.get('K', 1)), static, dev)
            log(f"candidate {cfg['name']} params {om.n_params(sur.model)}")
            tr = train(sur, dtr, cfg, a.tune_min * 60, log)
            pred = sur.window(val[0], Re['re100'], n_saves)
            e = native_error(pred, val, w_native, a.arm)
            fin = bool(np.isfinite(e).all())
            rec = {'config': cfg, 'params': om.n_params(sur.model), 'train': {k_: v for k_, v in tr.items() if k_ != 'history'},
                   'val_e_u_rms': oc.rms(e) if fin else None, 'val_e_u_max': float(e.max()) if fin else None,
                   'val_e_u_last': float(e[-1]) if fin else None, 'finite': fin}
            np.save(out / f"tune_{cfg['name']}_val_e_u.npy", e); np.save(out / f"tune_{cfg['name']}_loss.npy", np.array(tr['history']))
            summary['candidates'][cfg['name']] = rec
            log(f"  -> val e_u rms {rec['val_e_u_rms']} max {rec['val_e_u_max']} iters {tr['iterations']}")
            (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=float) + '\n')
            del sur, pred
            torch.cuda.empty_cache() if dev != 'cpu' else None
        ok = {n: r for n, r in summary['candidates'].items() if r['finite']}
        if not ok:
            summary['status'] = 'all candidates non-finite on validation'
            (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=float) + '\n')
            raise SystemExit(3)
        best = min(ok, key=lambda n: ok[n]['val_e_u_rms'])
        del dtr, val
    else:
        best = cands[0]['name']
    summary['selected'] = best
    cfg = [c for c in CONFIGS[a.arm] if c['name'] == best][0]
    log(f'selected {best}')

    # ---- final training on Re 90 + 100 + 110
    dall = Data(data_dir, rep, ['re90', 're100', 're110'], Re, ddev)
    sur = om.Surrogate(a.arm, cfg, meta, dall.stats(cfg.get('K', 1)), static, dev)
    tr = train(sur, dall, cfg, a.final_min * 60, log)
    np.save(out / 'final_loss.npy', np.array(tr['history']))
    torch.save(sur.checkpoint(), out / 'final.pt')
    summary['final'] = {'config': cfg, 'params': om.n_params(sur.model), 'train': {k_: v for k_, v in tr.items() if k_ != 'history'}}
    # training-set self check: rollout on Re 100 (seen in training)
    x100 = dall.X[dall.T:2 * dall.T]
    e = native_error(sur.window(x100[0], Re['re100'], n_saves), x100, w_native, a.arm)
    summary['final']['train_re100_native_e_u'] = {'rms': oc.rms(e), 'max': float(np.max(e))}
    del dall, x100
    summary['training_seconds_total'] = float(sum(r['train']['seconds'] for r in summary['candidates'].values()) + tr['seconds'])
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=float) + '\n')

    # ---- development case Re 95: common scorer on Q, timing
    dev_arr = np.load(data_dir / f're95_{rep}.npy', mmap_mode='r')
    x0 = torch.from_numpy(np.asarray(dev_arr[0]))
    pred = sur.window(x0, Re['re95'], n_saves)
    sc = oc.Scorer(geom, data_dir, 're95', Re['re95'])
    rec, series = sc.score((pred[i] for i in range(n_saves)), geom.grid_map() if a.arm == 'grid' else geom.node_map(k))
    rec['inference'] = time_window(sur, x0, Re['re95'], n_saves)
    summary['dev_re95'] = rec
    np.savez(out / 'series_re95.npz', **series)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, default=float) + '\n')
    log('dev ' + json.dumps({k_: rec.get(k_) for k_ in ('status', 'e_u', 'e_p', 'cycle_errors', 'phase', 'targets_met')}, default=float))
    log('arm_done')


if __name__ == '__main__':
    main()
