"""Speed/memory smoke (no data, no outcome): ms per training step at batch 8 and rough batch-1 forward ms, per arm.
  bench_speed.py --arms A,B,... --steps 40 --out FILE
"""
import argparse, json, time
import torch
import opnet

a = argparse.ArgumentParser(); a.add_argument('--arms', required=True); a.add_argument('--steps', type=int, default=40)
a.add_argument('--out', required=True); a = a.parse_args()
opnet.configure()
norm = dict(u_mean=0., u_std=1., nu_mean=.03, nu_std=.02, scale=.3)
res = {}
for arm in a.arms.split(','):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    cfg = opnet.ARMS[arm]
    try:
        net = opnet.make(cfg); opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
        y = torch.randn(8, 6, 257, 257, device='cuda'); nu = torch.full((8,), .03, device='cuda')
        ts = []
        for s in range(a.steps + 10):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            loss = opnet.relative_errors(opnet.predict(net, y[:, 0], nu, norm), y[:, 1:], y[:, 0]).square().mean()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
        net.eval(); fs = []
        with torch.no_grad():
            u = y[:1, 0].double(); v = nu[:1].double()
            for s in range(30):
                torch.cuda.synchronize(); t0 = time.perf_counter(); opnet.predict(net, u, v, norm); torch.cuda.synchronize()
                fs.append(time.perf_counter() - t0)
        ts, fs = sorted(ts[10:]), sorted(fs[10:])
        res[arm] = dict(params=opnet.real_parameters(net), ms_per_step=1e3 * ts[len(ts) // 2],
                        forward_b1_ms=1e3 * fs[len(fs) // 2], peak_gb=torch.cuda.max_memory_allocated() / 1e9)
        del net, opt
    except torch.cuda.OutOfMemoryError as e:
        res[arm] = dict(error='OOM')
    print(arm, res[arm], flush=True)
json.dump(dict(gpu=torch.cuda.get_device_name(), results=res), open(a.out, 'w'), indent=1)
