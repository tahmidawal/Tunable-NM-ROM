"""The released CROM online loop (online/online.py, branch exp == 'burger') as a function on our problem, plus the
validation grid and the single test evaluation. Imported unchanged from the released repository: Decoder, Encoder,
NonlinearSolver, SamplePoint. The loop below copies online.py's statements for the Burgers branch (proj_type
'nonlinear', nonlinear_initial_guess 'prev', 'full' sampling, warm start of 20 GN iterations against u0); file writing
is replaced by keeping the decoded fields at the five output times. The GN solver's per-call print is sent to a null
stream (it would print ~10^5 lines per query).

  online_ours.py grid --weights DIR --name C16-s64 --arm E|I --setting S --data DIR --out DIR [--dtype float64] [--cases a:b]
  online_ours.py test --selection FILE --weights-root DIR --test PREFIX --out DIR [--rows a:b]
"""
import argparse, contextlib, glob, hashlib, io, json, os, sys, time
import numpy as np

EXT = os.environ.get('EXT', '/cluster/tufts/paralab/tawal01/rebuttal-official/ext')
sys.path[:0] = [os.path.join(EXT, 'CROM_online_deployment', 'online'), os.path.join(EXT, 'CROM_online_deployment', 'common')]
import torch
from Encoder import Encoder
from Decoder import Decoder
from NonlinearSolver import NonlinearSolver
from SamplePoint import SamplePoint
from burgers_ours import Burger, L

DT, NSTEP, NGRID, NTRAIN = .005, 50, 32, 4352
TEST_SHA = 'cd058fd2a297c20296b897e54cb186033a44a8db9527d98dcb2c2dad6044527c'
SETTINGS = {   # PROTOCOL.md 5.2: (diff_threshold, step_size_threshold, max GN iterations per step)
    'E': {'ex': (1e-15, 1e-15, 50), 'def50': (1e-6, 1e-3, 50), 'def5': (1e-6, 1e-3, 5)},
    'I': {'def2': (1e-6, 1e-3, 2), 'def5': (1e-6, 1e-3, 5), 'def50': (1e-6, 1e-3, 50)},
}


class NonlinearSolverBE(NonlinearSolver):
    """CROM-I: the released Gauss-Newton solve (iteration, Armijo line search, stopping rule unchanged) on the
    backward-Euler residual instead of the distance to an explicit target (only computeLossAndR is replaced)."""
    def computeLossAndR(self, xhat, q_target, sample_point):
        q = self.problem.updateStateSample(xhat, self.decoder, sample_point)
        r = self.problem.be_residual(q)
        loss = self.computeLoss(r)
        return loss, r


def nsub(nu, umax):
    return int(np.ceil(DT * (4 * nu * L * L + 2 * umax * L) / 0.9))


def load_nets(wdir, dtype, device):
    dec_path = [p for p in glob.glob(os.path.join(wdir, '*_dec.pt'))][0]
    enc_path = dec_path.replace('_dec.pt', '_enc.pt')
    enc = torch.jit.load(enc_path, map_location=device)
    dec = torch.jit.load(dec_path, map_location=device)
    if dtype == torch.float64:
        enc, dec = enc.double(), dec.double()
    return Encoder(enc), Decoder(dec, dec_path, None), dec_path


def query(encoder, decoder, u0, nu, arm, setting, dtype, device):
    """u0 [257,257] on device -> fields [5,257,257] on device. online.py, exp == 'burger'."""
    diff_threshold, step_size_threshold, nonlinear_iter = SETTINGS[arm][setting]
    implicit = arm == 'I'
    n = 1 if implicit else nsub(nu, float(u0.abs().max()))
    problem = Burger(u0, nu, DT / n, dtype, device, implicit=False)
    problem.Nstep = NSTEP * n
    with contextlib.redirect_stdout(io.StringIO()) as sink:
        nonlinear_solver = NonlinearSolver(problem, decoder, diff_threshold, step_size_threshold)
        q0_gt = problem.getPhysicalInitialCondition()
        xhat = encoder.forward(q0_gt)
        q0_gt = q0_gt.view(-1, q0_gt.size(2))
        sample_point = SamplePoint(problem)
        sample_point.initialize('full', -1, -1, decoder, xhat, torch.zeros_like(xhat))
        xhat = nonlinear_solver.solve(xhat, q0_gt, sample_point, 1, 20)  # warm start, as released
        problem.updateState(xhat, decoder)
        sample_point = SamplePoint(problem)
        sample_point.initialize('full', -1, -1, decoder, xhat, torch.zeros_like(xhat))
        problem.updateStateSample(xhat, decoder, sample_point, torch.zeros_like(xhat))
        problem.updateStateSampleAll(xhat, decoder, sample_point, torch.zeros_like(xhat))
        if implicit:
            problem.implicit = True
            nonlinear_solver = NonlinearSolverBE(problem, decoder, diff_threshold, step_size_threshold)
        out = []
        for step in range(1, problem.Nstep + 1):
            if implicit:
                problem.q_prev_interior = problem.interior(problem.q_sample).clone()
                xhat = nonlinear_solver.solve(xhat, problem.q_prev_interior, sample_point, step_size=1, max_iters=nonlinear_iter)  # q_target only sets the dtype (type_as); the BE residual uses q_prev_interior
            else:
                res = problem.getResidualSample(xhat, decoder, sample_point, step)
                q_target = problem.q_sample.view_as(res) + res * problem.dt
                xhat = nonlinear_solver.solve(xhat, q_target, sample_point, step_size=1, max_iters=nonlinear_iter)
            problem.updateStateSampleAll(xhat, decoder, sample_point, None)
            if step % (10 * n) == 0:
                out.append(problem.q.view(L + 1, L + 1).clone())
            sink.seek(0); sink.truncate()
    return torch.stack(out)


def metric(f, ref, n0):
    return np.sqrt(((np.asarray(f, np.float64) - ref) ** 2).sum((-1, -2))).max(-1) / n0


def summary(err):
    err = np.asarray(err)
    fin = np.isfinite(err)
    return dict(worst_percent=float(100 * (err.max() if fin.all() else np.inf)), median_percent=float(100 * np.median(err)),
                mean_percent=float(100 * np.mean(err)), nonfinite=int((~fin).sum()))


def run_cases(encoder, decoder, u0s, nus, arm, setting, dtype, device, budget_s=None):
    fields, secs = [], []
    for i in range(len(u0s)):
        u0 = torch.as_tensor(u0s[i], dtype=dtype, device=device)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        f = query(encoder, decoder, u0, float(nus[i]), arm, setting, dtype, device)
        torch.cuda.synchronize(); secs.append(time.perf_counter() - t0)
        fields.append(f.cpu().numpy())
        print('CASE', i, round(secs[-1], 3), flush=True)
        if budget_s and sum(secs) > budget_s:
            break
    return np.stack(fields), np.array(secs)


def grid(a):
    device = torch.device('cuda'); assert torch.cuda.is_available()
    dtype = torch.float64 if a.dtype == 'float64' else torch.float32
    out = os.path.join(a.out, f'{a.name}-{a.arm}-{a.setting}-{a.dtype}-{a.cases.replace(":", "_")}.json')
    if os.path.exists(out):
        print('SKIP', out); return
    meta = np.load(os.path.join(a.data, 'pool26.meta.npz'))
    mm = np.load(os.path.join(a.data, 'pool26.fields.npy'), mmap_mode='r')
    lo, hi = [int(x) for x in a.cases.split(':')]
    sl = np.asarray(mm[NTRAIN + lo:NTRAIN + hi, ::5], np.float64).reshape(hi - lo, 6, L + 1, L + 1)
    params, n0 = meta['params'][NTRAIN + lo:NTRAIN + hi], meta['n0'][NTRAIN + lo:NTRAIN + hi]
    encoder, decoder, dec_path = load_nets(a.weights, dtype, device)
    t0 = time.time()
    f, secs = run_cases(encoder, decoder, sl[:, 0], params[:, 4], a.arm, a.setting, dtype, device, budget_s=a.budget_s)
    m = len(f)
    err = metric(f, sl[:m, 1:], n0[:m])
    r = dict(name=a.name, arm=a.arm, setting=a.setting, gn=SETTINGS[a.arm][a.setting], dtype=a.dtype, cases=[lo, lo + m], requested=[lo, hi],
             complete=m == hi - lo, **summary(err), rough_ms=float(1e3 * np.median(secs)), ms_per_case=(1e3 * secs).tolist(), errors=err.tolist(),
             decoder=dec_path, gpu=os.environ.get('NF_GPU_NAME'), job_id=os.environ.get('SLURM_JOB_ID'), seconds=time.time() - t0, torch=torch.__version__)
    os.makedirs(a.out, exist_ok=True)
    open(out, 'w').write(json.dumps(r, indent=1) + '\n')
    print('GRID', json.dumps({k: v for k, v in r.items() if k not in ('errors', 'ms_per_case')}), flush=True)


def test(a):
    device = torch.device('cuda'); assert torch.cuda.is_available()
    os.makedirs(a.out, exist_ok=True)
    lo, hi = [int(x) for x in a.rows.split(':')]
    tag = os.path.join(a.out, f'test-crom-{lo}-{hi}.json')
    assert not os.path.exists(tag), 'the test cohort is evaluated once'
    sel_text = open(a.selection).read()
    sel = json.loads(sel_text)['crom'][lo:hi]
    meta = np.load(a.test + '.meta.npz'); params = meta['params']
    assert hashlib.sha256(np.ascontiguousarray(params).tobytes()).hexdigest() == TEST_SHA
    fld = np.load(a.test + '.fields.npy'); assert fld.shape == (64, 6, L + 1, L + 1) and fld.dtype == np.float64
    u0, ref = fld[:, 0], fld[:, 1:]
    n0 = np.sqrt((u0 ** 2).sum((1, 2)))
    res = []
    for s in sel:
        dtype = torch.float64 if s['dtype'] == 'float64' else torch.float32
        encoder, decoder, dec_path = load_nets(os.path.join(a.weights_root, s['weights']), dtype, device)
        rough = s['rough_ms']
        if rough > 60_000:            # PROTOCOL.md section 6: the evaluation pass itself is timed, after 1 warm-up case
            run_cases(encoder, decoder, u0[:1], params[:1, 4], s['arm'], s['setting'], dtype, device)
            f, secs = run_cases(encoder, decoder, u0, params[:, 4], s['arm'], s['setting'], dtype, device)
            ms, warm, reps = 1e3 * secs, 1, 1
        else:
            f, _ = run_cases(encoder, decoder, u0, params[:, 4], s['arm'], s['setting'], dtype, device)
            warm, reps = (20, 5) if rough < 1000 else (3, 1)
            u0d = [torch.as_tensor(u0[i], dtype=dtype, device=device) for i in range(64)]
            for i in range(warm):
                query(encoder, decoder, u0d[i % 64], float(params[i % 64, 4]), s['arm'], s['setting'], dtype, device)
            ms = np.zeros((reps, 64))
            for r_ in range(reps):
                for i in range(64):
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    query(encoder, decoder, u0d[i], float(params[i, 4]), s['arm'], s['setting'], dtype, device)
                    torch.cuda.synchronize(); ms[r_, i] = 1e3 * (time.perf_counter() - t0)
        err = metric(f, ref, n0)
        msr = np.asarray(ms).reshape(-1)
        r = dict(selection=s, test=summary(err), errors=err.tolist(), timing_warmup=warm, timing_repetitions=reps,
                 ms_full=dict(median=float(np.median(msr)), p5=float(np.percentile(msr, 5)), p95=float(np.percentile(msr, 95)),
                              min=float(msr.min()), max=float(msr.max()), count=int(msr.size)), decoder=dec_path)
        res.append(r)
        print('TEST', json.dumps({k: v for k, v in r.items() if k != 'errors'}), flush=True)
    open(tag, 'w').write(json.dumps(dict(rows=[lo, hi], cohort='test64 = params_draw(20260916, 64)', cohort_sha256=TEST_SHA,
                                         metric='max over t=0.05..0.25 of |u-u_ref|_2/|u0|_2', selection_sha256=hashlib.sha256(sel_text.encode()).hexdigest(),
                                         gpu=os.environ.get('NF_GPU_NAME'), job_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURMD_NODENAME'),
                                         torch=torch.__version__, results=res), indent=1) + '\n')


def main():
    a = argparse.ArgumentParser()
    a.add_argument('cmd', choices=('grid', 'test'))
    a.add_argument('--weights'); a.add_argument('--weights-root'); a.add_argument('--name'); a.add_argument('--arm'); a.add_argument('--setting')
    a.add_argument('--dtype', default='float64'); a.add_argument('--cases', default=f'0:{NGRID}'); a.add_argument('--budget-s', type=float, default=None)
    a.add_argument('--data'); a.add_argument('--out'); a.add_argument('--selection'); a.add_argument('--test'); a.add_argument('--rows', default='0:999')
    a = a.parse_args()
    dict(grid=grid, test=test)[a.cmd](a)


if __name__ == '__main__':
    main()
