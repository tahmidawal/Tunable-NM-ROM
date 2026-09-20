"""Kim et al. NM-LSPG(-HR) baselines, POD-LSPG control, the frozen project ROM and the FOMs on the
shared Burgers 2D family, one mesh per job.  See DESIGN.md Section 3.

  family.py --config <json> --out output [--cache <pilot-data01>]

Config: intervals, cohort ('tune' | 'validation'), variants [ {name,K,ref,scale,b,db,M1,lr,patience,
seed,max_epochs,wall,dtype,hr:[[nr,nz],...]} ], pod_ks, rom_qs, timing {reps,cases,burn}.
Hyper-parameters are only ever chosen on the tuning subset (train cases 112-127).
"""
from __future__ import annotations
import argparse, hashlib, json, os, pickle, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROTO = dict(dataset_seed=20260914, pde_seed_code=22, split_codes=dict(calibration=0, train=1, validation=2),
             counts=dict(train=128, validation=32))
INDEX_SHA = dict(train='5333584b7162df622ec7bc2b08e68d8003036b1b05abbea520249ed408fc8d49',
                 validation='468b9e70df3392c4b5bbb41381c9062d3d697ac4772b7236c4ee59d0ca73ebad')
CKPT = ROOT / 'experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'
CKPT_SHA = '18f0266ae6f0454200ec0b7bf94a18cde531feac9d3170d5099adc5d68d6b589'
FIT, TUNE = np.arange(0, 112), np.arange(112, 128)
DT, STEPS, KEEP = .005, 50, 10


def sha_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for blk in iter(lambda: f.read(1 << 22), b''):
            h.update(blk)
    return h.hexdigest()


def case_seed(split, index):
    return int(np.random.SeedSequence([PROTO['dataset_seed'], PROTO['pde_seed_code'],
                                       PROTO['split_codes'][split], index]).generate_state(1, dtype=np.uint32)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--cache', default='/cluster/tufts/paralab/tawal01/no_burgers_20260914/pilot-data01')
    ap.add_argument('--allow-cpu', action='store_true')
    ap.add_argument('--gate', help='summary.json of the accepted Kim reproduction gate; required unless --smoke')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    backend = jax.default_backend()
    print(f'jax_backend={backend}', flush=True)
    if backend != 'gpu' and not a.allow_cpu:
        sys.exit(42)
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
    for d in ('experiments/mr-burgers2d', 'experiments/separable-decoder', 'experiments/nmrom-baselines/vendor'):
        sys.path.insert(0, str(ROOT / d))
    sys.path.insert(0, str(HERE))
    import engines as e, iterative_paths as ip, kimae, lspg
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    L = int(cfg['intervals']); m = L - 1; n = m * m
    act = kimae.ACT['swish']
    host = lambda t: jax.tree_util.tree_map(np.asarray, t)
    gpu_uuid = os.popen('nvidia-smi --query-gpu=name,uuid --format=csv,noheader').read().strip()
    report = dict(config=cfg, intervals=L, n=n, commit=os.environ.get('SOURCE_COMMIT'), job_id=os.environ.get('SLURM_JOB_ID'),
                  backend=backend, gpu=str(jax.devices()[0].device_kind), gpu_uuid=gpu_uuid, jax=jax.__version__,
                  x64=True, matmul='highest', gates={}, arms={}, training={}, dropped=[],
                  source_sha256={str(p.relative_to(ROOT)): sha_file(p) for p in
                                 [HERE / 'family.py', HERE / 'kimae.py', HERE / 'lspg.py', ROOT / 'experiments/mr-burgers2d/engines.py',
                                  ROOT / 'experiments/mr-burgers2d/iterative_paths.py']})
    for p_ in sorted(x for x in (HERE / 'vendor').glob('*') if x.is_file()):
        report['source_sha256'][str(p_.relative_to(ROOT))] = sha_file(p_)
    if a.smoke:
        report['gate_binding'] = dict(smoke=True, note='no result of a smoke run may be reported')
        hr_allowed = True
    else:
        g = json.loads(Path(a.gate).read_text())
        assert g['gate']['passed'], 'Kim reproduction gate has not passed: no Kim arm may be run for the comparison'
        hr_allowed = bool(g['gate']['hr_passed'])
        report['gate_binding'] = dict(gate_sha256=sha_file(a.gate), gate=g['gate'], gate_job=g['provenance']['job_id'],
                                      kimae_matches_gate=bool(g['source_sha256']['kimae.py'] == sha_file(HERE / 'kimae.py')),
                                      lspg_matches_gate=bool(g['source_sha256']['lspg.py'] == sha_file(HERE / 'lspg.py')))
        assert report['gate_binding']['kimae_matches_gate'] and report['gate_binding']['lspg_matches_gate'], 'code changed since the gate'
        tc = cfg.get('timing')
        assert tc is None or (tc['reps'] >= 5 and tc['cases'] >= 4 and tc['burn'] >= .25)
    save = lambda: (out / 'summary.json').write_text(json.dumps(report, indent=1, default=float) + '\n')
    T0 = time.monotonic()

    # ------------------------------------------------------------------ the shared split, from seed
    phys = {s: np.stack([e.params_draw(case_seed(s, i), 1)[0] for i in range(PROTO['counts'][s])]) for s in ('train', 'validation')}
    cache = Path(a.cache)
    for s in ('train', 'validation'):
        ip_ = cache / s / 'index.json'
        if ip_.exists():
            got = sha_file(ip_)
            recs = json.loads(ip_.read_text())['records']
            desc = np.array([[r['generation_descriptors'][k] for k in ('cx', 'cy', 'width', 'amplitude', 'nu')] for r in recs])
            seeds_ok = all(r['seed'] == case_seed(s, i) for i, r in enumerate(recs))
            dev = float(np.max(np.abs(desc - phys[s]) / np.abs(phys[s])))
            report['gates'][f'split_{s}_matches_cache'] = dict(passed=bool(got == INDEX_SHA[s] and seeds_ok and dev < 1e-12),
                                                              index_sha256=got, expected=INDEX_SHA[s], seeds_equal=seeds_ok,
                                                              max_relative_descriptor_deviation=dev)
            assert report['gates'][f'split_{s}_matches_cache']['passed'], report['gates']
        else:
            report['gates'][f'split_{s}_matches_cache'] = dict(passed=None, note='cache index not present; draws are from the protocol seeds only')
    save()
    cohort_name = cfg['cohort']
    assert cohort_name in ('tune', 'validation')
    eval_phys = phys['train'][TUNE] if cohort_name == 'tune' else phys['validation']
    tune_phys = phys['train'][TUNE]

    # ------------------------------------------------------------------ snapshots (all 51 states) and references
    fomq, _ = e.make_fom(L, DT, None, .25, DT)
    def trajectories(P):
        rows = []
        for p in P:
            f, it, rn = host(fomq(jnp.asarray(e.initial(L, p)), float(p[4]), 1e-9, 1e-7))
            assert np.isfinite(f).all() and rn.max() < 1e-7, rn.max()
            rows.append(f[:, 1:-1, 1:-1].reshape(STEPS + 1, n))
        return np.stack(rows)
    t0 = time.perf_counter()
    U_fit = trajectories(phys['train'][FIT])                       # (112, 51, n)
    report['snapshots'] = dict(fit=list(U_fit.shape), seconds=time.perf_counter() - t0,
                               sha256=hashlib.sha256(np.ascontiguousarray(U_fit[:, ::10]).tobytes()).hexdigest())
    del fomq; jax.clear_caches()
    ref_q, ref_pre = ip.make_fom(L, DT, 'fft')
    def reference(P):
        rows = []
        for p in P:
            f, it, rn = host(ref_q(jnp.asarray(e.initial(L, p)), float(p[4]), 1e-6, 1e-8, *ref_pre)[:3])
            assert np.isfinite(f).all() and rn.max() <= 1e-6 and it.max() < 20, (rn.max(), it.max())
            rows.append(f)
        return np.stack(rows)
    REF = reference(eval_phys)                                     # (cases, 6, L+1, L+1)
    REF_tune = REF if cohort_name == 'tune' else reference(tune_phys)
    print(f'DATA {time.monotonic()-T0:.0f}s fit={U_fit.shape} ref={REF.shape}', flush=True)
    np.savez(out / 'reference.npz', reference=REF, physical=eval_phys)

    def errors(fields, ref):
        """fixed-initial relative error per case and time: ||f-ref||_interior / ||ref(t=0)||_interior"""
        d = (fields - ref)[:, :, 1:-1, 1:-1].reshape(ref.shape[0], ref.shape[1], -1)
        e_ = np.linalg.norm(d, axis=2) / np.linalg.norm(ref[:, 0, 1:-1, 1:-1].reshape(ref.shape[0], -1), axis=1)[:, None]
        return dict(worst_evolved=float(e_[:, 1:].max()), median_evolved=float(np.median(e_[:, 1:].max(1))),
                    worst_all_times=float(e_.max()), worst_t0=float(e_[:, 0].max()),
                    per_case_evolved=e_[:, 1:].max(1).tolist(), finite=bool(np.isfinite(e_).all()))

    fdt = np.float64 if L <= 256 else np.float32       # 512: f32 storage, audit tolerance stated in the collector
    report['saved_field_dtype'] = str(np.dtype(fdt))
    pad = lambda V: jnp.pad(V.reshape(-1, m, m), ((0, 0), (1, 1), (1, 1)))

    def fom_res(u, prev, nu):
        return e.residual(u, prev, nu, DT, L)

    def run_cases(query, qargs, P):
        fields, its = [], []
        for p in P:
            f, it = query(jnp.asarray(e.initial(L, p)), float(p[4]), qargs)
            fields.append(np.asarray(f)); its.append(np.asarray(it))
        return np.stack(fields), np.stack(its)

    def mem_of(jitted, *args):
        try:
            ma = jitted.lower(*args).compile().memory_analysis()
            return dict(temp=int(ma.temp_size_in_bytes), arguments=int(ma.argument_size_in_bytes), output=int(ma.output_size_in_bytes),
                        total=int(ma.temp_size_in_bytes + ma.argument_size_in_bytes + ma.output_size_in_bytes))
        except Exception as exc:     # noqa: BLE001
            return dict(error=str(exc)[:200])

    subjects = {}     # name -> (jitted query(u0, nu, args)->(fields, its), args)

    # ------------------------------------------------------------------ POD-LSPG control
    Xfit = U_fit.reshape(-1, n)
    t0 = time.perf_counter()
    Gm = np.asarray(jax.jit(lambda X: X @ X.T)(jnp.asarray(Xfit)))
    w, V = np.linalg.eigh(Gm)
    order = np.argsort(w)[::-1][:max(cfg.get('pod_ks', [16]) + [8])]
    PhiAll = np.asarray(jax.jit(lambda X, C: X.T @ C)(jnp.asarray(Xfit), jnp.asarray(V[:, order] / np.sqrt(w[order]))))
    PhiAll, _ = np.linalg.qr(PhiAll)
    report['pod'] = dict(seconds=time.perf_counter() - t0, orthonormality=float(np.abs(PhiAll.T @ PhiAll - np.eye(PhiAll.shape[1])).max()))
    del Gm

    def make_pod_query(ref_mode):
        res = lambda z, zp, A: fom_res(A['ref'] + A['Phi'] @ z, A['ref'] + A['Phi'] @ zp, A['nu'])
        roll = lspg.make_rollout(res, STEPS, max_it=cfg.get('gn_cap', 20), save_every=KEEP)
        def query(u0, nu, A):
            ui = u0[1:-1, 1:-1].reshape(-1)
            ref = ui if ref_mode == 'ic' else jnp.zeros_like(ui)
            z0 = A['Phi'].T @ (ui - ref)
            Z, o = roll(z0, dict(A, nu=nu, ref=ref))
            return pad(ref + Z @ A['Phi'].T), o[0].reshape(-1)
        return jax.jit(query)
    # the linear control gets both reference conventions the Kim arms may use (audit finding 16)
    Dic = (U_fit - U_fit[:, :1]).reshape(-1, n)
    Gi = np.asarray(jax.jit(lambda X: X @ X.T)(jnp.asarray(Dic)))
    wi, Vi = np.linalg.eigh(Gi); oi = np.argsort(wi)[::-1][:PhiAll.shape[1]]
    PhiIC = np.linalg.qr(np.asarray(jax.jit(lambda X, C: X.T @ C)(jnp.asarray(Dic), jnp.asarray(Vi[:, oi] / np.sqrt(wi[oi])))))[0]
    del Dic, Gi
    for ref_mode, Pm in (('zero', PhiAll), ('ic', PhiIC)):
        podq = make_pod_query(ref_mode)
        for k in cfg.get('pod_ks', []):
            A = dict(Phi=jnp.asarray(Pm[:, :k]))
            F, its = run_cases(podq, A, eval_phys)
            nm_ = f'pod_lspg_{ref_mode}_K{k}'
            report['arms'][nm_] = dict(family='pod_lspg', K=k, solved_dimension=k, reference=ref_mode, cohort=cohort_name, **errors(F, REF),
                                       gn_mean=float(its.mean()), gn_cap_hits=int((its >= cfg.get('gn_cap', 20)).sum()))
            subjects[nm_] = (podq, host(A))
            if cohort_name == 'validation': np.savez(out / f'fields_{nm_}.npz', fields=F.astype(fdt))
            print('POD-LSPG', ref_mode, k, report['arms'][nm_]['worst_evolved'], flush=True); save()

    # ------------------------------------------------------------------ Kim et al. arms
    nbr = np.stack((np.where(np.arange(n) // m > 0, np.arange(n) - m, -1), np.where(np.arange(n) // m < m - 1, np.arange(n) + m, -1),
                    np.where(np.arange(n) % m > 0, np.arange(n) - 1, -1), np.where(np.arange(n) % m < m - 1, np.arange(n) + 1, -1)), 1)

    def make_nm_query(K, ref_mode, jac_batch):
        def dec(z, A):   # interior state
            return A['ref'] + kimae.decode(A['p'], z, A['idx'], act) * A['scale']
        res = lambda z, zp, A: fom_res(dec(z, A), dec(zp, A), A['nu'])
        roll = lspg.make_rollout(res, STEPS, max_it=cfg.get('gn_cap', 20), save_every=KEEP, jac_batch=jac_batch)
        def query(u0, nu, A):
            ui = u0[1:-1, 1:-1].reshape(-1)
            ref = ui if ref_mode == 'ic' else jnp.zeros_like(ui)
            z0 = kimae.encode(A['p'], (ui - ref) / A['scale'], act)
            B = dict(A, ref=ref, nu=nu)
            Z, o = roll(z0, B)
            return pad(jax.lax.map(lambda z: dec(z, B), Z)), o[0].reshape(-1)
        def rollout_residuals(u0, nu, A):
            ui = u0[1:-1, 1:-1].reshape(-1)
            ref = ui if ref_mode == 'ic' else jnp.zeros_like(ui)
            z0 = kimae.encode(A['p'], (ui - ref) / A['scale'], act)
            B = dict(A, ref=ref, nu=nu)
            Z, o = lspg.make_rollout(res, STEPS, max_it=cfg.get('gn_cap', 20), save_every=1, jac_batch=jac_batch)(z0, B)
            R = jax.lax.map(lambda zz: res(zz[1], zz[0], B), (Z[:-1], Z[1:]))
            return R, Z
        def project(U, u0, A):   # autoencode the true states: the manifold's own floor
            ui = u0[1:-1, 1:-1].reshape(-1)
            ref = ui if ref_mode == 'ic' else jnp.zeros_like(ui)
            B = dict(A, ref=ref)
            return pad(jax.lax.map(lambda x: dec(kimae.encode(A['p'], (x - ref) / A['scale'], act), B), U))
        return jax.jit(query), jax.jit(rollout_residuals), jax.jit(project)

    def make_hr_query(K, ref_mode):
        def nodes(z, H):
            return jnp.concatenate((H['refn'] + kimae.decode(H['sub'], z, H['remap'], act) * H['scalen'], jnp.zeros(1)))
        def raw(z, zp, H):
            u, up = nodes(z, H), nodes(zp, H)
            c, xm, xp, ym, yp = (u[H['loc'][:, j]] for j in range(5))
            adv = c * L * (jnp.where(c > 0, c - xm, xp - c) + jnp.where(c > 0, c - ym, yp - c))
            lap = L ** 2 * (xm + xp + ym + yp - 4 * c)
            return c - up[H['loc'][:, 0]] + DT * (adv - H['nu'] * lap)
        res = lambda z, zp, H: H['pinv'] @ raw(z, zp, H)
        roll = lspg.make_rollout(res, STEPS, max_it=cfg.get('gn_cap', 20), save_every=KEEP)
        def query(u0, nu, H):
            ui = u0[1:-1, 1:-1].reshape(-1)
            ref = ui if ref_mode == 'ic' else jnp.zeros_like(ui)
            z0 = kimae.encode(H['p'], (ui - ref) / H['scale'], act)
            Z, o = roll(z0, dict(H, refn=ref[H['need']], nu=nu))
            full = jax.lax.map(lambda z: ref + kimae.decode(H['p'], z, H['idx'], act) * H['scale'], Z)
            return pad(full), o[0].reshape(-1)
        return jax.jit(query), raw

    for v in cfg['variants']:
        name, K = v['name'], int(v['K'])
        t_arm = time.monotonic()
        try:
            idx, valid, M2 = kimae.mask_tables(m, m, v['b'], v['db'])
            M1 = 2 * n if v['M1'] == '2n' else int(v['M1'])
            assert v.get('dtype', 'float32') in ('float32', 'float64')
            tdt = jnp.float32 if v.get('dtype', 'float32') == 'float32' else jnp.float64
            bytes_ = 4 if tdt == jnp.float32 else 8
            need_gb = (M1 * n * bytes_ * 4 + idx.size * bytes_ * 4) / 1e9
            D = U_fit - (U_fit[:, :1] if v['ref'] == 'ic' else 0.)
            D = D.reshape(-1, n)
            rng = np.random.default_rng(20260920 + int(v['seed']))
            perm = rng.permutation(D.shape[0]); nva = D.shape[0] // 10
            Dfit = D[np.sort(perm[nva:])]
            if v['scale'] == 'feature':
                sc = np.abs(Dfit).max(0); sc = np.maximum(sc, float(v.get('scale_floor', 1e-3)) * sc.max())
            else:
                sc = np.full(n, np.abs(Dfit).max())
            del Dfit
            micro = max(d for d in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 30, 40, 48, 60, 80, 120, 240)
                        if d * idx.size * bytes_ <= float(cfg.get('micro_bytes', 5e9)) or d == 1)
            print(f'ARM {name}: n={n} K={K} M1={M1} M2={M2} nnz={int(valid.sum())} micro={micro} weights+adam~{need_gb:.1f}GB', flush=True)
            p0 = kimae.init(jax.random.PRNGKey(int(v['seed'])), n, K, M1, M2, idx, valid, tdt)
            Xn = (D / sc).astype(np.float32 if bytes_ == 4 else np.float64)
            del D
            p, info = kimae.train(p0, jnp.asarray(Xn[perm[nva:]]), jnp.asarray(Xn[perm[:nva]]), idx, valid, act, batch=240, micro=micro,
                                  max_epochs=int(v['max_epochs']), wall_seconds=float(v['wall']), seed=int(v['seed']),
                                  lr0=float(v['lr']), lr_patience=int(v['patience']), tag=name)
            del Xn, p0
            hist = info.pop('history'); np.save(out / f'history_{name}.npy', hist)
            info.update(params=kimae.n_params(p, valid), M1=M1, M2=M2, micro=micro, dtype=str(np.dtype(tdt)),
                        seconds_per_epoch=float(info['seconds'] / max(info['epochs'], 1)))
            report['training'][name] = info
            print('TRAINED', name, {k: info[k] for k in ('stop_reason', 'epochs', 'best_val', 'seconds')}, flush=True); save()
            with open(out / f'ae_{name}.pkl', 'wb') as fh:
                pickle.dump(dict(params=host(p), scale=sc, variant=v), fh)
            A = dict(p=jax.tree_util.tree_map(lambda w_: jnp.asarray(w_, jnp.float64), p), scale=jnp.asarray(sc), idx=jnp.asarray(idx))
            jac_batch = None if n * idx.shape[1] * K * 8 < 6e9 else max(1, int(6e9 // (n * idx.shape[1] * 8)))
            nmq, nmres, nmproj = make_nm_query(K, v['ref'], jac_batch)
            # manifold floor (autoencode the truth) on the evaluation cohort
            PF = np.stack([np.asarray(nmproj(jnp.asarray(REF[c][:, 1:-1, 1:-1].reshape(6, n)), jnp.asarray(REF[c][0]), A)) for c in range(len(REF))])
            F, its = run_cases(nmq, A, eval_phys)
            arm = dict(family='kim_nm_lspg', K=K, cohort=cohort_name, variant=v, **errors(F, REF), gn_mean=float(its.mean()), gn_max=int(its.max()),
                       autoencode=errors(PF, REF), jac_batch=jac_batch, solved_dimension=K,
                       gn_cap_hits=int((its >= cfg.get('gn_cap', 20)).sum()), training_dtype=v.get('dtype', 'float32'), online_dtype='float64')
            report['arms'][name] = arm; subjects[name] = (nmq, host(A))
            if cohort_name == 'validation': np.savez(out / f'fields_{name}.npz', fields=F.astype(fdt))
            print('NM-LSPG', name, 'worst_evolved', arm['worst_evolved'], 'autoencode', arm['autoencode']['worst_evolved'], flush=True); save()

            # ---------------- hyper-reduction (their online algorithm); grid chosen on the tuning subset only
            if v.get('hr') and hr_allowed:
                rcases = phys['train'][FIT][::max(1, len(FIT) // int(cfg.get('hr_residual_cases', 16)))]
                Rs = np.concatenate([np.asarray(nmres(jnp.asarray(e.initial(L, p_)), float(p_[4]), A)[0]) for p_ in rcases]).T
                hrq, hraw = make_hr_query(K, v['ref'])
                best = None
                for nr, nz in v['hr']:
                    Phir = kimae.pod_basis(Rs, nr)
                    rows = kimae.greedy_samples(Phir, nz)
                    need = np.unique(np.concatenate((rows, nbr[rows][nbr[rows] >= 0].ravel())))
                    pos = np.full(n + 1, need.size, np.int64); pos[need] = np.arange(need.size)
                    loc = np.stack([pos[rows]] + [pos[np.where(nbr[rows, j] >= 0, nbr[rows, j], n)] for j in range(4)], 1)
                    sub, remap, nact = kimae.subnet(A['p'], idx, valid, need)
                    H = dict(p=A['p'], scale=A['scale'], idx=A['idx'], sub=sub, remap=remap, scalen=A['scale'][need], need=jnp.asarray(need),
                             loc=jnp.asarray(loc), pinv=jnp.asarray(np.linalg.pinv(Phir[rows])))
                    # control: sub-network rows reproduce the full residual rows
                    p_ = tune_phys[0]; ui = jnp.asarray(e.initial(L, p_))[1:-1, 1:-1].reshape(-1)
                    refv = ui if v['ref'] == 'ic' else jnp.zeros_like(ui)
                    za, zb = jnp.full(K, .1), jnp.full(K, -.05)
                    fullr = fom_res(refv + kimae.decode(A['p'], za, A['idx'], act) * A['scale'],
                                    refv + kimae.decode(A['p'], zb, A['idx'], act) * A['scale'], float(p_[4]))[jnp.asarray(rows)]
                    subr = hraw(za, zb, dict(H, refn=refv[H['need']], nu=float(p_[4])))
                    parity = float(jnp.max(jnp.abs(fullr - subr)) / jnp.max(jnp.abs(fullr)))
                    assert parity < 1e-9, parity
                    Ft, itst = run_cases(hrq, H, tune_phys)
                    et = errors(Ft, REF_tune)
                    hname = f'{name}_hr{nr}x{nz}'
                    report['arms'][hname] = dict(family='kim_nm_lspg_hr', K=K, cohort='tune', residual_basis=nr, samples=nz,
                                                 nodes_evaluated=int(need.size), active_hidden=nact, subnet_parity=parity, **et, gn_mean=float(itst.mean()))
                    print('NM-LSPG-HR (tune)', hname, et['worst_evolved'], flush=True); save()
                    if et['finite'] and (best is None or et['worst_evolved'] < best[0]):
                        best = (et['worst_evolved'], hname, H, nr, nz)
                if best is not None:
                    _, hname, H, nr, nz = best
                    report['arms'][name]['hr_selected'] = hname
                    subjects[name + '_hr'] = (hrq, host(H))
                    if cohort_name == 'validation':
                        Fh, ith = run_cases(hrq, H, eval_phys)
                        report['arms'][name + '_hr'] = dict(family='kim_nm_lspg_hr', K=K, cohort='validation', residual_basis=nr, samples=nz,
                                                            selected_on='tune', **errors(Fh, REF), gn_mean=float(ith.mean()))
                        np.savez(out / f'fields_{name}_hr.npz', fields=Fh.astype(fdt))
                        print('NM-LSPG-HR (validation)', name, report['arms'][name + '_hr']['worst_evolved'], flush=True)
                del Rs
            del A
            report['arms'][name]['arm_seconds'] = time.monotonic() - t_arm
            save()
        except Exception as exc:   # noqa: BLE001 - an arm that does not fit is a result, not a crash
            msg = str(exc)
            if 'RESOURCE_EXHAUSTED' in msg or 'out of memory' in msg.lower() or 'Out of memory' in msg:
                report['dropped'].append(dict(name=name, reason='OOM', detail=msg[:400], seconds=time.monotonic() - t_arm))
                print('DROPPED (OOM)', name, flush=True); jax.clear_caches(); save()
            else:
                raise
    del U_fit, Xfit

    # ------------------------------------------------------------------ the frozen project ROM (K=16, R=512)
    if cfg.get('rom_qs') is not None and len(cfg['rom_qs']):
        import arms as AR, topfix as TF
        assert sha_file(CKPT) == CKPT_SHA
        ck = pickle.load(open(CKPT, 'rb'))
        params = jax.tree_util.tree_map(jnp.asarray, ck['params'])
        Zold = np.asarray(ck['Z_tr']); K0 = Zold.shape[1]; R0 = int(np.asarray(ck['params']['h_lin']).shape[1])
        bank = AR.CoordBank(params, K0, R0); G = bank.on_grid(L); Qb, Rb = AR.whiten(G)
        Cfull = jnp.asarray(np.load(HERE / 'vendor/directions_qtd02.npz')['C'])
        trust = .01 * float(np.max(np.linalg.norm(Zold - Zold.mean(0), axis=1)))
        Zsub = Zold[::max(1, len(Zold) // 8192)]
        for q in cfg['rom_qs']:
            M = 4 * (K0 + q)
            ph, lm, _ = e.modes(L, M); P_ = jnp.asarray(ph)
            data = dict(A=P_.T @ G, lam=jnp.asarray(lm), G=G, Phi=P_)
            head = TF.corrected_head(params, Cfull[:, :q], K0)
            cold, cinfo = AR.build_cold(bank, head, np.concatenate((Zsub, np.zeros((len(Zsub), q))), 1), 48)
            tq = TF.make_query(params, Cfull[:, :q], K0, q, L, DT, trust, 'dense', 'base', Rb=Rb, ic_budget=400, step_budget=600,
                               gtol=1e-6, ic_gtol=1e-6, linear='gj' if K0 + q <= 64 else 'lu', inner_damping=1e-10, tau_y=.1)
            romq = jax.jit(lambda u0, nu, D_, _tq=tq: _tq(u0, nu, D_[0], D_[1])[:2])
            F, its = run_cases(romq, (data, cold), eval_phys)
            nm_ = f'ours_q{q}'
            report['arms'][nm_] = dict(family='ours', K=K0, R=R0, q=q, M=M, solved_dimension=K0 + q, cohort=cohort_name, **errors(F, REF), iterations_mean=float(its.mean()),
                                       checkpoint_sha256=CKPT_SHA, query='topfix.make_query dense base, b-panel 25434a27 settings')
            subjects[nm_] = (romq, (data, cold))
            if cohort_name == 'validation': np.savez(out / f'fields_{nm_}.npz', fields=F.astype(fdt))
            print('OURS', nm_, report['arms'][nm_]['worst_evolved'], flush=True); save()

    # ------------------------------------------------------------------ FOMs
    for fname, (ntol, ltol) in dict(fom_fft_tight=(1e-6, 1e-8), fom_nt1e4_dt005=(1e-4, 1e-6)).items():
        fq = jax.jit(lambda u0, nu, A_, _n=ntol, _l=ltol: ref_q(u0, nu, _n, _l, *A_)[:2])
        F, its = run_cases(fq, ref_pre, eval_phys)
        report['arms'][fname] = dict(family='fom', ntol=ntol, ltol=ltol, dt=DT, cohort=cohort_name, **errors(F, REF), newton_total_mean=float(its.sum(1).mean()))
        subjects[fname] = (fq, ref_pre)
    save()

    # ------------------------------------------------------------------ timing, one allocation, interleaved
    tcfg = cfg.get('timing')
    if tcfg:
        names = list(subjects)
        rng = np.random.default_rng(20260920)
        burn = jax.jit(lambda x: x @ x); bx = jnp.eye(512)
        def burn_in(sec):
            until = time.perf_counter() + sec
            while time.perf_counter() < until: burn(bx).block_until_ready()
        U0 = [jax.device_put(np.array(e.initial(L, p_))) for p_ in eval_phys[:tcfg['cases']]]
        subjects = {k: (q_, jax.device_put(A_)) for k, (q_, A_) in subjects.items()}
        first_out = {}
        for nm_ in names:      # compile/warm every subject; record compiled-memory analysis
            q_, A_ = subjects[nm_]
            t0 = time.perf_counter(); jax.block_until_ready(q_(U0[0], float(eval_phys[0, 4]), A_))
            report['arms'][nm_]['first_timing_call_seconds'] = time.perf_counter() - t0   # may include recompilation; not a compile time
            if hasattr(q_, 'lower'):
                report['arms'][nm_]['memory_analysis'] = mem_of(q_, U0[0], float(eval_phys[0, 4]), A_)
            report['arms'][nm_]['argument_bytes'] = int(sum(x.nbytes for x in jax.tree_util.tree_leaves(A_) if hasattr(x, 'nbytes')))
        rows = []
        for rep in range(tcfg['reps']):
            for c in range(tcfg['cases']):
                for i in rng.permutation(len(names)):
                    q_, A_ = subjects[names[i]]
                    burn_in(tcfg['burn'])
                    jax.block_until_ready(U0[c])
                    t0 = time.perf_counter(); r = q_(U0[c], float(eval_phys[c, 4]), A_); jax.block_until_ready(r); gpu_s = time.perf_counter() - t0
                    t1 = time.perf_counter(); _ = np.asarray(r[0]); host_s = time.perf_counter() - t1
                    rows.append((names[i], rep, c, gpu_s, host_s))
                    fld = np.asarray(r[0])
                    key = (names[i], c)
                    if key not in first_out:
                        first_out[key] = fld
                        np.save(out / f'timed_{names[i]}_case{c}.npy', fld.astype(fdt))
                    else:
                        assert np.array_equal(first_out[key], fld, equal_nan=True), ('timed repetitions differ', key)
        for nm_ in names:
            g = np.array([r[3] for r in rows if r[0] == nm_]); h = np.array([r[4] for r in rows if r[0] == nm_])
            report['arms'][nm_]['timing'] = dict(gpu_ms_median=float(np.median(g) * 1e3), gpu_ms_min=float(g.min() * 1e3), gpu_ms_max=float(g.max() * 1e3),
                                                 fields_to_host_ms_median=float(np.median(h) * 1e3), invocations=int(g.size))
        report['timing_rows'] = [dict(arm=r[0], rep=r[1], case=r[2], gpu_seconds=r[3], host_seconds=r[4]) for r in rows]
        report['timing_contract'] = ('supplied dense initial field resident on GPU -> six dense (L+1)^2 fields resident on GPU, block_until_ready; '
                                     f"{tcfg['burn']} s GPU burn-in before every invocation; fresh random subject order per (repetition, case)")
        try:
            report['process_peak_bytes'] = int(jax.devices()[0].memory_stats()['peak_bytes_in_use'])
        except Exception:   # noqa: BLE001
            pass
    report['wall_seconds'] = time.monotonic() - T0
    save()
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
