"""Prepare immutable 3D reconstruction inputs and approximate multimesh POD controls.

No network training or reduced dynamics. Real execution is cluster/GPU only.
The references remain in the pre-existing read-only E0 store. Training FOMs
are regenerated from the original bank recipe, never copied from data/.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time

import numpy as np
from scipy.spatial import cKDTree
import jax
jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'vendor'
REF = ROOT / 'experiments' / 'jcp-references'
sys.path.insert(0, str(REF))
import cohorts as CH
import refcore as RC
import refstore as RS
C = CH.C3
LF = 64
RANKS = [128, 256, 512, 768, 1024]


def clean(v):
    if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)): return [clean(x) for x in v]
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, (np.floating, np.integer)): return v.item()
    return v


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(a):
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(str((a.shape, a.dtype.str)).encode())
    h.update(memoryview(a).cast('B'))
    return h.hexdigest()


def log(s):
    print(s, flush=True)


def guard():
    assert jax.default_backend() == 'gpu'
    assert jax.config.jax_enable_x64
    assert os.environ.get('JAX_DEFAULT_MATMUL_PRECISION') == 'highest'
    log(f'jax_backend=gpu x64=True precision=highest gpu={jax.devices()[0].device_kind}')


def sine_prolong(fields, source, target=LF):
    """Physical field amplitudes, not Euclidean basis-normalized resampling."""
    n = source - 1
    x = jnp.asarray(fields).reshape((-1, n, n, n))
    spec = C.dst3(x) * (target / source) ** 1.5
    d = target - source
    return np.asarray(C.dst3(jnp.pad(spec, ((0, 0), (0, d), (0, d), (0, d))))).reshape(len(fields), -1)


def group_indices(n, cfg):
    x = C.interior_coords(n)
    cap = cfg['group_points_cap']
    if len(x) > cap:
        idx = np.sort(np.random.default_rng(cfg['subset_seed']).choice(len(x), cap, replace=False))
    else:
        idx = np.arange(len(x))
    return idx, x[idx]


def idw_map(points):
    """Frozen eight-neighbor inverse-squared-distance extension, exact on coincidences."""
    target = C.interior_coords(LF + 1)
    dist, ix = cKDTree(points).query(target, k=8, workers=1)
    zero = dist[:, 0] < 1e-14
    w = 1 / np.maximum(dist, 1e-14) ** 2
    w[zero] = 0
    w[zero, 0] = 1
    w[~zero] /= w[~zero].sum(axis=1, keepdims=True)
    assert np.isfinite(w).all() and np.max(np.abs(w.sum(1) - 1)) < 1e-14
    return ix.astype(np.int32), w


def apply_idw(u, ix, weight):
    out = np.empty((len(u), len(ix)), np.float64)
    for a in range(0, len(ix), 16384):
        out[:, a:a+16384] = (u[:, ix[a:a+16384]] * weight[None, a:a+16384]).sum(-1)
    return out


def generate_training(cfg, out, smoke=False):
    ntraj = 4 if smoke else cfg['train_count']
    steps = cfg['train_steps']
    count = ntraj * len(steps)
    p = (LF-1)**3
    t0 = time.perf_counter()
    tab = C.table(cfg['train_seed'], cfg['train_count'])  # same exact original count even in smoke
    expected = json.loads((HERE/'configs/frozen_inputs.json').read_text())
    assert tab['sha256'] == expected['bank_training_table_sha256'], 'training table hash differs'
    train_info = dict(seed=cfg['train_seed'], original_count=cfg['train_count'], used_count=ntraj,
                      table_sha256=tab['sha256'], steps=steps, groups={},
                      weighting='equal groups and snapshots; each mapped field divided by original sampled-field RMS',
                      exact_info_extension='33-node sine; 65-node identity; 129-node fixed 8NN IDW inverse squared distance',
                      stronger_extension='33-node sine; 65-node identity; 129-node full native field injection; extra original-unsampled values')
    raw_path = out / 'regenerable'
    raw_path.mkdir(exist_ok=True)
    match = np.lib.format.open_memmap(raw_path / 'pod_matched.npy', mode='w+', dtype=np.float64, shape=(3*count, p))
    strong129 = np.lib.format.open_memmap(raw_path / 'pod_strong129.npy', mode='w+', dtype=np.float64, shape=(count, p))
    fields_hashes = []
    np.savez(out / 'training_table.npz', **{k:v for k,v in tab.items() if isinstance(v,np.ndarray)})
    for group, n in enumerate(cfg['meshes']):
        idx, points = group_indices(n, cfg)
        np.save(out / f'training_indices_n{n}.npy', idx)
        idw = idw_map(points) if n == 129 else None
        if idw is not None:
            np.savez(out / 'idw129_to64.npz', indices=idw[0], weights=idw[1])
        fom = C.make_fom(n, C.DT, cfg['data_ntol'], cfg['data_ltol'], save_steps=steps)
        idxj = jnp.asarray(idx)
        sampled_hash = hashlib.sha256()
        full_extra_hash = hashlib.sha256()
        norms = np.empty(count)
        max_res, gstart = 0., time.perf_counter()
        for i in range(ntraj):
            f, it, rn = fom(jnp.asarray(C.initial_interior(n, tab, i)), float(tab['nu'][i]))
            rn = np.asarray(rn)
            sampled = np.asarray(f[:, idxj])
            assert np.isfinite(sampled).all() and np.isfinite(rn).all()
            assert rn.max() <= cfg['data_ntol'] and int(jnp.max(it)) < C.MAX_NEWTON
            max_res = max(max_res, float(rn.max()))
            sampled_hash.update(memoryview(np.ascontiguousarray(sampled)).cast('B'))
            # Original sampled values only; each group enters with equal state weight.
            # RMS differs from the bank's Euclidean row normalization by a fixed
            # per-group factor; this declared common-grid transfer uses equal-group
            # physical mean-square weights, not a claim of identical VP/tail loss.
            norm = np.sqrt(np.mean(sampled**2, axis=1))
            assert np.all(norm > 0)
            a = i * len(steps)
            norms[a:a+len(steps)] = norm
            if n == 33:
                mapped = sine_prolong(sampled, n-1)
            elif n == 65:
                mapped = sampled
            else:
                mapped = apply_idw(sampled, *idw)
                native = np.asarray(f).reshape((-1,n-2,n-2,n-2))
                jj = np.arange(1, LF) * 2 - 1
                full = native[:, jj][:, :, jj][:, :, :, jj].reshape(len(steps), -1)
                full_extra_hash.update(memoryview(np.ascontiguousarray(full)).cast('B'))
                strong129[a:a+len(steps)] = full / norm[:, None]
            match[group*count+a:group*count+a+len(steps)] = mapped / norm[:, None]
            if i % 64 == 0 or i == ntraj-1:
                log(f'training n={n} rows={i+1}/{ntraj} sec={time.perf_counter()-gstart:.1f}')
        match.flush(); strong129.flush()
        np.save(out / f'training_norm_n{n}.npy', norms)
        train_info['groups'][str(n)] = dict(points=len(idx), snapshots=count, indices_sha256=digest(idx),
             sampled_values_sha256=sampled_hash.hexdigest(), normalization_sha256=digest(norms),
             stronger_common_grid_sha256=full_extra_hash.hexdigest() if n==129 else None,
             max_residual=max_res, seconds=time.perf_counter()-gstart)
        dump(out / 'training-provenance.json', train_info)
        del fom, idxj, idw
        gc.collect()
    train_info['total_seconds'] = time.perf_counter()-t0
    dump(out / 'training-provenance.json', train_info)
    return match, strong129, train_info


class SnapshotView:
    def __init__(self, matched, strong=None):
        self.matched, self.strong = matched, strong
        self.shape = matched.shape
        self.offset = matched.shape[0]//3*2
    def rows(self, a, b):
        if self.strong is None or b <= self.offset:
            return np.asarray(self.matched[a:b])
        if a >= self.offset:
            return np.asarray(self.strong[a-self.offset:b-self.offset])
        return np.concatenate((self.matched[a:self.offset], self.strong[:b-self.offset]))


def randomized_pod(view, out, name, ranks, seed, powers=2, oversample=128):
    """Fixed randomized range finder. Q1 and Q2 are retained, never claimed globally optimal."""
    m, p = view.shape
    k = min(max(ranks)+oversample, m, p)
    block = 128
    rng = np.random.default_rng(seed)
    omega = jnp.asarray(rng.standard_normal((p,k)))
    mul = jax.jit(lambda x, v: x @ v)
    acc = jax.jit(lambda b, q, x: b + q.T @ x, donate_argnums=(0,))
    qfun = jax.jit(lambda y: jnp.linalg.qr(y, mode='reduced')[0])
    t0 = time.perf_counter()
    def range_q(v):
        y = np.empty((m,k), np.float64)
        for a in range(0,m,block):
            y[a:a+block] = np.asarray(mul(jnp.asarray(view.rows(a,min(a+block,m))),v))
        return qfun(jnp.asarray(y))
    q = range_q(omega)
    del omega
    captures = {}
    for power in range(powers+1):
        b = jnp.zeros((k,p), jnp.float64)
        for a in range(0,m,block):
            b = acc(b,q[a:a+block],jnp.asarray(view.rows(a,min(a+block,m))))
        jax.block_until_ready(b)
        log(f'POD {name} range power={power} sec={time.perf_counter()-t0:.1f}')
        if power >= 1:
            lam, eig = jnp.linalg.eigh(b @ b.T)
            order = jnp.arange(k-1,-1,-1)
            lam, eig = lam[order], eig[:,order]
            r = min(max(ranks),k)
            sig = jnp.sqrt(jnp.maximum(lam[:r],0.))
            assert float(sig[-1]) > 1e-12*float(sig[0]), (name, power, 'rank deficient')
            phi0 = b.T @ (eig[:,:r]/sig[None,:])
            phi, rr = jnp.linalg.qr(phi0,mode='reduced')
            phi = phi * jnp.sign(jnp.diag(rr))[None,:]
            qr_correction=float(jnp.linalg.norm(phi-phi0)/jnp.linalg.norm(phi0))
            qr_offdiagonal=float(jnp.linalg.norm(rr-jnp.diag(jnp.diag(rr)))/jnp.linalg.norm(rr))
            ortho = float(jnp.linalg.norm(phi.T @ phi-jnp.eye(r))/jnp.sqrt(r))
            assert ortho <= 1e-9, ortho
            np.save(out/f'{name}_q{power}_basis.npy',np.asarray(phi))
            np.save(out/f'{name}_q{power}_singular.npy',np.asarray(sig))
            # Full training capture is evaluated independently of the range approximation.
            total, captured = 0., np.zeros(r)
            probe = rng.standard_normal((p,8))
            probe /= np.linalg.norm(probe,axis=0,keepdims=True)
            pj = jnp.asarray(probe)
            probe_residual=0.
            for a in range(0,m,block):
                x = jnp.asarray(view.rows(a,min(a+block,m)))
                z = mul(x,phi)
                total += float(jnp.sum(x*x))
                captured += np.asarray(jnp.sum(z*z,axis=0))
                eprobe = x @ pj - z @ (phi.T @ pj)
                probe_residual += float(jnp.sum(eprobe*eprobe))
            np.save(out/f'{name}_q{power}_variances.npy',captured/m)
            cum = np.cumsum(captured)
            assert np.min(total-cum) >= -1e-10*total
            captures[f'q{power}'] = dict(orthogonality=ortho, qr_correction=qr_correction,
                qr_offdiagonal=qr_offdiagonal, actual_column_variance_sha256=digest(captured/m),total_sq=total,
                residual_fraction={str(rr):max(0.,float((total-cum[rr-1])/total)) for rr in ranks if rr<=r},
                random_probe_residual_sq=probe_residual,
                projected_singular_values=np.asarray(sig).tolist(),seconds=time.perf_counter()-t0)
            dump(out/f'{name}_certificates.json',dict(seed=seed,powers=powers,oversampling=oversample,
                approximate=True, captures=captures, shape=[m,p], ranks=ranks))
            del phi, phi0, rr, pj, z
        if power < powers:
            right = qfun(b.T)
            del b, q
            q = range_q(right)
            del right
    return captures


def prepare_bank(bank_path, out):
    bank = pickle.loads(Path(bank_path).read_bytes())
    p, t = bank['params'], bank['rotation']
    info={}
    for l in [16,32,64]:
        t0=time.perf_counter()
        basis=np.asarray(C.bank_at(p,t,C.interior_coords(l+1),chunk=4096))
        np.save(out/f'bank_L{l}.npy',basis)
        info[str(l)]={'shape':list(basis.shape),'seconds':time.perf_counter()-t0,'array_sha256':digest(basis)}
    dump(out/'bank-evaluation.json',info)


def prepare_coarse_inputs(out, smoke=False):
    meta={}
    # Buffer capacity only; the approved fifth temporal rung has 1602 stages.
    RC.NST = 2000
    solvers={l:RC.make_solver(3,l,adv='central',max_newton=40) for l in (16,32)}
    for name,count in [('dev4',4),('val64',64)]:
        if smoke and name=='val64': continue
        coh=CH.Cohort(3,name)
        cases=range(1 if smoke else count)
        for case in cases:
            ref,rm=RS.load(3,name,case,'ST')
            assert RS.reference_status(3,name,case,'ST')['eligible']
            n0=rm['judge']['n0']
            sub=out/'inputs'/name/f'c{case:03d}'
            sub.mkdir(parents=True,exist_ok=True)
            rec={'descriptor':coh.desc(case),'cohort_sha256':coh.sha256,'reference_store':RS.store_version(),
                 'reference_eps':rm['judge']['eps_ref'],'n0_fine':n0,'meshes':{}}
            for l in [16,32]:
                solve=solvers[l]
                history=[]
                last=None
                previous=None
                accepted=False
                for dt in [.0025,.00125,.000625,.0003125,.00015625]:
                    rr=solve(coh.ic(l,case),coh.nu(case),dt,'cnr',1e-11,1e-12)
                    fields=np.asarray(rr['out']); ns=int(rr['stages'])
                    res=np.asarray(rr['res'])[:ns]; nit=np.asarray(rr['nwt'])[:ns]
                    valid=bool(np.isfinite(fields).all() and np.isfinite(res).all() and
                               np.max(res)<=2e-11 and ns==RC.n_stages(dt,'cnr') and int(np.max(nit))<40
                               and int(rr['units'])==2*round(.25/dt))
                    diff=None
                    if last is not None:
                        diff=float(np.max(np.linalg.norm(fields[1:]-last[1:],axis=1))*(LF/l)**1.5/n0)
                    history.append({'dt':dt,'valid':valid,'max_residual':float(res.max()),'max_newton':int(nit.max()),
                                    'physical_last_pair_gap':diff,'stages':ns,'array_sha256':digest(fields)})
                    np.savez(sub/f'coarse_L{l}_dt{dt:.7f}.npz',fields=fields,residual=res,newton=nit)
                    previous,last=last,fields
                    if valid and len(history)>1 and history[-2]['valid'] and diff<=1e-5:
                        accepted=True;break
                np.savez(sub/f'coarse_L{l}_accepted.npz',fields=last,previous=previous)
                rec['meshes'][str(l)]={'eligible':accepted,'history':history,'finest_dt':dt}
                log(f'coarse {name}/{case} L{l} eligible={accepted} gap={history[-1]["physical_last_pair_gap"]}')
                del solve
            dump(sub/'meta.json',rec)
            meta[f'{name}/{case}']=rec
            dump(out/'coarse-inputs.json',meta)
    return meta


def verify_inputs(bank_path, training_path):
    expected=json.loads((HERE/'configs/frozen_inputs.json').read_text())
    assert hashlib.sha256(Path(bank_path).read_bytes()).hexdigest()==expected['bank_sha256']
    assert hashlib.sha256(Path(training_path).read_bytes()).hexdigest()==expected['training_sha256']
    assert RS.store_version()==expected['reference_store_sha256']
    vendor=json.loads((HERE/'vendor/PROVENANCE.json').read_text())
    for rel,info in vendor.items():
        assert hashlib.sha256((HERE/'vendor'/rel).read_bytes()).hexdigest()==info['sha256'],rel
    cfg=json.loads(Path(training_path).read_text())['config']
    contract=dict(meshes=[33,65,129],train_seed=923701,train_count=1536,
        train_steps=[0,1,2,4,7,10,15,20,25,30,40,50],subset_seed=923002,
        group_points_cap=250047,data_ntol=1e-8,data_ltol=1e-9,bank_rank=1024,bank_width=2048,bank_depth=3)
    for key,value in contract.items(): assert cfg[key]==value,(key,cfg[key],value)
    return cfg


def disjointness(out):
    def rows(tab):
        result=[]
        for i in range(tab['m']):
            b=int(tab['B'][i]); c=np.zeros((3,3));w=np.zeros(3);rho=np.zeros(3)
            c[:b]=tab['c'][i,:b];w[:b]=tab['w'][i,:b];rho[:b]=tab['rho'][i,:b]
            result.append(np.r_[b,c.ravel(),w,rho,tab['A'][i],np.log(tab['nu'][i])])
        return np.asarray(result)
    train=rows(C.b3.draw_param_table(923701,1536))
    dev=rows(C.b3.draw_param_table(923651,4)); val=rows(C.b3.draw_param_table(923801,64))
    keys=lambda a:{digest(r) for r in a}
    assert not (keys(train)&keys(dev) or keys(train)&keys(val) or keys(dev)&keys(val))
    scale=np.std(train,axis=0);scale[scale==0]=1
    tree=cKDTree(train/scale)
    dd=tree.query(dev/scale)[0];dv=tree.query(val/scale)[0]
    result=dict(train_seed=923701,dev_seed=923651,val_seed=923801,exact_overlap=False,
        standardized_parameter_distance_to_training=dict(dev=dd.tolist(),val=dv.tolist()),
        interpretation='active blob descriptors; distance is descriptive, not an independence certificate')
    np.savez(out/'disjoint_parameters.npz',train=train,dev=dev,val=val)
    dump(out/'disjointness.json',result)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bank',required=True)
    ap.add_argument('--training',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    guard()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    cfg=verify_inputs(args.bank,args.training)
    disjointness(out)
    metadata=dict(job_id=os.environ.get('SLURM_JOB_ID'),gpu=jax.devices()[0].device_kind,
                  commit=os.environ.get('SOURCE_COMMIT'),jax=jax.__version__,precision='highest',f64=True)
    dump(out/'execution.json',metadata)
    prepare_bank(args.bank,out)
    prepare_coarse_inputs(out,args.smoke)
    matched,strong,info=generate_training(cfg,out,args.smoke)
    ranks=[16,32] if args.smoke else RANKS
    for name,view in [('pod_matched',SnapshotView(matched)),('pod_stronger',SnapshotView(matched,strong))]:
        randomized_pod(view,out,name,ranks,seed=2026102703,powers=2,oversample=16 if args.smoke else 128)
    dump(out/'COMPLETE.json',dict(complete=True,smoke=args.smoke,execution=metadata))

if __name__=='__main__': main()
