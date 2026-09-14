"""Same-case representation, best-found fit, stationary solve and paired FOM pilot.

Inherited checkpoint has unmatched training history. Oracle fits consume references
and are diagnostic only. Timing arrays always refer to the saved invocation fields.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from dataset import preflight, sha_file, relative, ROOT, HERE


def run(index_path, output, smoke=False, max_cases=None, protocol=None):
    core, jax, runtime = preflight(smoke)
    import jax.numpy as jnp
    import correction_core as c
    from iterative_core import make_cg, cg_query, verify_cg
    from scipy.optimize import least_squares
    index_path = Path(index_path)
    manifest = json.loads(index_path.read_text())
    assert manifest['pde'] == 'poisson' and manifest['split'] == 'validation'
    assert manifest['runtime']['purpose'] == runtime['purpose']
    records = manifest['records'][:max_cases]
    n = manifest['intervals']
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    (output/'fields').mkdir()
    cfg = json.loads(Path(protocol or HERE/'protocol.json').read_text())
    params, codes, metadata = c.sc.load_pkl(ROOT/cfg['checkpoint'])
    basis = np.load(ROOT/cfg['basis'])
    assert sha_file(ROOT/cfg['checkpoint']) == cfg['checkpoint_sha256']
    assert sha_file(ROOT/cfg['basis']) == cfg['basis_sha256']
    np.testing.assert_array_equal(codes, basis['training_latents'])
    rank_requested = metadata['r']
    assert metadata['k'] == 16 and rank_requested in (128,256)
    ops = c.assemble(params, codes, n, cfg['requested_modes'], cfg['lm_budget'])
    engine = c.prepare_correction(ops, codes, basis['coefficient_directions'], 32, cfg)
    U, R = jnp.linalg.qr(ops['bank'], mode='reduced')
    singular = np.asarray(jnp.linalg.svd(R, compute_uv=False))
    rank = int(np.sum(singular > singular[0]*max(ops['bank'].shape)*np.finfo(float).eps))
    assert rank == rank_requested
    C = engine['C']
    # All correction fields Qphys=G C lie in G. Rank-revealing SVD of the
    # small coordinate matrix avoids spurious directions from dependent QR.
    UA, sa, _ = jnp.linalg.svd(jnp.concatenate((R, R@C), axis=1), full_matrices=False)
    augmented_rank = int(np.sum(np.asarray(sa) > float(sa[0])*max(ops['bank'].shape[0],rank_requested+32)*np.finfo(float).eps))
    assert augmented_rank == rank
    UA = UA[:, :augmented_rank]
    L, LR = jnp.linalg.qr(R@C, mode='reduced')
    P = jnp.eye(rank_requested)-L@L.T
    reduced_R = P@R
    head = jax.jit(c.sc.head)
    residual = jax.jit(lambda z, target, matrix: matrix@head(params,z)-target)
    jacobian = jax.jit(jax.jacfwd(residual, argnums=0))
    cached = head(params, jnp.asarray(codes))@reduced_R.T
    cg = make_cg(n, cfg['cg_maxiter']); lam=jnp.asarray(core.eigenvalues(n))
    data = dict(complete=False, runtime=runtime, config=cfg,
        index_sha256=sha_file(index_path), setup=ops['info'], correction_setup=engine['info'],
        checks=dict(cg=verify_cg(), bank_rank=rank, augmented_bank_rank=augmented_rank,
                    correction_span='Qphys=G C; [G,Qphys] and G have the same span'),
        cohort='new independent validation only; final cohort sealed',
        comparison_status=cfg.get('comparison_status','inherited ROM unmatched training history; pilot only'),
        rows=[], oracles=[], repetitions=1 if smoke else cfg['repetitions'])
    def persist():
        temp=output/'result.tmp'
        temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
        temp.replace(output/'result.json')
    def store(field):
        h=core.sha(field); path=f'fields/{h}.npz'
        if not (output/path).exists(): np.savez_compressed(output/path,field=np.asarray(field))
        return dict(field_path=path,field_sha256=sha_file(output/path),array_sha256=h)
    persist()
    methods=['nmrom','dst']+[f'cg_{t}' for t in cfg['cg_tolerances']]
    def query(method, source):
        if method=='nmrom': return c.correction_query(source,ops,engine,cfg)
        if method=='dst':
            field,row=core.fom_query(source,lam)
            row['fused_device_seconds']=row['solver_seconds']; row['solver_valid']=bool(np.isfinite(field).all())
            return field,row
        return cg_query(source,cg,float(method.removeprefix('cg_')))
    for record in records:
        file=index_path.parent/record['path']; assert sha_file(file)==record['sha256']
        with np.load(file) as a: source=a['input'][0]; discrete=a['target'][0,0]
        ref=record['reference']; ref_path=index_path.parent/ref['path']
        assert sha_file(ref_path)==ref['sha256']
        with np.load(ref_path) as a: physical=a['target'][0,0]; refinement=a['refinement'][0,0]
        references={'discrete':discrete,'physical_candidate':physical}
        # Warm all kernels; every measured invocation has its own burn and output.
        for method in methods: query(method,source)
        online_field,online_row=query('nmrom',source)
        for repetition in range(data['repetitions']):
            for method in methods if repetition%2==0 else methods[::-1]:
                core.burn(.001 if smoke else cfg['burn_seconds'])
                field,row=query(method,source)
                data['rows'].append(dict(case_id=record['case_id'], seed=record['seed'], mesh=n,
                    method=method,repetition=repetition, **row, **store(field),
                    discrete_relative_error=relative(field,discrete),
                    physical_candidate_relative_error=relative(field,physical),
                    refinement_relative_error=relative(field,refinement),
                    empirical_reference_budget_pass=ref['empirical_reference_budget_pass'],
                    reference_relative_change=ref['reference_relative_change']))
        for ref_name, target in references.items():
            vector=jnp.asarray(target[1:-1,1:-1].ravel()); coordinates=U.T@vector
            projected=U@coordinates
            augmented=U@(UA@(UA.T@coordinates))
            field=np.pad(np.asarray(projected).reshape(n-1,n-1),1)
            augmented_field=np.pad(np.asarray(augmented).reshape(n-1,n-1),1)
            assert relative(field,augmented_field)<1e-10
            target_reduced=P@coordinates
            order=np.argsort(np.asarray(jnp.sum((cached-target_reduced)**2,axis=1)))
            starts=[np.asarray(codes[i]) for i in order[:1 if smoke else cfg['oracle_starts']]]
            starts.append(np.asarray(online_row['latent']))
            fits=[]; best=None
            for ordinal, z0 in enumerate(starts):
                fit=least_squares(lambda z:np.asarray(residual(jnp.asarray(z),target_reduced,reduced_R)),
                    z0,jac=lambda z:np.asarray(jacobian(jnp.asarray(z),target_reduced,reduced_R)),
                    max_nfev=50 if smoke else cfg['oracle_max_nfev'],ftol=1e-12,xtol=1e-12,gtol=1e-12)
                h=head(params,jnp.asarray(fit.x))
                y=jax.scipy.linalg.solve_triangular(LR,L.T@(coordinates-R@h),lower=False)
                fit_field=np.pad(np.asarray(ops['bank']@(h+C@y)).reshape(n-1,n-1),1)
                error=relative(fit_field,target)
                entry=dict(start=ordinal,source='online_latent' if ordinal==len(starts)-1 else 'nearest_field_training_code',
                    relative_error=error,nfev=int(fit.nfev),njev=int(fit.njev),status=int(fit.status),
                    success=bool(fit.success),optimality=float(fit.optimality),latent=fit.x.tolist(),
                    correction_coefficients=np.asarray(y).tolist(),**store(fit_field))
                fits.append(entry)
                if best is None or error < best['relative_error']: best=entry
            data['oracles'].append(dict(case_id=record['case_id'],reference=ref_name,
                bank_projection=dict(relative_error=relative(field,target),**store(field)),
                augmented_bank_projection=dict(relative_error=relative(augmented_field,target),**store(augmented_field)),
                best_found_nonlinear_linear_fit=best,fit_starts=fits,
                online_relative_error=relative(online_field,target),online_solver_valid=online_row['solver_valid'],
                note='Same case/reference comparisons; not an additive decomposition or certified global optimum.'))
        persist(); print('diagnosed',record['case_id'],flush=True)
    data['complete']=True; persist()


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--index',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path); p.add_argument('--smoke',action='store_true')
    p.add_argument('--max-cases',type=int); p.add_argument('--protocol',type=Path)
    a=p.parse_args(); run(a.index,a.output,a.smoke,a.max_cases,a.protocol)
