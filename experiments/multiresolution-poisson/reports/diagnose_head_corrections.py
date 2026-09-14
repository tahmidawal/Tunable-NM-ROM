"""Training-only nested correction spans from exact physical-metric residuals."""
from audit_iterative import *


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);args=ap.parse_args();run=args.run.resolve();out=run/'cluster/out/pilot';audit=json.loads((run/'audit.json').read_text());assert audit['passed'];d=json.loads((run/'result.json').read_text());assert d['complete'];cfg=d['config'];n=cfg['training_nodes']-1
    parameters=np.asarray(d['training']['parameters']);U=np.array([truth(n,param)[1:-1,1:-1].ravel() for param in parameters]);norm=np.linalg.norm(U,axis=1);models=[]
    for model in ['r128_head','r128_joint']:
        info=next(x for x in d['checkpoints'] if x['id']==model);payload=(out/info['path']).read_bytes();assert sha(payload)==info['sha256'];checkpoint=pickle.loads(payload);p=checkpoint['params'];z=checkpoint['Z_tr'];G=bank(p,n);Q,R=np.linalg.qr(G,mode='reduced');sv=np.linalg.svd(R,compute_uv=False);assert np.count_nonzero(sv>sv[0]*max(G.shape)*np.finfo(float).eps)==128
        target=U@Q;coeff=head(p,z);residual=(target-coeff@R.T)/norm[:,None];perp=np.linalg.norm(U-target@Q.T,axis=1)/norm;baseline=np.sqrt(np.sum(residual*residual,axis=1)+perp*perp)
        prior=next(x for x in audit['training_representation'] if x['model']==model);assert np.max(np.abs(baseline-prior['per_snapshot_checkpoint_error']))<1e-10 and np.max(np.abs(perp-prior['per_snapshot_bank_error']))<1e-10
        _,s,Vt=np.linalg.svd(residual,full_matrices=False);physical_directions=Vt[:32].T;directions=np.linalg.solve(R,physical_directions);assert np.linalg.norm((G@directions).T@(G@directions)-np.eye(32))<1e-9
        basispath=run/(model+'-training-correction-basis.npz');np.savez_compressed(basispath,coefficient_directions=directions,physical_metric_directions=physical_directions,R=R,singular_values=s,training_latents=z,training_norms=norm)
        item=dict(model=model,checkpoint_sha256=info['sha256'],basis_path=str(basispath),basis_sha256=file_sha(basispath),latent_code_status='retained jointly optimized training variables; no independent optimality/stationarity certification for these512 codes',training_cases=len(parameters),latent_dimension=z.shape[1],bank_rank=G.shape[1],worst_bank_projection_error=float(max(perp)),median_bank_projection_error=float(np.median(perp)),original_worst_reconstruction_error=float(max(baseline)),original_median_reconstruction_error=float(np.median(baseline)),singular_values=s.tolist(),normalized_inside_bank_residual_energy=float(np.sum(s*s)),correction_panels=[])
        for count in [0,8,16,32]:
            V=physical_directions[:,:count];left=residual-(residual@V)@V.T;corrected=np.sqrt(np.sum(left*left,axis=1)+perp*perp);fraction=float(np.sum(s[:count]**2)/np.sum(s*s));weak=[]
            for intervals in cfg['intervals']:
                B=np.load(out/f'cache_n{intervals}_{model}.npz')['B'];L=B@directions[:,:count];values=np.linalg.svd(L,compute_uv=False);rank=int(np.count_nonzero(values>values[0]*max(L.shape)*np.finfo(float).eps)) if count else 0
                weak.append(dict(intervals=intervals,actual_weak_modes=B.shape[0],correction_rank=rank,full_column_rank=rank==count,condition_number=float(values[0]/values[-1]) if count else None,remaining_projected_test_dimensions=B.shape[0]-count))
            item['correction_panels'].append(dict(correction_directions=count,nominal_augmented_latent_dimension=z.shape[1]+count,normalized_inside_bank_energy_captured=fraction,worst_fixed_code_corrected_error=float(max(corrected)),median_fixed_code_corrected_error=float(np.median(corrected)),rms_fixed_code_corrected_error=float(np.sqrt(np.mean(corrected*corrected))),training_cases_over_five_percent=int(np.count_nonzero(corrected>cfg['development_target'])),per_snapshot_corrected_error=corrected.tolist(),weak_direction_checks=weak))
        models.append(item);print(model,[(x['correction_directions'],x['normalized_inside_bank_energy_captured'],x['worst_fixed_code_corrected_error']) for x in item['correction_panels']],flush=True)
    result=dict(status='post-acceptance training-only correction diagnostic; no new network training or online PDE solve',source_result_sha256=file_sha(run/'result.json'),source_audit_sha256=file_sha(run/'audit.json'),script_sha256=file_sha(Path(__file__)),training_parameter_sha256=d['training']['parameter_sha256'],training_intervals=n,proposed_bank_target=cfg['proposed_gates']['bank_worst_relative_error'],online_physical_target=cfg['development_target'],models=models,
        construction='right singular vectors of normalized training residuals in exact QR physical metric; directions are nested prefixes; no development truth enters the construction',scope='Fixed saved training latents, with analytic linear correction coefficients chosen using training truth. These are constructive reconstruction diagnostics, not measured generic-input PDE results or certified nonlinear optima. The existing learned bank remains unchanged, including its missed3% development target. All42 development cases are already opened; final cohorts remain sealed.')
    (run/'head-correction-diagnostic.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
