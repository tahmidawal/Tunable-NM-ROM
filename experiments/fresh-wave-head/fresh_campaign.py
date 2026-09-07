"""Complete fresh learned-bank/head/rollout campaign; final cohort remains closed."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from fresh_fom import provenance
from fresh_models import count_parameters, head_apply
from fresh_learning import generate_data, train_bank, project_data, common_affine, train_head, latent_fits
from fresh_evaluate import stats, reconstruction_metrics, nonlinear_rollouts, randomized_pod, linear_baseline


def save_json(path, obj):
    # Every failure must be explicit; JSON NaN is prohibited.
    def clean(x):
        if isinstance(x, dict):
            return {str(k):clean(v) for k,v in x.items()}
        if isinstance(x, (list,tuple)):
            return [clean(v) for v in x]
        if isinstance(x, np.generic):
            return clean(x.item())
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return x
    path.write_text(json.dumps(clean(obj),indent=2,allow_nan=False)+"\n")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--bc",choices=("both","reflective","absorbing"),default="both")
    args=ap.parse_args()
    config=json.loads(Path(args.config).read_text())
    if not config["reference_approved"]:
        raise RuntimeError("Truth verification/design not yet approved for training")
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    meta=provenance()
    if meta["jax_backend"]!="gpu" or not meta["x64"] or meta["matmul_precision"]!="highest":
        raise RuntimeError("GPU/f64/highest required")
    source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}
    result={"provenance":meta,"config":config,"config_sha256":hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),"source_sha256":source,"final_test_opened":False,"boundary_results":{}}
    save_json(out/"result.json",result)
    bcs=["dirichlet","absorbing"] if args.bc=="both" else ["dirichlet" if args.bc=="reflective" else "absorbing"]
    for bc in bcs:
        bout=out/bc
        bout.mkdir()
        grid,data=generate_data(config,bc,bout)
        bank=train_bank(config,grid,data["train"],bout)
        projected=project_data(data,bank)
        common=common_affine(projected["train"]["a"],config["latent"])
        np.savez_compressed(bout/"common_initialization.npz",linear=common[0],center=common[1],codes=common[2],output_scale=common[3])
        floors={}
        for split,pj in projected.items():
            floors[split]={"displacement":stats(np.sqrt(pj["u_floor_squared"].ravel())/pj["u_scale"]),"velocity":stats(np.sqrt(pj["v_floor_squared"].ravel())/pj["v_scale"])}
        entry={"bank":{"raw_rank_ratio":bank["raw_rank_ratio"],"orthogonality_defect":bank["orthogonality_defect"],"floors":floors},"arms":[],"linear_baselines":[]}
        result["boundary_results"][bc]=entry
        save_json(out/"result.json",result)
        # Fresh randomized POD is an explicitly labeled comparator, never the
        # neural spatial bank or an architecture input.
        pod,singular=randomized_pod(config,grid,data["train"])
        np.savez_compressed(bout/"fresh_pod_singular_values.npz",singular=singular)
        for label,g in (("fresh_randomized_pod_k",pod[:,:config["latent"]]),("fresh_randomized_pod_r",pod),("learned_bank_linear_r",bank["g"])):
            entry["linear_baselines"].append(linear_baseline(config,grid,g,data["validation"],label,bout))
            save_json(out/"result.json",result)
        for seed in config["optimizer_seeds"]:
            for kind,velocity_weight in (("mlp",0.),("quadratic",0.),("mlp",1.),("quadratic",1.)):
                name=kind+("_velocity" if velocity_weight else "")
                armout=bout/f"{name}_{seed}"
                armout.mkdir()
                p,frozen,codes,history=train_head(config,projected["train"],common,kind,velocity_weight,seed,armout)
                arm={"name":name,"kind":kind,"velocity_weight":velocity_weight,"optimizer_seed":seed,"parameter_count":count_parameters(p),"training_history":history}
                pj=projected["validation"]
                targets=pj["a"].reshape(-1,config["rank"])
                z,fit=latent_fits(config,p,frozen,targets,pj["u_scale"],common,kind,armout)
                arm["latent_fit"]=fit
                arm["validation"],w=reconstruction_metrics(p,frozen,z,pj,kind,armout)
                train_pred=np.asarray(head_apply(p,frozen,jnp.asarray(codes),kind))
                training_error=np.sqrt(np.sum((train_pred-projected["train"]["a"].reshape(train_pred.shape))**2,axis=-1)+projected["train"]["u_floor_squared"].ravel())/projected["train"]["u_scale"]
                arm["training_reconstruction"]=stats(training_error)
                zs=z.reshape(*pj["a"].shape[:2],config["latent"])
                arm["rollout"]=nonlinear_rollouts(config,grid,bank,data["validation"],p,frozen,zs,w,kind,armout)
                primary=[s for s in arm["rollout"]["summaries"] if s["dt"]==arm["rollout"]["primary_dt"]][0]
                arm["accuracy_passed"]=(primary["failed"]==0 and all(primary[key]["outliers"]==0 for key in ("displacement","velocity","energy_state")) and fit["nonstationary"]==0 and arm["validation"]["rank_failures"]==0)
                # Zero displacement is a separate diagnostic for affine bias and
                # absorbing late-time behavior; not a training/test example.
                zeroout=armout/"zero_state"
                zeroout.mkdir()
                zz,zero_fit=latent_fits(config,p,frozen,np.zeros((1,config["rank"])),np.ones(1)*float(np.median(pj["u_scale"])),common,kind,zeroout)
                arm["zero_state"]={"fit":zero_fit,"mass_norm":float(np.linalg.norm(np.asarray(head_apply(p,frozen,jnp.asarray(zz[0]),kind))))}
                entry["arms"].append(arm)
                save_json(armout/"result.json",arm)
                save_json(out/"result.json",result)
        del data,projected,bank
    result["completed"]=True
    save_json(out/"result.json",result)
    print("fresh_campaign_complete",flush=True)


if __name__=="__main__":
    main()
