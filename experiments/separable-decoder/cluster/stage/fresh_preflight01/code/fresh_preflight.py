"""Engineering preflight; every checkpoint is smoke-only and never promoted."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from fresh_fom import Grid,provenance
from fresh_models import tree_from_npz,head_apply,head_init
from fresh_learning import generate_data,train_bank,project_data,common_affine,train_head,latent_fits
from fresh_evaluate import reconstruction_metrics,nonlinear_rollouts,randomized_pod,linear_baseline
from fresh_rom import rollout


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",required=True)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=False)
    base=json.loads(Path(args.config).read_text())
    config={**base,"n":16,"rank":8,"weak_tests":8,"latent":2,"bank_width":32,"head_width":16,"bank_steps":4,"bank_batch":8,"head_steps":4,"head_batch":8,"fit_batch":8,"fit_iterations":12,"train_count":4,"validation_count":2,"end_time":.1,"rom_dts":[.01,.005,.0025]}
    result={"provenance":provenance(),"status":"engineering_smoke_only_not_a_scientific_result","config":config}
    if result["provenance"]["jax_backend"]!="gpu" or not result["provenance"]["x64"] or result["provenance"]["matmul_precision"]!="highest":
        raise RuntimeError("GPU/f64/highest required")
    bout=out/"small_pipeline"
    bout.mkdir()
    grid,data=generate_data(config,"absorbing",bout)
    bank=train_bank(config,grid,data["train"],bout)
    projected=project_data(data,bank)
    common=common_affine(projected["train"]["a"],config["latent"])
    result["heads"]={}
    for kind in ("mlp","quadratic"):
        armout=bout/kind
        armout.mkdir()
        p,f,codes,history=train_head(config,projected["train"],common,kind,1.,691200,armout)
        restored=tree_from_npz(armout/"head.npz")
        for original,loaded in zip(jax.tree.leaves(p),jax.tree.leaves(restored["p"])):
            np.testing.assert_array_equal(original,loaded)
        pj=projected["validation"]
        targets=pj["a"].reshape(-1,config["rank"])
        z,fit=latent_fits(config,p,f,targets,pj["u_scale"],common,kind,armout,codes)
        with np.load(armout/"latent_fits.npz") as saved:
            assert saved["initial_starts"].shape[1]==8
            np.testing.assert_array_equal(saved["initial_starts"],saved["doubled_initial_starts"])
        recon,w=reconstruction_metrics(p,f,z,pj,kind,armout)
        roll=nonlinear_rollouts(config,grid,bank,data["validation"],p,f,z.reshape(2,3,2),w,kind,armout)
        result["heads"][kind]={"fit":fit,"reconstruction":recon,"rollout":roll}
    pod,singular=randomized_pod(config,grid,data["train"])
    result["baseline"]=linear_baseline(config,grid,pod[:,:config["latent"]],data["validation"],"smoke_randomized_pod",bout)
    # Full production-size bank update cost uses synthetic smooth fields only.
    big={**base,"n":256,"bank_steps":18,"measure_bank_updates":True}
    bigout=out/"synthetic_bank_cost"
    bigout.mkdir()
    grid=Grid(256)
    xy=grid.coordinates().reshape(-1,2)
    fields=np.stack([np.sin(i*np.pi*xy[:,0])*np.sin(j*np.pi*xy[:,1]) for i,j in ((1,1),(1,2),(2,1),(2,2),(3,1),(1,3),(3,2),(2,3))])
    synthetic={"u":fields.reshape(4,2,-1),"v":np.roll(fields,1,axis=0).reshape(4,2,-1),"scales":np.ones((4,2))}
    bank=train_bank(big,grid,synthetic,bigout)
    repetitions=bank["update_duration_repetitions"]
    result["bank_update_cost"]={"n":256,"rank":64,"repetitions_seconds":repetitions,"median_seconds":float(np.median(repetitions)),"purpose":"Job-resource planning only; no ROM speed comparison."}
    p,f=head_init(jax.random.PRNGKey(699901),np.eye(64,16),np.zeros(64),.1,"mlp",128)
    kr,dr=jnp.asarray(bank["stiffness_unit"]),jnp.asarray(bank["damping_unit"])
    z0,w0=jnp.full(16,.01),jnp.full(16,.01)
    kwargs=dict(kind="mlp",steps=960,stride=20)
    warm=rollout(p,f,z0,w0,kr,dr,.0025,**kwargs)
    warm["z"].block_until_ready()
    warm_matrix=jnp.ones((1024,1024),dtype=jnp.float64)
    until=time.monotonic()+1.
    while time.monotonic()<until:
        (warm_matrix@warm_matrix).block_until_ready()
    times=[]
    for _ in range(3):
        start=time.perf_counter()
        rr=rollout(p,f,z0,w0,kr,dr,.0025,**kwargs)
        rr["z"].block_until_ready()
        times.append(time.perf_counter()-start)
    result["one_rollout_cost"]={"repetitions_seconds":times,"median_seconds":float(np.median(times)),"completed":bool(np.all(np.asarray(rr["completed"]))) }
    (out/"result.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print("fresh_preflight_complete",flush=True)


if __name__=="__main__":
    main()
