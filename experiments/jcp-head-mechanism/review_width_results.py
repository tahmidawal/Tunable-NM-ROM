"""Independent CPU reconstruction of retained wider-prefix equation solutions."""
import argparse
import hashlib
import json
import itertools
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
WEAK = HERE.parents[2]/"2026-11-18-jcp-weak-visibility/experiments/jcp-weak-visibility"
parser = argparse.ArgumentParser()
parser.add_argument("width",type=int,choices=(256,512))
args=parser.parse_args()
RANK=args.width
JOB=WEAK/"runs"/f"w{RANK}"
OUT=JOB/"output"
BASE=WEAK/"runs/w128/output"
load=lambda p:json.loads(p.read_text())


def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):
            h.update(b)
    return h.hexdigest()


def bank(folder):
    return np.concatenate([np.load(p)["bank"] for p in sorted(folder.glob("bank-columns-*.npz"))],axis=1)


def metrics(field,truth,n):
    def derivatives(a):
        u=np.pad(a.reshape(-1,n-1,n-1),((0,0),(1,1),(1,1)))
        c=u[:,1:-1,1:-1]
        xminus,xplus=u[:,:-2,1:-1],u[:,2:,1:-1]
        yminus,yplus=u[:,1:-1,:-2],u[:,1:-1,2:]
        dx=np.where(c>0,c-xminus,xplus-c)*n
        dy=np.where(c>0,c-yminus,yplus-c)*n
        adv=c*(dx+dy)
        lap=(xminus+xplus+yminus+yplus-4*c)*n*n
        grad=np.stack(np.gradient(u,1/n,axis=(1,2)),axis=1)
        return [x.reshape(len(a),-1) for x in (grad,adv,lap)]
    norm=lambda x:np.linalg.norm(x,axis=1)
    fd,td=derivatives(field),derivatives(truth)
    return np.stack([norm(field-truth)/np.linalg.norm(truth[0]),
        *(norm(f-t)/np.maximum(norm(t),1e-300) for f,t in zip(fd,td)),
        .5*np.sum(field*field,axis=1)/n**2],axis=1)


verified=[]
for line in (JOB/"RESULTS.sha256").read_text().splitlines():
    expected,name=line.split(maxsplit=1)
    assert sha(JOB/name)==expected,name
    verified.append({"path":name,"sha256":expected})
collection=load(JOB/"collection.json")
assert collection["scheduler_state"]=="COMPLETED" and collection["exit_code"]=="0:0"
assert collection["checksum_verified"] and collection["gpu_log_valid"] and collection["no_environmental_warnings"]
env,summary=load(OUT/"environment.json"),load(OUT/"summary.json")
cfg=env["config"]
admission=load(JOB/"code/inputs/width-admission.json")
assert RANK in admission["admitted_widths"] and cfg["bank_width"]==RANK
for k,v in admission["fixed_contract"].items():assert cfg[k]==v,k
assert cfg["solve_rows"]==admission["allowed_test_spaces"][str(RANK)]
assert cfg["observer_start"]==8192 and cfg["observer_rows"]==1024
assert env["width_admission_sha256"]==sha(JOB/"code/inputs/width-admission.json")
assert env["checkpoint_sha256"]==admission["source_checkpoint_sha256"]
assert env["rotation_sha256"]==admission["source_rotation_sha256"]
assert env["source_sha256"]["core.py"]==admission["source_core_sha256"]
assert env["backend"]=="gpu" and env["x64"] and env["matmul_precision"]=="highest"
cohort=load(OUT/"cohort-drawing.json")
base_cohort=load(BASE/"cohort-drawing.json")
assert cohort["guard_passed"] and cohort["legacy_same_host_bitwise"] and cohort["first_four_columns_exact"]
assert cohort["selected_case_ids"]==list(range(6)) and cohort["local_portability_max_ulp"]<=4
assert cohort["parameter_sha256"]==base_cohort["parameter_sha256"]
assert env["buffer_residual_rows"]==9216
G,Gbase=bank(OUT),bank(BASE)
n=cfg["mesh_intervals"]
assert G.shape==((n-1)**2,RANK) and G.dtype==np.float64
prefix_gap=float(np.linalg.norm(G[:,:128]-Gbase)/np.linalg.norm(Gbase))
assert prefix_gap<=1e-10
prefix_gaps={"128":prefix_gap}
if RANK==512:
    G256=bank(WEAK/"runs/w256/output")
    prefix_gaps["256"]=float(np.linalg.norm(G[:,:256]-G256)/np.linalg.norm(G256))
    assert prefix_gaps["256"]<=1e-10
gram=G.T@G/n**2
ops=np.load(OUT/"operators.npz")
gram_gap=float(np.linalg.norm(gram-ops["gram"])/np.linalg.norm(gram))
R=np.load(OUT/"initialization-qr.npz")["Rmass"]
factor_gap=float(np.linalg.norm(R.T@R-gram)/np.linalg.norm(gram))
assert gram_gap<=1e-10 and factor_gap<=1e-10
eig=np.linalg.eigvalsh(gram);sing=np.linalg.svd(R,compute_uv=False)
ranks={str(t):{"gram":int(np.sum(eig>t*eig[-1])),"qr":int(np.sum(sing**2>t*sing[0]**2))} for t in (1e-10,1e-12,1e-14)}
assert all(r==RANK for values in ranks.values() for r in values.values())
assert all(r==RANK for r in [*env["tangent_ranks"].values(),*env["initialization_qr_ranks"].values()])
records,repetition_records,pointwise=[],[],[]
baseline_maxima=[]
baseline_curves={}
for c in range(6):
    folder=OUT/f"case-{c:02d}";base_folder=BASE/f"case-{c:02d}"
    cs=summary["cases"][str(c)]
    ref=np.load(folder/"reference-1.npz")["fields"]
    base_ref=np.load(base_folder/"reference-1.npz")["fields"]
    norm0=np.linalg.norm(ref[0])
    ref_gap=float(np.max(np.linalg.norm(ref-base_ref,axis=1))/norm0)
    assert cs["reference_eligible"] and ref_gap<=cfg["reference_max_disagreement"]
    assert cs["initialization_control"]["passed"]
    references=[]
    for level in (0,1):
        z=np.load(folder/f"reference-{level}.npz")
        assert np.isfinite(z["fields"]).all() and z["iterations"].max()<20 and z["residual"].max()<=1e-10
        assert cs["reference_controls"][level]["arithmetic_parity"]["passed"]
        references.append(z["fields"])
    assert np.max(np.linalg.norm(references[0]-references[1],axis=1))/norm0<=cfg["reference_max_disagreement"]
    projection=np.load(folder/"projection.npz")
    projected=projection["coeff"]@G.T
    projection_error=np.linalg.norm(projected-ref,axis=1)/norm0
    assert np.max(np.abs(projection_error-projection["error"]))<=1e-10
    orth=float(np.linalg.norm((projected-ref)@G)/(np.linalg.norm(ref)*np.linalg.norm(G)))
    assert orth<=1e-9
    base_z=np.load(base_folder/"m2048-tighter-rep0.npz")
    base_error=np.linalg.norm(base_z["coeff"]@Gbase.T-base_ref,axis=1)/np.linalg.norm(base_ref[0])
    baseline_maxima.append(float(base_error.max()))
    baseline_curves[c]=base_error
    errors={}
    for m in cfg["solve_rows"]:
        arm=cs["arms"][str(m)]
        assert arm["numerically_eligible"] and arm["solver_pair_eligible"]
        assert all(p["passed"] for p in arm["parity"])
        assert all(s["same_rank"] and s["rescaling_invariance_relative"]<=1e-8 for s in arm["reached_spectra"])
        fields={}
        for level,gtol in cfg["solver_gtols"].items():
            reps=cfg["timing_repetitions"] if level=="tight" else 1
            for rep in range(reps):
                z=np.load(folder/f"m{m}-{level}-rep{rep}.npz")
                assert np.isfinite(z["coeff"]).all() and np.all(z["stats"][:,1]==1) and np.all(z["stats"][:,2]<=gtol)
                assert np.array_equal(z["coeff"][0],projection["coeff"][0])
                assert float(z["elapsed_seconds"])==arm["levels"][level]["timing_repetitions"][rep]
                field=z["coeff"]@G.T
                error=np.linalg.norm(field-ref,axis=1)/norm0
                metric_gap=float(np.max(np.abs(error-z["metrics"][:,0])))
                assert metric_gap<=1e-10
                repetition_records.append({"case":c,"rows":m,"level":level,"rep":rep,"error_gap":metric_gap,
                    "maximum_stationarity":float(z["stats"][:,2].max())})
                fields[level]=field
        z=np.load(folder/f"m{m}-tighter-rep0.npz")
        physical_metrics=metrics(fields["tighter"],ref,n)
        metric_gap=float(np.max(np.abs(physical_metrics-z["metrics"])))
        assert metric_gap<=1e-9
        error=physical_metrics[:,0];errors[m]=error
        tol_gap=float(np.max(np.linalg.norm(fields["tight"]-fields["tighter"],axis=1))/norm0)
        assert tol_gap<=arm["tight_tighter_threshold"]
        evolution=np.linalg.norm(fields["tighter"]-projected,axis=1)/norm0
        assert np.max(np.abs(evolution-z["evolution_distance"]))<=1e-10
        identity=float(np.max(np.abs(error**2-projection_error**2-evolution**2)))
        assert identity<=1e-10
        residual=np.load(folder/f"m{m}-residuals.npz")
        assert residual["scaled"].shape==(50,9216)
        shell=float(np.max(np.linalg.norm(residual["scaled"][:,8192:9216],axis=1)))
        ushell=float(np.max(np.linalg.norm(residual["unscaled"][:,8192:9216],axis=1)))
        assert np.isclose(shell,arm["independent_scaled_shell_max"],rtol=1e-12,atol=1e-15)
        assert np.isclose(ushell,arm["independent_unscaled_shell_max"],rtol=1e-12,atol=1e-15)
        records.append({"case":c,"rows":m,"max_error":float(error.max()),"initial_error":float(error[0]),
            "baseline_max_error":float(base_error.max()),"case_maximum_ratio":float(error.max()/base_error.max()),
            "derivative_metric_max_gap":metric_gap,"tight_tighter_gap":tol_gap,"reference_baseline_gap":ref_gap,
            "pythagorean_residual":identity,"shell_max":shell,"unscaled_shell_max":ushell,"errors":error.tolist()})
        difference=error-base_error
        pointwise.append({"case":c,"previous_width":128,"previous_rows":2048,"width":RANK,"rows":m,
            "increased_time_count":int(np.sum(difference>1e-10)),"maximum_increase":float(max(0,difference.max()))})
    richer=max(cfg["solve_rows"])
    difference=errors[richer]-errors[2048]
    pointwise.append({"case":c,"previous_width":RANK,"previous_rows":2048,"width":RANK,"rows":richer,
        "increased_time_count":int(np.sum(difference>1e-10)),"maximum_increase":float(max(0,difference.max())),
        "time_of_maximum_increase":float(np.argmax(difference)*cfg["dt"])})

aggregates={}
for m in cfg["solve_rows"]:
    rows=[r for r in records if r["rows"]==m]
    worst=max(r["max_error"] for r in rows)
    ratios=[r["case_maximum_ratio"] for r in rows]
    aggregates[str(m)]={"case_count":len(rows),"maximum_error_worst":worst,"maximum_error_median":float(np.median([r["max_error"] for r in rows])),
        "worst_error_ratio_to_R128":worst/max(baseline_maxima),"maximum_case_error_ratio":max(ratios),
        "all_case_maxima_improve":all(r<1 for r in ratios),"primary_improvement_passed":worst<=.8*max(baseline_maxima) and max(ratios)<=1.02}
curves={(128,2048):baseline_curves}
for m in cfg["solve_rows"]:
    curves[(RANK,m)]={r["case"]:np.asarray(r["errors"]) for r in records if r["rows"]==m}
previous_review=None
if RANK==512:
    previous_review=HERE/"reviews/p2-R256-independent-result-review.json"
    admitted256=load(previous_review)
    assert admitted256["all_numerical_checks_passed"]
    for m in (2048,4096):
        curves[(256,m)]={r["case"]:np.asarray(r["errors"]) for r in admitted256["records"] if r["rows"]==m}
all_pairs=[]
for previous,current in itertools.combinations(sorted(curves),2):
    category="common-M width" if previous[1]==current[1] else ("within-width tests" if previous[0]==current[0] else "joint width and test changes")
    for c in range(6):
        difference=curves[current][c]-curves[previous][c]
        all_pairs.append({"case":c,"previous_width":previous[0],"previous_rows":previous[1],"width":current[0],"rows":current[1],
            "comparison_kind":category,"increased_time_count":int(np.sum(difference>1e-10)),
            "maximum_increase":float(max(0,difference.max())),"time_of_maximum_increase":float(np.argmax(difference)*cfg["dt"])})
review={"reviewer":"head_mechanism_run","decision":"Numerical results accepted; prospective primary improvement is reported separately for every setting.",
    "width":RANK,"job_id":env["job_id"],"source_commit":env["commit"],"verified_file_count":len(verified),
    "rehashed_files":verified,"collection":{k:v for k,v in collection.items() if k!="files"},
    "summary_sha256":sha(OUT/"summary.json"),"rank_checks":ranks,"R128_prefix_relative_gap":prefix_gap,
    "nested_prefix_relative_gaps":prefix_gaps,"cohort_parameter_sha256":cohort["parameter_sha256"],
    "gram_reconstruction_gap":gram_gap,"QR_factor_gap":factor_gap,"arm_count":len(records),"saved_repetition_count":len(repetition_records),
    "all_numerical_checks_passed":True,"aggregates":aggregates,"baseline_and_within_width_pointwise_comparisons":pointwise,
    "all_registered_arm_pair_comparisons":all_pairs,"all_registered_arm_pair_regressions":[p for p in all_pairs if p["increased_time_count"]>0],
    "previous_R256_review_sha256":sha(previous_review) if previous_review else None,
    "baseline_and_within_width_pointwise_regressions":[p for p in pointwise if p["increased_time_count"]>0],"records":records,"repetition_records":repetition_records,
    "scope":"All six exposed cases and all saved times/repetitions; same-grid same-step discrete target. No monotonic pointwise-error, continuum or cross-job timing claim."}
path=HERE/"reviews"/f"p2-R{RANK}-independent-result-review.json"
path.write_text(json.dumps(review,indent=2)+"\n")
print(json.dumps({k:v for k,v in review.items() if k not in ("rehashed_files","records","repetition_records","baseline_and_within_width_pointwise_comparisons","all_registered_arm_pair_comparisons","collection")},indent=2))
