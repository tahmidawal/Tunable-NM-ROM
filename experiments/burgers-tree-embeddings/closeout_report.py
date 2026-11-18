"""Generate the bounded-study decision only after all required repairs/audits finish."""
from pathlib import Path
import json,numpy as np,shutil,hashlib,re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
H=Path(__file__).resolve().parent;out=H/'reports';out.mkdir(exist_ok=True)
def read(job):return json.loads((H/'results'/job/'output/result.json').read_text())
ledger=json.loads((H/'cluster/ledger.json').read_text());assert all(r.get('state')=='COMPLETED' for r in ledger)
audits={r:read(job) for r,job in [('R0','audit01'),('R1','auditr1'),('R2','auditr2')]}
assert all(a['status']=='complete_representation_audit' for a in audits.values())
rows=[]
for recipe,a in audits.items():
    for key,method in [('fine_projection','Nonlinear child4'),('direct_fine_projection','Direct local4')]:
        for row in a.get(key,[]):
            e=np.asarray(row['projection']['per_snapshot'])
            rows.append(dict(recipe=recipe,method=method,allocation=row['allocation'],
                    variables=row['active_variables'][0],worst=float(e.max()),random=float(e[:16].max()),
                    alignment=float(e[16:].max()),median=float(np.median(e)),p90=float(np.quantile(e,.9)),
                    failing_snapshots=int(np.sum(e>.005)),failing_trajectories=int(np.sum(e.max(1)>.005)),
                    stationarity=row['max_stationarity']))
if any(r['worst']<=.005 for r in rows):
    raise RuntimeError('A candidate passed; do not close the study before justified PDE advancement.')
best=min(rows,key=lambda r:r['worst']);gpu_seconds=sum(r['elapsed_seconds'] for r in ledger)
cpu_hours=sum(r['elapsed_seconds']*r['allocated_cpus'] for r in ledger)/3600
references=[read(j) for j in ('refsdev01','refstrain01')]
allowance=max(max(c['allowance']) for r in references for c in r['cases'])
cross=read('crossref01');crossmax=max(max(c['levels'][-1]['relative_error']) for c in cross['cases'])
table='\n'.join(f'| {r["recipe"]} | {r["method"]} | {r["allocation"]} | {r["variables"]} | {100*r["random"]:.4f}% | {100*r["alignment"]:.4f}% | {100*r["median"]:.4f}% | {100*r["p90"]:.4f}% | {r["failing_trajectories"]}/28 |' for r in rows)
dense=[]
for recipe,a in audits.items():
    d=a['dense_witness'];q=d['grids']['4096']
    dense.append(dict(recipe=recipe,case=d['case'],time=d['time'],method=d.get('method','nonlinear_child'),
                      error=q['relative_l2'],pointwise=q['relative_linf'],change=d['scoring_change'],peak=q['peak_index']))
    f=a.get('final_time_output_witness')
    if f:dense.append(dict(recipe=recipe,case=f['case'],time=f['time'],method=f['method'],error=f['relative_l2'],pointwise=None,change=None,peak=None))
densetable='\n'.join(f'| {r["recipe"]} | {r["method"]} | {r["case"]} | {r["time"]:.3f} | {100*r["error"]:.5f}% |' for r in dense)
svtable='\n'.join(f'| {recipe} | {a["jacobian_zero"]["zero_child_max_field_change"]:.3g} | {a["jacobian_zero"]["rank"]}/{a["jacobian_zero"]["expected"]} | {a["jacobian_zero"]["useful_independent_rank"]} |' for recipe,a in audits.items())
decision=dict(status='SCOPED_NEGATIVE_STAGE1',advance_to_PDE=False,physical_speed_objective_met=False,
              projection_gate=.005,best_fine_projection=best,reference_allowance_max=allowance,
              independent_reference_difference_max=crossmax,allocated_gpu_seconds=gpu_seconds,
              allocated_gpu_hours=gpu_seconds/3600,allocated_cpu_hours=cpu_hours,jobs=len(ledger),
              final_test_solution_fields_opened=False,old_campaign='paused',merged=False,
              recipe_scope='R0 plus code-refit R1 and hidden-width128 R2; features16, child4, depth3, caps32/64/128',
              limitations=['finite-start nonlinear fits','global heads zero-anchored; not native unrestricted-bias reproductions',
                           'no PDE solve, deployable adaptation, or complete runtime benchmark',
                           'confirmation NM-ROM competitors not attempted after representation failure'])
(H/'DECISION-B0.json').write_text(json.dumps(decision,indent=2)+'\n')
fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained');labels=[];vals=[];colors=[]
for r in rows:
    labels.append(f'{r["recipe"]}\n{r["method"].replace(" ",chr(10),1)}\n{r["allocation"]}')
    vals.append(r['worst']*100);colors.append('#355d8c' if r['method']=='Direct local4' else '#5a7e55')
ax.bar(np.arange(len(vals)),vals,color=colors,width=.7)
ax.axhline(.5,color='#a53c3c',ls=':',label='0.5% advancement gate')
ax.set_xticks(np.arange(len(vals)),labels,fontsize=7)
ax.set(ylabel='Worst current-time relative L2 error (%)',title='Bounded recipes after independent128 × 128 Gauss scoring and code refitting')
ax.legend();ax.grid(axis='y',alpha=.15);fig.savefig(out/'bounded-repairs.png',dpi=170);fig.savefig(out/'bounded-repairs.svg');plt.close(fig)
text=f'''# Burgers tree embeddings: bounded representation study

The new study is implemented, trained and tested through reference qualification, the first decoder comparison and two registered repairs. **The tested recipe family does not pass the representation gate, so it does not advance to PDE solves or performance claims.** This is a scoped negative result for these checkpoints, not a rejection of hierarchical decoders.

The best independently rescored candidate is {best['recipe']} {best['method'].lower()} with {best['allocation']} allocation: worst{100*best['worst']:.4f}% at{best['variables']} active variables, versus the registered0.5% gate. The requested lower complete PDE-solve time at1% physical accuracy has not been established. Final-test solution fields remain unopened; the earlier campaign remains paused; no merge occurred.

## What ran

Fresh64-training and16-development trajectories plus12 alignment cases and two visible fixtures received the full independent space/time spectral ladders. Every trajectory contains11 snapshots. Maximum empirical spectral allowance is{100*allowance:.7f}%; the independent second-order reference checks all{len(cross['cases'])} required cases, with worst disagreement{100*crossmax:.5f}%, below0.1%. These are empirical convergence checks, not rigorous error certificates.

The initial comparison trained a128-feature global bank, global nonlinear heads16/32/64, shared16-channel local features with direct prefixes4/8/16, and a four-variable nonlinear child head. Uniform and prescribed allocations use the same checkpoint. Parents remain active and are refitted jointly with independent children. Prescribed allocation sees the initial-field callable only. Training uses a balanced set of uniform, connected-random and training-reference-guided configurations and their shallower truncations.

R0 trained codes with Adam alongside weights. A saved-checkpoint audit exposed a substantial training-code fitting gap, so its early stop is not a feature-learning plateau. R1 explicitly refits codes with up to25 damped GN/LM iterations before every assembled weight update. R2 retains that repair and widens only the local hidden layers from64 to128. Update ceilings, early stops, seed, data, feature width, child dimension, depth, windows and gates remain fixed. Every failed recipe and best/final checkpoint is retained.

## Fine scoring and advancement decision

All entries below refit codes on128-by128 Gauss integration with up to125 LM iterations and multiple cold/temporal-warm starts. Random and alignment panels stay separate; every snapshot counts. These are best-found fits, not certified global minima.

| Recipe | Decoder | Allocation | Variables | Worst random | Worst alignment | Median snapshot | 90th percentile | Failing trajectories |
|---|---|---|---:|---:|---:|---:|---:|---:|
{table}

![Bounded repairs](bounded-repairs.png)

No row passes the all-case0.5% screen. Stop this bounded recipe family after R1/R2. Larger feature banks, eight-dimensional children, depth4, different halos, unrestricted-bias global heads and expanded trajectory coverage were not tested; this study does not rule them out. Remaining fitting/conditioning limitations are recorded alongside the errors, rather than relabelled as a proof of an impossible manifold fit.

## Representation benefit versus cost

![Initial capacity comparison](representation-capacity.png)

The initial capacity plot uses64-by64 Gauss screening and is preserved as provisional history. It shows the root/global alternatives and actual variable counts; uniform trees can fall below a nominal cap. The fine table above governs the decision. Training-only global POD and optimistic local polynomial/linear spans distinguish data/span limits from learned-head limits. On R0, even the relaxed576-coefficient neural/polynomial controls failed the0.5% screen; they are diagnostics, not eligible128-variable methods.

![Parent and corrections](parent-children-witness.png)

The child correction adds to the jointly refitted parent shown separately. The visible error maps are reconstruction errors, not PDE rollout errors or refinement certificates.

![Component costs](representation-component-cost.png)

These are synchronized cached-feature field and Jacobian-vector-product calls at4096 integration points, with five saved repetitions in one GPU allocation. They include call/dispatch overhead and exclude compilation and feature-cache construction. They are **not** timings for4096-by4096 output or complete PDE queries. No cross-job wall-clock comparison is used. The central PDE-cost question remains unanswered because accuracy failed first.

## Actual dense outputs and implementation checks

Direct spectral evaluation and neural decoding produced actual4096-by4096 arrays at coordinates$j/4095$. They are reference-assisted projections; none was obtained by evolving a ROM. The final-time witness exercises the output shape at$t=0.25$ without calling it a PDE solution.

| Recipe | Decoder | Case | Time | Actual4096 relative L2 error |
|---|---|---|---:|---:|
{densetable}

![Actual dense witness](actual4096-witness.png)

Maximum pointwise errors, their locations,2048/4096 scoring changes, returned arrays and signed/absolute errors are retained in each audit. Pointwise diagnostics are not a1% pointwise guarantee. An independent NumPy audit reproduces the initial4096 error and rejects deliberate use of the old initial-field denominator.

| Recipe | Zero-child field change | Child Jacobian rank | Rank after removing parent directions |
|---|---:|---:|---:|
{svtable}

These Jacobian records audit the specified representative prescribed tree in the physical metric. Singular values and parent redundancy are saved. Geometry, physical weights, boundary behavior, AD/finite-difference derivatives and independent QR/SVD fitting fixtures pass. The common tested weak residual agrees with the exact inherited fine-upwind arithmetic on its dense fixture; this is not a production PDE accuracy qualification.

## Scope, failures and reproducibility

The global neural controls implement the planned matched-data Tunable form, using the first interface's zero-anchored linear-skip head. Unrestricted-bias native heads were not reproduced; the unrestricted direct-bank control is separate. These controls do not qualify a strong complete NM-ROM baseline by themselves. CROM, SNF-ROM and hyper-reduced masked-autoencoder confirmation remain unattempted because the preceding gate failed. They are not counted as defeated. No coarse-physics or truth-assisted online method was substituted.

Nonlinear fitting is finite-start and can retain nonzero stationarity residuals. Initial code-training failures, unsuccessful repairs and this reproduction limitation remain explicit. One local CUDA smoke attempt failed before numerical work because the shared local GPU was out of memory; its algebra check then passed on explicitly labelled CPU. Cluster numerical jobs all passed GPU preflight in float64/highest precision.

Resource accounting: {len(ledger)} owned allocations, {gpu_seconds} allocated GPU-seconds ({gpu_seconds/3600:.6f} GPU-hours) and{cpu_hours:.6f} allocated CPU-hours. The initial8-GPU-hour reservation was not expanded. Local GPU smoke durations were not instrumented; four attempts were each capped at55seconds (conservative upper bound220seconds, separate from measured cluster allocation). No local timing is used as a performance result.

Source, protocols, model checkpoints and figures are committed on `exp/2026-11-18-burgers-tree-embeddings` and backed up through its code-only mirror. Raw arrays/logs have checksum-verified local and designated owned-cluster copies. Job invocations, source commits, precision, GPUs, full fitting records and retention manifests remain in the lane. `REPRODUCE.md`, `DECISION-B0.json`, `configs/RECIPES.json` and `cluster/ledger.json` provide the exact replay and decision record.

## Glossary

- Active variables: independently adjustable global and retained-parent/child code entries.
- Projection: a reconstruction fitted with the reference field available; it does not solve the PDE.
- Current-time norm: the reference norm at the same time as the error being measured.
- Worst random/alignment: maximum over every snapshot in the respective panel.
- Failing trajectory: at least one snapshot exceeds the0.5% representation gate.
- Uniform/prescribed: complete-level allocation versus a connected tree selected from the initial field.
- Jacobian rank: number of independently changing field directions detected at zero child codes.
- Component cost: a timed portion of computation, insufficient for a complete-query speed claim.
- Reference allowance: an empirical numerical-convergence allowance, not a rigorous bound.
- Scoped negative: these registered models failed; untested variants and the broader idea remain open.
'''
text=re.sub(r'(?<=[a-z])(?=\d)',' ',text)
text=re.sub(r',(?=\d)',', ',text)
text=text.replace('Fresh 64-training and 16-development','Fresh 64 training and 16 development')
text=text.replace('returned arrays and signed/absolute errors are retained','returned arrays needed to recompute signed/absolute errors are retained')
report=out/'2026-11-18-tree-embedding-representation.md';report.write_text(text)
# Publish the generated report and its referenced figures in the canonical reports directory.
main=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/reports');assets=main/'2026-11-18-tree-embedding-representation-assets';assets.mkdir(exist_ok=True)
for file in ['bounded-repairs.png','representation-capacity.png','parent-children-witness.png','representation-component-cost.png','actual4096-witness.png']:
    shutil.copy2(out/file,assets/file);text=text.replace(']('+file+')',']('+assets.name+'/'+file+')')
(main/report.name).write_text(text)
(out/'closeout-source-hashes.json').write_text(json.dumps({str(p.relative_to(H)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [H/'cluster/ledger.json',H/'DECISION-B0.json']+[H/f'results/{j}/output/result.json' for j in ['audit01','auditr1','auditr2']]},indent=2)+'\n')
print(json.dumps(decision,indent=2))
