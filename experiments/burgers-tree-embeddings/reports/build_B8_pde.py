"""Publish verified B8 neural PDE diagnostic figures and exact measured scope."""
from pathlib import Path
import hashlib,json,shutil
H=Path(__file__).resolve().parents[1]
CAN=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verified(job,name):
    proof=json.loads((H/'manifests'/f'retention-{job}.json').read_text())
    assert proof['local_and_remote_verified'] and proof['terminal_state']=='COMPLETED'
    row=next(r for r in proof['files'] if r['file']=='output/'+name)
    p=H/'results'/job/'output'/name;assert sha(p)==row['sha256'];return p
source=verified('b8treepde','result.json');r=json.loads(source.read_text())
meta=verified('b8treeplots','report_metadata.json');m=json.loads(meta.read_text())
assert m['source_result_sha256']==sha(source) and m['complete_source']
assert len(m['verified_fields'])==5 and r['all_steps_converged'] and r['steps_completed']==500
name='2026-11-19-neural-tree-pde-diagnostic';assets=H/'reports'/(name+'-assets');assets.mkdir(exist_ok=True)
files=[]
for filename in m['plot_hashes']:
    p=verified('b8treeplots',filename);assert sha(p)==m['plot_hashes'][filename]
    dest=assets/filename;shutil.copy2(p,dest);files.append(dest)
rows='\n'.join(f"| {v['time']:.2f} | {100*v['relative_error']:.6f}% | {100*v['reference_allowance']:.3g}% |" for v in r['outputs'])
t=r['timing'];recon=sum(v['device_reconstruction_seconds'] for v in r['outputs'])
report=f'''# First actual nonlinear neural-tree PDE diagnostic

The smooth held-out development case `{r['case']}` completes all500 embedding-space PDE steps. Every requested actual4096-by4096 output is below1% relative L2 error, using the current-time reference norm. **This is one fixed-tree diagnostic, not all-case qualification or a speedup claim.** Narrow-pulse, ridge and two-pulse neural PDE qualification remains outstanding; final tests remain sealed.

The model uses the frozen32-code B7 neural parent and4 independently adjustable16-code nonlinear children (96 variables total), trained on the original primary cohort. Windows and weights stay fixed; no online adaptation occurs. Initialization uses only the common initial-field callable and training library; future reference fields enter only after model output, for scoring. The original fine upwind/weak PDE formulation usesL16384, integrationQ128,512test modes, dt0.0005 and unchanged stopping tolerances.

![Actual PDE fields and errors]({assets.name}/pde_fields_actual4096.png)

All retained arrays and error calculations are4096×4096. The image displays512×512 samples to fit the page; it does not represent a lower-resolution solve. Reference and model share one color scale; absolute errors share another.

| Output time | Actual4096 relative L2 error | Reference allowance |
|---|---:|---:|
{rows}

![Error over time]({assets.name}/pde_error_over_time.png)

Initial fitting error is{100*r['initialization']['initial_callable_relative_error']:.6f}% on itsQ128 integration grid, reported separately from actual4096 output scores. All500 PDE steps converged and used5 LM iterations each. Time-step halving, integration refinement, harder cases and final method freeze are still required.

## Runtime and advancement

The single cold online query costs **{t['complete_cold_query_seconds']:.3f}seconds**, including source loading, initialization, assembly, compilation, Python control, mandatory acceptance transfers,500PDE steps and allfive requested device reconstructions. PDE steps account for{t['step_seconds_sum']:.3f}seconds; reconstruction including its explicit compilation accounts for{recon:.3f}seconds. This is an enclosing wall measurement, not a sum advertised as complete time.

Separately excluded: reference scoring{t['reference_scoring_seconds']:.3f}s, optional output-to-host transfer{t['host_transfer_seconds']:.3f}s, archival I/O{t['archival_io_seconds']:.3f}s, and GPUconditioning{t['excluded_seconds']['gpu_conditioning']:.3f}s. The invocation-to-snapshot wall is{t['elapsed_to_timing_snapshot_seconds']:.3f}s. Reference/archival work occurs between outputs in this diagnostic; clean paired runtime runs remain necessary. There are no paired CROM/SNF timings, repetitions or eligible speed ratios here.

![Diagnostic runtime]({assets.name}/pde_timing.png)

**Decision:** advance the neural implementation to equivalent Jacobian/output-cache optimization and numerical-convergence checks. Continue expanded nonlinear-child training and capacity diagnostics before claiming success on localized cases. Locality alone does not establish a faster PDE solve. Preserve this correct but costly implementation as the control.

The CROM high-rate attempt collapsed near a constant predictor and was interrupted with artifacts retained; it is not a fair competing result. A fresh lower-rate native-network repair is prepared. SNF native training-path qualification and matched-data training remain separate prerequisites. No competitor superiority claim is justified.

## Provenance

Solver job{r['meta']['job']}, source `{r['meta']['commit']}`; model `{r['model_job']}` hash `{r['model_sha256']}`. Plot job{m['meta']['job']}, source `{m['meta']['commit']}`. All solver/plot outputs retained with local and remote hash verification in `manifests/retention-b8treepde.json` and `manifests/retention-b8treeplots.json`. Plotting independently recomputed allfive full-grid errors and verified every saved error array against prediction-minus-reference.
'''
p=H/'reports'/(name+'.md');p.write_text(report);files.append(p)
published=[]
for src in files:
    dst=CAN/src.relative_to(H);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);assert sha(src)==sha(dst)
    published.append(dict(source=str(src),published=str(dst),sha256=sha(src)))
(H/'manifests/B8-pde-publication.json').write_text(json.dumps(dict(source_result_sha256=sha(source),plot_metadata_sha256=sha(meta),files=published),indent=2)+'\n')
print(CAN/'reports'/(name+'.md'))
