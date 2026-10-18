"""Recompute historical heat aggregates and document exact implementation transfers."""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
HISTORY = ROOT/'worktrees/2026-09-07-mr-heat2d/experiments/mr-heat2d/history/cp-heat-transfer-20260910'
RAW = HISTORY/'neurips/share-bundle/results/raw'
TARGET = ROOT/'reports/2026-07-30-cp-heat-optimization-transfer.md'


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    sources = json.loads((HISTORY/'SOURCE-MANIFEST.json').read_text())
    for item in sources['files']:
        if not item.get('missing'): assert digest(HISTORY/item['saved']) == item['sha256']
    rows=[]
    for p in sorted(RAW.glob('*.json')):
        a=json.loads(p.read_text()); e=a['extra']; per_rom=e['per_traj_rom_s']; per_fom=e['per_traj_fom_s']
        for stored, repeats in [(per_rom,e['rom_repeat_times_s']), (per_fom,e['fom_repeat_times_s'])]:
            assert len(stored)==len(repeats)
            for value, samples in zip(stored,repeats):
                assert len(samples)==a['protocol']['repeats'] and min(samples)>0
                assert math.isclose(value,median(samples),rel_tol=1e-12)
        ratios=[f/r for f,r in zip(per_fom,per_rom)]
        assert all(math.isclose(x,y,rel_tol=1e-12) for x,y in zip(ratios,e['per_traj_speedup']))
        for actual,saved in [(median(ratios),a['speedup']),(median(per_rom),a['solve_time_s']),
                             (median(per_fom),a['fom_time_s']),(mean(e['per_traj_rel_l2']),a['rel_l2'])]:
            assert math.isclose(actual,saved,rel_tol=1e-12)
        counts={label:sum(sum(t>1.5*median(group) for t in group) for group in e[key])
                for label,key in [('rom','rom_repeat_times_s'),('fom','fom_repeat_times_s')]}
        rows.append(dict(path=str(p.relative_to(ROOT)),sha256=digest(p),pde=a['pde'],N=a['N'],variant=a['variant'],
            rom_median_ms=1000*median(per_rom),fom_median_ms=1000*median(per_fom),speedup_median_ratio=median(ratios),
            mean_final_relative_error=mean(e['per_traj_rel_l2']),maximum_final_relative_error=max(e['per_traj_rel_l2']),
            faster_cases=sum(v>1 for v in ratios),cases=len(ratios),outliers=counts,
            job=e['slurm_job_id'],node=e['node'],source_summary=e['source_summary'],exclusive_node=a['protocol']['exclusive_node'],
            repeats=a['protocol']['repeats'],xla_flags=e['xla_flags'],x64=e['x64']))
    # The dense/reduced pair in each 2D arm must actually share the same timing job.
    for arm in ['acc','fast']:
        pair=[r for r in rows if r['pde']=='heat2d' and r['variant'].startswith('recompiled_'+arm+'_')]
        assert len(pair)==2 and len({(r['job'],r['source_summary']) for r in pair})==1
    lines=['# Historical CP heat optimizations and transfer to the current NMROM','',
        'Recovered source and archived July timing records, reviewed in September. The table is a recomputation of saved timing/error summaries, not a new GPU experiment or independent full-field validation; all rows remain historical and provisional for present-day paper claims.','',
        '## Sources recovered','',
        'The local NeurIPS directory contains a later heat cost study, reduced-Gram implementations and raw timing repetitions. The original April/May project also remains on the cluster, including the May heat repair branch and session records. Selected sources were copied into the current heat worktree with verified hashes; originals were preserved.','',
        f"[Preserved source inventory](../{(HISTORY/'SOURCE-MANIFEST.json').relative_to(ROOT).as_posix()}) · [July cost-study chronology](../{(HISTORY/'neurips/HEAT_ROUNDS.md').relative_to(ROOT).as_posix()}) · [May heat diagnosis](../{(HISTORY/'may/2026-05-21-heat/DIAGNOSIS.md').relative_to(ROOT).as_posix()}) · [May training and quadrature follow-up](../{(HISTORY/'may/claude-lab/sessions/2026-05-26.tex').relative_to(ROOT).as_posix()})",'',
        'The May notes contain successive corrections, changing diagnoses and accuracy-focused training experiments; their prose is historical evidence, not an independently verified explanation of the current model. The July study is particularly useful because it retains paired implementation controls and timing arrays. The original large paper speedups were subsequently corrected; this review does not reinstate them.','',
        '## Recomputed historical records','',
        '| PDE | N | Variant | Median ROM ms | Median FOM ms | Median per-case speedup | Mean final error (%) | Worst final error (%) | Faster cases | ROM / FOM timing outliers |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['pde']} | {r['N']} | `{r['variant']}` | {r['rom_median_ms']:.6f} | {r['fom_median_ms']:.6f} | {r['speedup_median_ratio']:.6f} | {100*r['mean_final_relative_error']:.6f} | {100*r['maximum_final_relative_error']:.6f} | {r['faster_cases']} / {r['cases']} | {r['outliers']['rom']} / {r['outliers']['fom']} |")
    lines += ['',
        'Provisional historical evidence: every row reports a shared node rather than an exclusive node. Errors are final-state errors averaged across cases in the original headline, not the current worst-over-all-requested-times metric. The FOM is the compiled iterative CG time integrator used in that study, not the current direct spectral/coarse FOM. Reduced variants also use a different precision arrangement from the present all-float64 protocol. This prevents transplanting a historical speed ratio into the current benchmark.','',
        'Per-case medians are recomputed from all stored timed repetitions: the measurement source performs warmups separately before storing these arrays. The speedup is the median of per-case FOM/ROM ratios, not the ratio of the two cohort median times. Timing outliers above one-and-a-half times their own case/method median are counted here and retained. Error scalars are checked against their stored per-case arrays; full fields and per-repetition error arrays were not independently validated in this review.','',
        '## What actually transfers','',
        '| Mechanism | Historical implementation | Current heat status | Next controlled comparison |',
        '| --- | --- | --- | --- |',
        '| Precompute spatial algebra | CP basis or quadrature stencils are fixed offline; reduced variants contract the normal equations into the coefficient space | Exact weak spatial operator is already precomputed; online residuals already avoid the full grid | Compare fully contracted normal equations against the existing small weak-residual assembly |',
        '| Small positive-definite solve | Cholesky replaces the general LU solve | Current LM uses a general linear solve | Replace only the damped-system factorization and check fields, gradients and stopping reasons |',
        '| Compile the full query | Including the formerly eager encoder produced a large gain in some archived controls | Current input fit, latent rollout and readout are already compiled | Preserve this; there is no missing Python-loop fix to claim |',
        '| Graph execution and counted loops | Compiler flag and fixed-count loops helped selected configurations and hurt others that converged early | No matched graph/counted-loop study in this current branch | Test only after the algebra comparison, applying compiler settings to both FOM and ROM |',
        '| Learned initialization and explicit amplitude | Encoder supplies a latent guess; solver evolves latent plus scale | Current model fits initial coordinates and has no evolving scale | Separate architecture/training follow-up; not required to test Cholesky or contracted algebra |','',
        'The previous source review was incomplete: it identified the encoder and amplitude differences but omitted the later reduced-Gram, Cholesky and graph studies. These provide a more direct implementation comparison before changing the time integrator or retraining the decoder.','',
        '## The current weak objective can be retained','',
        r'For the existing weak solve, write $r(z)=B h(z)-b$, with fixed bank-to-test matrix $B$. Precompute $S=B^\top B$ and, once per target, $c=B^\top b$. If $D=\partial h/\partial z$, then', '',
        r'$$J^\top J=D^\top S D,\qquad J^\top r=D^\top(S h-c),$$','',
        r'$$\|r\|_2^2=h^\top S h-2h^\top c+b^\top b.$$','',
        'The existing target normalization must be carried through every expression. This is algebraic contraction of the same weak objective, not the archived strong-form residual or a free-coefficient linear ROM. It preserves the existing nonlinear decoder, time steps and stopping tolerance mathematically. Numerical parity must still be measured: cancellation near a small residual and changes in rounding can alter an acceptance decision.','',
        'Our current weak residual is already small and independent of the full mesh size. Thus the old reduction from a grid-sized Jacobian is already largely achieved here; the remaining contraction could reduce the constant cost, but its gain is not established. Cholesky is the simplest missing change to isolate first. No current accuracy or speed improvement is claimed for either transfer yet.','',
        '## Provenance','']
    for r in rows:
        lines.append(f"- [`{r['variant']}`](../{r['path']}): job `{r['job']}`, node `{r['node']}`, originating summary `{r['source_summary']}`; source and array aggregates checked, historical only.")
    lines += ['', '## Glossary','',
        '- **CP / NMROM / FOM:** tensor-product decoder / nonlinear-manifold reduced model / full-grid numerical solver.',
        '- **N / PDE:** archived grid size per axis / partial differential equation.',
        '- **Variant:** archived implementation label; `acc` and `fast` identify the original operating settings, not a new success classification.',
        '- **red / v4w / v5w:** contracted coefficient-space normal equations / compiled dense comparison with Cholesky / minimal Cholesky-only solve change alongside the compiled encoder.',
        '- **Median ROM/FOM time:** median of case-level median query times, in milliseconds.',
        '- **Median per-case speedup / faster cases:** median of each case\'s FOM time divided by its ROM time / number of cases whose ratio exceeds unity.',
        '- **Mean or worst final error:** average or maximum over stored final-state relative errors across cases; neither measures the entire time history.',
        '- **Timing outlier:** stored repetition above the declared within-case threshold; not removed.',
        '- **Gram / normal equations / Jacobian:** precomputed inner-product matrix / small linear system for a nonlinear least-squares update / derivative matrix.',
        '- **Cholesky / LU / positive-definite:** specialized matrix factorization / general matrix factorization / property ensured here by the damped least-squares system in exact arithmetic.',
        '- **Weak objective / coefficient space:** residual tested against smooth functions / coordinates multiplying the fixed spatial bank.',
        '- **LM / CG:** Levenberg–Marquardt nonlinear least squares / conjugate-gradient linear solver.',
        '- **Graph / counted loop:** compiled GPU execution mechanism / loop with a fixed iteration count, potentially doing unnecessary work after convergence.',
        '- **Encoder / amplitude:** learned initial-coordinate map / separate scalar scaling the decoded field.',
        '- **Shared node / parity:** machine also hosting other jobs / agreement of two implementation paths.']
    TARGET.write_text('\n'.join(lines)+'\n')
    output=dict(scope='Historical source and aggregate arithmetic audit only; not a new accepted GPU benchmark.',rows=rows,
        source_manifest_sha256=digest(HISTORY/'SOURCE-MANIFEST.json'),report_sha256=digest(TARGET),generator_sha256=digest(Path(__file__)))
    TARGET.with_suffix('.json').write_text(json.dumps(output,indent=2)+'\n')
    print(TARGET)
    print(json.dumps([{k:r[k] for k in ['pde','variant','speedup_median_ratio','mean_final_relative_error','outliers']} for r in rows],indent=2))


if __name__=='__main__': main()
