"""Build the historical report from pinned code and JSON evidence; no GPU imports.

Run with /home/tahmid/Dev/.venv/bin/python reports/2026-10-28-decoder-history/build.py
The first run pins source refs. Subsequent runs use those commits and check source hashes.
Copied code is a reading collection, not a standalone executable package.
"""
from pathlib import Path
import ast
import csv
import hashlib
import io
import json
import subprocess
import statistics

ROOT = Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude')
OUT = Path(__file__).resolve().parent
CON = 'exp/2026-10-20-jcp-consolidated'
MAY = 'origin/decoder-explorer'
SPEED = 'origin/inr-siren-speed-accuracy'
MAYDIR = 'Experiments/exploring-decoders/src/tunable_rom_decoders/decoders/'
SPEEDDIR = 'Experiments/2026-05-18-INR-SIREN-SPEED-ACCURACY-TRADEOFF/src/tunable_rom_speed/decoders/'

def git(*args):
    return subprocess.check_output(['git', '-c', 'gc.auto=0', '-C', str(ROOT), *args])

specs = {}
def src(key, ref, path):
    specs[key] = {'ref': ref, 'path': path}

src('cp', '839403136', 'heat/src/tunable_rom_heat/models/decoder.py')
src('linearcp', '839403136', 'poisson/src/tunable_rom_poisson/models/decoder.py')
src('vit', MAY, 'mirrored-vit-ablation/poisson/src/tunable_rom_vit_poisson/models/decoder.py')
for key, file in [('siren','modulated_siren'), ('xattn','cross_attn_inr'), ('default_siren','default_field_siren'), ('skip_siren','linear_skip_siren')]:
    src(key, MAY, MAYDIR + file + '.py')
for key,file in [('affine','affine_z'),('linear_affine','linear_affine_z')]:
    src(key, SPEED, SPEEDDIR + file + '.py')
src('may_journal', MAY, 'Experiments/exploring-decoders/JOURNAL.tex')
src('may_journey', MAY, 'Experiments/exploring-decoders/journey-to-good-manifold.md')
src('may_speed', SPEED, 'Experiments/2026-05-18-INR-SIREN-SPEED-ACCURACY-TRADEOFF/SUMMARY.md')
src('vit_status', MAY, 'mirrored-vit-ablation/STATUS.md')
src('concat', 'exp/2026-08-12-coord-decoder', 'experiments/coord-decoder/poisson2d_diag.py')
src('film', 'exp/2026-08-12-coord-decoder', 'experiments/coord-decoder/poisson2d_film.py')
src('film_heat', 'exp/2026-08-13-heat2d-coord-decoder', 'experiments/heat2d-coord-decoder/heat2d_film.py')
src('film_burgers', 'exp/2026-08-14-burgers2d-coord-rom', 'experiments/burgers2d-coord-rom/burgers2d_film.py')
src('film_wave', 'exp/2026-08-14-wave2d-coord-rom', 'experiments/wave2d-coord-rom/wave2d_film.py')
src('staged', 'exp/2026-08-14-multistage-precision', 'experiments/multistage-precision/ms_parametric.py')
src('staged_ad', 'exp/2026-08-14-multistage-precision', 'experiments/multistage-precision/ms_autodecoder.py')
for key,path in [
 ('film_arch','nonlinear-decoder-architecture/nda_arch.py'),
 ('nda_numbers','nonlinear-decoder-architecture/summary.json'),
 ('latent_burgers','burgers2d-rom-latent-stepping/blat_common.py'),
 ('latent_heat','heat2d-rom-latent-stepping/hlat_common.py'),
 ('latent_wave','wave2d-rom-latent-stepping/wlat_common.py'),
 ('separable','separable-decoder/sep_common.py'),
 ('headfit','separable-decoder/sep_hfit.py'),
 ('tensor','separable-decoder/b2d_tensor_common.py'),
 ('stokes','separable-decoder/stk2d_common.py'),
 ('wavebank','separable-decoder/wav2d_bank.py'),
 ('correction','multiresolution-poisson/correction_core.py'),
 ('linear_heat','mr-heat2d/run_linear.py'),
 ('ns2d','ns2d/ns2d_decoder.py'),
 ('shift','ns3d-shift/shift_rom.py'),
 ('shift_head','ns3d-shift-head/head_rom.py'),
 ('curlbank','ns3d-coordnet/coordnet.py'),
 ('banktrain','ns3d-coordnet/train_bank.py'),
 ('ordered','quadrature-study/qcore.py'),
 ('current_step','jcp-combined/t2core.py'),
 ('current_head','jcp-heads/hcore.py'),
 ('wide','jcp-wide-bank/w3d.py'),
 ('smooth','jcp-smooth-bank/smoothtrain.py'),
 ('pod_hr','jcp-hr-baselines/hrcore.py'),
 ('operators','ns3d-operators/ops3d.py')]:
    src(key, CON, 'experiments/' + path)
for arm in ('anchor','quadratic','encoder','mixture'):
    ref = 'exp/2026-09-06-b3d-' + arm
    src(arm, ref, 'experiments/separable-decoder/b3d_arch_' + arm + '.py')
    src(arm+'_numbers', ref, 'experiments/separable-decoder/runs/b3d_architecture/screen40/out/result.json')
for width in (128,192):
    src('mlp'+str(width)+'_numbers','exp/2026-09-06-burgers3d-repair',f'experiments/separable-decoder/runs/b3d_architecture/mlp{width}/out/result.json')
src('modcp','exp/2026-09-10-modcp-burgers2d','experiments/modcp-eq/common/decoders.py')
src('qman','exp/2026-09-22-quadratic-manifold','experiments/quadratic-manifold/qman.py')
src('masked','exp/2026-09-20-nmrom-baselines','experiments/nmrom-baselines/kimae.py')
src('heat_vp','exp/2026-09-21-heat3d-bank','experiments/heat3d-bank/train_vp.py')
src('prefix','exp/2026-10-27-burgers-prefix2d','experiments/burgers-prefix2d/prefixcore.py')
src('transfer','exp/2026-10-28-burgers-latent2d','experiments/burgers-latent2d/transfer.py')
src('latent_numbers','exp/2026-10-28-burgers-latent2d','experiments/burgers-latent2d/checks/summary.json')
for key,path in [
 ('old_podmlp','/home/tahmid/Dev/Neurips/share-bundle/code/lad/podmlp.py'),
 ('old_podmlp_rom','/home/tahmid/Dev/Neurips/share-bundle/code/lad/podmlp_rom.py'),
 ('old_wave','/home/tahmid/Dev/Tunable-NM-ROM/wave/src/tunable_rom_wave/models/pod_mlp.py'),
 ('old_heat','/home/tahmid/Dev/Neurips/h3red.py'),
 ('old_heat_log','/home/tahmid/Dev/Neurips/HEAT_ROUNDS.md'),
 ('old_ledger','/home/tahmid/Dev/Neurips/MASTER-RESULTS.md'),
 ('old_readme','/home/tahmid/Dev/Neurips/README.md'),
 ('tree_plan','/home/tahmid/Dev/QuadtreeNMROM/PLAN.md'),
 ('lablog',str(ROOT/'LAB-LOG.md'))]:
    src(key, None, path)

def pin():
    manifest_path = OUT/'sources.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = {}
    for key,spec in specs.items():
        if key in previous:
            item = previous[key]
            data = (OUT/item['copy']).read_bytes()
            assert hashlib.sha256(data).hexdigest() == item['sha256'], key
        else:
            item = dict(spec)
            if spec['ref']:
                item['commit'] = git('rev-parse', spec['ref']).decode().strip()
                data = git('show', item['commit']+':'+spec['path'])
            else:
                data = Path(spec['path']).read_bytes()
                item['commit'] = None
            dest = Path('sources')/key/Path(spec['path']).name
            (OUT/dest).parent.mkdir(parents=True, exist_ok=True)
            (OUT/dest).write_bytes(data)
            item['copy'] = str(dest)
            item['sha256'] = hashlib.sha256(data).hexdigest()
        manifest[key] = item
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

def link(key, label=None):
    return '['+(label or key)+']('+str(OUT/MANIFEST[key]['copy'])+')'

def tables():
    d=json.loads((OUT/MANIFEST['nda_numbers']['copy']).read_text())
    rows=[]
    for pde in ['poisson','burgers']:
        for group in [pde,pde+'_objectives']:
            for r in d[group]:
                rows.append(dict(r,pde=pde.title(),source_group=group))
    fields=['source_group','pde','cell','seed','architecture','hidden','layers','group_size','n_params','decoder','full','full_median','full_max','eq','eq_median','eq_max','eq256','eq512','M','m']
    with (OUT/'coordinate-search.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    # Use the recorded recommendation, not a new choice after seeing outcomes.
    selected=[]
    for pde in ['poisson','burgers']:
        rec=d[pde+'_gate_audit']['recommended']
        selected.extend(r for r in rows if r['source_group']==pde+'_objectives' and all(r.get(k)==rec[k] for k in ['architecture','hidden','group_size','M','m']))
    lines=['| PDE / cell | Seed | Decoder | Parameters | M / m | Fitted mean error | Full weak mean error | EQ mean / median / worst error |','|---|---:|---|---:|---|---:|---:|---|']
    for r in selected:
        pct=lambda k:f"{100*r[k]:.6f}%"
        lines.append(f"| {r['pde']} / `{r['cell']}` | {r['seed']} | {r['architecture']} | {r['n_params']} | {r['M']} / {r['m']} | {pct('decoder')} | {pct('full')} | {pct('eq')} / {pct('eq_median')} / {pct('eq_max')} |")
    a=json.loads((OUT/MANIFEST['latent_numbers']['copy']).read_text())
    recent=['| Arm | Median query (ms) | Median paired time ratio to direct | Slower timing pairs / all pairs | Fine corrections total | Cost gate |','|---|---:|---:|---|---:|---|']
    for name,r in a['arms'].items():
        recent.append(f"| {name} | {r['median_query_ms']:.5f} | {r['median_paired_ratio']:.6f} | {r['slower_pairs']} / {r['total_pairs']} | {r['total_fine_corrections']} | {'baseline' if r['cost_gate'] is None else ('pass' if r['cost_gate'] else 'fail')} |")
    heads=['| Head | Optimizer seed | Fitted validation mean | Median | Worst | States above 15% | Nonstationary fits |','|---|---:|---:|---:|---:|---:|---:|']
    for key in ['mlp128','mlp192','anchor','quadratic','encoder','mixture']:
        h=json.loads((OUT/MANIFEST[key+'_numbers']['copy']).read_text())
        for run in h['runs']:
            f=run['fits'][-1]
            errors=f['error']
            mean,median,worst=statistics.mean(errors),statistics.median(errors),max(errors)
            assert abs(mean-f['summary']['mean'])<1e-12
            assert abs(median-f['summary']['median'])<1e-12
            assert abs(worst-f['summary']['worst'])<1e-12
            outliers=sum(x>.15 for x in errors)
            assert outliers==f['outliers_above_15pct']
            heads.append(f"| {key} | {run['seed']} | {100*mean:.4f}% | {100*median:.4f}% | {100*worst:.4f}% | {outliers} | {f['nonstationary']} |")
    return '\n'.join(lines), '\n'.join(recent), '\n'.join(heads)

def code_index():
    lines=['# Decoder code and evidence index','',
           'Pinned reading copies accompany the report. They are not an independently runnable package: use the original commit and its configuration, dependencies, data generation and checkpoints to reproduce a run. External files are identified by content hash rather than an invented Git commit.','',
           '| Key | Reading copy | Original ref and file | Pinned commit | SHA-256 |','|---|---|---|---|---|']
    for key,item in MANIFEST.items():
        if key=='lablog':continue
        lines.append(f"| `{key}` | {link(key,Path(item['path']).name)} | `{item['ref'] or 'external local snapshot'}` · `{item['path']}` | `{item['commit'] or 'not asserted'}` | `{item['sha256']}` |")
    lines+=['','## Recover an original source file','',
            'Use the exact commit and original repository-relative path from the table. `git show COMMIT:PATH` reads the file without switching or modifying any worktree. Code-only mirrors may use a different commit; do not substitute hashes without verifying file contents.','',
            '## Glossary','',
            '**Key:** short identifier used by the report. **Ref:** branch or remote-tracking reference when this collection was pinned. **Pinned commit:** immutable Git revision whose file was copied. **SHA-256:** fingerprint of the exact copied bytes. **Reading copy:** preserved source for inspection, with dependencies left in its original project.']
    (OUT/'CODE-INDEX.md').write_text('\n'.join(lines)+'\n')

if __name__ == '__main__':
    MANIFEST=pin()
    code_index()
    report=(OUT/'report.source.md').read_text()
    import re
    report=re.sub(r'\{\{code:([a-z_0-9]+)\}\}',lambda m:link(m[1]),report)
    coordinate,recent,heads=tables()
    report=report.replace('{{coordinate_table}}',coordinate).replace('{{recent_table}}',recent).replace('{{head_table}}',heads)
    assert '{{' not in report
    (OUT/'2026-10-28-decoder-history.md').write_text(report)
    # Syntax-check preserved Python as text; do not import historical GPU code.
    for key,item in MANIFEST.items():
        p=OUT/item['copy']
        if p.suffix=='.py': ast.parse(p.read_text(), filename=str(p))
    missing=[]
    for target in re.findall(r'\]\((/[^)]+)\)',report):
        if not Path(target.split(':')[0]).exists():missing.append(target)
    assert not missing,missing
    print(f'Built report; pinned {len(MANIFEST)} sources; Python syntax and report links pass.')
