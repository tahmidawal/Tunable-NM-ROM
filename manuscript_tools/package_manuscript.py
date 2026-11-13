"""Validate, document and package the locally revised Overleaf manuscript."""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import difflib
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper_revision'
TOOLS = ROOT / 'manuscript_tools'
aux = (PAPER/'main.aux').read_text()
log = (PAPER/'main.log').read_text()
assert not re.search(r'Overfull|undefined|^!', log, re.M), 'Inspect LaTeX diagnostics'
pdfinfo = subprocess.check_output(['pdfinfo', str(PAPER/'main.pdf')], text=True)
pages = int(re.search(r'Pages:\s+(\d+)', pdfinfo)[1])
def label(name):
    m = re.search(r'\\newlabel\{'+re.escape(name)+r'\}\{\{([^}]+)\}\{([^}]+)\}', aux)
    assert m, name
    return {'number':m[1], 'page':int(m[2])}

refs = {k:label(k) for k in ['sec:conclusion','fig:ns-head-span','tab:g1','tab:g2',
                            'tab:baselines','tab:ablation','tab:cyl-extra',
                            'eq:bank-loss-vp','tab:breakeven','fig:meshflat']}
assert refs['sec:conclusion']['page'] <= 10
assert refs['fig:ns-head-span']['page'] <= 10
assert refs['eq:bank-loss-vp']['number'] == '22'
assert refs['tab:cyl-extra']['number'] == 'E.2'
assert refs['fig:meshflat']['number'] == 'G.3'
original = TOOLS/'original'
assert (original/'main.tex').read_bytes() == (PAPER/'submitted.tex').read_bytes()
old_changes = PAPER/'CHANGES.md'
if not (TOOLS/'previous-changes.md').exists():
    (TOOLS/'previous-changes.md').write_bytes(old_changes.read_bytes())
old_changes.write_text(f'''# Rebuttal manuscript revision, 2026-11-13

The updated manuscript builds to {pages} pages. Its conclusion ends on page
{refs['sec:conclusion']['page']}, within the discussion-phase main-text limit.
The source ZIP supplied by the author is preserved under `manuscript_tools/original`;
its `main.tex` exactly matches `submitted.tex`, which remains unchanged.
The author's `todonotes` and `soul` preamble additions are retained.

## Changes from the uploaded source

- Incorporated the existing rebuttal revision: harder interacting-vortex Navier–Stokes,
  cylinder flow, matched solvers, neural-field and matched-data operator comparisons,
  ablations, loss definitions, offline costs and supporting figures.
- Included the completed cylinder CPU AMG, Navier–Stokes batched-throughput and
  reduced-solve scaling studies. No pending experiment is presented as a result.
- Added main-text Figure {refs['fig:ns-head-span']['number']} from saved fresh-cohort JSONs.
  Its generator retains all error seeds and uses only the two globally admitted
  timing seeds; no new solver or training run was performed.
- Clarified the interacting-vortex family, P4/P3 cylinder truth versus P2/P1
  export mesh, and the separate accuracy-matched finite-element comparator.
- Removed the stale statement that multigrid was untested; distinguish measured
  CPU AMG from the pending GPU sparse-direct/AMG comparisons.
- Corrected Eq. ({refs['eq:bank-loss-vp']['number']}) against the preserved heat training code:
  the mean term uses weighted POD-mode projection errors, and the tail term
  uses a snapshot minibatch. Exact thin-QR elimination is unchanged.
- Clarified total-cost amortisation and identified the heat modal-comparator arm.
- Qualified scaling: the reduced computation is distinct from mesh-dependent
  projection/output, and cross-allocation wall times are not paired speedups.
- Kept CROM/CoLoRA re-implementation, budget, failed-timing, transfer and
  extrapolation disclosures. Older cylinder field plots retain their explicit
  earlier-development-bank caption and do not depict the held-out headline bank.
- Following the author's presentation choice, neural operators are compared
  by error only. Their inference times and relative-speed claims are removed
  from the main text, historical/matched-data appendix tables and cylinder/NS
  descriptions; the cost plot now shows ROMs and numerical solvers only.
  All prediction-error values and training-budget disclosures are retained.
  Numerical-solver and reduced-model timing comparisons remain in place.

## Build and upload

Run `bash build.sh` in this directory. The source archive contains all referenced
styles and figures, with `main.tex` at its root. The compiled PDF is supplied separately.
Changes relative to the original submission remain marked in blue; moved/deleted
passages are described in the existing revision footnote.

Sparse-solver and equal-median-error operator experiments remain pending.
Overleaf has not been edited or uploaded by this session.
''')

patch = []
for name in ['main.tex','preamble.tex']:
    patch.extend(difflib.unified_diff((original/name).read_text().splitlines(True),
                                    (PAPER/name).read_text().splitlines(True),
                                    fromfile='uploaded/'+name,tofile='revised/'+name))
(TOOLS/'attachment-to-revision.patch').write_text(''.join(patch))
archive = ROOT/'manuscript-rebuttal-source.zip'
files = [p for p in PAPER.rglob('*') if p.is_file() and
         (p.suffix in {'.tex','.sty','.bst','.png','.pdf','.md','.sh'} or p.name=='.gitignore')
         and p.name not in {'main.pdf','submitted.tex'} and p.name != 'F8_ns_head_span.png']
with ZipFile(archive,'w',ZIP_DEFLATED) as z:
    for p in sorted(files):z.write(p,p.relative_to(PAPER))
    assert z.testzip() is None
    assert 'main.tex' in z.namelist()

meta = {'source_archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'pdf_sha256':hashlib.sha256((PAPER/'main.pdf').read_bytes()).hexdigest(),
        'pdf_pages':pages, 'verified_labels':refs, 'latex_errors':False,
        'undefined_references':False,'overfull_boxes':False,
        'pending_experiments_included':False,'overleaf_uploaded':False,
        'archive_files':len(files)}
package_build = TOOLS/'checks/package-build'
if (package_build/'main.log').exists():
    with ZipFile(archive) as z:
        matching_source = all((package_build/n).read_bytes() == z.read(n) for n in z.namelist())
    built_log = (package_build/'main.log').read_text()
    meta['source_zip_independent_build_verified'] = matching_source and bool(
        re.search(r'Output written on main.pdf', built_log)) and not bool(
        re.search(r'Overfull|undefined|^!', built_log, re.M))
(TOOLS/'checks/validation.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2))
