"""Reproduce the scoped integration from the approved, immutable source heads.

Run only in this worktree. Existing evidence is copied, never regenerated here.
"""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CANONICAL = Path('/home/tahmid/Dev/pod-ae-nmrom/Tunable-NM-ROM-Claude')
HEADS = {
    'heat': ('2026-09-07-mr-heat2d', '5974d3e0df3f2e0906059564a72511de375ea2a5'),
    'poisson': ('2026-09-07-mr-poisson2d', 'ef2552800756bd9e6be1b9f44776b0406f056d51'),
    'burgers': ('2026-09-07-mr-burgers2d', 'ad9e2e7efa7894230cc0494db948ce17fd5f67ee'),
    'wave': ('2026-09-07-mr-wave2d', '277ef65c267dd1fee9090c50b6294dcc83ba2531'),
}
CELLS = {'heat': 'mr-heat2d', 'poisson': 'multiresolution-poisson',
         'burgers': 'mr-burgers2d', 'wave': 'multiresolution-wave'}
RUNS = {
    'heat': 'experiments/mr-heat2d/runs/accuracy10/archive/outputs',
    'poisson': 'experiments/multiresolution-poisson/runs/correction_accuracy10',
    'burgers': 'experiments/mr-burgers2d/runs/accuracy09/archive/out',
    'wave': 'experiments/multiresolution-wave/runs/accel12/cluster/out/pilot',
}


def git(*args):
    return subprocess.check_output(['git', '-C', str(CANONICAL), *args])


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert ROOT != CANONICAL and ROOT.name == '2026-09-13-nmrom-consolidated'
    manifest = dict(schema=1, base_commit=HEADS['heat'][1], source_heads=HEADS,
                    scope='Selected accuracy-campaign implementations and artifacts; no new scientific run',
                    files=[], models={})
    blobs = {}
    for pde, (name, commit) in HEADS.items():
        source = CANONICAL/'worktrees'/name
        assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD']).decode().strip() == commit
        blobs[pde] = {}
        for line in git('ls-tree', '-r', commit).decode().splitlines():
            meta, path = line.split('\t', 1)
            mode, kind, blob = meta.split()
            if kind == 'blob':
                blobs[pde][path] = (mode, blob)

    def copy(source, destination, role, pde=None):
        source, destination = Path(source), Path(destination)
        target = ROOT/destination
        record = dict(path=str(destination), source=str(source.relative_to(CANONICAL)),
                      sha256=digest(source), bytes=source.stat().st_size, role=role)
        if pde:
            relative = str(source.relative_to(CANONICAL/'worktrees'/HEADS[pde][0]))
            if relative in blobs[pde]:
                mode, blob = blobs[pde][relative]
                assert git('hash-object', str(source)).decode().strip() == blob, relative
                record.update(source_commit=HEADS[pde][1], git_blob=blob)
            else:
                record['source_commit'] = None
                record['provenance_note'] = 'Restored archive member; linked by the accepted result/input hashes'
        else:
            record['source_commit'] = git('rev-parse', 'HEAD').decode().strip()
            tracked = subprocess.run(['git', '-C', str(CANONICAL), 'ls-files', '--error-unmatch', record['source']], capture_output=True).returncode == 0
            record['source_working_copy'] = not tracked or subprocess.run(
                ['git', '-C', str(CANONICAL), 'diff', '--quiet', 'HEAD', '--', record['source']]).returncode != 0
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or digest(target) != record['sha256']:
            shutil.copy2(source, target)
        manifest['files'].append(record)
        return str(destination)

    def native(pde, path, role='artifact'):
        return copy(CANONICAL/'worktrees'/HEADS[pde][0]/path, path, role, pde)

    for pde, cell in CELLS.items():
        prefixes = [f'experiments/{cell}/']
        if pde == 'wave':
            prefixes.append('experiments/fresh-wave-head/')
        for path in sorted(blobs[pde]):
            if any(path.startswith(prefix) for prefix in prefixes) and not set(Path(path).parts).intersection({'runs', 'history', 'stages'}):
                native(pde, path, 'implementation')
    # Shared modules are retained from the corrected base, with equality checked
    # against every source that consumes them.
    shared = ['experiments/separable-decoder/sep_common.py',
              'experiments/cost-to-tolerance/ctol_tol.py',
              'experiments/wave2d-rom-latent-stepping/deps/multistage-precision/ms_parametric.py']
    for path in shared:
        ids = {blobs[pde][path][1] for pde in HEADS}
        assert len(ids) == 1, (path, ids)
        native('heat', path, 'shared_dependency')

    paths = {
        'heat': ['experiments/mr-heat2d/runs/accuracy10/archive/outputs/checkpoints/initial_tail.pkl'],
        'poisson': ['experiments/multiresolution-poisson/runs/correction_accuracy10/checkpoints/r128_joint.pkl',
                    'experiments/multiresolution-poisson/runs/correction_accuracy10/checkpoints/original_relative.pkl',
                    'experiments/multiresolution-poisson/runs/correction_accuracy10/basis.npz'],
        'burgers': ['experiments/separable-decoder/runs/dn256b/out/sep_hfit_dense_mid_N256_dense.pkl'],
        'wave': ['experiments/multiresolution-wave/runs/accel10/cluster/out/pilot/head_trained_nested40.npz',
                 'experiments/multiresolution-wave/runs/accel10/cluster/out/pilot/initializer_trained_nested40.npz'],
    }
    for pde, artifacts in paths.items():
        manifest['models'][pde] = dict(cell=f'experiments/{CELLS[pde]}',
                                     artifacts=[native(pde, path, 'selected_model') for path in artifacts])
    wave_in = 'experiments/multiresolution-wave/runs/accel12/cluster/in'
    wave_root = CANONICAL/'worktrees'/HEADS['wave'][0]
    for source in sorted((wave_root/wave_in/'dirichlet').iterdir()):
        native('wave', str(source.relative_to(wave_root)), 'wave_rebuild_input')
    native('wave', wave_in+'/ORIGIN.json', 'wave_rebuild_provenance')
    # Canonical reports remain on main. This evidence mirror makes their exact
    # source generators independently replayable without other live worktrees.
    campaign = json.loads((CANONICAL/'reports/2026-09-11-accuracy-improvements-and-wave-speed.json').read_text())
    reports = ['2026-09-11-accuracy-campaign-handoff.md',
               '2026-09-11-accuracy-improvements-and-wave-speed.md',
               '2026-09-11-accuracy-improvements-and-wave-speed.json',
               '2026-09-11-accuracy-integration.json',
               '2026-09-11-accuracy-aggregates.md', '2026-09-11-accuracy-aggregates.json',
               'generate_accuracy_campaign.py', 'generate_accuracy_aggregates.py']
    for filename in reports:
        copy(CANONICAL/'reports'/filename, Path('reports')/filename, 'report_snapshot')
    for path, expected in campaign['sources'].items():
        assert digest(CANONICAL/path) == expected, path
        owner = next((pde for pde, (name, _) in HEADS.items() if path.startswith('worktrees/'+name+'/')), None)
        copy(CANONICAL/path, Path('consolidated/evidence')/path, 'accepted_report_source', owner)
    integration = json.loads((CANONICAL/'reports/2026-09-11-accuracy-integration.json').read_text())
    extra = [r['path'] for r in integration['heat']['files']]
    extra += [integration[pde]['inventory'] for pde in ('poisson','burgers','wave')]
    for path in extra:
        owner = next(pde for pde, (name, _) in HEADS.items() if path.startswith('worktrees/'+name+'/'))
        copy(CANONICAL/path, Path('consolidated/evidence')/path, 'report_integration_dependency', owner)
    for filename in reports:
        if filename.endswith('.json'):
            copy(CANONICAL/'reports'/filename, Path('consolidated/evidence/reports')/filename, 'report_evidence')
    copy(CANONICAL/'AGENTS.md', 'AGENTS.md', 'operating_instructions')

    for pde, run in RUNS.items():
        src = CANONICAL/'worktrees'/HEADS[pde][0]/run
        rawpath = src/('results.json' if pde == 'heat' else 'result.json')
        raw = json.loads(rawpath.read_text())
        evidence_path = str(Path('consolidated/evidence')/rawpath.relative_to(CANONICAL))
        fdir = Path('consolidated/fixtures')/pde
        def fixture(path, name):
            return copy(src/path, fdir/name, 'saved_case_fixture', pde)
        case = dict(intervals=64, case=0, raw_result=evidence_path)
        if pde == 'heat':
            row = next(r for r in raw['rows'] if (r['intervals'],r['case'],r['method']) == (64,0,'nmrom_initial_tail'))
            initial = next(r for r in raw['case_fields'] if (r['intervals'],r['case']) == (64,0))
            case.update(expected=fixture(row['repetitions'][0]['field']['path'], 'expected.npz'),
                        initial=fixture(initial['initial']['path'], 'initial.npz'))
        elif pde == 'poisson':
            row = next(r for r in raw['rows'] if (r['intervals'],r['case'],r['method'],r['repetition']) == (64,0,'r128_q32',0))
            case.update(expected=fixture('cluster/out/pilot/fields/'+row['field_sha256']+'.npz', 'expected.npz'))
        elif pde == 'burgers':
            case.update(expected=fixture('L64_frozen_stationary_case0_rep0.npz','expected.npz'),
                        operators=fixture('operators_frozen_L64.npz','operators.npz'))
        else:
            case.update(case='opened_0', expected=fixture('64_opened_0_trained_nested40_0.01.npz','expected.npz'),
                        operators=fixture('mesh_dirichlet_64.npz','operators.npz'),
                        initial=fixture('reference_64_opened_0.npz','initial.npz'))
        manifest['models'][pde]['replay'] = case

    manifest['files'].sort(key=lambda r: r['path'])
    assert len({r['path'] for r in manifest['files']}) == len(manifest['files'])
    (ROOT/'consolidated/manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(imported_files=len(manifest['files']), total_bytes=sum(r['bytes'] for r in manifest['files']))))


if __name__ == '__main__':
    main()
