"""Verify imported bytes and regenerate the accepted reports on CPU."""
import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'consolidated/evidence'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(filename):
    spec = importlib.util.spec_from_file_location(Path(filename).stem,ROOT/'reports'/filename)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def verify_fixtures(manifest):
    methods = dict(heat='nmrom_initial_tail',poisson='r128_q32',
                   burgers='frozen_stationary',wave='trained_nested40')
    checks = {}
    for pde, model in manifest['models'].items():
        spec = model['replay']
        raw = json.loads((ROOT/spec['raw_result']).read_text())
        rows = raw.get('rows',raw.get('invocations'))
        row = next(r for r in rows if r['intervals']==spec['intervals'] and
                   r['case']==spec['case'] and r.get('name',r.get('method'))==methods[pde])
        saved = np.load(ROOT/spec['expected'])
        if pde=='heat':
            value = np.ascontiguousarray(saved['field'])
            h = hashlib.sha256(str((value.shape,value.dtype.str)).encode()+value.tobytes()).hexdigest()
            assert h == row['repetitions'][0]['field']['sha256_array']
            expected = next(r['sha256'] for r in raw['models'] if r['name']=='initial_tail')
        elif pde=='wave':
            for k in ('u','v'):
                assert hashlib.sha256(saved[k].tobytes()).hexdigest()==row['output_sha256'][k]
            expected = raw['input_sha256']['dirichlet/trained_nested40.npz']
            native = ROOT/'experiments/multiresolution-wave/runs/accel12/cluster/in/dirichlet'
            for file in native.iterdir():
                assert digest(file)==raw['input_sha256']['dirichlet/'+file.name]
            original = np.load(ROOT/model['artifacts'][1])
            staged = np.load(native/'initializer_trained_nested40.npz')
            for k in original.files:
                np.testing.assert_array_equal(original[k],staged[k])
        else:
            k = 'field' if pde=='poisson' else 'fields'
            assert hashlib.sha256(saved[k].tobytes()).hexdigest()==row['field_sha256']
            expected = raw['config']['checkpoint_sha256'] if pde=='poisson' else raw['checkpoint_sha256']
        assert digest(ROOT/model['artifacts'][0])==expected
        if pde=='poisson':
            assert digest(ROOT/model['artifacts'][2])==raw['config']['basis_sha256']
        checks[pde] = 'Selected checkpoint and saved field agree with accepted raw result'
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT/'consolidated/manifest.json').read_text())
    fixture_checks = verify_fixtures(manifest)
    syntax = 0
    for record in manifest['files']:
        path = ROOT/record['path']
        assert digest(path) == record['sha256'], path
        if 'git_blob' in record:
            actual = subprocess.check_output(['git','-C',str(ROOT),'hash-object',str(path)]).decode().strip()
            assert actual == record['git_blob'], path
        if path.suffix == '.py':
            ast.parse(path.read_text(),filename=str(path))
            syntax += 1
    report_files = ['2026-09-11-accuracy-improvements-and-wave-speed.json',
                    '2026-09-11-accuracy-improvements-and-wave-speed.md',
                    '2026-09-11-accuracy-integration.json',
                    '2026-09-11-accuracy-aggregates.json',
                    '2026-09-11-accuracy-aggregates.md']
    # Preserve the exact relative paths used by the original generators while
    # redirecting all writes into a disposable directory in this worktree.
    with tempfile.TemporaryDirectory(prefix='report-check-',dir=ROOT/'consolidated') as tmp:
        target = Path(tmp)
        (target/'worktrees').symlink_to(EVIDENCE/'worktrees',target_is_directory=True)
        (target/'reports').mkdir()
        for source in (EVIDENCE/'reports').iterdir():
            if source.name not in report_files:
                (target/'reports'/source.name).symlink_to(source)
        campaign = module('generate_accuracy_campaign.py')
        campaign.ROOT = target
        campaign.REPORT = target/'reports/2026-09-11-accuracy-improvements-and-wave-speed'
        with contextlib.redirect_stdout(io.StringIO()):
            campaign.build()
        aggregates = module('generate_accuracy_aggregates.py')
        aggregates.ROOT = target
        aggregates.CAMPAIGN = campaign.REPORT.with_suffix('.json')
        aggregates.OUTPUT = target/'reports/2026-09-11-accuracy-aggregates'
        with contextlib.redirect_stdout(io.StringIO()):
            aggregates.main()
        hashes = {}
        for name in report_files:
            assert (target/'reports'/name).read_bytes() == (ROOT/'reports'/name).read_bytes(), name
            hashes[name] = digest(target/'reports'/name)
    wrappers = {str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'consolidated').glob('*.py')}
    for name in wrappers:
        ast.parse((ROOT/name).read_text(),filename=name)
    result = dict(passed=True,verified_imported_files=len(manifest['files']),parsed_python_files=syntax,
                  fixture_checks=fixture_checks,integration_tool_sha256=wrappers,
                  report_byte_parity=hashes,manifest_sha256=digest(ROOT/'consolidated/manifest.json'),
                  scope='CPU content/provenance and exact report regeneration; GPU parity is separate')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
