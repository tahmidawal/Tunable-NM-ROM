"""Bundle the audited campaign's recorded results without duplicating field archives."""
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
STEM = '2026-09-07-multiresolution-pilots'
STUDIES = [
    ('Heat', 'Initial mesh transfer', 'heat'),
    ('Heat', 'Stopping tolerance and compilation', 'heat_runtime'),
    ('Heat', 'Head training coverage', 'heat_head'),
    ('Heat', 'Wider frozen transfer and fresh inputs', 'heat_transfer'),
    ('Burgers', 'Initial mesh transfer', 'burgers'),
    ('Burgers', 'Reference refinement and multiple initial guesses', 'burgers_followup'),
    ('Burgers', 'Initial fitting objective diagnostics', 'burgers_cold'),
    ('Burgers', 'Corrected initial fit and full rollouts', 'burgers_rollout'),
    ('Burgers', 'Larger time steps', 'burgers_steps'),
    ('Poisson', 'Initial mesh transfer', 'poisson'),
    ('Poisson', 'Weak modes, representation and compilation', 'poisson_followup'),
    ('Poisson', 'Guarded small-system solver', 'poisson_kernel'),
    ('Poisson', 'Training coverage and loss factorial', 'poisson_training'),
    ('Poisson', 'Source projection and initialization factorial', 'poisson_speed'),
    ('Fresh waves', 'Initial mesh transfer', 'wave_native'),
    ('Fresh waves', 'Nonlinear, affine and full-bank dynamics', 'wave_dynamics'),
    ('Fresh waves', 'Larger nonlinear heads', 'wave_heads'),
]


def main():
    manifest_path = HERE/f'{STEM}.json'
    report_path = HERE/f'{STEM}.md'
    manifest = json.loads(manifest_path.read_text())
    files = {manifest_path, report_path, Path(__file__).resolve()}
    records = {}
    for key, item in manifest['artifacts'].items():
        path = ROOT/item['path']
        blob = path.read_bytes()
        assert hashlib.sha256(blob).hexdigest() == item['sha256'], key
        files.add(path)
        records[key] = json.loads(blob)

    inventory = []
    for pde, study, key in STUDIES:
        record = records[key]
        meta = record.get('provenance', record.get('metadata', {}))
        job = record.get('job_id', meta.get('job_id'))
        commit = record.get('commit', meta.get('source_commit', meta.get('commit',
            record.get('source_manifest', {}).get('source_commit'))))
        assert job and commit, key
        inventory.append(dict(pde=pde, experiment=study, job_id=str(job), source_commit=commit,
            result_json=manifest['artifacts'][key]['path'], sha256=manifest['artifacts'][key]['sha256']))
    assert len({r['job_id'] for r in inventory}) == len(STUDIES)
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
    writer.writeheader()
    writer.writerows(inventory)
    inventory_path = HERE/'2026-09-07-multiresolution-experiment-index.csv'
    inventory_path.write_text(stream.getvalue())
    files.add(inventory_path)

    # Include the canonical report's directly linked findings and figures.
    # Full fields/checkpoints remain in their separately checked experiment archives.
    for target in re.findall(r'\]\(([^)]+)\)', report_path.read_text()):
        path = (HERE/target.split('#')[0]).resolve()
        path.relative_to(ROOT)
        assert path.is_file(), target
        files.add(path)
        if path.suffix == '.png' and path.with_suffix('.pdf').is_file():
            files.add(path.with_suffix('.pdf'))
    payloads = {str(path.relative_to(ROOT)): path.read_bytes() for path in files}
    checksums = {name: hashlib.sha256(blob).hexdigest() for name, blob in sorted(payloads.items())}
    payloads['BUNDLE-MANIFEST.json'] = (json.dumps(checksums, indent=2)+'\n').encode()
    payloads['README.md'] = (f'''# Recorded multiresolution experiment results

This bundle contains provisional development results for the current separable NM-ROM campaign. Numerical results were finalized on the date in the canonical report filename; later packaging does not represent new experiments.

Open `reports/{STEM}.md` for the experiment-level findings and linked plots. The experiment index lists all {len(inventory)} completed GPU studies, with source revisions and exact result-file hashes. The {len(manifest['artifacts'])} source JSON artifacts retain raw timing repetitions, physical errors, configurations, audits and diagnostic records in their native schemas. The absorbing-wave moment analysis is included as saved-run postprocessing, not another GPU study.

The original ViT + CP comparison and reset older-wave evidence are outside this campaign. Proposed Jacobian reuse, shared wave derivative calculations and new learned predictors have no new results in this bundle. No nonlinear-ROM complete-query advantage over the tested efficient FOMs is established. Error norms and cohorts differ across PDEs, and independent final confirmation remains unopened.

Full field arrays, trained checkpoints, staged source trees and cluster logs remain in the separately tracked experiment archives. This bundle is a results export, not a standalone reproduction environment. Relative links to those omitted artifacts and original absolute provenance paths still require the project workspace. `BUNDLE-MANIFEST.json` gives checksums for every included report, source JSON, index, figure and exporter script.

Timing arrays inside different result schemas have different roles: repeated comparisons, separate accuracy controls, warmups and component diagnostics must not be pooled as equivalent measurements. The source records retain those roles. Stored row counts are not necessarily invocation counts.

## Plain-language glossary

- **PDE / experiment / result_json:** partial differential equation / the numerical question tested / the path to its original machine-readable result file.
- **GPU study / job ID:** one recorded cluster experiment / its scheduler identifier.
- **Source revision / SHA256:** the exact saved code version / a fingerprint of file contents.
- **ROM / FOM:** reduced mathematical model / full spatial solver.
- **NM-ROM:** a reduced model whose solution is constrained by a nonlinear decoder.
- **Query:** supplied input field through the requested returned solution fields.
- **Development / final confirmation:** inputs used while improving the method / independent inputs reserved for the final test.
- **Checkpoint / field array:** trained network weights / numerical solution values on a grid.
- **Timing repetition / accuracy control / warmup:** repeated measurement / separate check of numerical accuracy / preparation before comparable timing.
- **Native schema / provenance:** an experiment's original record format / the history identifying how a result was produced.
''').encode()
    output = HERE/'2026-09-07-multiresolution-results.zip'
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, blob in sorted(payloads.items()):
            info = zipfile.ZipInfo(name, date_time=(2026, 9, 7, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, blob)
    with zipfile.ZipFile(output) as archive:
        assert set(archive.namelist()) == set(payloads)
        for name, blob in payloads.items():
            assert archive.read(name) == blob, name
    print(json.dumps(dict(studies=len(inventory), source_json_artifacts=len(manifest['artifacts']),
        archive_members=len(payloads), archive_bytes=output.stat().st_size,
        archive_sha256=hashlib.sha256(output.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
