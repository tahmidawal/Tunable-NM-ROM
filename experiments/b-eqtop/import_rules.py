"""Import the q-ridge `qrg304` reachable-state rules into this lane, checksum-verified.

The rule files live only inside qrg304's checksum-verified archive (`runs/` is gitignored),
so they are copied here, each SHA256 checked against the `OUTPUTS.sha256` the job itself
wrote, and recorded in `rules/qrg304/MANIFEST.json`. Nothing is recomputed.

    python import_rules.py
"""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SRC = ROOT / 'experiments/q-ridge/runs/qrg304/archive'
ART = ROOT / 'experiments/q-ridge/artifacts/qrg304'


def main():
    listed = {}
    for line in (ART / 'OUTPUTS.sha256').read_text().splitlines():
        h, name = line.split(maxsplit=1)
        listed[name.strip()] = h
    r = json.loads((ART / 'result.json').read_text())
    out = HERE / 'rules/qrg304'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for x in r['rules']:
        name = f"rule_q{x['q']}_{x['population']}_m{x['m_target']}.npz"
        src = SRC / 'output' / name
        digest = hashlib.sha256(src.read_bytes()).hexdigest()
        assert digest == listed[f'output/{name}'], name
        shutil.copy2(src, out / name)
        rows.append(dict(file=name, sha256=digest, q=x['q'], population=x['population'],
                         m=x['m'], m_target=x['m_target'], M=x['M'],
                         relative_fit=x['relative_fit'], rho_max=x['certification']['rho_max'],
                         rho_p95=x['certification']['rho_p95'],
                         rho_median=x['certification']['rho_median'],
                         certified_primary=x['certified_primary'],
                         certified_secondary=x['certified_secondary'],
                         nodes_sha256=x['nodes_sha256'], weights_sha256=x['weights_sha256'],
                         fit_states=x['fit_states'], candidates=x['candidates'],
                         design_rows=x['design_rows'], fit_seconds=x['seconds']))
    (out / 'MANIFEST.json').write_text(json.dumps(dict(
        source_job_id=r['job_id'], source_attempt='qrg304', source_commit=r['commit'],
        source_gpu=r['gpu'], archive_sha256=json.loads((ART / 'archive.json').read_text())['sha256'],
        rho_bar=r['rho_bar'], rules=rows), indent=2) + '\n')
    print(len(rows), 'rules imported to', out)


if __name__ == '__main__':
    main()
