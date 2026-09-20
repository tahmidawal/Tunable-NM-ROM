"""Estimate frozen final storage from archived development member sizes."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def main():
    p = argparse.ArgumentParser(); p.add_argument('attempt'); a = p.parse_args()
    assert a.attempt.isalnum()
    lane = Path(__file__).resolve().parent
    run = lane/'runs'/a.attempt
    manifest_path = lane/'retained-fields'/a.attempt/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    files = {row['path']: row['bytes'] for row in manifest['files']}
    rows = []
    for label, prefix, config_file in [('primary', '', 'finalA.json'), ('seedB', 'seedB/', 'finalB.json')]:
        cfg = json.loads((lane/config_file).read_text())
        record = json.loads((run/'archive/out'/prefix/'result.json').read_text())
        k = cfg['latent_dimensions'][0]
        ranks = {cfg['bank_rank']} | {k+q for q in cfg['q_ladder'] if k+q <= cfg['bank_rank']}
        selected = {}; sample = count = 0
        for mesh in record['meshes']:
            n = mesh['intervals']
            if n not in cfg['evaluation_intervals']: continue
            names = []
            for name, meta in mesh['methods'].items():
                if name.startswith('nmrom_') and not name.startswith(f'nmrom_K{k}_'): continue
                if name.startswith('pod') and int(re.match(r'pod(\d+)', name)[1]) not in ranks: continue
                if name.startswith('fom_cn_cg_') and name not in cfg['cg_methods_by_mesh'][str(n)]: continue
                if meta['kind'] == 'neural_operator' and name not in cfg['operator_methods_by_mesh'][str(n)]: continue
                names.append(name)
            selected[str(n)] = names
            for case in range(cfg['validation_count']):
                for name in names+['reference']:
                    sample += files[f'{prefix}fields/N{n}_case{case}_{name}.npz']; count += 1
        other = sum(size for name, size in files.items() if name.startswith(prefix)
                    and 'fields' not in Path(name[len(prefix):]).parts
                    and 'seedB' not in Path(name[len(prefix):]).parts)
        rows.append(dict(cohort=label, selected_methods_by_mesh=selected,
            development_selected_field_bytes=sample, development_selected_files=count,
            nonfield_bytes_upper_bound=other,
            estimated_final_output_bytes=sample*cfg['reserved_final_count']/cfg['validation_count']+other))
    total = sum(r['estimated_final_output_bytes'] for r in rows)
    result = dict(schema='heat3d-storage-forecast-v1', final_fields_accessed=False,
        source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        basis='Archived development NPZ sizes scaled by reserved final count; frozen selected methods and checkpoint artifacts. Final compression can vary. No materialized fields are required.',
        rows=rows, estimated_final_output_bytes=total, estimated_three_copy_peak_bytes=3*total,
        three_copy_peak_with_15_percent_margin_bytes=3*total*1.15)
    destination = run/'FINAL-STORAGE-FORECAST.json'
    if destination.exists():
        previous = json.loads(destination.read_text())
        assert previous['estimated_final_output_bytes'] == total, 'Investigate changed forecast inputs rather than silently replace the budget.'
    destination.write_text(json.dumps(result, indent=2)+'\n')
    print('FINAL_OUTPUT_BYTES_FORECAST', total)


if __name__ == '__main__': main()
