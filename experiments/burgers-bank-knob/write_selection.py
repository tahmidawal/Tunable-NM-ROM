"""Freeze the 4096^2 dev6 selection (DESIGN section 6, and the A4 sensitivity arms) into selection-4096.json, from the
audited summary only. Commit the output BEFORE generating and submitting the hold64 job.

    python write_selection.py checks/bk4096-summary.json
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    p = Path(sys.argv[1])
    s = json.loads(p.read_text())
    assert s['accepted'], s['failed_gates']
    se = s['selection']
    s2 = se['sensitivity_k_ge_j_plus_1']
    out = dict(mesh=s['intervals'], job_id=s['job_id'], source_summary=str(p.name),
               source_summary_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
               rule='DESIGN.md section 6 (pre-registered)', accurate=se['accurate']['name'], fast=se['fast']['name'],
               current_accurate=se['current_accurate_arm']['name'], current_fast=se['current_fast_arm']['name'],
               sensitivity_rule='DESIGN A3/A4: certificate on k >= exact_steps + 1 (labelled sensitivity, not the rule)',
               sensitivity_accurate=(s2['accurate']['name'] if s2.get('accurate') and s2['accurate']['name'] != se['accurate']['name'] else None),
               sensitivity_fast=(s2['fast']['name'] if s2.get('fast') and s2['fast']['name'] != se['fast']['name'] else None))
    (HERE / 'selection-4096.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
