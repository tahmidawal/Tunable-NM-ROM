"""Separate present-invocation validity from the retained failed replay gate.

Does not alter the original umbrella audit or its tolerance. This disposition
cannot support old/new implementation-speed ratios or exact replay claims.
"""
import hashlib,json
from pathlib import Path


def main():
    base=Path(__file__).parent;run=base/'runs/extra03'
    raw=json.loads((run/'audit.json').read_text());hist=json.loads((run/'history_audit.json').read_text())
    source=json.loads((run/'source_audit.json').read_text());archive=json.loads((base/'artifacts/extra03/archive.json').read_text())
    disposition=json.loads((run/'replay_disposition.json').read_text())
    independent={k:v for k,v in raw['checks'].items() if k!='frozen_method_replay'}
    failed=[k for k,v in raw['checks'].items() if not v['passed']]
    assert failed==['frozen_method_replay'],failed
    passed=all(v['passed'] for v in independent.values()) and hist['passed'] and source['passed'] and source['remote_original_manifest_passed'] and archive['passed']
    result=dict(passed=bool(passed),source_commit=raw['source_commit'],job_id=raw['job_id'],
        scope='independently checked new paired measurements only; not cross-run equivalence',
        qualification='The predeclared frozen-field replay gate failed for NM-ROM q16 case1. Its threshold and failed umbrella audit are unchanged. Do not claim parity or compare implementation speed across jobs.',
        original_umbrella_audit_passed=raw['passed'],cross_run_replay=raw['checks']['frozen_method_replay'],
        independent_checks=independent,history_audit=dict(passed=hist['passed'],scope=hist['scope'],probes=len(hist['checks'])),
        source_audit=dict(passed=source['passed'],source_files=len(source['checks']),remote_manifest_passed=source['remote_original_manifest_passed']),
        durable_archive=dict(passed=archive['passed'],files=archive['retained_files'],parts=len(archive['parts']),bytes=archive['retained_bytes']),
        replay_disposition=disposition,derived_summary=raw['derived_summary'],
        linked_audit_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [run/'audit.json',run/'history_audit.json',run/'source_audit.json',run/'replay_disposition.json']})
    (run/'numerical_validity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k in ('passed','scope','qualification','original_umbrella_audit_passed','history_audit','durable_archive')}))
    raise SystemExit(0 if passed else 2)


if __name__=='__main__':main()
