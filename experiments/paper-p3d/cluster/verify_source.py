"""Verify a collected attempt against historical Git blobs, never remote ancestry."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]


def verify(attempt):
    assert attempt.isalnum()
    folder=ROOT/'experiments/paper-p3d/runs'/attempt
    archive=folder/'archive'
    assert json.loads((folder/'COLLECTED.json').read_text())['checksums_verified']
    proof=json.loads((archive/'PROVENANCE.json').read_text())
    record=json.loads((archive/'out/result.json').read_text())
    commit=(archive/'COMMIT.txt').read_text().strip()
    assert commit==proof['source_commit']==record['source_commit']
    rows=[]
    for entry in proof['files']:
        staged=archive/entry['staged'];actual=hashlib.sha256(staged.read_bytes()).hexdigest()
        assert actual==entry['sha256']
        if entry['path'].startswith('experiments/paper-p3d/'):
            blob=subprocess.check_output(['git','-C',str(ROOT),'show',f"{commit}:{entry['path']}"])
            assert hashlib.sha256(blob).hexdigest()==actual
            rows.append(dict(path=entry['path'],sha256=actual))
    assert rows
    assert json.loads((archive/'code/config.json').read_text())==record['config']
    result=dict(passed=True,source_commit=commit,verified_git_source_files=len(rows),files=rows,
        complete_source_manifest_verified=True,configuration_matches_measured_result=True,
        scope='Every staged provenance entry hashes to its retained bytes; each scientific source entry matches the recorded historical Git commit.')
    (folder/'source-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('attempt');a=p.parse_args();verify(a.attempt)
