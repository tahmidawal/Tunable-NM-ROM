"""Verify collected scientific sources against their recorded Git blobs."""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('collected');p.add_argument('--out',required=True);a=p.parse_args()
    root=Path(a.collected);commit=(root/'COMMIT.txt').read_text().strip();checks=[]
    for row in json.loads((root/'PROVENANCE.json').read_text()):
        blob=subprocess.check_output(['git','show',commit+':'+row['path']])
        actual=(root/row['path']).read_bytes()
        checks.append(dict(path=row['path'],git_match=actual==blob,
            stage_match=hashlib.sha256(actual).hexdigest()==row['sha256'],source_commit_match=row['source_commit']==commit))
    result=dict(passed=all(all(r[k] for k in ('git_match','stage_match','source_commit_match')) for r in checks),source_commit=commit,checks=checks)
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=result['passed'],files=len(checks),source_commit=commit)))
    raise SystemExit(0 if result['passed'] else 2)

if __name__=='__main__':main()
