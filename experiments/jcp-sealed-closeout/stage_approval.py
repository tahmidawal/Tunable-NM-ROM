"""Check approved, committed study settings and selections before staging restricted jobs."""
import json
import os
from pathlib import Path
import subprocess
from open_sealed import sha, verify_approval
from roster import ROSTER_PATH, build
from open_sealed import packed
HERE=Path(__file__).resolve().parent
def main():
    review=(HERE/'F1-REVIEW.json').read_bytes()
    expected=sha(review)
    os.environ['JCP_APPROVED_REVIEW_SHA256']=expected
    receipt=json.loads(review)
    source=receipt['freeze_commit']
    subprocess.run(['git','merge-base','--is-ancestor',source,'HEAD'],check=True)
    if subprocess.check_output(['git','rev-parse',source+'^{commit}']).decode().strip()!=source:
        raise ValueError('Expected complete Git commit for coordinator freeze')
    freeze=json.loads((HERE/'F1.json').read_bytes())
    for name,digest in dict(freeze['files'],**{'F1.json':sha((HERE/'F1.json').read_bytes())}).items():
        if sha(subprocess.check_output(['git','show',source+':experiments/jcp-sealed-closeout/'+name]))!=digest:
            raise ValueError('Frozen local Git bytes disagree: '+name)
    if ROSTER_PATH.read_bytes()!=packed(build(json.loads(ROSTER_PATH.read_bytes()))):
        raise ValueError('Selection evidence does not match original pinned sources')
    verify_approval(HERE)
    print(expected)
if __name__=='__main__': main()
