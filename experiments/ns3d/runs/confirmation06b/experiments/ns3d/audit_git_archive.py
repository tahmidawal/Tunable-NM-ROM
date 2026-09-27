"""Verify archive parts from actual Git object bytes before remote cleanup."""
import argparse,hashlib,json,subprocess
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--commit',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    archive=Path(a.archive);manifest=json.loads((archive/'archive.json').read_text());assert manifest['passed']
    checks=[]
    for part in manifest['parts']:
        path=archive/part['path'];proc=subprocess.Popen(['git','show',a.commit+':'+str(path)],stdout=subprocess.PIPE)
        h=hashlib.sha256();size=0
        while block:=proc.stdout.read(8*1024**2):h.update(block);size+=len(block)
        status=proc.wait();checks.append(dict(path=part['path'],passed=status==0 and h.hexdigest()==part['sha256'] and size==part['bytes']))
    result=dict(passed=bool(checks) and all(row['passed'] for row in checks),complete=True,archive_commit=a.commit,checks=checks,
        verified_bytes=sum(row['bytes'] for row in manifest['parts']))
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='checks'}))
    raise SystemExit(0 if result['passed'] else 2)


if __name__=='__main__':main()
