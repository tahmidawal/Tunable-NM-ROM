"""Verify every retained output byte before numerical analysis or reuse."""
import argparse,hashlib,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('collected');p.add_argument('--out',required=True)
    p.add_argument('--manifest',choices=('OUTPUTS.sha256','MANIFEST.sha256'),default='OUTPUTS.sha256')
    a=p.parse_args();root=Path(a.collected);checks=[]
    for line in (root/a.manifest).read_text().splitlines():
        expected,name=line.split(None,1);name=name.strip();h=hashlib.sha256()
        with (root/name).open('rb') as f:
            while block:=f.read(8*1024**2):h.update(block)
        checks.append(dict(path=name,passed=h.hexdigest()==expected))
    result=dict(passed=bool(checks) and all(x['passed'] for x in checks),complete=True,manifest=a.manifest,checks=checks)
    Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=result['passed'],files=len(checks))))
    raise SystemExit(0 if result['passed'] else 2)


if __name__=='__main__':main()
