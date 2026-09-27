"""One bounded family-resolution smoke; full reference cohort runs on cluster."""
import json
from pathlib import Path
import ns3d_verify as V


if __name__=='__main__':
    out=Path(__file__).resolve().parent/'checks'
    result=V.reference_checks(n=32,count=1,outdir=None)
    (out/'local_reference_amended.json').write_text(json.dumps(result,indent=2)+'\n')
    if not result['passed']:
        raise SystemExit(2)
