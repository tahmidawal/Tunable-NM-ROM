"""Record native-control completion or failure without physical/training admission."""
import argparse
import json
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--results',required=True,type=Path)
parser.add_argument('--exit-code',required=True,type=int)
parser.add_argument('--last-stage',required=True)
args=parser.parse_args()
path=args.results/'control/result.json'
result=json.loads(path.read_text()) if path.exists() else None
output_bytes=sum(p.stat().st_size for p in (args.results/'control').glob('*') if p.is_file())
passed=args.exit_code==0 and result is not None and result.get('passed') is True and output_bytes<=64*2**20
report={'scope':'native implementation control only','passed':passed,'physical_admission':False,
        'training_admission':False,'exit_code':args.exit_code,'last_stage':args.last_stage,
        'output_bytes':output_bytes,'output_cap_bytes':64*2**20,'native_result_present':result is not None}
(args.results/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(0 if passed else 1)
