import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('out',type=Path);p.add_argument('--exit-code',type=int);p.add_argument('--failed-stage');a=p.parse_args();a.out.mkdir(exist_ok=True);(a.out/'batch_summary.json').write_text(json.dumps({'exit_code':a.exit_code,'failed_stage':a.failed_stage,'physical_admission':False},indent=2)+'\n')
