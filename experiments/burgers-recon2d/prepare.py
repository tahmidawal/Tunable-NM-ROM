"""Prepare the frozen 2D screen inputs and matched-data POD; no neural training."""
from pathlib import Path
import argparse
import adapter as A

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',required=True); ap.add_argument('--phase',choices=['bases','coarse','all'],default='all'); ap.add_argument('--cohorts',nargs='+',choices=['dev6','val32'],default=['dev6','val32']); args=ap.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    A.verify_inputs(); A.write_json(out/'runtime.json',A.runtime()); A.write_json(out/'disjointness.json',A.disjointness())
    if args.phase in ['bases','all']: A.build_bases(out/'bases')
    if args.phase in ['coarse','all']: A.coarse_fields(out/'coarse',cohorts=args.cohorts)
    A.write_json(out/'COMPLETE.json',dict(phase=args.phase,cohorts=args.cohorts))
if __name__=='__main__': main()
