"""Fresh-process primary reconstruction used by the prospective native proof."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from reference_primary import reconstruct_primary
from immutable_rule import array_digest
p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
import ngsolve as ng
cfg=json.loads((a.case/'config.json').read_text());ng.SetNumThreads(cfg['threads']);tick=time.perf_counter()
with ng.TaskManager():fields,record=reconstruct_primary(a.case,ng)
for region,arrays in fields.items():np.savez_compressed(a.out/(region+'.npz'),**arrays)
record.update(complete_reconstruction_and_persistence_seconds=time.perf_counter()-tick,physical_admission=False,content_sha256={r:{k:array_digest(v) for k,v in x.items()} for r,x in fields.items()})
(a.out/'replay.json').write_text(json.dumps(record,indent=2)+'\n')
