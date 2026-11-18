"""Finite sequential family-A/B quadtree span cells in one allocated GPU job."""
import subprocess,sys,argparse,json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
for family in ('A','B'):
 subprocess.run([sys.executable,'c1_tree.py','--family',family,'--out',str(o/family)],check=True)
(o/'result.json').write_text(json.dumps(dict(complete=True,families=['A','B'],contract='separate matched-data banks; initial-only complete leaf partitions; no PDE claim'))+'\n')
