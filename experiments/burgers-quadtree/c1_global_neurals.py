import subprocess,sys,argparse,json
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();o=Path(a.out);o.mkdir(parents=True,exist_ok=True)
for family in ('A','B'):subprocess.run([sys.executable,'c1_global_neural.py','--family',family,'--out',str(o/family)],check=True)
(o/'result.json').write_text(json.dumps(dict(complete=True,families=['A','B'],contract='separate matched-data global neural banks; frozen existing training/dev tables; no tests'))+'\n')
