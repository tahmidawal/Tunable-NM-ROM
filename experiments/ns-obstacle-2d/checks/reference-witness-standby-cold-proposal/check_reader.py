"""Verify unchanged recovery reader on all three proposed archive descriptors."""
import hashlib, importlib.util, json
from pathlib import Path
out=Path(__file__).resolve().parent
w=out.parents[3]
reader=w/'experiments/ns-obstacle-2d/g2/git_archive_recovery.py'
h=hashlib.sha256(reader.read_bytes()).hexdigest()
assert h=='da6d150859f688ad7fba8767d249d9c7fd8f4f237af8c20e80e93351777365f2'
s=importlib.util.spec_from_file_location('recovery',reader)
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
rows=[]
for p in sorted(out.glob('*-RECOVERY.json')):
    rows.append({'manifest':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
                 'result':m.verify(w,json.loads(p.read_text()),output=None)})
assert len(rows)==3
r={'reader_sha256':h,'passed':True,'archives':rows,'archive_materialized':False}
(out/'READER-CHECK.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
