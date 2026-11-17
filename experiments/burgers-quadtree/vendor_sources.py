from pathlib import Path
import subprocess,hashlib,json
H=Path(__file__).resolve().parent
src=Path('/home/tahmid/Dev/Tunable-NM-ROM-Claude/worktrees/2026-11-05-burgers-quadrature-plateau/experiments/burgers-quadrature-plateau')
m=json.loads((src/'vendor/PROVENANCE.json').read_text())
rows=[]
for f in m['files']:
 p=f['source_path']
 if '/store/' in p or '/checks/' in p or p.endswith('DESIGN.md'): continue
 if any(s in p for s in ['quadrature-burgers3d/','jcp-speed3d/']): continue
 data=subprocess.check_output(['git','-c','gc.auto=0','show',f"{f['source_commit']}:{p}"])
 assert hashlib.sha256(data).hexdigest()==f['sha256']
 dst=H/'vendor/tree'/p; dst.parent.mkdir(parents=True,exist_ok=True); dst.write_bytes(data)
 rows.append(f)
(H/'vendor/PROVENANCE.json').write_text(json.dumps({'files':rows},indent=2)+'\n')
print(len(rows),sum(f['bytes'] for f in rows))
