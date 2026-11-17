from pathlib import Path
import json,numpy as np
H=Path(__file__).resolve().parent;r=json.loads((H/'evidence/dynamics01-result.json').read_text());rows=[]
for row in r['rows']:
 i=row['case']
 if i>=2:continue
 data=np.load(H/f"results/dynamics01/output/{row['method']}_{i}.npz")
 pred=data['final4096'];truth=np.load(H/f'results/dense01/output/dev{i}_final4096.npz');ref=truth['field'];cx,cy,w,amp,nu=truth['parameters']
 x=np.arange(4096)/4095
 u0=amp*np.exp(-((x[:,None]-cx)**2+(x[None,:]-cy)**2)/(2*w*w));u0[[0,-1],:]=0;u0[:,[0,-1]]=0
 error=float(np.linalg.norm(pred-ref)/np.linalg.norm(u0));discrepancy=abs(error-row['final4096_error'])
 assert pred.shape==(4096,4096) and np.isfinite(pred).all()
 assert not np.any(pred[[0,-1],:]) and not np.any(pred[:,[0,-1]])
 assert discrepancy<1e-12
 rows.append(dict(method=row['method'],case=i,independent_error=error,reported_discrepancy=discrepancy))
(H/'evidence/dense-independent-audit.json').write_text(json.dumps(dict(passed=True,rows=rows),indent=2)+'\n');print(rows)
