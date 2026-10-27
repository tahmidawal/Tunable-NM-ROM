"""Fresh-process entry point: load the independently reviewed POD modules before generic imports."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
for pin in json.loads((HERE/'POD-PROVENANCE.json').read_text()):
    h=hashlib.sha256()
    with (ROOT/pin['path']).open('rb') as f:
        for block in iter(lambda:f.read(1<<22),b''):h.update(block)
    assert h.hexdigest()==pin['sha256'],pin['path']
sys.path.insert(0,str(ROOT/'experiments/b3d-pod'))
import pod_subjects as POD
from run import Context

def module_provenance():
    modules=[POD,POD.I,POD.H,POD.R,POD.I.P,POD.H.C3,POD.H.C3.b3,POD.H.E2,POD.R.T2]
    allowed={p['sha256'] for p in json.loads((HERE/'POD-PROVENANCE.json').read_text())}
    rows=[]
    for m in modules:
        path=Path(m.__file__).resolve();digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest in allowed,(m.__name__,str(path),digest)
        rows.append(dict(module=m.__name__,path=str(path),sha256=digest))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();cp=Path(a.config).resolve();cfg=json.loads(cp.read_text());cfg['_config_path']=str(cp)
    assert cfg['seed']==923801
    cx=Context(cfg,a.out);cx.rep['module_provenance']=module_provenance();cx.save()
    try:
        if cfg['mode']=='pod_tune':
            import pod_tune
            pod_tune.run(cx,POD)
        else:raise ValueError(cfg['mode'])
        cx.rep.update(execution_complete=True,complete=True);cx.save()
    except BaseException as e:
        cx.rep['failure']=repr(e);cx.save();raise

if __name__=='__main__':main()
