"""Bounded cluster rendering of already verified results; no new training or fits."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import argparse,subprocess,sys
from common import *
from neural.build_data import resolve


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);meta=preflight();ns=Path(os.environ['STUDY_NAMESPACE']);H=Path(__file__).resolve().parent
    (H/'results').mkdir(exist_ok=True);inputs=[]
    for name in ['b7parentfull','b7parentrepair','b7namedparent','b7parentstarts','b7snff64']:
        rp=resolve(ns/name/'output/result.json',ns);root=rp.parent.parent
        m=json.loads((H/f'manifests/retention-{name}.json').read_text());assert m['local_and_remote_verified']
        for row in m['files']:assert digest(root/row['file'])==row['sha256'],(name,row['file'])
        target=H/'results'/name;assert not target.exists();target.symlink_to(root,target_is_directory=True)
        inputs.append(dict(name=name,result_sha256=digest(rp),verified_files=len(m['files'])))
        print('verified_render_input',name,flush=True)
    result=dict(meta=meta,inputs=inputs,status='rendering',trained=False,new_reconstructions=False,final_test_opened=False)
    save(out/'result.json',result)
    subprocess.run([sys.executable,str(H/'reports/final_parent_round.py'),'--output-root',str(out)],check=True,timeout=480)
    exports=json.loads((out/'full-resolution-exports.json').read_text());assert len(exports)==72
    from PIL import Image
    for row in exports:
        with Image.open(out/row['file']) as im:assert im.size==(4096,4096)
    result.update(status='complete',full_resolution_png_count=len(exports),scientific_results_changed=False)
    save(out/'result.json',result)


if __name__=='__main__':main()
