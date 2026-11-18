"""Pre-registered prefix-stable extension; old raw reference artifacts stay immutable."""
from pathlib import Path
import argparse,json,subprocess,sys
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    script=str(Path(__file__).with_name('references.py'))
    for group,start,count in [('train',64,128),('dev',16,32)]:
        subprocess.run([sys.executable,script,'--out',str(out/group),'--group',group,'--start',str(start),'--count',str(count),'--random-only'],check=True)
    rows={g:json.loads((out/g/'result.json').read_text()) for g in ['train','dev']}
    assert all(c['provisional_ladder_pass'] for r in rows.values() for c in r['cases'])
    (out/'result.json').write_text(json.dumps(dict(status='complete_extended_spectral_ladders',groups=rows,
       independent_crosscheck='additional worst-uncertainty cases pending',final_test_opened=False),indent=2)+'\n')
if __name__=='__main__':main()
