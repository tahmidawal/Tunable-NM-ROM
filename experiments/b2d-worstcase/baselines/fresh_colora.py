"""Run the CoLoRA lane's single test evaluation (baselines/nfrom/rom.py `test`) on another cohort, model and online
stage unchanged. The only change is the cohort identity the script asserts: the two literals naming test64 (its hash
and its draw) are replaced in the source text before it is executed, and --keep picks rows of the frozen selection.

  fresh_colora.py --seed S --sha HASH --keep name:role[,name:role] --selection FILE --out-selection FILE -- <rom.py test arguments without --selection>
"""
import json, sys
from pathlib import Path

i = sys.argv.index('--')
opt = dict(zip(sys.argv[1:i:2], sys.argv[2:i:2]))
src = (Path(__file__).parent / 'nfrom' / 'rom.py').read_text()
old_sha = "TEST_SHA = 'cd058fd2a297c20296b897e54cb186033a44a8db9527d98dcb2c2dad6044527c'"
old_draw = 'e.params_draw(20260916, 64)'
assert src.count(old_sha) == 1 and src.count(old_draw) == 1
src = src.replace(old_sha, f"TEST_SHA = '{opt['--sha']}'").replace(old_draw, f"e.params_draw({int(opt['--seed'])}, 64)")
sel = json.loads(Path(opt['--selection']).read_text())
keep = [tuple(k.split(':')) for k in opt['--keep'].split(',')]
sel['selected'] = [s for s in sel['selected'] if (s['name'], s['role']) in keep and s['arm'] == 'L']
assert len(sel['selected']) == len(keep), [(s['name'], s['role']) for s in sel['selected']]
out_sel = Path(opt['--out-selection']); out_sel.write_text(json.dumps(sel, indent=1) + '\n')
sys.argv = ['rom.py', 'test', '--selection', str(out_sel)] + sys.argv[i + 1:]
sys.path.insert(0, str(Path(__file__).parent / 'nfrom'))
exec(compile(src, 'rom.py (cohort literals replaced)', 'exec'), {'__name__': '__main__'})
