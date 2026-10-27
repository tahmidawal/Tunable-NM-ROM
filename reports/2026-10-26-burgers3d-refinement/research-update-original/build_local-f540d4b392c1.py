from pathlib import Path
import json, re, base64
root=Path(__file__).resolve().parents[2]
here=Path(__file__).resolve().parent
final=root/'reports/2026-10-17-full-experiment-plan/final/figs'
method=root/'reports/2026-10-22-method-presentation/figs'
quad=root/'reports/2026-10-01-offmesh-quadrature-plots/figs'
assets={
'db24dc0167cc8ed61a4464fc66064f6e':method/'error_vs_r.png',
'8ef6d364d6fdd02c3c06390d6b519bd0':method/'mesh_cost_iters.png',
'cc2c9df27cc9f988fe15071c375ae566':final/'fom2d_fom_pareto.png',
'1045285ee59ea88117766f439f8b7ca0':final/'fom2d_speedup_vs_mesh.png',
'1384991a552e7eefa6f06053e505220c':final/'hrb_front-c2d1024.png',
'50f6ccebb30a8475519d377f3893f1fc':final/'dims_front_3d.png',
'705c17d8e9b74c03d3de6505ce435344':final/'hrb_front-c3d64.png',
'49ca0e980cd1f160fcd01bff9b58b226':final/'summary_union_speedup.png',
'ed71b71f74f58d02203d5c5741e534e2':root/'reports/2026-10-21-improvement-campaign/final/figs/imp-box__S_vs_W_worst.png',
'5fad1d732f75889cc125d918dbb7bdf8':quad/'2d_rule_ladder.png',
'bae25b691332b768c521140afaa983b9':quad/'2d_solve_time.png'}
deck=json.loads((here/'project/deck.json').read_text())
slides=[]
for name in deck['order']:
    s=(here/'project/slides'/f'{name}.html').read_text()
    def image(m):
        p=assets[m[1]]
        return 'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()
    s=re.sub(r'/_blob/([a-z0-9]+)', image,s)
    slides.append(s)
page='''<!doctype html><html><head><meta charset="utf-8"><title>Research Update</title><meta name="viewport" content="width=device-width, initial-scale=1"><style>
*{box-sizing:border-box}body{margin:0;background:#18202c;font-family:Arial,sans-serif}header{height:58px;display:flex;align-items:center;gap:14px;padding:8px 18px;color:white}button,select{font:inherit;padding:8px;border-radius:5px;border:1px solid #aaa}header strong{margin-right:auto}#viewport{position:relative;margin:auto;overflow:hidden}#slides{width:1920px;height:1080px;transform-origin:top left}#slides>section{width:1920px;height:1080px;position:relative;overflow:hidden}#slides>section[hidden]{display:none!important}aside{display:none}h1,h2,p{margin:0}ul{margin:0;padding-left:30px}table{border-collapse:collapse;width:100%}td,th{padding:20px;text-align:left;border-bottom:1px solid #ccc}
</style></head><body><header><strong>Research Update</strong><button id="prev">←</button><select id="pick" aria-label="Slide"></select><button id="next">→</button><span id="count"></span></header><div id="viewport"><main id="slides">'''+''.join(slides)+'''</main></div><script>
const slides=[...document.querySelectorAll('#slides>section')], pick=document.querySelector('#pick'); let index=0;
slides.forEach((s,i)=>{const o=document.createElement('option');o.value=i;o.textContent=(i+1)+' · '+(s.querySelector('h1,h2')?.textContent||s.id).slice(0,65);pick.append(o)});
function show(i){index=Math.max(0,Math.min(slides.length-1,i));slides.forEach((s,j)=>s.hidden=j!==index);pick.value=index;document.querySelector('#count').textContent=(index+1)+' / '+slides.length;location.hash=slides[index].id;}
function resize(){let scale=Math.min(innerWidth/1920,(innerHeight-64)/1080);let v=document.querySelector('#viewport');v.style.width=1920*scale+'px';v.style.height=1080*scale+'px';document.querySelector('#slides').style.transform='scale('+scale+')'}
pick.onchange=()=>show(+pick.value);document.querySelector('#prev').onclick=()=>show(index-1);document.querySelector('#next').onclick=()=>show(index+1);onkeydown=e=>{if(e.target.tagName==='SELECT')return;if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();show(index+1)}if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();show(index-1)}};onresize=resize;show(Math.max(0,slides.findIndex(s=>s.id===location.hash.slice(1))));resize();
</script></body></html>'''
(here/'research-update-local.html').write_text(page)
print(f'Built {len(slides)} slides with embedded local images.')
