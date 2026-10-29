"""Export the collaborator edition with portable GitHub links and rendered math.
Run with /home/tahmid/Dev/.venv/bin/python reports/2026-10-28-decoder-history/export_pdf.py
"""
from pathlib import Path
import hashlib, html, io, json, re, subprocess
from urllib.parse import quote
from markdown_it import MarkdownIt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image, KeepTogether
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from matplotlib.mathtext import math_to_image
from matplotlib.font_manager import FontProperties
from PIL import Image as PILImage
from pypdf import PdfReader

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
REV='dd6c6c0931dbbb21f3da39a940d2df90781e1a49'
BASE='https://github.com/tahmidawal/Tunable-NM-ROM'
MD=MarkdownIt('commonmark').enable('table')
FONT='/usr/share/fonts/truetype/dejavu/'
for name,file in [('DejaVu','DejaVuSans.ttf'),('DejaVu-Bold','DejaVuSans-Bold.ttf'),('DejaVu-Oblique','DejaVuSans-Oblique.ttf'),('DejaVu-BoldOblique','DejaVuSans-BoldOblique.ttf'),('DejaVuMono','DejaVuSansMono.ttf')]:
    pdfmetrics.registerFont(TTFont(name,FONT+file))
pdfmetrics.registerFontFamily('DejaVu',normal='DejaVu',bold='DejaVu-Bold',italic='DejaVu-Oblique',boldItalic='DejaVu-BoldOblique')

def github(m):
    path=Path(m[1]);rel=path.relative_to(ROOT).as_posix()
    # The shared index is itself converted to portable links in this edition.
    if path.name=='CODE-INDEX.md':
        return ']('+BASE+'/blob/main/'+quote(rel.replace('CODE-INDEX.md','CODE-INDEX.github.md'),safe='/')+')'
    subprocess.run(['git','-c','gc.auto=0','-C',str(ROOT),'cat-file','-e',REV+':'+rel],check=True,capture_output=True)
    return ']('+BASE+'/blob/'+REV+'/'+quote(rel,safe='/')+')'

def portable(s):
    return re.sub(r'\]\((/[^)]+)\)',github,s)

original=(HERE/'2026-10-28-decoder-history.md').read_text()
technical=original[original.index('## The architectural distinctions'):original.index('## The beginning:')]
technical=re.sub(r'```mermaid.*?```','The development branches are described chronologically below. They are not all merged into a single implementation.',technical,flags=re.S)
technical=technical.replace('Arrows indicate conceptual development, not a claim that every branch was merged into the next.','')
body=original[original.index('## The beginning:'):]
body=body.replace('## Reproducibility of this report','## Source and reproduction notes')
body=body.replace('**Current', '**Historical')
body=body.replace('### Obstacle-flow ROM — active work, no accepted decoder result yet','### Obstacle-flow ROM — status at the historical cutoff')
body=body.replace('## Source and reproduction notes','## Source and reproduction notes\n\nThe collaborator PDF is built by `export_pdf.py`, using the simpler opening and dated update beside the original report. The GitHub links point to committed evidence; the companion GitHub code index provides original branch and file locations. The paragraphs below describe how the underlying historical report and its numerical tables were generated.')
intro=(HERE/'collaborator-introduction.md').read_text()
update=(HERE/'collaborator-update.md').read_text()
text=portable(intro+'\n\n'+update+'\n\n<!-- PAGEBREAK -->\n\n'+technical+body)
# Do not expose workstation paths in inline source-location prose either.
text=re.sub(r'`worktrees/([^`]+)`',r'`exp/\1`',text)
assert '/home/tahmid' not in text
(HERE/'Decoder-History-Collaborators.md').write_text(text)
index=portable((HERE/'CODE-INDEX.md').read_text())
(HERE/'CODE-INDEX.github.md').write_text(index)

styles={
 'p':ParagraphStyle('p',fontName='DejaVu',fontSize=9.5,leading=14,spaceAfter=7,textColor=colors.HexColor('#263442')),
 'h1':ParagraphStyle('h1',fontName='DejaVu-Bold',fontSize=25,leading=30,spaceAfter=15,textColor=colors.HexColor('#152f43'),keepWithNext=True),
 'h2':ParagraphStyle('h2',fontName='DejaVu-Bold',fontSize=16,leading=21,spaceBefore=15,spaceAfter=10,keepWithNext=True,textColor=colors.HexColor('#152f43')),
 'h3':ParagraphStyle('h3',fontName='DejaVu-Bold',fontSize=11.5,leading=16,spaceBefore=12,spaceAfter=7,keepWithNext=True,textColor=colors.HexColor('#152f43')),
 'cell':ParagraphStyle('cell',fontName='DejaVu',fontSize=7,leading=10,spaceAfter=0,wordWrap='CJK'),
 'bullet':ParagraphStyle('bullet',fontName='DejaVu',fontSize=9.5,leading=14,leftIndent=12,firstLineIndent=-9,spaceAfter=6),
}
ASSETS=HERE/'pdf-assets';ASSETS.mkdir(exist_ok=True)
def mathimage(expr,inline=False):
    expr=' '.join(expr.split()).replace(r'\operatorname{vech}',r'\mathrm{vech}').replace(r'\bigl','').replace(r'\bigr','').replace(r'\mathcal A',r'\mathcal{A}')
    name=hashlib.sha256((expr+str(inline)).encode()).hexdigest()[:16]+'.png'
    path=ASSETS/name
    math_to_image('$'+expr+'$',path,prop=FontProperties(size=11 if inline else 14),dpi=220,format='png',color='#152f43')
    im=PILImage.open(path);w,h=im.size
    scale=72/220
    return path,w*scale,h*scale

def inline(tokens):
    out=[]
    for t in tokens or []:
        if t.type=='text':
            bits=re.split(r'(\$[^$]+\$)',t.content)
            for bit in bits:
                if bit.startswith('$') and bit.endswith('$') and len(bit)>2:
                    p,w,h=mathimage(bit[1:-1],True)
                    out.append(f'<img src="{p}" width="{w:.2f}" height="{h:.2f}" valign="middle"/>')
                else:out.append(html.escape(bit))
        elif t.type=='code_inline':out.append('<font name="DejaVuMono" size="8">'+html.escape(t.content)+'</font>')
        elif t.type=='strong_open':out.append('<b>')
        elif t.type=='strong_close':out.append('</b>')
        elif t.type=='em_open':out.append('<i>')
        elif t.type=='em_close':out.append('</i>')
        elif t.type=='link_open':out.append('<a color="#12638a" href="'+html.escape(t.attrGet('href'),quote=True)+'">')
        elif t.type=='link_close':out.append('</a>')
        elif t.type in ['softbreak','hardbreak']:out.append(' ')
    return ''.join(out)

# Extract display math before Markdown parsing.
equations=[]
def eqmark(m):
    equations.append(m[1]);return '\n\nMATHDISPLAY'+str(len(equations)-1)+'\n\n'
parsed=re.sub(r'\$\$(.*?)\$\$',eqmark,text,flags=re.S)
tokens=MD.parse(parsed);story=[];i=0;list_depth=0
width=483
while i<len(tokens):
    t=tokens[i]
    if t.type=='html_block' and 'PAGEBREAK' in t.content:story.append(PageBreak())
    elif t.type in ['bullet_list_open','ordered_list_open']:list_depth+=1
    elif t.type in ['bullet_list_close','ordered_list_close']:list_depth-=1
    elif t.type=='heading_open':
        content=inline(tokens[i+1].children)
        story.append(Paragraph(content,styles.get(t.tag,styles['h3'])))
        if t.tag=='h1':story.append(Paragraph('Research history · Collaborator edition · October 29, 2026',styles['p']))
        i+=2
    elif t.type=='paragraph_open':
        nxt=tokens[i+1]
        if nxt.content.startswith('MATHDISPLAY'):
            n=int(nxt.content.removeprefix('MATHDISPLAY'));p,w,h=mathimage(equations[n]);factor=min(1,width/w)
            story.extend([Spacer(1,5),Image(str(p),w*factor,h*factor),Spacer(1,10)])
        else:story.append(Paragraph(('• ' if list_depth else '')+inline(nxt.children),styles['bullet' if list_depth else 'p']))
        i+=2
    elif t.type=='table_open':
        rows=[];row=[];i+=1
        while tokens[i].type!='table_close':
            q=tokens[i]
            if q.type=='tr_open':row=[]
            elif q.type=='inline':row.append(q)
            elif q.type=='tr_close':rows.append(row)
            i+=1
        # Long internal cell identifiers stay in the source CSV, not this print table.
        if rows[0][0].content=='PDE / cell':
            for row in rows:
                row[0].content=row[0].content.split(' / ')[0]
                row[0].children=MD.parseInline(row[0].content)[0].children
        n=len(rows[0]);ws=[width/n]*n
        if n==2:ws=[width*.40,width*.60]
        data=[[Paragraph(('<b>'+inline(c.children)+'</b>') if ri==0 else inline(c.children),styles['cell']) for c in row] for ri,row in enumerate(rows)]
        tab=Table(data,colWidths=ws,repeatRows=1,hAlign='LEFT')
        tab.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf0f4')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#8da1ae')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d8e0e5')),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        story.extend([KeepTogether([tab]) if tab.wrap(width,740)[1]<450 else tab,Spacer(1,10)])
    i+=1

def footer(canvas,doc):
    canvas.saveState();canvas.setFont('DejaVu',8);canvas.setFillColor(colors.HexColor('#536675'))
    canvas.drawString(56,29,'Decoder history | Tunable NM-ROM')
    canvas.drawRightString(539,29,str(doc.page));canvas.restoreState()

pdf=HERE/'Decoder-History-Collaborators.pdf'
class ReportDoc(SimpleDocTemplate):
    def afterFlowable(self,flowable):
        if isinstance(flowable,Paragraph) and flowable.style.name in ['h1','h2','h3']:
            title=flowable.getPlainText();key=hashlib.sha256(title.encode()).hexdigest()
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(title,key,level=int(flowable.style.name[-1])-1,closed=False)
doc=ReportDoc(str(pdf),pagesize=(595,842),rightMargin=56,leftMargin=56,topMargin=49,bottomMargin=48,title='How our decoders evolved — collaborator edition',author='Tunable NM-ROM research project')
doc.build(story,onFirstPage=footer,onLaterPages=footer)
r=PdfReader(pdf);links=[]
for page in r.pages:
    for a in page.get('/Annots',[]):
        act=a.get_object().get('/A',{})
        if act.get('/URI'):links.append(str(act['/URI']))
assert links and all(u.startswith(BASE) for u in links)
assert '/home/' not in '\n'.join(p.extract_text() for p in r.pages)
check={'pages':len(r.pages),'clickable_github_links':len(links),'unique_github_links':sorted(set(links)),'pinned_history_revision':REV,'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'no_local_path_links':True}
(HERE/'PDF-VALIDATION.json').write_text(json.dumps(check,indent=2)+'\n')
print(json.dumps({k:v for k,v in check.items() if k!='unique_github_links'},indent=2))
