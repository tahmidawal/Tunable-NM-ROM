// Install marked, katex, mermaid and playwright in a separate tools directory.
// Usage: node reports/export_adaptive_trees_pdf.cjs /path/to/tools/node_modules
const fs = require('node:fs');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
const modules = path.resolve(process.argv[2]);
const {chromium} = require(path.join(modules, 'playwright'));
const katex = require(path.join(modules, 'katex'));
const stem = path.resolve(__dirname, '../understand/adaptive-trees-proposal-guide');
const esc = s => s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');
(async () => {
  const {marked} = await import(pathToFileURL(path.join(modules,'marked/lib/marked.esm.js')));
  let md = fs.readFileSync(stem+'.md','utf8');
  const blocks = [];
  const protect = html => {const key = `PRINTBLOCK${blocks.length}END`; blocks.push(html); return key;};
  md = md.replace(/```mermaid\n([\s\S]*?)```/g, (_,s) => protect(`<div class="mermaid">${esc(s)}</div>`));
  md = md.replace(/\$\$([\s\S]*?)\$\$/g, (_,s) => protect(katex.renderToString(s,{displayMode:true,throwOnError:true})));
  md = md.replace(/\$([^$\n]+)\$/g, (_,s) => protect(katex.renderToString(s,{displayMode:false,throwOnError:true})));
  let body = marked.parse(md).replaceAll("<!-- PAGE -->", '<div class="pagebreak"></div>');
  blocks.forEach((html,i) => {body = body.replace(`<p>PRINTBLOCK${i}END</p>`,html).replace(`PRINTBLOCK${i}END`,html);});
  let css = fs.readFileSync(path.join(modules,'katex/dist/katex.min.css'),'utf8');
  css = css.replace(/url\(([^)]+)\)/g,(_,file) => {
    const f = file.replace(/["']/g,'');
    const bytes = fs.readFileSync(path.join(modules,'katex/dist',f));
    return `url(data:font/${path.extname(f).slice(1)};base64,${bytes.toString('base64')})`;
  });
  const html = `<!doctype html><html><head><meta charset="utf-8"><title>Adaptive trees: proposal guide</title><style>${css}
  @page{size:Letter;margin:0.68in 0.65in 0.7in}
  *{box-sizing:border-box}body{font-family:Arial,sans-serif;color:#171717;font-size:11.5pt;line-height:1.45;margin:0}
  .pagebreak{break-before:page}h1{font-size:22pt;line-height:1.15;margin:0 0 15pt}h2{font-size:16pt;line-height:1.2;border-bottom:1px solid #999;padding-bottom:6pt;margin:22pt 0 10pt}h3{font-size:12pt;margin:16pt 0 7pt}
  h1,h2,h3{break-after:avoid}p{margin:7pt 0;orphans:3;widows:3}a{color:#222;text-decoration:underline;text-decoration-color:#aaa}strong{font-weight:700}
  table{width:100%;border-collapse:collapse;table-layout:fixed;margin:12pt 0;font-size:10.5pt;line-height:1.35}thead{display:table-header-group}tr{break-inside:avoid}th,td{padding:6pt;border:0.6pt solid #aaa;vertical-align:top;overflow-wrap:anywhere}th{background:#eee;text-align:left}li{margin:5pt 0}code{font-size:9pt;overflow-wrap:anywhere}pre{white-space:pre-wrap}
  .katex{font-size:1.02em}.katex-display{margin:14pt 0;break-inside:avoid}.mermaid{break-inside:avoid;text-align:center;margin:12pt 0}.mermaid svg{max-width:100%;max-height:200px}
  .printnote{font-size:9pt;color:#555;margin-bottom:16pt}.references{break-before:page;font-size:8.5pt}.references li{overflow-wrap:anywhere;margin:8pt 0}
  </style></head><body><p class="printnote">RESEARCH PROPOSAL GUIDE · 27 OCTOBER 2026 · Proposed, unvalidated design</p>${body}</body></html>`;
  fs.writeFileSync(stem+'.print.html',html);
  const browser = await chromium.launch({headless:true,...(process.env.PDF_CHROMIUM ? {executablePath:process.env.PDF_CHROMIUM}: {})});
  const page = await browser.newPage({viewport:{width:1000,height:1000}});
  await page.goto(pathToFileURL(stem+'.print.html').href);
  await page.addScriptTag({path:path.join(modules,'mermaid/dist/mermaid.min.js')});
  await page.evaluate(async()=>{
    mermaid.initialize({startOnLoad:false,theme:'neutral',flowchart:{useMaxWidth:true},securityLevel:'strict'});
    await mermaid.run();
    /* Explicit sources already included in this guide.
    const refs = new Map();
    document.querySelectorAll('a[href^="https://"]').forEach(a=>{
      const u=a.href;if(!refs.has(u))refs.set(u,refs.size+1);
      a.append(document.createTextNode(` [${refs.get(u)}]`));
    });
    const section=document.createElement('section');section.className='references';
    section.innerHTML='<h2>Source links for the printed copy</h2><p>Numbers correspond to links in the report. Links remain clickable in this PDF.</p>';
    const ol=document.createElement('ol');refs.forEach((n,u)=>{const li=document.createElement('li');const a=document.createElement('a');a.href=u;a.textContent=u;li.append(a);ol.append(li)});section.append(ol);
    const glossary=[...document.querySelectorAll('h2')].find(h=>h.textContent==='Plain-language glossary');
    if(glossary){glossary.style.breakBefore='page';glossary.before(section)}else{document.body.append(section)}
    */
    await document.fonts.ready;
  });
  await page.emulateMedia({media:'print'});
  // Keep unusually long display formulas inside the printable text width.
  await page.addStyleTag({content:'body{width:7.2in}'});
  await page.evaluate(()=>document.querySelectorAll('.katex-display').forEach(el=>{if(el.scrollWidth>el.clientWidth)el.style.fontSize=`${el.clientWidth/el.scrollWidth*100}%`;}));
  await page.evaluate(()=>document.querySelectorAll('script').forEach(s=>s.remove()));
  fs.writeFileSync(stem+'.print.html',(await page.content()).split('\n').map(s=>s.trimEnd()).join('\n')+'\n');
  await page.pdf({path:stem+'.pdf',format:'Letter',printBackground:true,displayHeaderFooter:true,headerTemplate:'<span></span>',footerTemplate:'<div style="width:100%;text-align:center;font-size:8px;color:#555">Adaptive trees: proposal guide · <span class="pageNumber"></span> / <span class="totalPages"></span></div>',margin:{top:'0.68in',bottom:'0.7in',left:'0.65in',right:'0.65in'}});
  console.log(JSON.stringify({pdf:stem+'.pdf',equations:blocks.length-1,mermaid:await page.locator('.mermaid svg').count()}));
  await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
