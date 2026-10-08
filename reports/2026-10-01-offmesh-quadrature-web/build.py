"""Render the two 2026-10-01 off-mesh quadrature reports (generated markdown) into one web page.
Content is copied verbatim from the reports; this script only converts markdown to HTML.
Run with any Python that has `markdown` installed."""
import html, os, re
import markdown

HERE = os.path.dirname(os.path.abspath(__file__))
REP = os.path.dirname(HERE)
SRC = [("b3d", "Burgers 3D", "2026-10-01-burgers3d-offmesh-quadrature.md"),
       ("b2d", "Burgers 2D", "2026-10-01-burgers2d-offmesh-quadrature.md")]


def convert(text, prefix):
    stash = []
    def keep(s):
        stash.append(s); return f"@@KEEP{len(stash) - 1}@@"
    # mermaid fences -> native mermaid blocks
    text = re.sub(r"```mermaid\n(.*?)```", lambda m: keep('<pre class="mermaid">' + html.escape(m.group(1)) + "</pre>"), text, flags=re.S)
    # protect math from markdown (display first, then inline)
    text = re.sub(r"\$\$(.+?)\$\$", lambda m: keep("$$" + html.escape(m.group(1)) + "$$"), text, flags=re.S)
    text = re.sub(r"(?<![\\$])\$([^$\n]+?)\$", lambda m: keep("$" + html.escape(m.group(1)) + "$"), text)
    out = markdown.markdown(text, extensions=["tables", "fenced_code", "toc"],
                            extension_configs={"toc": {"slugify": lambda v, s: prefix + "-" + re.sub(r"[^a-z0-9]+", "-", v.lower()).strip("-")}})
    out = re.sub(r"@@KEEP(\d+)@@", lambda m: stash[int(m.group(1))], out)
    out = re.sub(r"<table>", '<div class="tbl"><table>', out).replace("</table>", "</table></div>")
    return out


tmpl = open(os.path.join(HERE, "template.html")).read()
tabs, panels = [], []
for i, (key, name, fn) in enumerate(SRC):
    body = convert(open(os.path.join(REP, fn)).read(), key)
    tabs.append(f'<button class="tab" role="tab" id="tab-{key}" aria-controls="{key}" aria-selected="{"true" if i == 0 else "false"}">{name}</button>')
    panels.append(f'<article class="report" id="{key}" role="tabpanel" aria-labelledby="tab-{key}"{"" if i == 0 else " hidden"}>'
                  f'<p class="src">Source: <code>reports/{fn}</code> (generated from run JSONs; rendered verbatim)</p>{body}</article>')
page = tmpl.replace("{{TABS}}", "\n".join(tabs)).replace("{{PANELS}}", "\n".join(panels))
open(os.path.join(HERE, "index.html"), "w").write(page)
print("wrote", os.path.join(HERE, "index.html"), len(page) // 1024, "KB")
