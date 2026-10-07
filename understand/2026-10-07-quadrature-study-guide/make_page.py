"""Build web/index.html from page.template.html and the generated tables/*.tex (no hand-typed numbers)."""
import os, re
HERE = os.path.dirname(os.path.abspath(__file__))

def tex_table_to_html(path):
    lines = [l.strip() for l in open(path) if l.strip()]
    body = [l for l in lines if not l.startswith("\\") or l.startswith("\\textbf")]
    rows = [[c.strip() for c in re.sub(r"\\\\$", "", l).split("&")] for l in body]
    head, rest = rows[0], rows[1:]
    h = "<table><thead><tr>" + "".join(f"<th>{c}</th>" for c in head) + "</tr></thead><tbody>"
    h += "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rest)
    return h + "</tbody></table>"

page = open(os.path.join(HERE, "page.template.html")).read()
for name in sorted(os.listdir(os.path.join(HERE, "tables"))):
    key = "{{" + name[:-4] + "}}"
    if key in page:
        page = page.replace(key, tex_table_to_html(os.path.join(HERE, "tables", name)))
left = re.findall(r"\{\{[a-z0-9_]+\}\}", page)
assert not left, left
os.makedirs(os.path.join(HERE, "web"), exist_ok=True)
open(os.path.join(HERE, "web", "index.html"), "w").write(page)
print("wrote web/index.html")
