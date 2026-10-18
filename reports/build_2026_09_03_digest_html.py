#!/usr/bin/env python3
"""Render `2026-09-03-separable-decoder-results-digest.md` to a self-contained artifact page.

    /home/tahmid/Dev/.venv/bin/python reports/build_2026_09_03_digest_html.py

The markdown is the source (regenerate it with `gen_2026-09-03-separable-decoder-digest.py`);
this file only renders.  The subset of markdown used by the digest is handled here: headings,
paragraphs, bullet lists, block quotes, pipe tables, ```mermaid fences, $$ display math, and the
inline forms **bold**, *italic*, `code`, <sub>.  Math is left for MathJax; PASS / FAIL /
INCOMPLETE tokens in table cells become chips.
"""
from __future__ import annotations

import html
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "2026-09-03-separable-decoder-results-digest.md"
OUT = HERE / "2026-09-03-separable-decoder-results-digest.html"


def inline(s: str) -> str:
    # protect code spans first
    codes = []

    def keep(m):
        codes.append(html.escape(m.group(1)))
        return f"\x00{len(codes) - 1}\x00"

    s = re.sub(r"`([^`]*)`", keep, s)
    s = html.escape(s, quote=False)
    s = s.replace("&lt;sub&gt;", "<sub>").replace("&lt;/sub&gt;", "</sub>")
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)
    s = re.sub(r"\x00(\d+)\x00", lambda m: f"<code>{codes[int(m.group(1))]}</code>", s)
    return s


def chip(cell: str) -> str:
    cell = re.sub(r"\bPASS\b", '<span class="chip pass">PASS</span>', cell)
    cell = re.sub(r"\bFAIL\b", '<span class="chip fail">FAIL</span>', cell)
    cell = re.sub(r"\bINCOMPLETE\b", '<span class="chip inc">INCOMPLETE</span>', cell)
    return cell


def table(lines):
    rows = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
    head, body = rows[0], [r for r in rows[2:]]
    prose = any(len(c) > 70 for r in body for c in r)
    out = [f'<div class="tscroll{" prose" if prose else ""}"><table>', "<thead><tr>"]
    out += [f"<th>{inline(c)}</th>" for c in head]
    out.append("</tr></thead><tbody>")
    for r in body:
        cells = r + [""] * (len(head) - len(r))
        out.append("<tr>" + "".join(f"<td>{chip(inline(c))}</td>" for c in cells) + "</tr>")
    out.append("</tbody></table></div>")
    return "\n".join(out)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def render(md: str):
    lines = md.splitlines()
    out, toc = [], []
    i = 0
    para = []

    def flush():
        nonlocal para
        if para:
            txt = " ".join(para).strip()
            cls = ' class="source"' if txt.startswith("*Source:") or txt.startswith("*Phase") else ""
            out.append(f"<p{cls}>{inline(txt)}</p>")
            para = []

    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```mermaid"):
            flush()
            j = i + 1
            buf = []
            while not lines[j].startswith("```"):
                buf.append(lines[j]); j += 1
            out.append('<div class="fig"><pre class="mermaid">' + "\n".join(buf) + "</pre></div>")
            i = j + 1
            continue
        if ln.strip() == "$$":
            flush()
            j = i + 1
            buf = []
            while lines[j].strip() != "$$":
                buf.append(lines[j]); j += 1
            out.append('<div class="math">$$' + "\n".join(buf) + "$$</div>")
            i = j + 1
            continue
        m = re.match(r"^(#{1,3}) (.*)$", ln)
        if m:
            flush()
            level = len(m.group(1)); text = m.group(2)
            sid = slug(re.sub(r"[$\\]", "", text))
            if level == 1:
                out.append(f"<h1>{inline(text)}</h1>")
            else:
                if level == 2:
                    toc.append((sid, text))
                out.append(f'<h{level} id="{sid}">{inline(text)}</h{level}>')
            i += 1
            continue
        if ln.startswith("|"):
            flush()
            j = i
            buf = []
            while j < len(lines) and lines[j].startswith("|"):
                buf.append(lines[j]); j += 1
            out.append(table(buf))
            i = j
            continue
        if ln.startswith("- "):
            flush()
            j = i
            items = []
            while j < len(lines) and (lines[j].startswith("- ") or (lines[j].startswith("  ") and items)):
                if lines[j].startswith("- "):
                    items.append(lines[j][2:])
                else:
                    items[-1] += " " + lines[j].strip()
                j += 1
            out.append("<ul>" + "".join(f"<li>{inline(t)}</li>" for t in items) + "</ul>")
            i = j
            continue
        if ln.startswith("> "):
            flush()
            j = i
            buf = []
            while j < len(lines) and lines[j].startswith(">"):
                buf.append(lines[j][1:].strip()); j += 1
            paras = [p for p in " ".join(buf).split("  ") if p.strip()]
            out.append("<blockquote>" + "".join(f"<p>{inline(p)}</p>" for p in re.split(r"\s{2,}|(?<=\.) (?=\*\*)", " ".join(buf)) if p.strip()) + "</blockquote>")
            i = j
            continue
        if not ln.strip():
            flush()
            i += 1
            continue
        para.append(ln)
        i += 1
    flush()
    return "\n".join(out), toc


CSS = r"""
:root{
  --ground:#f3f4f1; --surface:#ffffff; --ink:#1b2229; --muted:#5b6670; --rule:#d3d8d4;
  --accent:#1f5f7a; --accent-ink:#ffffff; --sand:#8a7350; --leaf:#2f7a48; --rust:#a4462d;
  --chip-pass-bg:#dcefe0; --chip-pass-ink:#14401f; --chip-fail-bg:#f3ddd5; --chip-fail-ink:#5e2415;
  --chip-inc-bg:#ece7dc; --chip-inc-ink:#3d3220; --thead:#e9ecea; --zebra:#f7f8f6; --code:#eceeeb;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --ground:#14181b; --surface:#1b2125; --ink:#e4e6e2; --muted:#9aa4ab; --rule:#2e363c;
    --accent:#7fb7d1; --accent-ink:#0f1a20; --sand:#c9b088; --leaf:#7cc492; --rust:#e08a6f;
    --chip-pass-bg:#1f3b28; --chip-pass-ink:#bfe6c9; --chip-fail-bg:#4a221a; --chip-fail-ink:#f2c4b4;
    --chip-inc-bg:#3a3225; --chip-inc-ink:#e3d6bf; --thead:#232b30; --zebra:#1f262a; --code:#262e33;
  }
}
:root[data-theme="dark"]{
  --ground:#14181b; --surface:#1b2125; --ink:#e4e6e2; --muted:#9aa4ab; --rule:#2e363c;
  --accent:#7fb7d1; --accent-ink:#0f1a20; --sand:#c9b088; --leaf:#7cc492; --rust:#e08a6f;
  --chip-pass-bg:#1f3b28; --chip-pass-ink:#bfe6c9; --chip-fail-bg:#4a221a; --chip-fail-ink:#f2c4b4;
  --chip-inc-bg:#3a3225; --chip-inc-ink:#e3d6bf; --thead:#232b30; --zebra:#1f262a; --code:#262e33;
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);font-family:"IBM Plex Sans",system-ui,-apple-system,"Segoe UI",sans-serif;font-size:16px;line-height:1.55}
.mast{border-bottom:1px solid var(--rule);background:var(--surface)}
.mast-in{max-width:1180px;margin:0 auto;padding:36px 28px 22px;display:grid;gap:10px}
.eyebrow{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
h1{font-family:"IBM Plex Serif",Georgia,serif;font-weight:500;font-size:clamp(30px,4vw,44px);line-height:1.12;margin:0;text-wrap:balance;max-width:20ch}
.lede{color:var(--muted);max-width:68ch;margin:0}
nav.toc{position:sticky;top:0;z-index:5;background:var(--surface);border-bottom:1px solid var(--rule)}
nav.toc ul{list-style:none;margin:0 auto;padding:0 20px;max-width:1180px;display:flex;gap:2px;overflow-x:auto;scrollbar-width:thin}
nav.toc a{display:block;padding:11px 10px;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12.5px;color:var(--muted);text-decoration:none;white-space:nowrap;border-bottom:2px solid transparent}
nav.toc a:hover,nav.toc a:focus-visible{color:var(--ink);border-bottom-color:var(--accent);outline:none}
main{max-width:1180px;margin:0 auto;padding:28px 28px 80px}
main > p, main > ul, main > blockquote, main > h2, main > h3, .math{max-width:72ch}
h2{font-family:"IBM Plex Serif",Georgia,serif;font-weight:500;font-size:28px;line-height:1.2;margin:56px 0 12px;padding-top:18px;border-top:1px solid var(--rule);text-wrap:balance}
h3{font-family:"IBM Plex Serif",Georgia,serif;font-weight:500;font-size:20px;margin:34px 0 8px;text-wrap:balance}
p{margin:0 0 14px}
p.source{color:var(--muted);font-size:14px;margin:6px 0 22px}
ul{padding-left:22px;margin:0 0 14px}
li{margin:0 0 6px}
blockquote{margin:0 0 16px;padding:10px 16px;border-left:3px solid var(--sand);background:var(--surface);color:var(--muted);font-size:15px}
blockquote p{margin:0 0 8px}
code{font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:.88em;background:var(--code);padding:1px 5px;border-radius:3px}
strong{font-weight:600}
.math{margin:10px 0 18px;overflow-x:auto}
.fig{background:var(--surface);border:1px solid var(--rule);padding:14px;margin:8px 0 22px;overflow-x:auto}
.tscroll{overflow-x:auto;margin:10px 0 8px;border:1px solid var(--rule);background:var(--surface)}
table{border-collapse:collapse;width:max-content;min-width:100%;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12.5px;font-variant-numeric:tabular-nums;line-height:1.4}
thead th{position:sticky;top:0;background:var(--thead);text-align:left;font-weight:600;padding:8px 10px;border-bottom:1px solid var(--rule);white-space:nowrap}
tbody td{padding:6px 10px;border-bottom:1px solid var(--rule);vertical-align:top;white-space:nowrap}
tbody tr:nth-child(even){background:var(--zebra)}
td sub{font-size:10px;color:var(--muted)}
.prose table{width:100%}
.prose td,.prose th{white-space:normal;min-width:12ch;max-width:44ch}
.prose td:first-child{white-space:nowrap}
.chip{display:inline-block;padding:0 7px;border-radius:3px;font-size:11px;font-weight:600;letter-spacing:.04em}
.chip.pass{background:var(--chip-pass-bg);color:var(--chip-pass-ink)}
.chip.fail{background:var(--chip-fail-bg);color:var(--chip-fail-ink)}
.chip.inc{background:var(--chip-inc-bg);color:var(--chip-inc-ink)}
.legend{display:flex;gap:18px;flex-wrap:wrap;font-family:"IBM Plex Mono",ui-monospace,monospace;font-size:12px;color:var(--muted);margin:-10px 0 18px}
.legend span::before{content:"";display:inline-block;width:10px;height:10px;margin-right:6px;vertical-align:-1px;border-radius:2px}
.legend .k1::before{background:#2b5d8c}.legend .k2::before{background:#7a6a4a}.legend .k3::before{background:#2f7a48}
a{color:var(--accent)}
@media (max-width:640px){.mast-in,main{padding-left:16px;padding-right:16px} h2{font-size:24px}}
@media (prefers-reduced-motion: no-preference){html{scroll-behavior:smooth}}
"""


def main():
    body, toc = render(SRC.read_text())
    nav = "".join(f'<li><a href="#{sid}">{html.escape(t)}</a></li>' for sid, t in toc)
    # the mermaid legend sentence in the source becomes a proper legend row
    body = body.replace(
        "<p>Blue = trained offline, sand = frozen and cached, green = solved online.</p>",
        '<div class="legend"><span class="k1">trained offline</span><span class="k2">frozen, cached once</span><span class="k3">solved online</span></div>')
    page = f"""<title>Separable Decoder Digest</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:wght@400;500&family=IBM+Plex+Sans:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>{CSS}</style>
<script>
window.MathJax = {{ tex: {{ inlineMath: [['$','$']], displayMath: [['$$','$$']] }}, svg: {{ fontCache: 'global' }},
  options: {{ skipHtmlTags: ['script','noscript','style','textarea','pre','code'] }} }};
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-svg.js"></script>
<header class="mast"><div class="mast-in">
  <div class="eyebrow">Tunable NM-ROM · results as of 2026-09-03 · every table generated from run JSONs</div>
  {body.split("</h1>")[0].replace("<h1>", "<h1>")}</h1>
  <p class="lede">The separable EQ-decoder reduced-order model on every PDE it has been run on. Final numbers unless a table says otherwise; two superseded tables are kept and labelled.</p>
</div></header>
<nav class="toc"><ul>{nav}</ul></nav>
<main>
{body.split("</h1>", 1)[1]}
</main>
"""
    OUT.write_text(page)
    print(f"wrote {OUT.name} ({len(page) // 1024} KB)")


if __name__ == "__main__":
    main()
