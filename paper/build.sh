#!/usr/bin/env bash
# Regenerate every table and prose number from the lanes' machine-readable
# outputs, then compile the manuscript with the ICLR 2027 style.
#   ./build.sh            regenerate + compile
#   ./build.sh --figures  also regenerate the family figure (matplotlib, CPU only)
set -euo pipefail
cd "$(dirname "$0")"
PY=/home/tahmid/Dev/.venv/bin/python
"$PY" gen_tables.py
"$PY" gen_campaign_supplement.py
"$PY" gen_rewrite_tables.py
"$PY" gen_main_experiments.py
"$PY" gen_headline.py
"$PY" figures/gen_fig_speedup_resolution.py
if [[ "${1:-}" == "--figures" ]]; then
  "$PY" figures/gen_fig_tunability_family.py
fi
"$PY" gen_paper_md.py >/dev/null
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/dev/null
grep -c "pending:" main.log >/dev/null && echo "placeholders in the PDF: $(grep -o 'pending: [^}]*' tables/*.tex | wc -l) (see tables/PENDING.md)" || true
grep -i "undefined" main.log | grep -v "^\\s*$" | head -20 || true
pdfinfo main.pdf 2>/dev/null | grep Pages || true
