#!/usr/bin/env bash
# Rebuild the JCP campaign notebook: include every entry in notebook/entries/ in filename order,
# then render notebook/jcp-notebook.pdf. Run after every notebook update and commit both files.
set -euo pipefail
cd "$(dirname "$0")"
ls entries/*.tex | sort | sed 's|^\(.*\)\.tex$|\\input{\1}|' > entries.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error jcp-notebook.tex > build.log 2>&1 || { tail -30 build.log; exit 1; }
if grep -q "There were undefined references" jcp-notebook.log; then echo "warning: undefined references"; fi
latexmk -c > /dev/null 2>&1
rm -f build.log entries.tex.bak
echo "built $(pwd)/jcp-notebook.pdf ($(pdfinfo jcp-notebook.pdf | awk '/Pages/{print $2}') pages)"
