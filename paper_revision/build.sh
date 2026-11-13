#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
latexmk -pdf -halt-on-error -interaction=nonstopmode main.tex
