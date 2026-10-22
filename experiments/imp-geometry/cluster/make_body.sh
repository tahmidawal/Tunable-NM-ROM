#!/bin/bash
# usage: make_body.sh <dom> <fam> <lev> <stages comma list> [bank steppers] [extra args] > body.sh
set -eu
if [ "$1" = test ]; then C="$2_$3_$4"; sed -e "s#@DOM@#$2#; s#@FAM@#$3#; s#@LEV@#$4#; s#@CELL@#$C#g" "$(dirname "$0")/bodies/test.sh.in"; exit 0; fi
ST=$(echo "$4" | tr ',' ' ')
sed -e "s#@DOM@#$1#; s#@FAM@#$2#; s#@LEV@#$3#; s#@STAGES@#$ST#; s#@BST@#${5:-lspg_cnr}#; s#@EXTRA@#${6:-}#" "$(dirname "$0")/bodies/cell.sh.in"
# test bodies: make_body.sh test <dom> <fam> <lev>  (restores the archived cell, then runs the test stage)
