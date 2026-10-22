#!/bin/bash
# usage: make_body.sh <W> <lev> <stages comma list> [extra args] > bodies/<name>.sh
#        make_body.sh test <W> <lev> <archived cell name> > bodies/test_<cell>.sh
set -eu
if [ "$1" = test ]; then sed -e "s#@W@#$2#g; s#@LEV@#$3#g; s#@CELL@#$4#g" "$(dirname "$0")/bodies/test.sh.in"; exit 0; fi
case ",$3," in *,time,*) case ",$3," in *,controls,*) ;; *) echo "a stage list with time must include controls" >&2; exit 2;; esac;; esac
ST=$(echo "$3" | tr ',' ' ')
sed -e "s#@W@#$1#; s#@LEV@#$2#; s#@STAGES@#$ST#; s#@EXTRA@#${4:-}#" "$(dirname "$0")/bodies/cell.sh.in"
