#!/bin/bash
# jcp-references: submit the listed staged attempts in order as lane slots free up (cap 4 enforced by submit.sh).
# Prints one line per submission. Skips attempts whose remote directory already exists.
cd "$(dirname "$0")"
for A in "$@"; do
  while true; do
    if ssh tufts-login "test -e /cluster/tufts/paralab/tawal01/jcp-ref/$A"; then echo "skip $A (exists)"; break; fi
    OUT=$(bash submit.sh "$A" 2>&1); RC=$?
    if [ $RC -eq 0 ]; then echo "SUBMITTED $A $(echo "$OUT" | grep -o 'Submitted batch job [0-9]*')"; break; fi
    if [ $RC -eq 5 ]; then sleep 120; continue; fi
    echo "ERROR $A rc=$RC: $(echo "$OUT" | tail -2 | tr '\n' ' ')"; break
  done
done
echo FILL-DONE
