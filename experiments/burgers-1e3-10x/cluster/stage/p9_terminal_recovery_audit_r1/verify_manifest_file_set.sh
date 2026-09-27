#!/usr/bin/env bash
# verify_manifest_file_set.sh ROOT MANIFEST [EXACT_EXTRA_PATH ...]
set -euo pipefail
root="$1"; manifest_path="$2"; shift 2
[[ -d "$root" && -f "$manifest_path" ]] || exit 2
expected_file="$(mktemp)"; actual_file="$(mktemp)"
trap 'rm -f "$expected_file" "$actual_file"' EXIT
while read -r digest path extra; do
  [[ "$digest" =~ ^[0-9a-f]{64}$ && "$path" == ./* && -z "${extra:-}" ]] || {
    echo "malformed manifest row" >&2; exit 3;
  }
  printf '%s\n' "$path"
done < "$manifest_path" > "$expected_file"
for path in "$@"; do
  [[ "$path" == ./* && "$path" != ./MANIFEST.sha256 ]] || exit 3
  printf '%s\n' "$path" >> "$expected_file"
done
LC_ALL=C sort -o "$expected_file" "$expected_file"
[[ ! -s <(uniq -d "$expected_file") ]] || { echo "duplicate expected path" >&2; exit 3; }
(cd "$root" && LC_ALL=C find . -type f ! -path './MANIFEST.sha256' -print | LC_ALL=C sort) \
  > "$actual_file"
if ! cmp -s "$expected_file" "$actual_file"; then
  echo "manifest file-set mismatch" >&2
  diff -u "$expected_file" "$actual_file" >&2 || true
  exit 4
fi
echo "manifest_file_set=exact files=$(wc -l < "$actual_file")"
