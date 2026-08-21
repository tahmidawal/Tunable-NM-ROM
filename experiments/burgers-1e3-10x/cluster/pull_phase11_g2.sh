#!/usr/bin/env bash
# pull_phase11_g2.sh p11_g2_s11_r1 JOB COMMIT MANIFEST_SHA256
set -euo pipefail

verify_pulled_bundle() {
  local root="$1" job="$2" commit="$3" expected_manifest="$4"
  [[ -d "$root/out" && -d "$root/logs" && "$job" =~ ^[0-9]+$ \
     && "$commit" =~ ^[0-9a-f]{40}$ && "$expected_manifest" =~ ^[0-9a-f]{64}$ ]] || return 20
  [[ "$(sha256sum "$root/MANIFEST.sha256" | cut -d' ' -f1)" == "$expected_manifest" ]] || return 21
  (cd "$root" && sha256sum -c REMOTE.sha256) || return 22
  local expected actual
  expected="$(mktemp)"; actual="$(mktemp)"
  {
    printf '%s\n' "logs/$job.err" "logs/$job.out" out/AUDIT.json out/PROGRESS.json \
      out/checkpoint.pkl out/phase11_g2.json out/phase11_g2.npz
    [[ ! -f "$root/out/work_checkpoint.pkl" ]] || printf '%s\n' out/work_checkpoint.pkl
  } | LC_ALL=C sort > "$expected"
  (cd "$root" && find out logs -type f -print | LC_ALL=C sort) > "$actual"
  cmp -s "$expected" "$actual" || { rm -f "$expected" "$actual"; return 23; }
  awk '{print $2}' "$root/REMOTE.sha256" | LC_ALL=C sort > "$actual.remote"
  cmp -s "$expected" "$actual.remote" || { rm -f "$expected" "$actual" "$actual.remote"; return 24; }
  rm -f "$expected" "$actual" "$actual.remote"
  awk -F'|' -v job="$job" '$1==job && $3=="COMPLETED" && $4=="0:0" {ok=1} END {exit !ok}' "$root/SACCT.txt" || return 25
  local audit="$root/out/AUDIT.json"
  jq -e --arg commit "$commit" --arg job "$job" --arg manifest "$expected_manifest" \
    '.status=="pass" and .negative_aware==true and .expected_commit==$commit
     and .expected_job==$job and .manifest_sha256==$manifest
     and .decision.retracted_capacity_used==false
     and .decision.third_architecture_licensed==false' "$audit" >/dev/null || return 26
  [[ "$(jq -r .source_json_sha256 "$audit")" == "$(sha256sum "$root/out/phase11_g2.json" | cut -d' ' -f1)" ]] || return 27
  [[ "$(jq -r .source_npz_sha256 "$audit")" == "$(sha256sum "$root/out/phase11_g2.npz" | cut -d' ' -f1)" ]] || return 28
  [[ "$(jq -r .checkpoint_sha256 "$audit")" == "$(sha256sum "$root/out/checkpoint.pkl" | cut -d' ' -f1)" ]] || return 29
}

main() {
  [[ $# -eq 4 ]] || exit 2
  local cell="$1" job="$2" commit="$3" expected_manifest="$4"
  [[ "$cell" == p11_g2_s11_r1 && "$job" =~ ^[0-9]+$ && "$commit" =~ ^[0-9a-f]{40}$ \
     && "$expected_manifest" =~ ^[0-9a-f]{64}$ ]] || exit 2
  local here exp remote local_dir transfer state
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; exp="$(dirname "$here")"
  remote="/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/$cell"; local_dir="$exp/runs/$cell"
  [[ ! -e "$local_dir" ]] || exit 3
  state="$(ssh tufts-login "sacct -j '$job' -X -n -P -o JobIDRaw,State,ExitCode" | awk -F'|' -v job="$job" '$1==job {print $2"|"$3; exit}')"
  [[ "$state" == "COMPLETED|0:0" ]] || exit 4
  ssh tufts-login "cd '$remote' && test \"\$(sha256sum MANIFEST.sha256 | cut -d' ' -f1)\" = '$expected_manifest' \
    && sacct -j '$job' -X -o JobIDRaw,JobName,State,ExitCode,Elapsed,AllocTRES,NodeList -P > SACCT.txt \
    && find out logs -type f -exec sha256sum {} \; | sort > REMOTE.sha256"
  transfer="$(mktemp -d "$exp/runs/.${cell}.pull.XXXXXX")"
  trap '[[ -z "${transfer:-}" ]] || rm -rf -- "$transfer"' EXIT
  scp -qr "tufts-login:$remote/out" "tufts-login:$remote/logs" \
    "tufts-login:$remote/MANIFEST.sha256" "tufts-login:$remote/REMOTE.sha256" \
    "tufts-login:$remote/SACCT.txt" "$transfer/"
  verify_pulled_bundle "$transfer" "$job" "$commit" "$expected_manifest"
  (cd "$transfer" && find out logs MANIFEST.sha256 REMOTE.sha256 SACCT.txt -type f \
    ! -name LOCAL.sha256 -exec sha256sum {} \; | sort > LOCAL.sha256 && sha256sum -c LOCAL.sha256)
  mv "$transfer" "$local_dir"; transfer=""
  ssh tufts-login "test '$remote' = '/cluster/tufts/paralab/tawal01/burgers_nmrom_1e3_10x/$cell' \
    && rm -rf -- '$remote' && test ! -e '$remote'"
  echo "pulled=$local_dir commit=$commit job=$job manifest=$expected_manifest remote_removed=true"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
