#!/usr/bin/env bash
# Upload the tested wheels in <dist> to the request's pre-release, then regenerate its
# SHA256SUMS. A wheel already on the release is never replaced; every upload is read back.
# Needs: gh with GH_TOKEN (contents: write), GITHUB_REPOSITORY.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=request.sh
source "$HERE/request.sh"
load_request "$HERE/request.env"
DIST=${1:?usage: publish.sh <dir of tested wheels>}
repo=${GITHUB_REPOSITORY:?}
SUMMARY=${GITHUB_STEP_SUMMARY:-/dev/null}

release() { gh api "repos/$repo/releases/tags/$RELEASE_TAG"; }
# The sha256 GitHub serves for asset $1, from its digest or else by downloading it.
served_sha256() {
  local d
  jq -e --arg n "$1" '.assets[] | select(.name == $n)' <<<"$rel" >/dev/null ||
    die "read-back: $1 is not on release $RELEASE_TAG"
  d=$(jq -r --arg n "$1" '.assets[] | select(.name == $n) | .digest // empty' <<<"$rel")
  if [[ $d == sha256:* ]]; then
    echo "${d#sha256:}"
  else
    gh release download "$RELEASE_TAG" -R "$repo" -p "$1" -O - | sha256sum | cut -d' ' -f1
  fi
}

rel=$(release)
[ "$(jq -r .prerelease <<<"$rel")" = true ] || die "release $RELEASE_TAG is not a pre-release"
expected=""
for plat in $PLATFORMS; do
  for py in $PYTHONS; do expected+="$(wheel_name "$py" "$plat")"$'\n'; done
done

uploaded=()
shopt -s nullglob
for f in "$DIST"/*.whl; do
  n=$(basename "$f")
  grep -qxF "$n" <<<"$expected" || die "$n is not a wheel this request builds"
  if jq -e --arg n "$n" '.assets[] | select(.name == $n)' <<<"$rel" >/dev/null; then
    echo "already on the release, left as it is: $n"
    continue
  fi
  gh release upload "$RELEASE_TAG" "$f" -R "$repo"
  uploaded+=("$f")
done
[ "${#uploaded[@]}" -gt 0 ] || echo "no new wheel to upload"

rel=$(release)
for f in ${uploaded[@]+"${uploaded[@]}"}; do
  n=$(basename "$f")
  want=$(sha256sum "$f" | cut -d' ' -f1)
  got=$(served_sha256 "$n")
  [ "$got" = "$want" ] || die "read-back of $n: GitHub serves $got, the tested wheel is $want"
  echo "read back: $n $got"
done

# SHA256SUMS lists every wheel on the release. Lines it already had must still hold.
sums=$(mktemp)
while read -r n; do
  echo "$(served_sha256 "$n")  $n"
done < <(jq -r '.assets[].name | select(endswith(".whl"))' <<<"$rel" | LC_ALL=C sort) >"$sums"
if jq -e '.assets[] | select(.name == "SHA256SUMS")' <<<"$rel" >/dev/null; then
  old=$(gh release download "$RELEASE_TAG" -R "$repo" -p SHA256SUMS -O -)
  while read -r line; do
    [ -z "$line" ] || grep -qxF "$line" "$sums" || die "SHA256SUMS line no longer matches the release: $line"
  done <<<"$old"
fi
if [ "${old:-}" != "$(cat "$sums")" ]; then
  d=$(mktemp -d) && cp "$sums" "$d/SHA256SUMS"
  gh release upload "$RELEASE_TAG" "$d/SHA256SUMS" -R "$repo" --clobber
  rel=$(release)
  [ "$(served_sha256 SHA256SUMS)" = "$(sha256sum "$sums" | cut -d' ' -f1)" ] || die "read-back of SHA256SUMS differs"
  echo "SHA256SUMS now lists $(wc -l <"$sums") wheels"
fi

{
  echo "### $RELEASE_TAG"
  echo
  echo "| wheel | sha256 | |"
  echo "|---|---|---|"
  while read -r sha n; do
    new=""
    for f in ${uploaded[@]+"${uploaded[@]}"}; do [ "$(basename "$f")" = "$n" ] && new="new"; done
    echo "| \`$n\` | \`$sha\` | $new |"
  done <"$sums"
} >>"$SUMMARY"
cat "$sums"

# Every requested wheel should now be on the release.
missing=$(grep -vxF -f <(cut -d' ' -f3 "$sums") <<<"${expected%$'\n'}" || true)
if [ -n "$missing" ]; then
  echo "::error::Still missing from $RELEASE_TAG (see the failed or unfinished platform jobs; re-running the workflow builds only these, from the saved caches): $(tr '\n' ' ' <<<"$missing")"
  exit 1
fi
echo "All requested wheels are on $RELEASE_TAG."
