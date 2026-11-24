#!/usr/bin/env bash
# Check the request against the live release and list, per platform, the Python versions
# whose wheel the release still lacks. Writes job outputs to $GITHUB_OUTPUT.
# Needs: gh with GH_TOKEN, GITHUB_REPOSITORY.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=request.sh
source "$HERE/request.sh"
load_request "$HERE/request.env"
OUT=${GITHUB_OUTPUT:-/dev/stdout}
repo=${GITHUB_REPOSITORY:?}

release=$(gh api "repos/$repo/releases/tags/$RELEASE_TAG") ||
  die "release $RELEASE_TAG does not exist on $repo; publish the candidate's pre-release first"
[ "$(jq -r .prerelease <<<"$release")" = true ] || die "release $RELEASE_TAG is not a pre-release; refusing to add wheels to it"
[ "$(jq -r .draft <<<"$release")" = false ] || die "release $RELEASE_TAG is a draft"
tag_commit=$(gh api "repos/$repo/commits/$RELEASE_TAG" --jq .sha)
[ "$tag_commit" = "$SOURCE_SHA" ] || die "tag $RELEASE_TAG is at $tag_commit, but the request names $SOURCE_SHA"
assets=$(jq -r '.assets[].name' <<<"$release")

{
  echo "source_sha=$SOURCE_SHA"
  echo "version=$VERSION"
  echo "release_tag=$RELEASE_TAG"
  echo "all_pythons=$PYTHONS"
  echo "tests=${TESTS:-}"
} >>"$OUT"

any=false
for plat in $KNOWN_PLATFORMS; do
  missing=""
  if [[ " $PLATFORMS " == *" $plat "* ]]; then
    for py in $PYTHONS; do
      name=$(wheel_name "$py" "$plat")
      if grep -qxF "$name" <<<"$assets"; then
        echo "on the release already: $name"
      else
        echo "to build: $name"
        missing="${missing:+$missing }$py"
      fi
    done
  fi
  [ -z "$missing" ] || any=true
  echo "$plat=$missing" >>"$OUT"
done
echo "any=$any" >>"$OUT"
