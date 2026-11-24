# Sourced by plan.sh and publish.sh: read and validate request.env, and name the wheels.
# shellcheck shell=bash

KNOWN_PLATFORMS="manylinux_2_28_x86_64 manylinux_2_28_aarch64 macosx_14_0_arm64"

die() {
  echo "::error::$*" >&2
  exit 1
}

load_request() {
  local file=$1
  [ -f "$file" ] || die "no request file at $file"
  # shellcheck disable=SC1090
  source "$file"
  [[ ${SOURCE_SHA:-} =~ ^[0-9a-f]{40}$ ]] || die "SOURCE_SHA must be a full 40-hex commit, got '${SOURCE_SHA:-}'"
  [[ ${VERSION:-} =~ ^[0-9A-Za-z.+]+$ ]] || die "VERSION has unexpected characters: '${VERSION:-}'"
  [[ ${RELEASE_TAG:-} =~ ^[0-9A-Za-z._-]+$ ]] || die "RELEASE_TAG has unexpected characters: '${RELEASE_TAG:-}'"
  [ -n "${PYTHONS:-}" ] || die "PYTHONS is empty"
  [ -n "${PLATFORMS:-}" ] || die "PLATFORMS is empty"
  local p
  for p in $PYTHONS; do
    [[ $p =~ ^3\.1[0-9]$ ]] || die "unsupported Python version '$p'"
  done
  for p in $PLATFORMS; do
    [[ " $KNOWN_PLATFORMS " == *" $p "* ]] || die "unknown platform '$p' (known: $KNOWN_PLATFORMS)"
  done
  for p in ${TESTS:-}; do
    [[ $p =~ ^[A-Za-z0-9_]+$ ]] || die "test name '$p' has unexpected characters"
  done
}

# wheel_name <python X.Y> <platform>: the file name a finished wheel must have.
wheel_name() {
  local tag="cp${1//./}"
  echo "enzyme_ad-${VERSION}-${tag}-${tag}-${2}.whl"
}
