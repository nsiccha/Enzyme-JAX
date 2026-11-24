#!/usr/bin/env bash
# The runner side of one platform's build attempt. Linux builds run build.sh inside the
# manylinux_2_28 container; macOS runs it on the runner itself.
#
#   host.sh setup         tools: bazelisk, the container image (Linux), GNU timeout (macOS)
#   host.sh build         build $PYTHONS into $ROOT/out/wheelhouse; sets output status=
#                         done|incomplete, exits 1 on a failed build
#   host.sh smoke         test every built wheel on this runner (smoke.py); the passing
#                         ones move to $ROOT/out/passed and set output artifact=
#   host.sh prune-cache   delete this platform's bazel caches other than $CACHE_KEY
#
# Environment: PLATFORM, PYTHONS, ALL_PYTHONS, VERSION, TESTS, ROOT (the job's working
# directory, holding src/ = the candidate checkout), JOB_START (epoch seconds), and
# optionally JOB_MINUTES (355), RESERVE_MINUTES (50), CPUS, DOCKER_CPUS.
set -euo pipefail
WF=$(cd "$(dirname "$0")" && pwd)
: "${PLATFORM:?}" "${ROOT:?}"
SRC=$ROOT/src WORK=$ROOT/work OUT=$ROOT/out TOOLS=$ROOT/tools
GITHUB_OUTPUT=${GITHUB_OUTPUT:-/dev/null}

# Pinned inputs. The manylinux image is the one that built the published cp314
# manylinux_2_28_x86_64 wheels (tag 2026.11.21-1); bazelisk runs the .bazelversion bazel.
BAZELISK_VERSION=v1.29.0
case "$PLATFORM" in
  manylinux_2_28_x86_64)
    os=linux arch=x86_64 bazelisk=linux-amd64
    bazelisk_sha256=5a408715e932c0250d28bd84555f12edbf70117de42f9181691c736eacc4a992
    manylinux=quay.io/pypa/manylinux_2_28_x86_64@sha256:55a36963bc0504cdf004011001c1072306402f0b8b83be5a5eba7068f93eec53
    ;;
  manylinux_2_28_aarch64)
    os=linux arch=aarch64 bazelisk=linux-arm64
    bazelisk_sha256=e20e8b0f4f240091b7a55bf17b9398bd4f40ee70ae0208dff95dd4c445fb4010
    manylinux=quay.io/pypa/manylinux_2_28_aarch64@sha256:126d9eea9d6b1717f3e17b2b3ac51da04ef159a5f935dac36fe92d621dc8f246
    ;;
  macosx_14_0_arm64)
    os=macos arch=arm64 bazelisk=darwin-arm64
    bazelisk_sha256=cee851f726789227d5561004e9904a52be45c3efb56f8b38b6993d6adbaa0409
    manylinux=
    ;;
  *) echo "unknown platform $PLATFORM"; exit 1 ;;
esac
IMAGE=flatppl-wheels-manylinux_2_28_$arch

sha256() { if command -v sha256sum >/dev/null; then sha256sum "$@"; else shasum -a 256 "$@"; fi; }

setup() {
  mkdir -p "$TOOLS" "$WORK" "$OUT"
  if [ -n "${GITHUB_ACTIONS:-}" ] && [ "$os" = linux ]; then
    # A cold build needs about 25 GB; make room on the runner's disk.
    df -h / "$ROOT"
    sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc /usr/local/.ghcup \
      /opt/hostedtoolcache/CodeQL /usr/local/share/boost /usr/local/share/powershell
    df -h / "$ROOT"
  fi
  local b="$TOOLS/bazelisk"
  if ! [ -x "$b" ] || [ "$(sha256 "$b" | cut -d' ' -f1)" != "$bazelisk_sha256" ]; then
    curl -fsSL --retry 3 -o "$b" \
      "https://github.com/bazelbuild/bazelisk/releases/download/$BAZELISK_VERSION/bazelisk-$bazelisk"
    echo "$bazelisk_sha256  $b" | sha256 -c -
    chmod +x "$b"
  fi
  if [ "$os" = linux ]; then
    docker build -q -t "$IMAGE" - <<EOF
FROM $manylinux
RUN dnf -y install clang lld && dnf clean all
EOF
  elif ! command -v timeout >/dev/null && ! command -v gtimeout >/dev/null; then
    HOMEBREW_NO_AUTO_UPDATE=1 brew install coreutils
  fi
}

build() {
  : "${PYTHONS:?}" "${ALL_PYTHONS:?}" "${VERSION:?}" "${JOB_START:?}"
  local deadline=$((JOB_START + (${JOB_MINUTES:-355} - ${RESERVE_MINUTES:-50}) * 60)) rc=0
  echo "Building CPython $PYTHONS for $PLATFORM until $(date -u -d "@$deadline" 2>/dev/null || date -u -r "$deadline")"
  mkdir -p "$WORK" "$OUT"
  if [ "$os" = linux ]; then
    docker run --rm --name "flatppl-wheels-$arch" ${DOCKER_CPUS:+--cpus="$DOCKER_CPUS"} \
      -v "$SRC:/src" -v "$WORK:/work" -v "$OUT:/out" -v "$WF:/wf:ro" \
      -v "$TOOLS/bazelisk:/usr/local/bin/bazelisk:ro" \
      -e PLATFORM -e PYTHONS -e ALL_PYTHONS -e VERSION -e "DEADLINE=$deadline" ${CPUS:+-e CPUS} \
      -e SRC=/src -e WORK=/work -e OUT=/out -e BAZEL=/usr/local/bin/bazelisk \
      "$IMAGE" bash /wf/build.sh || rc=$?
    # A rootful docker leaves root-owned files; the cache and artifact steps run as the runner user.
    if [ -n "${GITHUB_ACTIONS:-}" ]; then sudo chown -R "$(id -u):$(id -g)" "$SRC" "$WORK" "$OUT"; fi
  else
    SRC=$SRC WORK=$WORK OUT=$OUT BAZEL=$TOOLS/bazelisk DEADLINE=$deadline bash "$WF/build.sh" || rc=$?
  fi
  case $rc in
    0) echo "status=done" >>"$GITHUB_OUTPUT" ;;
    75) echo "status=incomplete" >>"$GITHUB_OUTPUT" ;;
    *) echo "status=failed" >>"$GITHUB_OUTPUT"; return 1 ;;
  esac
}

smoke() {
  : "${VERSION:?}"
  local whl name py tag passed=() failed=()
  mkdir -p "$OUT/passed"
  shopt -s nullglob
  for whl in "$OUT"/wheelhouse/*.whl; do
    name=$(basename "$whl")
    tag=$(cut -d- -f3 <<<"$name")
    py="${tag:2:1}.${tag:3}"
    echo "::group::Test $name"
    local cmd=(--wheel "$whl" --version "$VERSION")
    # shellcheck disable=SC2206 # TESTS is a space-separated list of plain names
    [ -z "${TESTS:-}" ] || cmd+=(--tests-dir "$SRC/test" --tests $TESTS)
    local rc=0
    if [ "$os" = linux ]; then
      docker run --rm -v "$OUT:$OUT:ro" -v "$WF:/wf:ro" -v "$SRC/test:$SRC/test:ro" \
        "$IMAGE" "/opt/python/$tag-$tag/bin/python" /wf/smoke.py "${cmd[@]}" || rc=$?
    else
      "python$py" "$WF/smoke.py" "${cmd[@]}" || rc=$?
    fi
    echo "::endgroup::"
    if [ "$rc" = 0 ]; then
      mv "$whl" "$OUT/passed/"
      passed+=("$tag")
      echo "passed: $name"
    else
      failed+=("$tag")
      echo "::error::$name failed its tests on $PLATFORM; it will not be uploaded"
    fi
  done
  if [ "${#passed[@]}" -gt 0 ]; then
    (IFS=-; echo "artifact=wheels-$PLATFORM-${passed[*]}") >>"$GITHUB_OUTPUT"
  fi
  [ "${#failed[@]}" = 0 ]
}

prune_cache() {
  : "${CACHE_PREFIX:?}" "${CACHE_KEY:?}" "${GITHUB_REPOSITORY:?}"
  local id
  gh cache list -R "$GITHUB_REPOSITORY" --key "$CACHE_PREFIX" --limit 100 --json id,key \
    --jq ".[] | select(.key != \"$CACHE_KEY\") | .id" |
    while read -r id; do gh cache delete -R "$GITHUB_REPOSITORY" "$id"; done
}

case "${1:-}" in
  setup) setup ;;
  build) build ;;
  smoke) smoke ;;
  prune-cache) prune_cache ;;
  *) echo "usage: $0 setup|build|smoke|prune-cache"; exit 64 ;;
esac
