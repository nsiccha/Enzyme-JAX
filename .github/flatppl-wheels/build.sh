#!/usr/bin/env bash
# Build the enzyme_ad wheels of one platform, one CPython version after another, sharing
# one bazel output base and disk cache. For Linux it runs inside the manylinux_2_28
# container (clang/lld, static libstdc++, then auditwheel repair), the recipe of the
# published cp314 manylinux_2_28_x86_64 wheels; for macOS it runs on the arm64 runner with
# the flags of upstream's tag.yml plus the deployment target that the platform tag names.
#
# Environment:
#   PLATFORM     manylinux_2_28_x86_64 | manylinux_2_28_aarch64 | macosx_14_0_arm64
#   PYTHONS      versions to build now, e.g. "3.12 3.15"
#   ALL_PYTHONS  every version of the request (the source edits depend only on these)
#   VERSION      the version the source's BUILD must set
#   SRC WORK OUT the source checkout (edited in place), bazel's state, the results
#   BAZEL        a bazelisk binary
#   DEADLINE     epoch seconds; bazel is interrupted then, and its progress stays in the
#                disk cache for the next attempt
# Finished wheels land in $OUT/wheelhouse, one line per built version in $OUT/built.
# Exit status: 0 all built, 75 the deadline left some for a later attempt, 1 a failure.
set -euo pipefail
: "${PLATFORM:?}" "${PYTHONS:?}" "${ALL_PYTHONS:?}" "${VERSION:?}" "${SRC:?}" "${WORK:?}" "${OUT:?}" "${BAZEL:?}" "${DEADLINE:?}"
# A version is started only with at least this many seconds left (the first one always).
MIN_SECONDS=${MIN_SECONDS:-1800}
TIMEOUT=$(command -v timeout || command -v gtimeout) || { echo "need GNU timeout"; exit 1; }

mkdir -p "$WORK/home" "$OUT/wheelhouse"
export HOME="$WORK/home" BAZELISK_HOME="$WORK/bazelisk-home" XDG_CACHE_HOME="$WORK/xdg-cache"
BZ=("$BAZEL" --output_user_root="$WORK/bazel-user" --max_idle_secs=600)
CPUS=${CPUS:-$(getconf _NPROCESSORS_ONLN)}
flags=(--repository_cache="$WORK/repo-cache" --disk_cache="$WORK/disk-cache"
  --check_visibility=false "--jobs=$CPUS" --color=no --curses=no --show_progress_rate_limit=60)

cd "$SRC"
grep -qF "version = \"$VERSION\"," BUILD || { echo "BUILD at $(git rev-parse HEAD) does not set version $VERSION"; exit 1; }

# --- Source edits: the same for every attempt, so the disk cache stays valid across them.
# CPU-only jax, as upstream's CPU runners do.
sed -i.bak "s/jax\[cuda12\]/jax[cpu]/g" builddeps/requirements.in
sed -i.bak "/pypi_jax_cuda12_plugin/d" test/BUILD
# LLVM backends the platform cannot use, as upstream's tag.yml strips them.
strip_targets() {
  local t
  for t in "$@"; do sed -i.bak -e "s/\"$t\", //g" -e "s/, \"$t\"//g" workspace.bzl; done
}
case "$PLATFORM" in
  manylinux_2_28_x86_64) ;;
  manylinux_2_28_aarch64) strip_targets X86 ;;
  macosx_14_0_arm64) strip_targets AMDGPU X86 ;;
  *) echo "unknown platform $PLATFORM"; exit 1 ;;
esac
grep -n '^LLVM_TARGETS' workspace.bzl
# The WORKSPACE maps each hermetic Python version to its requirements lock; flatppl-main
# has locks up to 3.14. A missing version gets an entry and a seed copy of the newest lock,
# which requirements.update below then regenerates for that version.
locks=(builddeps/requirements_lock_3_1*.txt) # glob order is sorted
newest_lock=${locks[${#locks[@]} - 1]}
for py in $ALL_PYTHONS; do
  lock="requirements_lock_${py//./_}.txt"
  [ -f "builddeps/$lock" ] || cp "$newest_lock" "builddeps/$lock"
  if ! grep -qF "\"$py\": \"//builddeps:$lock\"" WORKSPACE; then
    sed -i.bak "/\"3\.14\": \"\/\/builddeps:requirements_lock_3_14\.txt\",/a\\
        \"$py\": \"//builddeps:$lock\",
" WORKSPACE
    grep -qF "\"$py\": \"//builddeps:$lock\"" WORKSPACE || { echo "could not add $py to WORKSPACE"; exit 1; }
  fi
done
sed -n '/python_init_repositories(/,/^)/p' WORKSPACE

# --- Platform toolchain.
case "$PLATFORM" in
  manylinux_2_28_*)
    export CC=clang CXX=clang++
    LINKLIBS="$(clang++ -print-file-name=libstdc++.a):-lm"
    flags+=(--repo_env=CC=clang --repo_env=CXX=clang++ "--repo_env=BAZEL_LINKLIBS=$LINKLIBS"
      --linkopt=-fuse-ld=lld --host_linkopt=-fuse-ld=lld --spawn_strategy=local)
    echo "toolchain: $(clang --version | head -1); libstdc++: $LINKLIBS; glibc: $(ldd --version | head -1)"
    ;;
  macosx_14_0_arm64)
    # The oldest macOS the platform tag promises, e.g. 14.0 for macosx_14_0_arm64. Bazel's
    # macOS toolchain passes --macos_minimum_os to every compile and link as
    # -mmacosx-version-min, and without it uses the SDK's version, which is newer than the
    # tag. The build's own tools (the exec configuration) keep that default.
    MACOS_MIN=$(sed -E 's/^macosx_([0-9]+)_([0-9]+)_.*$/\1.\2/' <<<"$PLATFORM")
    flags+=("--repo_env=CC=$(command -v clang)" --define=enable_triton=false "--macos_minimum_os=$MACOS_MIN"
      --cxxopt=-Wno-nullability-completeness --host_cxxopt=-Wno-nullability-completeness)
    echo "toolchain: $(clang --version | head -1); $(xcrun --show-sdk-path); deployment target $MACOS_MIN"
    ;;
esac

# macos_minos_ok <Mach-O file>: whether the file runs on MACOS_MIN, as the platform tag says.
macos_minos_ok() {
  local minos
  vtool -show-build "$1"
  minos=$(vtool -show-build "$1" | awk '$1 == "minos" {print $2; exit}')
  awk -v m="${minos:-99}" -v t="$MACOS_MIN" \
    'BEGIN { split(m, v, "."); split(t, w, "."); exit !(v[1] < w[1] || (v[1] == w[1] && v[2] + 0 <= w[2] + 0)) }' ||
    { echo "$(basename "$1") needs macOS ${minos:-?}, newer than the tag's $MACOS_MIN"; return 1; }
}

# run_until <command…>: run it, interrupting it at DEADLINE (status 124 then).
run_until() {
  local left=$((DEADLINE - $(date +%s)))
  [ "$left" -gt 60 ] || return 124
  "$TIMEOUT" -s INT -k 300 "$left" "$@"
}

# --- Check one finished wheel and put it in $OUT/wheelhouse under its final name.
finish_wheel() {
  local py=$1 raw=$2 tag="cp${1//./}" name
  name="enzyme_ad-${VERSION}-${tag}-${tag}-${PLATFORM}.whl"
  case "$PLATFORM" in
    manylinux_2_28_*)
      auditwheel show "$raw"
      auditwheel repair --plat "$PLATFORM" -w "$OUT/wheelhouse" "$raw"
      ;;
    macosx_14_0_arm64)
      # The library is the wheel's only native file.
      local so rc=0
      so=$(mktemp -d)
      unzip -q -o "$raw" 'enzyme_ad/jax/enzyme_call.so' -d "$so"
      otool -L "$so/enzyme_ad/jax/enzyme_call.so"
      macos_minos_ok "$so/enzyme_ad/jax/enzyme_call.so" || rc=1
      rm -rf "$so"
      [ "$rc" = 0 ] || return 1
      cp "$raw" "$OUT/wheelhouse/"
      ;;
  esac
  [ -f "$OUT/wheelhouse/$name" ] || { echo "expected $name, got: $(ls "$OUT/wheelhouse")"; return 1; }
  if command -v sha256sum >/dev/null; then sha256sum "$OUT/wheelhouse/$name"; else shasum -a 256 "$OUT/wheelhouse/$name"; fi
}

# --- On macOS, link a one-function library with the build's flags and check its deployment
# target first, so a target that does not reach the toolchain fails in minutes, not after
# hours of building. It uses the first version's flags, as the first build does.
if [ "$PLATFORM" = macosx_14_0_arm64 ]; then
  echo "::group::Deployment target of a probe library"
  probe_py=${PYTHONS%% *}
  export HERMETIC_PYTHON_VERSION=$probe_py
  probe_flags=("${flags[@]}" "--repo_env=HERMETIC_PYTHON_VERSION=$probe_py")
  mkdir -p flatppl_probe
  echo 'int flatppl_probe(void) { return 0; }' >flatppl_probe/probe.c
  printf 'cc_binary(\n    name = "probe.so",\n    srcs = ["probe.c"],\n    linkshared = True,\n)\n' >flatppl_probe/BUILD
  rc=0
  "${BZ[@]}" build "${probe_flags[@]}" //flatppl_probe:probe.so || rc=$?
  if [ "$rc" = 0 ]; then
    if ! probe=$("${BZ[@]}" cquery "${probe_flags[@]}" --output=files //flatppl_probe:probe.so 2>/dev/null |
      grep -m1 '/probe\.so$') || ! macos_minos_ok "$probe"; then
      rc=1
    fi
  fi
  echo "::endgroup::"
  if [ "$rc" != 0 ]; then
    echo "::error::the probe library did not build or does not target macOS $MACOS_MIN; no wheel is built"
    "${BZ[@]}" shutdown || true
    exit 1
  fi
fi

failed=() deferred=() first=1
for py in $PYTHONS; do
  left=$((DEADLINE - $(date +%s)))
  if [ "$first" = 0 ] && [ "$left" -lt "$MIN_SECONDS" ]; then
    deferred+=("$py")
    continue
  fi
  first=0
  echo "::group::CPython $py: $((left / 60)) min left"
  export HERMETIC_PYTHON_VERSION=$py
  pyflags=("${flags[@]}" "--repo_env=HERMETIC_PYTHON_VERSION=$py")
  rc=0
  run_until "${BZ[@]}" run "${pyflags[@]}" //builddeps:requirements.update || rc=$?
  if [ "$rc" = 0 ]; then
    t0=$(date +%s)
    run_until "${BZ[@]}" build --keep_going "${pyflags[@]}" :wheel || rc=$?
    echo "bazel build exit=$rc seconds=$(($(date +%s) - t0))"
    if [ "$rc" != 0 ] && [ "$rc" != 124 ]; then
      # A compiler killed for memory (16 GB for 4 jobs on Linux, 7 GB for 3 on macOS)
      # fails the build; retry once with half the jobs, from what the cache already has.
      rc=0
      run_until "${BZ[@]}" build --keep_going "${pyflags[@]}" "--jobs=$(((CPUS + 1) / 2))" :wheel || rc=$?
      echo "bazel build retry exit=$rc seconds=$(($(date +%s) - t0))"
    fi
  fi
  if [ "$rc" = 0 ]; then
    if raw=$("${BZ[@]}" cquery "${pyflags[@]}" --output=files :wheel 2>/dev/null | grep '\.whl$') &&
      finish_wheel "$py" "$raw"; then
      echo "$py" >>"$OUT/built"
    else
      rc=1
    fi
  fi
  echo "::endgroup::"
  if [ "$rc" = 124 ]; then
    echo "The deadline interrupted CPython $py; its progress is in the disk cache."
    deferred+=("$py")
    "${BZ[@]}" shutdown || true
    # Later versions would hit the same deadline.
    for later in $PYTHONS; do
      [[ " ${deferred[*]:-} ${failed[*]:-} " == *" $later "* ]] || grep -qxF "$later" "$OUT/built" 2>/dev/null || deferred+=("$later")
    done
    break
  elif [ "$rc" != 0 ]; then
    echo "::error::CPython $py failed on $PLATFORM (exit $rc)"
    failed+=("$py")
  fi
done
"${BZ[@]}" shutdown || true
df -h "$WORK" || true
unique=()
for py in ${deferred[@]+"${deferred[@]}"}; do
  [[ " ${unique[*]:-} " == *" $py "* ]] || unique+=("$py")
done
deferred=(${unique[@]+"${unique[@]}"})

echo "built: $(tr '\n' ' ' 2>/dev/null <"$OUT/built" || true)"
echo "failed: ${failed[*]:-none}"
echo "left for a later attempt: ${deferred[*]:-none}"
[ "${#failed[@]}" = 0 ] || exit 1
[ "${#deferred[@]}" = 0 ] || exit 75
