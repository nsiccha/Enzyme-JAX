"""Test one enzyme_ad wheel in a fresh virtual environment before it is published.

Run it with the CPython the wheel is built for:

    python smoke.py --wheel W.whl --version V --tests-dir SRC/test --tests test_a test_b

It creates a virtual environment, installs the wheel (which pins jax and jaxlib) and
absl-py, and then, inside that environment:

* checks that the installed enzyme_ad is version V and comes from the wheel;
* differentiates two loops whose trip count depends on the data through
  ``enzyme_ad.jax.hlo_call`` in reverse mode, and compares the gradient with forward
  mode and with the closed form (a wheel without the adaptive-while fix refuses these
  with "WhileOp does not have known iteration count for cache removal");
* runs each named test file from the candidate's ``test/`` directory.

It exits non-zero if anything fails.
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

TEST_TIMEOUT_SECONDS = 1800


def run(cmd, **kwargs):
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, **kwargs).returncode


def outer(args):
    tag = "cp%d%d" % sys.version_info[:2]
    name = os.path.basename(args.wheel)
    if f"-{tag}-{tag}-" not in name:
        print(f"FAIL: {name} is not a {tag} wheel; run this with its own Python")
        return 1
    print(f"Python {sys.version.split()[0]} at {sys.executable}", flush=True)
    work = tempfile.mkdtemp(prefix="enzyme-ad-smoke-")
    venv = os.path.join(work, "venv")
    bindir = "Scripts" if os.name == "nt" else "bin"
    py = os.path.join(venv, bindir, "python")
    failures = []
    if run([sys.executable, "-m", "venv", venv]) != 0:
        return 1
    pip = [py, "-m", "pip", "--disable-pip-version-check", "--no-cache-dir"]
    if run(pip + ["install", args.wheel, "absl-py"]) != 0:
        print(f"FAIL: could not install {name}")
        return 1
    run(pip + ["list"])

    # The checks run from an empty directory, so only the installed wheel is importable.
    inner_dir = os.path.join(work, "inner")
    os.mkdir(inner_dir)
    inner = [py, "-I", os.path.abspath(__file__), "--inner", "--version", args.version]
    if run(inner, cwd=inner_dir) != 0:
        failures.append("reverse-mode checks")

    if args.tests:
        # The test files read sibling .mlir inputs; copy them away from the source tree.
        tests_dir = os.path.join(work, "tests")
        os.mkdir(tests_dir)
        for pattern in ("*.py", "*.mlir"):
            for f in glob.glob(os.path.join(args.tests_dir, pattern)):
                shutil.copy(f, tests_dir)
        for test in args.tests:
            path = os.path.join(tests_dir, test + ".py")
            if not os.path.exists(path):
                print(f"FAIL: {test}.py is not in {args.tests_dir}")
                failures.append(test)
                continue
            try:
                rc = run([py, path], cwd=tests_dir, timeout=TEST_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                print(f"FAIL: {test} ran longer than {TEST_TIMEOUT_SECONDS} s")
                rc = 1
            if rc != 0:
                failures.append(test)

    shutil.rmtree(work, ignore_errors=True)
    if failures:
        print(f"SMOKE FAILED for {name}: {', '.join(failures)}")
        return 1
    print(f"SMOKE PASSED for {name}")
    return 0


def inner(args):
    import importlib.metadata

    import numpy as np

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    import enzyme_ad.jax
    from enzyme_ad.jax import hlo_call

    ok = True

    def check(label, passed, detail):
        nonlocal ok
        ok = ok and passed
        print(f"{'ok  ' if passed else 'FAIL'} {label}: {detail}", flush=True)

    installed = importlib.metadata.version("enzyme_ad")
    check("version", installed == args.version, f"installed {installed}")
    check("import", "site-packages" in enzyme_ad.jax.__file__, enzyme_ad.jax.__file__)

    def rel(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))

    def gradient_checks(label, fun, args_, exact, exact_tol):
        """Reverse mode through hlo_call against forward mode and the closed form."""
        src = jax.jit(fun).lower(*args_).as_text()
        f = lambda *a: hlo_call(*a, source=src)[0]
        n = len(args_)
        value = jax.jit(f)(*args_)
        check(f"{label}: value", rel(value, fun(*args_)) < 1e-12, f"{float(value)!r}")
        try:
            rev = jax.jit(jax.grad(f, argnums=tuple(range(n))))(*args_)
        except Exception as e:  # report the refusal itself
            lines = [line for line in str(e).splitlines() if line.strip()][:3]
            check(f"{label}: reverse mode", False, " | ".join(lines))
            return
        rev = np.array([float(g) for g in rev])
        fwd = np.array(
            [
                float(jax.jit(lambda *a: jax.jvp(f, a, t)[1])(*args_))
                for t in (
                    tuple(
                        jnp.ones_like(a) if i == j else jnp.zeros_like(a)
                        for j, a in enumerate(args_)
                    )
                    for i in range(n)
                )
            ]
        )
        check(
            f"{label}: reverse vs forward",
            rel(rev, fwd) < 1e-9,
            f"{rev} vs {fwd}, rel {rel(rev, fwd):.1e}",
        )
        check(
            f"{label}: reverse vs exact",
            rel(rev, exact) < exact_tol,
            f"{rev} vs {exact}, rel {rel(rev, exact):.1e}",
        )

    # A float accumulator loop: from s = 0, add a until s >= 2.4. At a = 0.5 it runs five
    # trips and ends at s = 2.5, so ds/da = 5 exactly.
    def accumulate(a):
        return jax.lax.while_loop(lambda s: s < 2.4, lambda s: s + a, jnp.zeros_like(a))

    gradient_checks("accumulator loop", accumulate, [jnp.float64(0.5)], [5.0], 1e-12)

    # An adaptive Heun-Euler solve of y' = -k y on [0, 1]: the number of steps depends on
    # k and y0 through the error control. The exact gradient of y0 exp(-k) differs from
    # the discretised program's by about the solver tolerance.
    def solve(k, y0):
        def cond(s):
            t, y, h, n = s
            return (t < 1.0) & (n < 100000)

        def body(s):
            t, y, h, n = s
            h = jnp.minimum(h, 1.0 - t)
            f0 = -k * y
            y1 = y + h * f0
            f1 = -k * y1
            err = jnp.abs(0.5 * h * (f1 - f0))
            tol = 1e-6 + 1e-6 * jnp.abs(y)
            accept = err <= tol
            fac = jnp.clip(0.9 * jnp.sqrt(tol / jnp.maximum(err, 1e-300)), 0.2, 5.0)
            t = jnp.where(accept, t + h, t)
            y = jnp.where(accept, y + 0.5 * h * (f0 + f1), y)
            return (t, y, h * fac, n + 1)

        return jax.lax.while_loop(cond, body, (0.0, y0, 1e-3, 0))[1]

    exact = [-2.0 * np.exp(-1.3), np.exp(-1.3)]
    gradient_checks(
        "adaptive ODE solve", solve, [jnp.float64(1.3), jnp.float64(2.0)], exact, 1e-4
    )

    print("reverse-mode checks " + ("passed" if ok else "FAILED"), flush=True)
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--wheel")
    p.add_argument("--version", required=True)
    p.add_argument("--tests-dir")
    p.add_argument("--tests", nargs="*", default=[])
    p.add_argument("--inner", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.inner:
        return inner(args)
    if not args.wheel:
        p.error("--wheel is required")
    if args.tests and not args.tests_dir:
        p.error("--tests needs --tests-dir")
    return outer(args)


if __name__ == "__main__":
    sys.exit(main())
