from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest, parameterized
from jax import lax
from jaxlib.mlir import ir

from enzyme_ad.jax import hlo_call
from enzyme_ad.jax._hlo_solve import recognize_linalg

jax.config.update("jax_enable_x64", True)

RECOGNISED = "__enzyme_linalg_eigh_v1_{precision}_cpu_n{n}"
# kTieTolerance in CompositeEighDerivative, relative to the largest |eigenvalue|.
TIE_TOLERANCE = 1e-10


def jacobi(a, sweeps):
    """Cyclic Jacobi eigendecomposition, the kind of body FlatPPL emits.

    Returns the eigenvalues in ascending order, the eigenvectors as columns
    and whether the eigenvalues are finite.
    """
    n = a.shape[0]
    pairs = [(p, q) for p in range(n) for q in range(p + 1, n)]
    v = jnp.eye(n, dtype=a.dtype)
    if pairs:
        rows = jnp.array([p for p, _ in pairs])
        cols = jnp.array([q for _, q in pairs])

        def rotate(k, carry):
            a, v = carry
            p, q = rows[k % len(pairs)], cols[k % len(pairs)]
            # Zeroes a[p, q]: tan(2 theta) = 2 a_pq / (a_qq - a_pp).
            theta = 0.5 * jnp.arctan2(2 * a[p, q], a[q, q] - a[p, p])
            c, s = jnp.cos(theta), jnp.sin(theta)
            j = jnp.eye(n, dtype=a.dtype)
            j = j.at[p, p].set(c).at[q, q].set(c).at[p, q].set(s).at[q, p].set(-s)
            return j.T @ a @ j, v @ j

        a, v = lax.fori_loop(0, sweeps * len(pairs), rotate, (a, v))
    w = jnp.diagonal(a)
    order = jnp.argsort(w)
    return w[order], v[:, order], jnp.all(jnp.isfinite(w))


def source_for(n, dtype, name):
    """A module whose @main calls the Jacobi body as the private function name."""
    lowered = (
        jax.jit(partial(jacobi, sweeps=10))
        .lower(jax.ShapeDtypeStruct((n, n), dtype))
        .compiler_ir("stablehlo")
    )
    functions = []
    with lowered.context:
        for op in lowered.body.operations:
            if op.operation.name != "func.func":
                continue
            if ir.StringAttr(op.attributes["sym_name"]).value == "main":
                op.attributes["sym_name"] = ir.StringAttr.get(name)
                op.attributes["sym_visibility"] = ir.StringAttr.get("private")
                del op.attributes["res_attrs"]
            functions.append(str(op))
    element = "f64" if dtype == np.float64 else "f32"
    m, w = f"tensor<{n}x{n}x{element}>", f"tensor<{n}x{element}>"
    types = f"{w}, {m}, tensor<i1>"
    return "\n".join(
        [
            "module {",
            *functions,
            f"""func.func @main(%a: {m}) -> ({types}) {{
              %w, %v, %valid = func.call @{name}(%a) : ({m}) -> ({types})
              return %w, %v, %valid : {types}
            }}""",
            "}",
        ]
    )


def symmetric(rng, n, dtype):
    m = rng.standard_normal((n, n))
    return (m + m.T).astype(dtype)


def spectral_score(w, v):
    # Depends on the eigenvalues only.
    return jnp.sum(jnp.sin(w))


def projected_score(weight):
    # sum_k exp(-w_k) v_k^T B v_k = tr(exp(-A) B): independent of the sign of
    # each eigenvector, so the gradient is defined for distinct eigenvalues.
    return lambda w, v: jnp.sum(jnp.exp(-w) * jnp.einsum("ik,ij,jk->k", v, weight, v))


def adjoint(w, v, wbar, vbar):
    """The rule's Abar, with eigenvalues closer than the tolerance tied."""
    gap = w[None, :] - w[:, None]
    tied = np.abs(gap) <= TIE_TOLERANCE * np.max(np.abs(w))
    f = np.where(tied, 0.0, 1.0 / np.where(tied, 1.0, gap))
    a = v @ (np.diag(wbar) + f * (v.T @ vbar)) @ v.T
    return (a + a.T) / 2


class EighCompositeTest(parameterized.TestCase):
    def test_recognises_only_the_reserved_name(self):
        n, precision = 3, "f64"
        name = RECOGNISED.format(precision=precision, n=n)
        recognised = recognize_linalg(source_for(n, np.float64, name))
        self.assertIn('stablehlo.composite "enzyme.linalg.eigh"', recognised)
        self.assertIn(f"decomposition = @{name}", recognised)
        self.assertNotIn(f"func.call @{name}", recognised)
        # Another version of the helper is an ordinary call.
        other = name.replace("_v1_", "_v2_")
        source = source_for(n, np.float64, other)
        self.assertEqual(recognize_linalg(source), source)

    @parameterized.named_parameters(
        ("public", "", "tensor<2xf64>", "tensor<i1>"),
        ("status", "private", "tensor<2xf64>", "tensor<i64>"),
        ("values", "private", "tensor<3xf64>", "tensor<i1>"),
    )
    def test_rejects_a_reserved_name_with_another_contract(
        self, visibility, values, valid
    ):
        types = f"{values}, tensor<2x2xf64>, {valid}"
        name = RECOGNISED.format(precision="f64", n=2)
        source = f"""
        module {{
          func.func @main(%a: tensor<2x2xf64>) -> ({types}) {{
            %w, %v, %valid = func.call @{name}(%a) : (tensor<2x2xf64>) -> ({types})
            return %w, %v, %valid : {types}
          }}
          func.func {visibility} @{name}(%a: tensor<2x2xf64>) -> ({types}) {{
            %w = stablehlo.constant dense<1.0> : {values}
            %valid = stablehlo.constant dense<1> : {valid}
            return %w, %a, %valid : {types}
          }}
        }}
        """
        with self.assertRaisesRegex(ValueError, "invalid reserved eigh"):
            recognize_linalg(source)

    @parameterized.named_parameters(
        *(
            (f"{precision}_n{n}", n, precision, dtype, tolerance)
            for n in (1, 2, 3, 5)
            for precision, dtype, tolerance in (
                ("f32", np.float32, 2e-4),
                ("f64", np.float64, 1e-10),
            )
        )
    )
    def test_derivatives_match_the_reference_decomposition(
        self, n, precision, dtype, tolerance
    ):
        rng = np.random.default_rng(n)
        a = symmetric(rng, n, dtype)
        direction = symmetric(rng, n, dtype)
        weight = jnp.asarray(symmetric(rng, n, dtype))
        recognised = source_for(n, dtype, RECOGNISED.format(precision=precision, n=n))
        # Without the reserved name the body itself is differentiated.
        iterative = source_for(n, dtype, f"jacobi_eigh_n{n}")
        for score in (spectral_score, projected_score(weight)):
            reference = lambda a: score(*jnp.linalg.eigh(a))
            exact = (a.astype(np.float64), direction.astype(np.float64))
            value, gradient = jax.value_and_grad(reference)(exact[0])
            tangent = jax.jvp(reference, exact[:1], exact[1:])[1]
            for source in (recognised, iterative):
                function = lambda a: score(*hlo_call(a, source=source)[:2])
                checkpointed = (
                    (jax.checkpoint(function),) if source is recognised else ()
                )
                for f in (function, *checkpointed):
                    actual_value, actual_gradient = jax.jit(jax.value_and_grad(f))(a)
                    actual_tangent = jax.jit(lambda a, d: jax.jvp(f, (a,), (d,))[1])(
                        a, direction
                    )
                    if source is recognised:
                        # The rule symmetrises its gradient, up to the rounding
                        # of the two triangles; the body's gradient is not.
                        np.testing.assert_allclose(
                            actual_gradient,
                            actual_gradient.T,
                            rtol=0,
                            atol=16
                            * np.finfo(dtype).eps
                            * np.max(np.abs(actual_gradient)),
                        )
                    else:
                        # The body treats A as a general matrix, so only the
                        # symmetric part of its gradient is the reference's.
                        actual_gradient = (actual_gradient + actual_gradient.T) / 2
                    scale = 1 + np.max(np.abs(gradient))
                    np.testing.assert_allclose(
                        actual_value, value, rtol=tolerance, atol=tolerance
                    )
                    np.testing.assert_allclose(
                        actual_gradient, gradient, rtol=0, atol=tolerance * scale
                    )
                    np.testing.assert_allclose(
                        actual_tangent,
                        tangent,
                        rtol=tolerance,
                        atol=tolerance * scale * np.max(np.abs(direction)),
                    )

    def test_batched_derivatives(self):
        n = 3
        rng = np.random.default_rng(7)
        matrices = np.stack([symmetric(rng, n, np.float64) for _ in range(4)])
        weight = jnp.asarray(symmetric(rng, n, np.float64))
        score = projected_score(weight)
        source = source_for(n, np.float64, RECOGNISED.format(precision="f64", n=n))
        function = lambda a: score(*hlo_call(a, source=source)[:2])
        reference = lambda a: score(*jnp.linalg.eigh(a))
        values, gradients = jax.jit(jax.vmap(jax.value_and_grad(function)))(matrices)
        expected_values, expected_gradients = jax.vmap(jax.value_and_grad(reference))(
            matrices
        )
        np.testing.assert_allclose(values, expected_values, rtol=1e-10)
        np.testing.assert_allclose(gradients, expected_gradients, atol=1e-10)

    def test_equal_eigenvalues_keep_spectral_gradients_finite(self):
        # The Jacobi body differentiates atan2(0, 0) here; the rule needs only
        # the results, and a score of the eigenvalues alone has gradient
        # V diag(cos w) V^T = diag(cos w).
        n = 3
        a = np.diag([1.0, 1.0, 2.0])
        source = source_for(n, np.float64, RECOGNISED.format(precision="f64", n=n))
        function = lambda a: spectral_score(*hlo_call(a, source=source)[:2])
        gradient = jax.jit(jax.grad(function))(a)
        np.testing.assert_allclose(gradient, np.diag(np.cos(np.diag(a))), atol=1e-12)
        direction = np.ones((n, n))
        tangent = jax.jit(lambda a, d: jax.jvp(function, (a,), (d,))[1])(a, direction)
        np.testing.assert_allclose(tangent, np.sum(np.cos(np.diag(a))), rtol=1e-12)

    @parameterized.named_parameters(("tied", 1e-12, True), ("distinct", 1e-6, False))
    def test_eigenvalues_closer_than_the_tolerance_are_tied(self, gap, tied):
        n = 3
        rng = np.random.default_rng(11)
        q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        a = (q * np.array([1.0, 1.0 + gap, 3.0])) @ q.T
        a = (a + a.T) / 2
        weight = jnp.asarray(symmetric(rng, n, np.float64))
        score = projected_score(weight)
        source = source_for(n, np.float64, RECOGNISED.format(precision="f64", n=n))
        w, v, _ = jax.jit(lambda a: hlo_call(a, source=source))(a)
        w, v = np.asarray(w), np.asarray(v)
        self.assertEqual(abs(w[1] - w[0]) <= TIE_TOLERANCE * np.max(np.abs(w)), tied)
        wbar, vbar = jax.grad(score, argnums=(0, 1))(w, v)
        gradient = jax.jit(jax.grad(lambda a: score(*hlo_call(a, source=source)[:2])))(
            a
        )
        expected = adjoint(w, v, np.asarray(wbar), np.asarray(vbar))
        scale = np.max(np.abs(expected))
        np.testing.assert_allclose(gradient, expected, rtol=0, atol=1e-8 * scale)
        if not tied:
            # A distinct pair keeps the coupling term of tr(exp(-A) B).
            oracle = jax.grad(lambda a: jnp.trace(jax.scipy.linalg.expm(-a) @ weight))(
                a
            )
            oracle = (oracle + oracle.T) / 2
            np.testing.assert_allclose(gradient, oracle, rtol=0, atol=1e-8 * scale)

    def test_clustered_spectrum_keeps_kernel_parameter_gradients(self):
        # An RBF kernel on a grid has most of its eigenvalues within rounding
        # of zero, so their computed gaps (about 1e-18 here) are noise. The
        # gradient of a Gaussian process log density in the kernel parameters
        # still matches the dense formulation, which uses no eigh.
        n, noise = 30, 0.1
        x = jnp.linspace(0.0, 1.0, n)
        y = jnp.asarray(np.random.default_rng(5).standard_normal(n))
        source = source_for(n, np.float64, RECOGNISED.format(precision="f64", n=n))

        def kernel(theta):
            distance = (x[:, None] - x[None, :]) / jnp.exp(theta[1])
            return jnp.exp(theta[0]) * jnp.exp(-0.5 * distance**2)

        def spectral(theta):
            w, v = hlo_call(kernel(theta), source=source)[:2]
            return -0.5 * jnp.sum((v.T @ y) ** 2 / (w + noise)) - 0.5 * jnp.sum(
                jnp.log(w + noise)
            )

        def dense(theta):
            c = kernel(theta) + noise * jnp.eye(n)
            return -0.5 * y @ jnp.linalg.solve(c, y) - 0.5 * jnp.linalg.slogdet(c)[1]

        # Log variance 0 and log lengthscale 0.
        theta = jnp.zeros(2)
        np.testing.assert_allclose(
            jax.jit(jax.grad(spectral))(theta), jax.grad(dense)(theta), rtol=1e-4
        )


if __name__ == "__main__":
    absltest.main()
