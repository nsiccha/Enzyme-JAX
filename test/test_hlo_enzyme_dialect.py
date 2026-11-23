"""Imported HLO may use Enzyme's own dialect, in custom or generic form."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from enzyme_ad.jax import hlo_call

jax.config.update("jax_enable_x64", True)

DUP = "#enzyme.activity<enzyme_dup>"
DUPNONEED = "#enzyme.activity<enzyme_dupnoneed>"


def ignore(value, generic):
    if generic:
        return (
            f'"enzyme.ignore_derivatives"({value}) : (tensor<f64>) -> tensor<f64>'
        )
    return f"enzyme.ignore_derivatives {value} : tensor<f64> -> tensor<f64>"


def fwddiff(fn, operands, types, result, generic, width=None):
    properties = f"activity=[{DUP}], ret_activity=[{DUPNONEED}]"
    if width is not None:
        properties += f", width = {width} : i64"
    if generic:
        return (
            f'"enzyme.fwddiff"({operands}) <{{fn = @{fn}, {properties}}}> '
            f": ({types}) -> {result}"
        )
    return f"enzyme.fwddiff @{fn}({operands}) <{{{properties}}}> : ({types}) -> {result}"


CUBE = """
  func.func private @cube(%x: tensor<f64>) -> tensor<f64> {
    %a = stablehlo.multiply %x, %x : tensor<f64>
    %b = stablehlo.multiply %a, %x : tensor<f64>
    return %b : tensor<f64>
  }"""


def stop_gradient(generic):
    # stop(x) * x: value x^2, derivative stop(x) = x.
    return f"""module {{
  func.func @main(%x: tensor<f64>) -> tensor<f64> {{
    %s = {ignore("%x", generic)}
    %y = stablehlo.multiply %s, %x : tensor<f64>
    return %y : tensor<f64>
  }}
}}"""


def two_solves(generic):
    # FlatPPL's combination stop(v) + (c - stop(c)): the value of v = x^2 and
    # the derivative of c = x^3.
    return f"""module {{{CUBE}
  func.func @main(%x: tensor<f64>) -> tensor<f64> {{
    %v = stablehlo.multiply %x, %x : tensor<f64>
    %c = func.call @cube(%x) : (tensor<f64>) -> tensor<f64>
    %sv = {ignore("%v", generic)}
    %sc = {ignore("%c", generic)}
    %d = stablehlo.subtract %c, %sc : tensor<f64>
    %y = stablehlo.add %sv, %d : tensor<f64>
    return %y : tensor<f64>
  }}
}}"""


def derivative(generic):
    # d/dx x^3 = 3 x^2 from enzyme.fwddiff.
    call = fwddiff(
        "cube", "%x, %one", "tensor<f64>, tensor<f64>", "tensor<f64>", generic
    )
    return f"""module {{{CUBE}
  func.func @main(%x: tensor<f64>) -> tensor<f64> {{
    %one = stablehlo.constant dense<1.0> : tensor<f64>
    %d = {call}
    return %d : tensor<f64>
  }}
}}"""


def batched_derivative(generic):
    # Two JVP directions of v -> v * v in one call: the Jacobian diag(2 v).
    # main(v) sums it to 2 (v0 + v1).
    call = fwddiff(
        "square",
        "%v, %seed",
        "tensor<2xf64>, tensor<2x2xf64>",
        "tensor<2x2xf64>",
        generic,
        width=2,
    )
    return f"""module {{
  func.func private @square(%v: tensor<2xf64>) -> tensor<2xf64> {{
    %a = stablehlo.multiply %v, %v : tensor<2xf64>
    return %a : tensor<2xf64>
  }}
  func.func @main(%v: tensor<2xf64>) -> tensor<f64> {{
    %seed = stablehlo.constant dense<[[1.0, 0.0], [0.0, 1.0]]> : tensor<2x2xf64>
    %j = {call}
    %z = stablehlo.constant dense<0.0> : tensor<f64>
    %s = stablehlo.reduce(%j init: %z) applies stablehlo.add across dimensions = [0, 1] : (tensor<2x2xf64>, tensor<f64>) -> tensor<f64>
    return %s : tensor<f64>
  }}
}}"""


def loop_derivative(generic):
    # Three steps of y <- y + 0.1 * d/dy y^3, with the derivative in the body.
    call = fwddiff(
        "cube", "%y, %one", "tensor<f64>, tensor<f64>", "tensor<f64>", generic
    )
    return f"""module {{{CUBE}
  func.func @main(%x: tensor<f64>) -> tensor<f64> {{
    %c0 = stablehlo.constant dense<0> : tensor<i64>
    %c1 = stablehlo.constant dense<1> : tensor<i64>
    %n = stablehlo.constant dense<3> : tensor<i64>
    %h = stablehlo.constant dense<0.1> : tensor<f64>
    %one = stablehlo.constant dense<1.0> : tensor<f64>
    %r:2 = stablehlo.while(%i = %c0, %y = %x) : tensor<i64>, tensor<f64>
      cond {{
        %p = stablehlo.compare LT, %i, %n : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %p : tensor<i1>
      }} do {{
        %d = {call}
        %s = stablehlo.multiply %h, %d : tensor<f64>
        %y2 = stablehlo.add %y, %s : tensor<f64>
        %i2 = stablehlo.add %i, %c1 : tensor<i64>
        stablehlo.return %i2, %y2 : tensor<i64>, tensor<f64>
      }}
    return %r#1 : tensor<f64>
  }}
}}"""


def loop_reference(x):
    for _ in range(3):
        x = x + 0.3 * x * x
    return x


SCALAR_CASES = (
    # name, module, value function, derivative function
    ("stop_gradient", stop_gradient, lambda x: x * x, lambda x: x),
    ("two_solves", two_solves, lambda x: x * x, lambda x: 3 * x * x),
    ("derivative", derivative, lambda x: 3 * x * x, lambda x: 6 * x),
    (
        "loop_derivative",
        loop_derivative,
        loop_reference,
        jax.grad(loop_reference),
    ),
)


class EnzymeDialect(absltest.TestCase):
    def check_scalar(self, source, value, slope):
        x = jnp.asarray(3.0)

        def f(a, passes=None):
            kwargs = {} if passes is None else {"passes": passes}
            return hlo_call(a, source=source, **kwargs)[0]

        np.testing.assert_allclose(jax.jit(f)(x), value(x), rtol=1e-14)
        np.testing.assert_allclose(
            jax.jit(lambda a: f(a, passes=""))(x), value(x), rtol=1e-14
        )
        np.testing.assert_allclose(jax.jit(jax.grad(f))(x), slope(x), rtol=1e-14)
        primal, tangent = jax.jit(
            lambda a: jax.jvp(f, (a,), (jnp.asarray(2.0),))
        )(x)
        np.testing.assert_allclose(primal, value(x), rtol=1e-14)
        np.testing.assert_allclose(tangent, 2 * slope(x), rtol=1e-14)

    def test_scalar_cases(self):
        for name, module, value, slope in SCALAR_CASES:
            for generic in (False, True):
                with self.subTest(name=name, generic=generic):
                    self.check_scalar(module(generic), value, slope)

    def test_batched_derivative(self):
        v = jnp.asarray([1.0, 2.0])
        for generic in (False, True):
            with self.subTest(generic=generic):
                source = batched_derivative(generic)

                def f(a):
                    return hlo_call(a, source=source)[0]

                np.testing.assert_allclose(jax.jit(f)(v), 6.0, rtol=1e-14)
                np.testing.assert_allclose(
                    jax.jit(jax.grad(f))(v), [2.0, 2.0], rtol=1e-14
                )

    def test_plain_module_pipeline_is_unchanged(self):
        from enzyme_ad.jax.primitives import _with_enzyme_lowering

        plain = """module {
  func.func @main(%x: tensor<f64>) -> tensor<f64> {
    %y = stablehlo.multiply %x, %x : tensor<f64>
    return %y : tensor<f64>
  }
}"""
        self.assertEqual(_with_enzyme_lowering(plain, "canonicalize"), "canonicalize")
        self.assertNotEqual(
            _with_enzyme_lowering(stop_gradient(False), "canonicalize"),
            "canonicalize",
        )


if __name__ == "__main__":
    absltest.main()
