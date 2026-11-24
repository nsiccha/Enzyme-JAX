"""Imported HLO may use Enzyme's own dialect, in custom or generic form."""

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np

from enzyme_ad.jax import hlo_call

jax.config.update("jax_enable_x64", True)

DUP = "#enzyme.activity<enzyme_dup>"
DUPNONEED = "#enzyme.activity<enzyme_dupnoneed>"
ACTIVE = "#enzyme.activity<enzyme_active>"
ACTIVENONEED = "#enzyme.activity<enzyme_activenoneed>"


def ignore(value, generic):
    if generic:
        return (
            f'"enzyme.ignore_derivatives"({value}) : (tensor<f64>) -> tensor<f64>'
        )
    return f"enzyme.ignore_derivatives {value} : tensor<f64> -> tensor<f64>"


def fwddiff(fn, operands, types, result, generic, width=None, op="fwddiff"):
    # An enzyme.fwddiff call, or an enzyme.autodiff call with op="autodiff".
    activity, ret_activity = (
        (DUP, DUPNONEED) if op == "fwddiff" else (ACTIVE, ACTIVENONEED)
    )
    properties = f"activity=[{activity}], ret_activity=[{ret_activity}]"
    if width is not None:
        properties += f", width = {width} : i64"
    if generic:
        return (
            f'"enzyme.{op}"({operands}) <{{fn = @{fn}, {properties}}}> '
            f": ({types}) -> {result}"
        )
    return f"enzyme.{op} @{fn}({operands}) <{{{properties}}}> : ({types}) -> {result}"


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


def tensor_type(shape):
    return "tensor<" + "".join(f"{n}x" for n in shape) + "f64>"


def batched_module(body, shape, out_shape, width, generic, op):
    # @main(y, seed) returns `width` derivatives of @f at y, one per row of
    # seed: JVPs (op = "fwddiff", seed of y's shape) or VJPs (op =
    # "autodiff", seed of the result's shape).
    y_t, out_t = tensor_type(shape), tensor_type(out_shape)
    if op == "fwddiff":
        seed_shape, result_shape = shape, out_shape
    else:
        seed_shape, result_shape = out_shape, shape
    seed_t = tensor_type((width,) + seed_shape)
    result_t = tensor_type((width,) + result_shape)
    call = fwddiff(
        "f", "%y, %seed", f"{y_t}, {seed_t}", result_t, generic, width, op
    )
    return f"""module {{
  func.func private @f(%y: {y_t}) -> {out_t} {{
    {body}
    return %r : {out_t}
  }}
  func.func @main(%y: {y_t}, %seed: {seed_t}) -> {result_t} {{
    %j = {call}
    return %j : {result_t}
  }}
}}"""


BATCHED_LAYOUT_CASES = (
    # name, body of @f(%y) -> %r, operand shape, result shape, JAX reference,
    # derivative ops to check
    (
        "reverse dims=[1]",
        "%r = stablehlo.reverse %y, dims = [1] : tensor<2x4xf64>",
        (2, 4),
        (2, 4),
        lambda y: y[:, ::-1],
        ("fwddiff", "autodiff"),
    ),
    (
        "reverse dims=[0, 1]",
        "%r = stablehlo.reverse %y, dims = [0, 1] : tensor<2x4xf64>",
        (2, 4),
        (2, 4),
        lambda y: y[::-1, ::-1],
        ("fwddiff", "autodiff"),
    ),
    (
        "y * reverse(y)",
        "%v = stablehlo.reverse %y, dims = [0] : tensor<4xf64>\n"
        "    %r = stablehlo.multiply %y, %v : tensor<4xf64>",
        (4,),
        (4,),
        lambda y: y * y[::-1],
        ("fwddiff", "autodiff"),
    ),
    (
        "transpose",
        "%r = stablehlo.transpose %y, dims = [1, 0] : "
        "(tensor<2x4xf64>) -> tensor<4x2xf64>",
        (2, 4),
        (4, 2),
        lambda y: y.T,
        # A batched enzyme.autodiff of transpose is not supported yet: its
        # rule builds the unbatched result type.
        ("fwddiff",),
    ),
)


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
        xs = jnp.asarray([1.0, 2.0, 3.0])

        def f(a, passes=None):
            kwargs = {} if passes is None else {"passes": passes}
            return hlo_call(a, source=source, **kwargs)[0]

        np.testing.assert_allclose(jax.jit(f)(x), value(x), rtol=1e-14)
        np.testing.assert_allclose(
            jax.jit(lambda a: f(a, passes=""))(x), value(x), rtol=1e-14
        )
        np.testing.assert_allclose(jax.jit(jax.grad(f))(x), slope(x), rtol=1e-14)
        primal, tangent = jax.jit(lambda a: jax.jvp(f, (a,), (2.0,)))(x)
        np.testing.assert_allclose(primal, value(x), rtol=1e-14)
        np.testing.assert_allclose(tangent, 2 * slope(x), rtol=1e-14)
        np.testing.assert_allclose(
            jax.jit(jax.vmap(f))(xs), jax.vmap(value)(xs), rtol=1e-14
        )
        np.testing.assert_allclose(
            jax.jit(jax.vmap(jax.grad(f)))(xs), jax.vmap(slope)(xs), rtol=1e-14
        )

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

    def test_batched_derivative_layout(self):
        # Row k of a width-3 derivative is the one along row k of the seed. A
        # random seed (not the identity) and a width that matches no operand
        # dimension catch a rule that mixes up the batch and operand
        # dimensions, as the reverse rule did.
        rng = np.random.default_rng(0)
        width = 3
        for name, body, shape, out_shape, reference, ops in BATCHED_LAYOUT_CASES:
            y = jnp.asarray(rng.normal(size=shape))
            seeds = {
                "fwddiff": rng.normal(size=(width,) + shape),
                "autodiff": rng.normal(size=(width,) + out_shape),
            }
            _, vjp = jax.vjp(reference, y)
            expected = {
                "fwddiff": jax.vmap(lambda d: jax.jvp(reference, (y,), (d,))[1]),
                "autodiff": jax.vmap(lambda d: vjp(d)[0]),
            }
            for op in ops:
                seed = seeds[op]
                for generic in (False, True):
                    with self.subTest(name=name, op=op, generic=generic):
                        source = batched_module(
                            body, shape, out_shape, width, generic, op
                        )
                        result = jax.jit(
                            lambda a, s: hlo_call(a, s, source=source)[0]
                        )(y, jnp.asarray(seed))
                        np.testing.assert_allclose(
                            result, expected[op](seed), rtol=1e-14, atol=1e-14
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
