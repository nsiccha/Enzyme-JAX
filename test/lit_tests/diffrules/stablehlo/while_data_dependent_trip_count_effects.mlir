// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --cse | FileCheck %s

// A loop whose trip count is data-dependent is replayed in the reverse pass
// only if its body has no effects. This one draws a random number on every
// trip, which a replay would draw anew, so it is reversed by the plain loop
// over its counted trips and the draw stays in the forward pass. The loop is
// uncounted because it steps a floating-point value; its trip counter is still
// an integer. Its step adjoint reads no primal values, so nothing is taped.

module {
  func.func private @noisy(%a: tensor<f64>, %s0: tensor<f64>) -> tensor<f64> {
    %stop = stablehlo.constant dense<2.400000e+00> : tensor<f64>
    %lo = stablehlo.constant dense<0.000000e+00> : tensor<f64>
    %hi = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %shape = stablehlo.constant dense<> : tensor<0xi64>
    %r:2 = stablehlo.while(%s = %s0, %n = %lo) : tensor<f64>, tensor<f64>
    cond {
      %c = stablehlo.compare LT, %s, %stop : (tensor<f64>, tensor<f64>) -> tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      %u = stablehlo.rng %lo, %hi, %shape, distribution = UNIFORM : (tensor<f64>, tensor<f64>, tensor<0xi64>) -> tensor<f64>
      %n2 = stablehlo.add %n, %u : tensor<f64>
      %s2 = stablehlo.add %s, %a : tensor<f64>
      stablehlo.return %s2, %n2 : tensor<f64>, tensor<f64>
    }
    return %r#0 : tensor<f64>
  }

  func.func @main(%a: tensor<f64>, %s0: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
    %seed = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %g:2 = enzyme.autodiff @noisy(%a, %s0, %seed) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    return %g#0, %g#1 : tensor<f64>, tensor<f64>
  }
}

// CHECK-LABEL: func.func private @diffenoisy(
// CHECK:         %[[FWD:[0-9]+]]:3 = stablehlo.while({{.*}}) : tensor<f64>, tensor<f64>, tensor<i64>{{$}}
// CHECK:           stablehlo.rng
// CHECK:         stablehlo.while({{.*}}) : tensor<i64>, tensor<f64>, tensor<f64>{{$}}
// CHECK-NEXT:    cond {
// CHECK-NEXT:      stablehlo.compare LT, %{{.+}}, %[[FWD]]#2
// CHECK-NOT:     {{stablehlo.rng|binomial_progress|tensor<[0-9]+x}}
// CHECK:         {{ }}return
