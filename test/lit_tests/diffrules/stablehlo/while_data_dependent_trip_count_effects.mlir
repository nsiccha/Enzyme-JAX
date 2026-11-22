// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --canonicalize --cse | FileCheck %s

// A loop whose trip count is data-dependent is replayed in the reverse pass
// only if its body has no effects. This one calls the host on every trip, so
// it is reversed by the plain loop over its counted trips, and the call stays
// in the forward pass. The loop is uncounted because it steps a floating-point
// value; its trip counter is still an integer. Its step adjoint reads no
// primal values, so nothing is taped.

module {
  func.func private @ticking(%a: tensor<f64>, %s0: tensor<f64>) -> tensor<f64> {
    %stop = stablehlo.constant dense<2.400000e+00> : tensor<f64>
    %r = stablehlo.while(%s = %s0) : tensor<f64>
    cond {
      %c = stablehlo.compare LT, %s, %stop : (tensor<f64>, tensor<f64>) -> tensor<i1>
      stablehlo.return %c : tensor<i1>
    } do {
      stablehlo.custom_call @tick() {has_side_effect = true} : () -> ()
      %s2 = stablehlo.add %s, %a : tensor<f64>
      stablehlo.return %s2 : tensor<f64>
    }
    return %r : tensor<f64>
  }

  func.func @main(%a: tensor<f64>, %s0: tensor<f64>) -> (tensor<f64>, tensor<f64>) {
    %seed = stablehlo.constant dense<1.000000e+00> : tensor<f64>
    %g:2 = enzyme.autodiff @ticking(%a, %s0, %seed) <{activity = [#enzyme.activity<enzyme_active>, #enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]}> : (tensor<f64>, tensor<f64>, tensor<f64>) -> (tensor<f64>, tensor<f64>)
    return %g#0, %g#1 : tensor<f64>, tensor<f64>
  }
}

// CHECK-LABEL: func.func private @diffeticking(
// CHECK:         %[[FWD:[0-9]+]]:2 = stablehlo.while({{.*}}) : tensor<f64>, tensor<i64>{{$}}
// CHECK:           stablehlo.custom_call @tick
// CHECK:         stablehlo.while({{.*}}) : tensor<i64>, tensor<f64>, tensor<f64>{{$}}
// CHECK-NEXT:    cond {
// CHECK-NEXT:      stablehlo.compare LT, %{{.+}}, %[[FWD]]#1
// CHECK-NOT:     {{custom_call|binomial_progress|tensor<[0-9]+x}}
// CHECK:         return
