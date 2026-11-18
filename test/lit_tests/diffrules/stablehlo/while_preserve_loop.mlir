// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise="stablehlo=true" --enzyme-hlo-generate-td="patterns=while_simplify(1);enzyme_hlo_unroll(4)" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// The augmented primal and reverse sweep both inherit frontend retention,
// including when cache removal rebuilds their carries.
module {
  func.func private @loop(%x: tensor<f64>) -> tensor<f64> {
    %zero = stablehlo.constant dense<0> : tensor<i64>
    %one = stablehlo.constant dense<1> : tensor<i64>
    %two = stablehlo.constant dense<2> : tensor<i64>
    %r:2 = stablehlo.while(%i = %zero, %acc = %x) : tensor<i64>, tensor<f64> attributes {enzyme.disable_mincut, enzymexla.preserve_loop}
    cond {
      %cond = stablehlo.compare LT, %i, %two : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
    } do {
      %next = stablehlo.add %i, %one : tensor<i64>
      %prod = stablehlo.multiply %acc, %x : tensor<f64>
      stablehlo.return %next, %prod : tensor<i64>, tensor<f64>
    }
    return %r#1 : tensor<f64>
  }
  func.func @main(%x: tensor<f64>) -> tensor<f64> {
    %seed = stablehlo.constant dense<1.0> : tensor<f64>
    %dx = enzyme.autodiff @loop(%x, %seed) {activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]} : (tensor<f64>, tensor<f64>) -> tensor<f64>
    return %dx : tensor<f64>
  }
}

// CHECK-LABEL: func.func private @diffeloop
// CHECK: stablehlo.while
// CHECK-SAME: enzymexla.preserve_loop
// CHECK: stablehlo.while
// CHECK-SAME: enzymexla.preserve_loop
// CHECK-NOT: enzyme.push
// CHECK-NOT: enzyme.pop
// CHECK: return
