// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise="stablehlo=true" | FileCheck %s

// Nested single-iteration loops each need a leading tape axis for tensor caches.
module {
  func.func private @loop(%x: tensor<2xf64>) -> tensor<2xf64> {
    %zero = stablehlo.constant dense<0> : tensor<i64>
    %one = stablehlo.constant dense<1> : tensor<i64>
    %r:2 = stablehlo.while(%i = %zero, %acc = %x) : tensor<i64>, tensor<2xf64> attributes {enzyme.disable_mincut, enzymexla.preserve_loop}
    cond {
      %cond = stablehlo.compare LT, %i, %one : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %cond : tensor<i1>
    } do {
      %next = stablehlo.add %i, %one : tensor<i64>
      %inner:2 = stablehlo.while(%j = %zero, %value = %acc) : tensor<i64>, tensor<2xf64> attributes {enzyme.disable_mincut, enzymexla.preserve_loop}
      cond {
        %more = stablehlo.compare LT, %j, %one : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %more : tensor<i1>
      } do {
        %next_j = stablehlo.add %j, %one : tensor<i64>
        %updated = stablehlo.exponential %value : tensor<2xf64>
        stablehlo.return %next_j, %updated : tensor<i64>, tensor<2xf64>
      }
      stablehlo.return %next, %inner#1 : tensor<i64>, tensor<2xf64>
    }
    return %r#1 : tensor<2xf64>
  }
  func.func @main(%x: tensor<2xf64>) -> tensor<2xf64> {
    %seed = stablehlo.constant dense<1.0> : tensor<2xf64>
    %dx = enzyme.autodiff @loop(%x, %seed) {activity = [#enzyme.activity<enzyme_active>], ret_activity = [#enzyme.activity<enzyme_activenoneed>]} : (tensor<2xf64>, tensor<2xf64>) -> tensor<2xf64>
    return %dx : tensor<2xf64>
  }
}

// CHECK-LABEL: func.func private @diffeloop
// CHECK: stablehlo.while
// CHECK-SAME: tensor<1x1x2xf64>
// CHECK: stablehlo.while
// CHECK-SAME: tensor<1x2xf64>
// CHECK: stablehlo.dynamic_update_slice
// CHECK-SAME: tensor<1x2xf64>
// CHECK: stablehlo.while
// CHECK: stablehlo.dynamic_slice
// CHECK-SAME: tensor<1x1x2xf64>
// CHECK: stablehlo.while
// CHECK: stablehlo.dynamic_slice
// CHECK-SAME: tensor<1x2xf64>
// CHECK: return
