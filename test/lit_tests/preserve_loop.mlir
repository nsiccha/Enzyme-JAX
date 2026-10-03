// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=enzyme_hlo_unroll(4);greedy_while_loop_batch_fission;while_elementwise_reduction_to_reduce" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme-hlo-unroll="max-num-iterations=4" | FileCheck %s

// Shape-specialized frontend loops retain one body even when the default
// optimizer would otherwise materialize their small iteration space.
module {
  func.func @unmarked(%x: tensor<f64>) -> tensor<f64> {
    %zero = stablehlo.constant dense<0> : tensor<i64>
    %one = stablehlo.constant dense<1> : tensor<i64>
    %two = stablehlo.constant dense<2> : tensor<i64>
    %r:2 = stablehlo.while(%i = %zero, %acc = %x) : tensor<i64>, tensor<f64>
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
  func.func @marked(%x: tensor<f64>) -> tensor<f64> {
    %zero = stablehlo.constant dense<0> : tensor<i64>
    %one = stablehlo.constant dense<1> : tensor<i64>
    %two = stablehlo.constant dense<2> : tensor<i64>
    %r:2 = stablehlo.while(%i = %zero, %acc = %x) : tensor<i64>, tensor<f64> attributes {enzymexla.preserve_loop}
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
}

// CHECK-LABEL: func.func @unmarked
// CHECK-NOT: stablehlo.while
// CHECK-COUNT-2: stablehlo.multiply
// CHECK: return
// CHECK-LABEL: func.func @marked
// CHECK: stablehlo.while
// CHECK-SAME: attributes {enzymexla.preserve_loop}
// CHECK: stablehlo.multiply
// CHECK-NOT: stablehlo.multiply
// CHECK: return
