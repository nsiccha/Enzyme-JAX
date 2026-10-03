// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=remove_no_ops_from_while_loop;if_inline;while_simplify(1)" --transform-interpreter --enzyme-hlo-remove-transform --canonicalize | FileCheck %s

// Shape bounds must not discard the conditional body of a retained loop.
module {
  func.func @unmarked(%x: tensor<f64>) -> tensor<f64> {
    %zero = stablehlo.constant dense<0> : tensor<i64>
    %one = stablehlo.constant dense<1> : tensor<i64>
    %zero_f = stablehlo.constant dense<0.0> : tensor<f64>
    %r:2 = stablehlo.while(%i = %zero, %acc = %zero_f) : tensor<i64>, tensor<f64>
    cond {
      %more = stablehlo.compare LT, %i, %one : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %more : tensor<i1>
    } do {
      %next = stablehlo.add %i, %one : tensor<i64>
      %active = stablehlo.compare LT, %next, %one : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %sum = "stablehlo.if"(%active) ({
        %updated = stablehlo.add %acc, %x : tensor<f64>
        stablehlo.return %updated : tensor<f64>
      }, {
        stablehlo.return %acc : tensor<f64>
      }) : (tensor<i1>) -> tensor<f64>
      stablehlo.return %next, %sum : tensor<i64>, tensor<f64>
    }
    return %r#1 : tensor<f64>
  }
  func.func @marked(%x: tensor<f64>) -> tensor<f64> {
    %zero = stablehlo.constant dense<0> : tensor<i64>
    %one = stablehlo.constant dense<1> : tensor<i64>
    %zero_f = stablehlo.constant dense<0.0> : tensor<f64>
    %r:2 = stablehlo.while(%i = %zero, %acc = %zero_f) : tensor<i64>, tensor<f64> attributes {enzymexla.preserve_loop}
    cond {
      %more = stablehlo.compare LT, %i, %one : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %more : tensor<i1>
    } do {
      %next = stablehlo.add %i, %one : tensor<i64>
      %active = stablehlo.compare LT, %next, %one : (tensor<i64>, tensor<i64>) -> tensor<i1>
      %sum = "stablehlo.if"(%active) ({
        %updated = stablehlo.add %acc, %x : tensor<f64>
        stablehlo.return %updated : tensor<f64>
      }, {
        stablehlo.return %acc : tensor<f64>
      }) : (tensor<i1>) -> tensor<f64>
      stablehlo.return %next, %sum : tensor<i64>, tensor<f64>
    }
    return %r#1 : tensor<f64>
  }
}

// CHECK-LABEL: func.func @unmarked
// CHECK-NOT: stablehlo.while
// CHECK: return
// CHECK-LABEL: func.func @marked
// CHECK: stablehlo.while
// CHECK-SAME: enzymexla.preserve_loop
// CHECK: stablehlo.if
// CHECK: stablehlo.add
// CHECK: return
