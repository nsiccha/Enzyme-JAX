// RUN: enzymexlamlir-opt %s --drop-unsupported-attributes | FileCheck %s

// The final export stage preserves the existing frontend metadata and carries
// control-flow requirements into XLA. Ordinary selects and unmarked loops
// keep their normal optimizer behavior.
module {
  func.func @marked(%x: tensor<f64>, %take: tensor<i1>) -> tensor<f64> {
    %zero = stablehlo.constant dense<0> : tensor<i64>
    %one = stablehlo.constant dense<1> : tensor<i64>
    %r = "stablehlo.if"(%take) ({
      %loop:2 = stablehlo.while(%i = %zero, %acc = %x) : tensor<i64>, tensor<f64> attributes {enzymexla.preserve_loop, mhlo.frontend_attributes = {client = "kept"}}
      cond {
        %more = stablehlo.compare LT, %i, %one : (tensor<i64>, tensor<i64>) -> tensor<i1>
        stablehlo.return %more : tensor<i1>
      } do {
        %next = stablehlo.add %i, %one : tensor<i64>
        %value = stablehlo.log %acc : tensor<f64>
        stablehlo.return %next, %value : tensor<i64>, tensor<f64>
      }
      stablehlo.return %loop#1 : tensor<f64>
    }, {
      stablehlo.return %x : tensor<f64>
    }) : (tensor<i1>) -> tensor<f64>
    return %r : tensor<f64>
  }
  func.func @unmarked(%x: tensor<f64>, %take: tensor<i1>) -> tensor<f64> {
    %zero = stablehlo.constant dense<0> : tensor<i64>
    %one = stablehlo.constant dense<1> : tensor<i64>
    %loop:2 = stablehlo.while(%i = %zero, %acc = %x) : tensor<i64>, tensor<f64>
    cond {
      %more = stablehlo.compare LT, %i, %one : (tensor<i64>, tensor<i64>) -> tensor<i1>
      stablehlo.return %more : tensor<i1>
    } do {
      %next = stablehlo.add %i, %one : tensor<i64>
      %value = stablehlo.log %acc : tensor<f64>
      stablehlo.return %next, %value : tensor<i64>, tensor<f64>
    }
    %selected = stablehlo.select %take, %loop#1, %x : tensor<i1>, tensor<f64>
    return %selected : tensor<f64>
  }
}

// CHECK-LABEL: func.func @marked
// CHECK-DAG: mhlo.frontend_attributes = {xla_preserve_conditional = "true"}
// CHECK-DAG: mhlo.frontend_attributes = {_xla_disable_loop_instr_hoisting = "true", client = "kept", "skip-simplify-while-loops_trip-count-one" = "true", xla_disable_while_loop_dce = "true"}
// CHECK-LABEL: func.func @unmarked
// CHECK: stablehlo.while
// CHECK-NOT: mhlo.frontend_attributes
// CHECK: stablehlo.select
// CHECK-NOT: mhlo.frontend_attributes
