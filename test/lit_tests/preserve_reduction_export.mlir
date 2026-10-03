// RUN: enzymexlamlir-opt %s --drop-unsupported-attributes | FileCheck %s

// Keep the frontend's existing metadata while requiring one reduction region.
// Vector contractions can become reductions later in the XLA pipeline.
module {
  func.func @reduction(%x: tensor<2048xf64>) -> tensor<f64> {
    %zero = stablehlo.constant dense<0.0> : tensor<f64>
    %sum = "stablehlo.reduce"(%x, %zero) <{dimensions = array<i64: 0>}> ({
    ^bb0(%lhs: tensor<f64>, %rhs: tensor<f64>):
      %add = stablehlo.add %lhs, %rhs : tensor<f64>
      stablehlo.return %add : tensor<f64>
    }) {mhlo.frontend_attributes = {client = "kept"}} : (tensor<2048xf64>, tensor<f64>) -> tensor<f64>
    return %sum : tensor<f64>
  }
  func.func @contraction(%x: tensor<2048xf64>) -> tensor<f64> {
    %dot = "stablehlo.dot_general"(%x, %x) {
      dot_dimension_numbers = #stablehlo.dot<
        lhs_batching_dimensions = [], rhs_batching_dimensions = [],
        lhs_contracting_dimensions = [0], rhs_contracting_dimensions = [0]>,
      precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>],
      mhlo.frontend_attributes = {client = "kept"}
    } : (tensor<2048xf64>, tensor<2048xf64>) -> tensor<f64>
    return %dot : tensor<f64>
  }
  func.func @elementwise(%x: tensor<2048xf64>) -> tensor<2048xf64> {
    %add = stablehlo.add %x, %x : tensor<2048xf64>
    return %add : tensor<2048xf64>
  }
}

// CHECK-LABEL: func.func @reduction
// CHECK: mhlo.frontend_attributes = {client = "kept", xla_preserve_reduction = "true"}
// CHECK-LABEL: func.func @contraction
// CHECK: mhlo.frontend_attributes = {client = "kept", xla_preserve_reduction = "true"}
// CHECK-LABEL: func.func @elementwise
// CHECK: stablehlo.add
// CHECK-NOT: mhlo.frontend_attributes
