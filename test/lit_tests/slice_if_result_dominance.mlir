// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=slice_if" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// The second result is consumed before the slice of the first result. The new
// if must be placed at the original if, so every replacement result dominates
// its existing uses. All inputs and both branch computations are synthetic.
// CHECK-LABEL: func.func @slice_if_result_dominance
// CHECK: %[[IF:.*]]:2 = "stablehlo.if"
// CHECK: stablehlo.slice
// CHECK: stablehlo.slice
// CHECK: stablehlo.add %[[IF]]#1
// CHECK: return %[[IF]]#0
func.func @slice_if_result_dominance(%pred: tensor<i1>, %a: tensor<2x2xf64>, %b: tensor<2x2xf64>) -> (tensor<1x2xf64>, tensor<2x2xf64>) {
  %choice:2 = "stablehlo.if"(%pred) ({
    %first = stablehlo.add %a, %b : tensor<2x2xf64>
    %second = stablehlo.subtract %a, %b : tensor<2x2xf64>
    stablehlo.return %first, %second : tensor<2x2xf64>, tensor<2x2xf64>
  }, {
    %first = stablehlo.multiply %a, %b : tensor<2x2xf64>
    %second = stablehlo.divide %a, %b : tensor<2x2xf64>
    stablehlo.return %first, %second : tensor<2x2xf64>, tensor<2x2xf64>
  }) : (tensor<i1>) -> (tensor<2x2xf64>, tensor<2x2xf64>)
  %before = stablehlo.add %choice#1, %a : tensor<2x2xf64>
  %slice = stablehlo.slice %choice#0 [0:1, 0:2] : (tensor<2x2xf64>) -> tensor<1x2xf64>
  return %slice, %before : tensor<1x2xf64>, tensor<2x2xf64>
}

// A slice can also be nested under a consumer of another conditional result.
// Rebuilding inside that consumer would make its predicate depend on a value
// defined in its own branch.
// CHECK-LABEL: func.func @slice_if_nested_result_use
// CHECK: %[[OUTER:.*]]:2 = "stablehlo.if"
// CHECK: stablehlo.slice
// CHECK: stablehlo.slice
// CHECK: "stablehlo.if"(%[[OUTER]]#1)
func.func @slice_if_nested_result_use(%pred: tensor<i1>, %a: tensor<2x2xf64>, %b: tensor<2x2xf64>, %x: tensor<f64>, %y: tensor<f64>) -> tensor<1x2xf64> {
  %choice:2 = "stablehlo.if"(%pred) ({
    %matrix = stablehlo.add %a, %b : tensor<2x2xf64>
    %flag = stablehlo.compare GT, %x, %y : (tensor<f64>, tensor<f64>) -> tensor<i1>
    stablehlo.return %matrix, %flag : tensor<2x2xf64>, tensor<i1>
  }, {
    %matrix = stablehlo.multiply %a, %b : tensor<2x2xf64>
    %flag = stablehlo.compare GT, %y, %x : (tensor<f64>, tensor<f64>) -> tensor<i1>
    stablehlo.return %matrix, %flag : tensor<2x2xf64>, tensor<i1>
  }) : (tensor<i1>) -> (tensor<2x2xf64>, tensor<i1>)
  %result = "stablehlo.if"(%choice#1) ({
    %slice = stablehlo.slice %choice#0 [0:1, 0:2] : (tensor<2x2xf64>) -> tensor<1x2xf64>
    stablehlo.return %slice : tensor<1x2xf64>
  }, {
    %fallback = stablehlo.slice %a [0:1, 0:2] : (tensor<2x2xf64>) -> tensor<1x2xf64>
    stablehlo.return %fallback : tensor<1x2xf64>
  }) : (tensor<i1>) -> tensor<1x2xf64>
  return %result : tensor<1x2xf64>
}
