// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=reduce_pad" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// A constant adjoint pads a row before its lanes are summed. The row's
// contribution is -3, rather than a single -1.
func.func @sum_padded_row(%arg: tensor<1x3xf64>, %init: tensor<f64>) -> tensor<2xf64> {
  %pv = stablehlo.constant dense<-1.0> : tensor<f64>
  %p = stablehlo.pad %arg, %pv, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x3xf64>, tensor<f64>) -> tensor<2x3xf64>
  %r = stablehlo.reduce(%p init: %init) applies stablehlo.add across dimensions = [1] : (tensor<2x3xf64>, tensor<f64>) -> tensor<2xf64>
  return %r : tensor<2xf64>
}
// CHECK-LABEL: func.func @sum_padded_row(
// CHECK-DAG: %[[PV:.*]] = stablehlo.constant dense<-1.000000e+00> : tensor<f64>
// CHECK-DAG: %[[N:.*]] = stablehlo.constant dense<3.000000e+00> : tensor<f64>
// CHECK: %[[R:.*]] = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [1]
// CHECK: %[[M:.*]] = stablehlo.multiply %[[PV]], %[[N]] : tensor<f64>
// CHECK: %[[F:.*]] = stablehlo.add %arg1, %[[M]] : tensor<f64>
// CHECK: stablehlo.pad %[[R]], %[[F]], low = [0], high = [1], interior = [0]

// With several reduced axes, the number of padded elements is the difference
// between slice volumes, not the sum of padding widths: 6*5 - 2*3 = 24.
func.func @sum_multiple_axes(%arg: tensor<2x2x3xf64>, %pv: tensor<f64>, %init: tensor<f64>) -> tensor<3xf64> {
  %p = stablehlo.pad %arg, %pv, low = [0, 1, 1], high = [1, 3, 1], interior = [0, 0, 0] : (tensor<2x2x3xf64>, tensor<f64>) -> tensor<3x6x5xf64>
  %r = stablehlo.reduce(%p init: %init) applies stablehlo.add across dimensions = [1, 2] : (tensor<3x6x5xf64>, tensor<f64>) -> tensor<3xf64>
  return %r : tensor<3xf64>
}
// CHECK-LABEL: func.func @sum_multiple_axes(
// CHECK-DAG: %[[N:.*]] = stablehlo.constant dense<2.400000e+01> : tensor<f64>
// CHECK-DAG: %[[TOTAL:.*]] = stablehlo.constant dense<3.000000e+01> : tensor<f64>
// CHECK: %[[R:.*]] = stablehlo.reduce(%arg0 init: %arg2) applies stablehlo.add across dimensions = [1, 2]
// CHECK: %[[M:.*]] = stablehlo.multiply %arg1, %[[N]] : tensor<f64>
// CHECK: %[[B:.*]] = stablehlo.broadcast_in_dim %[[M]], dims = []
// CHECK: %[[A:.*]] = stablehlo.add %[[R]], %[[B]]
// CHECK: %[[FULL:.*]] = stablehlo.multiply %arg1, %[[TOTAL]] : tensor<f64>
// CHECK: %[[F:.*]] = stablehlo.add %arg2, %[[FULL]] : tensor<f64>
// CHECK: stablehlo.pad %[[A]], %[[F]], low = [0], high = [1], interior = [0]

func.func @product_padded_row(%arg: tensor<1x3xf64>, %pv: tensor<f64>, %init: tensor<f64>) -> tensor<2xf64> {
  %p = stablehlo.pad %arg, %pv, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x3xf64>, tensor<f64>) -> tensor<2x3xf64>
  %r = stablehlo.reduce(%p init: %init) applies stablehlo.multiply across dimensions = [1] : (tensor<2x3xf64>, tensor<f64>) -> tensor<2xf64>
  return %r : tensor<2xf64>
}
// CHECK-LABEL: func.func @product_padded_row(
// CHECK-DAG: %[[N:.*]] = stablehlo.constant dense<3.000000e+00> : tensor<f64>
// CHECK: %[[R:.*]] = stablehlo.reduce(%arg0 init: %arg2) applies stablehlo.multiply across dimensions = [1]
// CHECK: %[[P:.*]] = stablehlo.power %arg1, %[[N]] : tensor<f64>
// CHECK: %[[F:.*]] = stablehlo.multiply %arg2, %[[P]] : tensor<f64>
// CHECK: stablehlo.pad %[[R]], %[[F]], low = [0], high = [1], interior = [0]

// No padding along the reduced axes means the fill never participates in
// the original row's minimum or maximum. It still combines with init in a
// padded output row.
func.func @minimum_padded_row(%arg: tensor<1x3xf64>, %pv: tensor<f64>, %init: tensor<f64>) -> tensor<2xf64> {
  %p = stablehlo.pad %arg, %pv, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x3xf64>, tensor<f64>) -> tensor<2x3xf64>
  %r = stablehlo.reduce(%p init: %init) applies stablehlo.minimum across dimensions = [1] : (tensor<2x3xf64>, tensor<f64>) -> tensor<2xf64>
  return %r : tensor<2xf64>
}
// CHECK-LABEL: func.func @minimum_padded_row(

// CHECK: %[[R:.*]] = stablehlo.reduce(%arg0 init: %arg2) applies stablehlo.minimum across dimensions = [1]
// CHECK-NOT: stablehlo.minimum %[[R]]
// CHECK: %[[F:.*]] = stablehlo.minimum %arg2, %arg1 : tensor<f64>
// CHECK: stablehlo.pad %[[R]], %[[F]], low = [0], high = [1], interior = [0]

func.func @maximum_padded_row(%arg: tensor<1x3xf64>, %pv: tensor<f64>, %init: tensor<f64>) -> tensor<2xf64> {
  %p = stablehlo.pad %arg, %pv, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x3xf64>, tensor<f64>) -> tensor<2x3xf64>
  %r = stablehlo.reduce(%p init: %init) applies stablehlo.maximum across dimensions = [1] : (tensor<2x3xf64>, tensor<f64>) -> tensor<2xf64>
  return %r : tensor<2xf64>
}
// CHECK-LABEL: func.func @maximum_padded_row(

// CHECK: %[[R:.*]] = stablehlo.reduce(%arg0 init: %arg2) applies stablehlo.maximum across dimensions = [1]
// CHECK-NOT: stablehlo.maximum %[[R]]
// CHECK: %[[F:.*]] = stablehlo.maximum %arg2, %arg1 : tensor<f64>
// CHECK: stablehlo.pad %[[R]], %[[F]], low = [0], high = [1], interior = [0]

// Interior padding contributes both reduced elements and output lanes.
func.func @sum_interior(%arg: tensor<2x3xi32>, %pv: tensor<i32>, %init: tensor<i32>) -> tensor<5xi32> {
  %p = stablehlo.pad %arg, %pv, low = [1, 0], high = [1, 0], interior = [1, 1] : (tensor<2x3xi32>, tensor<i32>) -> tensor<5x5xi32>
  %r = stablehlo.reduce(%p init: %init) applies stablehlo.add across dimensions = [1] : (tensor<5x5xi32>, tensor<i32>) -> tensor<5xi32>
  return %r : tensor<5xi32>
}
// CHECK-LABEL: func.func @sum_interior(
// CHECK-DAG: %[[N:.*]] = stablehlo.constant dense<2> : tensor<i32>
// CHECK-DAG: %[[TOTAL:.*]] = stablehlo.constant dense<5> : tensor<i32>
// CHECK: %[[R:.*]] = stablehlo.reduce(%arg0 init: %arg2) applies stablehlo.add across dimensions = [1]
// CHECK: %[[M:.*]] = stablehlo.multiply %arg1, %[[N]] : tensor<i32>
// CHECK: %[[B:.*]] = stablehlo.broadcast_in_dim %[[M]], dims = []
// CHECK: %[[A:.*]] = stablehlo.add %[[R]], %[[B]]
// CHECK: %[[FULL:.*]] = stablehlo.multiply %arg1, %[[TOTAL]] : tensor<i32>
// CHECK: %[[F:.*]] = stablehlo.add %arg2, %[[FULL]] : tensor<i32>
// CHECK: stablehlo.pad %[[A]], %[[F]], low = [1], high = [1], interior = [1]

// An empty reduced axis has no fill elements: padded rows return init too.
func.func @empty_axis(%arg: tensor<1x0xf64>, %pv: tensor<f64>, %init: tensor<f64>) -> tensor<2xf64> {
  %p = stablehlo.pad %arg, %pv, low = [0, 0], high = [1, 0], interior = [0, 0] : (tensor<1x0xf64>, tensor<f64>) -> tensor<2x0xf64>
  %r = stablehlo.reduce(%p init: %init) applies stablehlo.minimum across dimensions = [1] : (tensor<2x0xf64>, tensor<f64>) -> tensor<2xf64>
  return %r : tensor<2xf64>
}
// CHECK-LABEL: func.func @empty_axis(

// CHECK: %[[R:.*]] = stablehlo.reduce(%arg0 init: %arg2) applies stablehlo.minimum across dimensions = [1]
// CHECK-NOT: stablehlo.minimum %[[R]]
// CHECK: stablehlo.pad %[[R]], %arg2, low = [0], high = [1], interior = [0]
