// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-generate-td{patterns=slice_dot_general<1>;slice_reshape_dot_general<1>},transform-interpreter,enzyme-hlo-remove-transform)" %s | FileCheck %s --check-prefix=FOCUSED
// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{passses=65535})" %s | FileCheck %s --check-prefix=ALL
// Nondivisible extents round up; all three helper dimension families and
// the reshape caller must preserve the verified output lengths.

// ALL-LABEL: func.func @lhs_even(
// FOCUSED-LABEL: func.func @lhs_even(
// FOCUSED: stablehlo.dot_general{{.*}} -> tensor<2xf64>
func.func @lhs_even(%a: tensor<5x2xf64>, %b: tensor<2xf64>) -> tensor<2xf64> {
  %d = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<5x2xf64>, tensor<2xf64>) -> tensor<5xf64>
  %s = stablehlo.slice %d [1:4:2] : (tensor<5xf64>) -> tensor<2xf64>
  return %s : tensor<2xf64>
}

// ALL-LABEL: func.func @rhs_even(
// FOCUSED-LABEL: func.func @rhs_even(
// FOCUSED: stablehlo.dot_general{{.*}} -> tensor<2xf64>
func.func @rhs_even(%a: tensor<2xf64>, %b: tensor<2x5xf64>) -> tensor<2xf64> {
  %d = stablehlo.dot_general %a, %b, contracting_dims = [0] x [0] : (tensor<2xf64>, tensor<2x5xf64>) -> tensor<5xf64>
  %s = stablehlo.slice %d [1:4:2] : (tensor<5xf64>) -> tensor<2xf64>
  return %s : tensor<2xf64>
}

// ALL-LABEL: func.func @batch_even(
// FOCUSED-LABEL: func.func @batch_even(
// FOCUSED: stablehlo.dot_general{{.*}} -> tensor<2xf64>
func.func @batch_even(%a: tensor<5x2xf64>, %b: tensor<5x2xf64>) -> tensor<2xf64> {
  %d = stablehlo.dot_general %a, %b, batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<5x2xf64>, tensor<5x2xf64>) -> tensor<5xf64>
  %s = stablehlo.slice %d [1:4:2] : (tensor<5xf64>) -> tensor<2xf64>
  return %s : tensor<2xf64>
}

// ALL-LABEL: func.func @reshape_even(
// FOCUSED-LABEL: func.func @reshape_even(
// FOCUSED: stablehlo.dot_general{{.*}} -> tensor<2x1xf64>
func.func @reshape_even(%a: tensor<5x2xf64>, %b: tensor<2x1xf64>) -> tensor<2xf64> {
  %d = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<5x2xf64>, tensor<2x1xf64>) -> tensor<5x1xf64>
  %r = stablehlo.reshape %d : (tensor<5x1xf64>) -> tensor<5xf64>
  %s = stablehlo.slice %r [1:4:2] : (tensor<5xf64>) -> tensor<2xf64>
  return %s : tensor<2xf64>
}

// ALL-LABEL: func.func @lhs_odd(
// FOCUSED-LABEL: func.func @lhs_odd(
// FOCUSED: stablehlo.dot_general{{.*}} -> tensor<3xf64>
func.func @lhs_odd(%a: tensor<5x2xf64>, %b: tensor<2xf64>) -> tensor<3xf64> {
  %d = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<5x2xf64>, tensor<2xf64>) -> tensor<5xf64>
  %s = stablehlo.slice %d [0:5:2] : (tensor<5xf64>) -> tensor<3xf64>
  return %s : tensor<3xf64>
}

// ALL-LABEL: func.func @rhs_odd(
// FOCUSED-LABEL: func.func @rhs_odd(
// FOCUSED: stablehlo.dot_general{{.*}} -> tensor<3xf64>
func.func @rhs_odd(%a: tensor<2xf64>, %b: tensor<2x5xf64>) -> tensor<3xf64> {
  %d = stablehlo.dot_general %a, %b, contracting_dims = [0] x [0] : (tensor<2xf64>, tensor<2x5xf64>) -> tensor<5xf64>
  %s = stablehlo.slice %d [0:5:2] : (tensor<5xf64>) -> tensor<3xf64>
  return %s : tensor<3xf64>
}

// ALL-LABEL: func.func @batch_odd(
// FOCUSED-LABEL: func.func @batch_odd(
// FOCUSED: stablehlo.dot_general{{.*}} -> tensor<3xf64>
func.func @batch_odd(%a: tensor<5x2xf64>, %b: tensor<5x2xf64>) -> tensor<3xf64> {
  %d = stablehlo.dot_general %a, %b, batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<5x2xf64>, tensor<5x2xf64>) -> tensor<5xf64>
  %s = stablehlo.slice %d [0:5:2] : (tensor<5xf64>) -> tensor<3xf64>
  return %s : tensor<3xf64>
}

// ALL-LABEL: func.func @reshape_odd(
// FOCUSED-LABEL: func.func @reshape_odd(
// FOCUSED: stablehlo.dot_general{{.*}} -> tensor<3x1xf64>
func.func @reshape_odd(%a: tensor<5x2xf64>, %b: tensor<2x1xf64>) -> tensor<3xf64> {
  %d = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<5x2xf64>, tensor<2x1xf64>) -> tensor<5x1xf64>
  %r = stablehlo.reshape %d : (tensor<5x1xf64>) -> tensor<5xf64>
  %s = stablehlo.slice %r [0:5:2] : (tensor<5xf64>) -> tensor<3xf64>
  return %s : tensor<3xf64>
}

// ALL-LABEL: func.func @lhs_stride_three(
// FOCUSED-LABEL: func.func @lhs_stride_three(
// FOCUSED: stablehlo.dot_general{{.*}} -> tensor<2xf64>
func.func @lhs_stride_three(%a: tensor<5x2xf64>, %b: tensor<2xf64>) -> tensor<2xf64> {
  %d = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<5x2xf64>, tensor<2xf64>) -> tensor<5xf64>
  %s = stablehlo.slice %d [1:5:3] : (tensor<5xf64>) -> tensor<2xf64>
  return %s : tensor<2xf64>
}

// ALL-LABEL: func.func @rhs_stride_three(
// FOCUSED-LABEL: func.func @rhs_stride_three(
// FOCUSED: stablehlo.dot_general{{.*}} -> tensor<2xf64>
func.func @rhs_stride_three(%a: tensor<2xf64>, %b: tensor<2x5xf64>) -> tensor<2xf64> {
  %d = stablehlo.dot_general %a, %b, contracting_dims = [0] x [0] : (tensor<2xf64>, tensor<2x5xf64>) -> tensor<5xf64>
  %s = stablehlo.slice %d [1:5:3] : (tensor<5xf64>) -> tensor<2xf64>
  return %s : tensor<2xf64>
}

// ALL-LABEL: func.func @batch_stride_three(
// FOCUSED-LABEL: func.func @batch_stride_three(
// FOCUSED: stablehlo.dot_general{{.*}} -> tensor<2xf64>
func.func @batch_stride_three(%a: tensor<5x2xf64>, %b: tensor<5x2xf64>) -> tensor<2xf64> {
  %d = stablehlo.dot_general %a, %b, batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<5x2xf64>, tensor<5x2xf64>) -> tensor<5xf64>
  %s = stablehlo.slice %d [1:5:3] : (tensor<5xf64>) -> tensor<2xf64>
  return %s : tensor<2xf64>
}

// ALL-LABEL: func.func @reshape_stride_three(
// FOCUSED-LABEL: func.func @reshape_stride_three(
// FOCUSED: stablehlo.dot_general{{.*}} -> tensor<2x1xf64>
func.func @reshape_stride_three(%a: tensor<5x2xf64>, %b: tensor<2x1xf64>) -> tensor<2xf64> {
  %d = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<5x2xf64>, tensor<2x1xf64>) -> tensor<5x1xf64>
  %r = stablehlo.reshape %d : (tensor<5x1xf64>) -> tensor<5xf64>
  %s = stablehlo.slice %r [1:5:3] : (tensor<5xf64>) -> tensor<2xf64>
  return %s : tensor<2xf64>
}

// ALL-LABEL: func.func @lhs_divisible(
// FOCUSED-LABEL: func.func @lhs_divisible(
// FOCUSED: stablehlo.dot_general{{.*}} -> tensor<2xf64>
func.func @lhs_divisible(%a: tensor<5x2xf64>, %b: tensor<2xf64>) -> tensor<2xf64> {
  %d = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<5x2xf64>, tensor<2xf64>) -> tensor<5xf64>
  %s = stablehlo.slice %d [0:4:2] : (tensor<5xf64>) -> tensor<2xf64>
  return %s : tensor<2xf64>
}

// ALL-LABEL: func.func @rhs_divisible(
// FOCUSED-LABEL: func.func @rhs_divisible(
// FOCUSED: stablehlo.dot_general{{.*}} -> tensor<2xf64>
func.func @rhs_divisible(%a: tensor<2xf64>, %b: tensor<2x5xf64>) -> tensor<2xf64> {
  %d = stablehlo.dot_general %a, %b, contracting_dims = [0] x [0] : (tensor<2xf64>, tensor<2x5xf64>) -> tensor<5xf64>
  %s = stablehlo.slice %d [0:4:2] : (tensor<5xf64>) -> tensor<2xf64>
  return %s : tensor<2xf64>
}

// ALL-LABEL: func.func @batch_divisible(
// FOCUSED-LABEL: func.func @batch_divisible(
// FOCUSED: stablehlo.dot_general{{.*}} -> tensor<2xf64>
func.func @batch_divisible(%a: tensor<5x2xf64>, %b: tensor<5x2xf64>) -> tensor<2xf64> {
  %d = stablehlo.dot_general %a, %b, batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<5x2xf64>, tensor<5x2xf64>) -> tensor<5xf64>
  %s = stablehlo.slice %d [0:4:2] : (tensor<5xf64>) -> tensor<2xf64>
  return %s : tensor<2xf64>
}

// ALL-LABEL: func.func @reshape_divisible(
// FOCUSED-LABEL: func.func @reshape_divisible(
// FOCUSED: stablehlo.dot_general{{.*}} -> tensor<2x1xf64>
func.func @reshape_divisible(%a: tensor<5x2xf64>, %b: tensor<2x1xf64>) -> tensor<2xf64> {
  %d = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<5x2xf64>, tensor<2x1xf64>) -> tensor<5x1xf64>
  %r = stablehlo.reshape %d : (tensor<5x1xf64>) -> tensor<5xf64>
  %s = stablehlo.slice %r [0:4:2] : (tensor<5xf64>) -> tensor<2xf64>
  return %s : tensor<2xf64>
}

// ALL-LABEL: func.func @lhs_unit(
// FOCUSED-LABEL: func.func @lhs_unit(
// FOCUSED: stablehlo.dot_general{{.*}} -> tensor<3xf64>
func.func @lhs_unit(%a: tensor<5x2xf64>, %b: tensor<2xf64>) -> tensor<3xf64> {
  %d = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<5x2xf64>, tensor<2xf64>) -> tensor<5xf64>
  %s = stablehlo.slice %d [1:4:1] : (tensor<5xf64>) -> tensor<3xf64>
  return %s : tensor<3xf64>
}

// ALL-LABEL: func.func @rhs_unit(
// FOCUSED-LABEL: func.func @rhs_unit(
// FOCUSED: stablehlo.dot_general{{.*}} -> tensor<3xf64>
func.func @rhs_unit(%a: tensor<2xf64>, %b: tensor<2x5xf64>) -> tensor<3xf64> {
  %d = stablehlo.dot_general %a, %b, contracting_dims = [0] x [0] : (tensor<2xf64>, tensor<2x5xf64>) -> tensor<5xf64>
  %s = stablehlo.slice %d [1:4:1] : (tensor<5xf64>) -> tensor<3xf64>
  return %s : tensor<3xf64>
}

// ALL-LABEL: func.func @batch_unit(
// FOCUSED-LABEL: func.func @batch_unit(
// FOCUSED: stablehlo.dot_general{{.*}} -> tensor<3xf64>
func.func @batch_unit(%a: tensor<5x2xf64>, %b: tensor<5x2xf64>) -> tensor<3xf64> {
  %d = stablehlo.dot_general %a, %b, batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<5x2xf64>, tensor<5x2xf64>) -> tensor<5xf64>
  %s = stablehlo.slice %d [1:4:1] : (tensor<5xf64>) -> tensor<3xf64>
  return %s : tensor<3xf64>
}

// ALL-LABEL: func.func @reshape_unit(
// FOCUSED-LABEL: func.func @reshape_unit(
// FOCUSED: stablehlo.dot_general{{.*}} -> tensor<3x1xf64>
func.func @reshape_unit(%a: tensor<5x2xf64>, %b: tensor<2x1xf64>) -> tensor<3xf64> {
  %d = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<5x2xf64>, tensor<2x1xf64>) -> tensor<5x1xf64>
  %r = stablehlo.reshape %d : (tensor<5x1xf64>) -> tensor<5xf64>
  %s = stablehlo.slice %r [1:4:1] : (tensor<5xf64>) -> tensor<3xf64>
  return %s : tensor<3xf64>
}

// ALL-LABEL: func.func @lhs_empty(
// FOCUSED-LABEL: func.func @lhs_empty(
func.func @lhs_empty(%a: tensor<5x2xf64>, %b: tensor<2xf64>) -> tensor<0xf64> {
  %d = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<5x2xf64>, tensor<2xf64>) -> tensor<5xf64>
  %s = stablehlo.slice %d [1:1:2] : (tensor<5xf64>) -> tensor<0xf64>
  return %s : tensor<0xf64>
}

// ALL-LABEL: func.func @rhs_empty(
// FOCUSED-LABEL: func.func @rhs_empty(
func.func @rhs_empty(%a: tensor<2xf64>, %b: tensor<2x5xf64>) -> tensor<0xf64> {
  %d = stablehlo.dot_general %a, %b, contracting_dims = [0] x [0] : (tensor<2xf64>, tensor<2x5xf64>) -> tensor<5xf64>
  %s = stablehlo.slice %d [1:1:2] : (tensor<5xf64>) -> tensor<0xf64>
  return %s : tensor<0xf64>
}

// ALL-LABEL: func.func @batch_empty(
// FOCUSED-LABEL: func.func @batch_empty(
func.func @batch_empty(%a: tensor<5x2xf64>, %b: tensor<5x2xf64>) -> tensor<0xf64> {
  %d = stablehlo.dot_general %a, %b, batching_dims = [0] x [0], contracting_dims = [1] x [1] : (tensor<5x2xf64>, tensor<5x2xf64>) -> tensor<5xf64>
  %s = stablehlo.slice %d [1:1:2] : (tensor<5xf64>) -> tensor<0xf64>
  return %s : tensor<0xf64>
}

// ALL-LABEL: func.func @reshape_empty(
// FOCUSED-LABEL: func.func @reshape_empty(
func.func @reshape_empty(%a: tensor<5x2xf64>, %b: tensor<2x1xf64>) -> tensor<0xf64> {
  %d = stablehlo.dot_general %a, %b, contracting_dims = [1] x [0] : (tensor<5x2xf64>, tensor<2x1xf64>) -> tensor<5x1xf64>
  %r = stablehlo.reshape %d : (tensor<5x1xf64>) -> tensor<5xf64>
  %s = stablehlo.slice %r [1:1:2] : (tensor<5xf64>) -> tensor<0xf64>
  return %s : tensor<0xf64>
}

// ALL-LABEL: func.func @batch_and_both_free(
// FOCUSED-LABEL: func.func @batch_and_both_free(
// FOCUSED: stablehlo.dot_general{{.*}} -> tensor<2x3x2xf64>
func.func @batch_and_both_free(%a: tensor<5x5x2xf64>, %b: tensor<5x2x5xf64>) -> tensor<2x3x2xf64> {
  %d = stablehlo.dot_general %a, %b, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<5x5x2xf64>, tensor<5x2x5xf64>) -> tensor<5x5x5xf64>
  %s = stablehlo.slice %d [1:4:2, 0:5:2, 2:5:2] : (tensor<5x5x5xf64>) -> tensor<2x3x2xf64>
  return %s : tensor<2x3x2xf64>
}
