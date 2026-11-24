// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops | FileCheck %s
// RUN: enzymexlamlir-opt %s --enzyme --canonicalize --remove-unnecessary-enzyme-ops --arith-raise | stablehlo-translate - --interpret

// With width = 4 the shadow of a tensor<2x3xf32> is a tensor<4x2x3xf32>, one
// direction per row, so reversing operand dimension 1 reverses shadow
// dimension 2.

func.func @reverse(%x : tensor<2x3xf32>) -> tensor<2x3xf32> {
  %y = stablehlo.reverse %x, dims = [1] : (tensor<2x3xf32>) -> tensor<2x3xf32>
  func.return %y : tensor<2x3xf32>
}

// CHECK:  func.func private @fwddiffe4reverse(%arg0: tensor<2x3xf32>, %arg1: tensor<4x2x3xf32>) -> tensor<4x2x3xf32> {
// CHECK-NEXT:    %0 = stablehlo.reverse %arg1, dims = [2] : tensor<4x2x3xf32>
// CHECK-NEXT:    return %0 : tensor<4x2x3xf32>
// CHECK-NEXT:  }

// CHECK:  func.func private @diffe4reverse(%arg0: tensor<2x3xf32>, %arg1: tensor<4x2x3xf32>) -> tensor<4x2x3xf32> {
// CHECK:    stablehlo.reverse %{{.+}}, dims = [2] : tensor<4x2x3xf32>
// CHECK:  }

func.func @main() {
  %input = stablehlo.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>

  %seed = stablehlo.constant dense<[
    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
    [[7.0, -8.0, 9.0], [0.5, 0.0, -1.0]],
    [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
    [[-2.0, 3.0, 5.0], [7.0, 11.0, 13.0]]
  ]> : tensor<4x2x3xf32>
  %expected = stablehlo.constant dense<[
    [[3.0, 2.0, 1.0], [6.0, 5.0, 4.0]],
    [[9.0, -8.0, 7.0], [-1.0, 0.0, 0.5]],
    [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    [[5.0, 3.0, -2.0], [13.0, 11.0, 7.0]]
  ]> : tensor<4x2x3xf32>

  %fwd = enzyme.fwddiff @reverse(%input, %seed) <{
    activity=[#enzyme.activity<enzyme_dup>],
    ret_activity=[#enzyme.activity<enzyme_dupnoneed>],
    width = 4 : i64
  }> : (tensor<2x3xf32>, tensor<4x2x3xf32>) -> tensor<4x2x3xf32>

  check.expect_almost_eq %fwd, %expected : tensor<4x2x3xf32>

  %rev = enzyme.autodiff @reverse(%input, %seed) <{
    activity=[#enzyme.activity<enzyme_active>],
    ret_activity=[#enzyme.activity<enzyme_activenoneed>],
    width = 4 : i64
  }> : (tensor<2x3xf32>, tensor<4x2x3xf32>) -> tensor<4x2x3xf32>

  check.expect_almost_eq %rev, %expected : tensor<4x2x3xf32>

  func.return
}
