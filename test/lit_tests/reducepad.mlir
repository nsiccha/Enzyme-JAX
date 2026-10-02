// RUN: enzymexlamlir-opt --pass-pipeline="builtin.module(enzyme-hlo-opt{max_constant_expansion=1})" %s | FileCheck %s

module {
  func.func @main(%a : tensor<2x3x1xf32>, %b : tensor<f32>) -> tensor<6x1xf32> {
    %pv = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %pad = stablehlo.pad %a, %pv, low = [1, 2, 0], high = [3, 4, 0], interior = [0, 1, 0] : (tensor<2x3x1xf32>, tensor<f32>) -> tensor<6x11x1xf32>
    %conv = stablehlo.reduce(%pad init: %b) applies stablehlo.add across dimensions = [1] : (tensor<6x11x1xf32>, tensor<f32>) -> tensor<6x1xf32>
    return %conv : tensor<6x1xf32>
  }
}

// CHECK:  func.func @main(%arg0: tensor<2x3x1xf32>, %arg1: tensor<f32>) -> tensor<6x1xf32> {
// CHECK-NEXT:    %[[i1:.+]] = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [1] : (tensor<2x3x1xf32>, tensor<f32>) -> tensor<2x1xf32>
// CHECK-NEXT:    %[[i2:.+]] = stablehlo.pad %[[i1]], %arg1, low = [1, 0], high = [3, 0], interior = [0, 0] : (tensor<2x1xf32>, tensor<f32>) -> tensor<6x1xf32>
// CHECK-NEXT:    return %[[i2]] : tensor<6x1xf32>
// CHECK-NEXT:  }

module {
  func.func @main(%a : tensor<2x3x1xf32>, %b : tensor<f32>) -> tensor<6x1xf32> {
    %pv = stablehlo.constant dense<3.000000e+00> : tensor<f32>
    %pad = stablehlo.pad %a, %pv, low = [1, 2, 0], high = [3, 4, 0], interior = [0, 1, 0] : (tensor<2x3x1xf32>, tensor<f32>) -> tensor<6x11x1xf32>
    %conv = stablehlo.reduce(%pad init: %b) applies stablehlo.add across dimensions = [1] : (tensor<6x11x1xf32>, tensor<f32>) -> tensor<6x1xf32>
    return %conv : tensor<6x1xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<2x3x1xf32>, %arg1: tensor<f32>) -> tensor<6x1xf32> {
// CHECK-DAG: %[[ADDED:.+]] = stablehlo.constant dense<2.400000e+01> : tensor<2x1xf32>
// CHECK-DAG: %[[FULL:.+]] = stablehlo.constant dense<3.300000e+01> : tensor<f32>
// CHECK: %[[RED:.+]] = stablehlo.reduce(%arg0 init: %arg1) applies stablehlo.add across dimensions = [1] : (tensor<2x3x1xf32>, tensor<f32>) -> tensor<2x1xf32>
// CHECK: %[[SUM:.+]] = stablehlo.add %[[RED]], %[[ADDED]] : tensor<2x1xf32>
// CHECK: %[[FILL:.+]] = stablehlo.add %arg1, %[[FULL]] : tensor<f32>
// CHECK: %[[PAD:.+]] = stablehlo.pad %[[SUM]], %[[FILL]], low = [1, 0], high = [3, 0], interior = [0, 0] : (tensor<2x1xf32>, tensor<f32>) -> tensor<6x1xf32>
// CHECK: return %[[PAD]] : tensor<6x1xf32>
// CHECK: }

module {
  func.func @main(%arg0: tensor<3xf32>) -> tensor<27xf32> {
    %cst = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reshape %arg0 : (tensor<3xf32>) -> tensor<1x3xf32>
    %1 = stablehlo.pad %0, %cst_0, low = [2, 3], high = [24, 26], interior = [0, 0] : (tensor<1x3xf32>, tensor<f32>) -> tensor<27x32xf32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.maximum across dimensions = [1] : (tensor<27x32xf32>, tensor<f32>) -> tensor<27xf32>
    return %2 : tensor<27xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<3xf32>) -> tensor<27xf32> {
// CHECK-DAG: %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG: %[[BROADCAST:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<1xf32>
// CHECK-DAG: %[[INIT:.+]] = stablehlo.constant dense<0xFF800000> : tensor<f32>
// CHECK: %[[RESHAPE:.+]] = stablehlo.reshape %arg0 : (tensor<3xf32>) -> tensor<1x3xf32>
// CHECK: %[[RED:.+]] = stablehlo.reduce(%[[RESHAPE]] init: %[[INIT]]) applies stablehlo.maximum across dimensions = [1] : (tensor<1x3xf32>, tensor<f32>) -> tensor<1xf32>
// CHECK: %[[FOLD:.+]] = stablehlo.maximum %[[RED]], %[[BROADCAST]] : tensor<1xf32>
// CHECK: %[[PAD:.+]] = stablehlo.pad %[[FOLD]], %[[ZERO]], low = [2], high = [24], interior = [0] : (tensor<1xf32>, tensor<f32>) -> tensor<27xf32>
// CHECK: return %[[PAD]] : tensor<27xf32>
// CHECK: }

module {
  func.func @main(%arg0: tensor<3xf32>) -> tensor<27xf32> {
    %cst = stablehlo.constant dense<0x7F800000> : tensor<f32>
    %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %0 = stablehlo.reshape %arg0 : (tensor<3xf32>) -> tensor<1x3xf32>
    %1 = stablehlo.pad %0, %cst_0, low = [2, 3], high = [24, 26], interior = [0, 0] : (tensor<1x3xf32>, tensor<f32>) -> tensor<27x32xf32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.minimum across dimensions = [1] : (tensor<27x32xf32>, tensor<f32>) -> tensor<27xf32>
    return %2 : tensor<27xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<3xf32>) -> tensor<27xf32> {
// CHECK-DAG: %[[ZERO:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG: %[[BROADCAST:.+]] = stablehlo.constant dense<0.000000e+00> : tensor<1xf32>
// CHECK-DAG: %[[INIT:.+]] = stablehlo.constant dense<0x7F800000> : tensor<f32>
// CHECK: %[[RESHAPE:.+]] = stablehlo.reshape %arg0 : (tensor<3xf32>) -> tensor<1x3xf32>
// CHECK: %[[RED:.+]] = stablehlo.reduce(%[[RESHAPE]] init: %[[INIT]]) applies stablehlo.minimum across dimensions = [1] : (tensor<1x3xf32>, tensor<f32>) -> tensor<1xf32>
// CHECK: %[[FOLD:.+]] = stablehlo.minimum %[[RED]], %[[BROADCAST]] : tensor<1xf32>
// CHECK: %[[PAD:.+]] = stablehlo.pad %[[FOLD]], %[[ZERO]], low = [2], high = [24], interior = [0] : (tensor<1xf32>, tensor<f32>) -> tensor<27xf32>
// CHECK: return %[[PAD]] : tensor<27xf32>
// CHECK: }

module {
  func.func @main(%arg0: tensor<3xf32>) -> tensor<27xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %0 = stablehlo.reshape %arg0 : (tensor<3xf32>) -> tensor<1x3xf32>
    %1 = stablehlo.pad %0, %cst, low = [2, 3], high = [24, 26], interior = [0, 0] : (tensor<1x3xf32>, tensor<f32>) -> tensor<27x32xf32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.multiply across dimensions = [1] : (tensor<27x32xf32>, tensor<f32>) -> tensor<27xf32>
    return %2 : tensor<27xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<3xf32>) -> tensor<27xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.reshape %arg0 : (tensor<3xf32>) -> tensor<1x3xf32>
// CHECK-NEXT:     %1 = stablehlo.reduce(%0 init: %cst) applies stablehlo.multiply across dimensions = [1] : (tensor<1x3xf32>, tensor<f32>) -> tensor<1xf32>
// CHECK-NEXT:     %2 = stablehlo.pad %1, %cst, low = [2], high = [24], interior = [0] : (tensor<1xf32>, tensor<f32>) -> tensor<27xf32>
// CHECK-NEXT:     return %2 : tensor<27xf32>
// CHECK-NEXT: }

module {
  func.func @main(%arg0: tensor<3xf32>) -> tensor<27xf32> {
    %cst = stablehlo.constant dense<1.000000e+00> : tensor<f32>
    %cst_0 = stablehlo.constant dense<1.100000e+00> : tensor<f32>
    %0 = stablehlo.reshape %arg0 : (tensor<3xf32>) -> tensor<1x3xf32>
    %1 = stablehlo.pad %0, %cst_0, low = [2, 3], high = [24, 26], interior = [0, 0] : (tensor<1x3xf32>, tensor<f32>) -> tensor<27x32xf32>
    %2 = stablehlo.reduce(%1 init: %cst) applies stablehlo.multiply across dimensions = [1] : (tensor<27x32xf32>, tensor<f32>) -> tensor<27xf32>
    return %2 : tensor<27xf32>
  }
}

// CHECK: func.func @main(%arg0: tensor<3xf32>) -> tensor<27xf32> {
// CHECK-DAG: %[[ADDED:.+]] = stablehlo.constant dense<15.8631029> : tensor<1xf32>
// CHECK-DAG: %[[INIT:.+]] = stablehlo.constant dense<1.000000e+00> : tensor<f32>
// CHECK-DAG: %[[FULL:.+]] = stablehlo.constant dense<{{(21[.]113[0-9]+|2[.]1113[0-9]+e\+01)}}> : tensor<f32>
// CHECK: %[[RESHAPE:.+]] = stablehlo.reshape %arg0 : (tensor<3xf32>) -> tensor<1x3xf32>
// CHECK: %[[RED:.+]] = stablehlo.reduce(%[[RESHAPE]] init: %[[INIT]]) applies stablehlo.multiply across dimensions = [1] : (tensor<1x3xf32>, tensor<f32>) -> tensor<1xf32>
// CHECK: %[[PROD:.+]] = stablehlo.multiply %[[RED]], %[[ADDED]] : tensor<1xf32>
// CHECK: %[[PAD:.+]] = stablehlo.pad %[[PROD]], %[[FULL]], low = [2], high = [24], interior = [0] : (tensor<1xf32>, tensor<f32>) -> tensor<27xf32>
// CHECK: return %[[PAD]] : tensor<27xf32>
// CHECK: }
