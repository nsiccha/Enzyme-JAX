// RUN: enzymexlamlir-opt %s --enzyme-hlo-generate-td="patterns=convert_convert_float" --transform-interpreter --enzyme-hlo-remove-transform | FileCheck %s

// f64 -> f32 rounds, so the pair is not the identity.
func.func @narrow_then_widen(%x: tensor<3xf64>) -> tensor<3xf64> {
  %0 = stablehlo.convert %x : (tensor<3xf64>) -> tensor<3xf32>
  %1 = stablehlo.convert %0 : (tensor<3xf32>) -> tensor<3xf64>
  return %1 : tensor<3xf64>
}

// CHECK-LABEL: func.func @narrow_then_widen(
// CHECK-SAME:      %arg0: tensor<3xf64>) -> tensor<3xf64> {
// CHECK-NEXT:    %0 = stablehlo.convert %arg0 : (tensor<3xf64>) -> tensor<3xf32>
// CHECK-NEXT:    %1 = stablehlo.convert %0 : (tensor<3xf32>) -> tensor<3xf64>
// CHECK-NEXT:    return %1 : tensor<3xf64>
// CHECK-NEXT:  }

// Rounding to f32 first and to f16 second can differ from rounding to f16
// directly.
func.func @narrow_then_narrow(%x: tensor<3xf64>) -> tensor<3xf16> {
  %0 = stablehlo.convert %x : (tensor<3xf64>) -> tensor<3xf32>
  %1 = stablehlo.convert %0 : (tensor<3xf32>) -> tensor<3xf16>
  return %1 : tensor<3xf16>
}

// CHECK-LABEL: func.func @narrow_then_narrow(
// CHECK-SAME:      %arg0: tensor<3xf64>) -> tensor<3xf16> {
// CHECK-NEXT:    %0 = stablehlo.convert %arg0 : (tensor<3xf64>) -> tensor<3xf32>
// CHECK-NEXT:    %1 = stablehlo.convert %0 : (tensor<3xf32>) -> tensor<3xf16>
// CHECK-NEXT:    return %1 : tensor<3xf16>
// CHECK-NEXT:  }

// bf16 is as wide as f16 and has fewer significand bits.
func.func @same_width_other_format(%x: tensor<3xf16>) -> tensor<3xf16> {
  %0 = stablehlo.convert %x : (tensor<3xf16>) -> tensor<3xbf16>
  %1 = stablehlo.convert %0 : (tensor<3xbf16>) -> tensor<3xf16>
  return %1 : tensor<3xf16>
}

// CHECK-LABEL: func.func @same_width_other_format(
// CHECK-SAME:      %arg0: tensor<3xf16>) -> tensor<3xf16> {
// CHECK-NEXT:    %0 = stablehlo.convert %arg0 : (tensor<3xf16>) -> tensor<3xbf16>
// CHECK-NEXT:    %1 = stablehlo.convert %0 : (tensor<3xbf16>) -> tensor<3xf16>
// CHECK-NEXT:    return %1 : tensor<3xf16>
// CHECK-NEXT:  }

// f32 -> f64 is exact, so the pair is the identity.
func.func @widen_then_narrow(%x: tensor<3xf32>) -> tensor<3xf32> {
  %0 = stablehlo.convert %x : (tensor<3xf32>) -> tensor<3xf64>
  %1 = stablehlo.convert %0 : (tensor<3xf64>) -> tensor<3xf32>
  return %1 : tensor<3xf32>
}

// CHECK-LABEL: func.func @widen_then_narrow(
// CHECK-SAME:      %arg0: tensor<3xf32>) -> tensor<3xf32> {
// CHECK-NEXT:    return %arg0 : tensor<3xf32>
// CHECK-NEXT:  }

// f16 -> f32 is exact, so one conversion is left.
func.func @widen_then_widen(%x: tensor<3xf16>) -> tensor<3xf64> {
  %0 = stablehlo.convert %x : (tensor<3xf16>) -> tensor<3xf32>
  %1 = stablehlo.convert %0 : (tensor<3xf32>) -> tensor<3xf64>
  return %1 : tensor<3xf64>
}

// CHECK-LABEL: func.func @widen_then_widen(
// CHECK-SAME:      %arg0: tensor<3xf16>) -> tensor<3xf64> {
// CHECK-NEXT:    %0 = stablehlo.convert %arg0 : (tensor<3xf16>) -> tensor<3xf64>
// CHECK-NEXT:    return %0 : tensor<3xf64>
// CHECK-NEXT:  }

// The first conversion keeps the type.
func.func @same_type_then_widen(%x: tensor<3xf32>) -> tensor<3xf64> {
  %0 = stablehlo.convert %x : (tensor<3xf32>) -> tensor<3xf32>
  %1 = stablehlo.convert %0 : (tensor<3xf32>) -> tensor<3xf64>
  return %1 : tensor<3xf64>
}

// CHECK-LABEL: func.func @same_type_then_widen(
// CHECK-SAME:      %arg0: tensor<3xf32>) -> tensor<3xf64> {
// CHECK-NEXT:    %0 = stablehlo.convert %arg0 : (tensor<3xf32>) -> tensor<3xf64>
// CHECK-NEXT:    return %0 : tensor<3xf64>
// CHECK-NEXT:  }
